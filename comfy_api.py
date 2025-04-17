"""
自定义 ComfyUI 接口方法
"""

import json
import uuid
import httpx
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from fastapi import status
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ===== 配置参数 =====
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "http://127.0.0.1:8188")
TIMEOUT = int(os.getenv("COMIFY_TIMEOUT", "300"))
CURRENT_WORKFLOW_FILE = os.getenv("CURRENT_WORKFLOW_FILE","flux_workflow.json")

# ===== 内存存储任务状态 =====
tasks = {}


# ===== 自定义异常类型 =====
class ServiceConnectionError(Exception):
    """服务连接异常"""

    pass


class ImageGenerationError(Exception):
    """生图逻辑异常"""

    pass


class DataStructureError(Exception):
    """响应数据结构异常"""

    pass


# 在HTTPException中映射错误类型
ERROR_MAPPING = {
    ServiceConnectionError: (status.HTTP_503_SERVICE_UNAVAILABLE, "服务连接失败"),
    ImageGenerationError: (status.HTTP_502_BAD_GATEWAY, "生图流程失败"),
    DataStructureError: (status.HTTP_500_INTERNAL_SERVER_ERROR, "数据解析失败"),
}


# ===== 数据模型 =====
class FluxParams(BaseModel):
    # 核心参数
    prompt: str = Field(..., min_length=1, example="A beautiful sunset")
    negative_prompt: str = ""
    width: int = Field(1024, ge=512, le=2048)
    height: int = Field(1024, ge=512, le=2048)
    seed: int = Field(-1, description="-1表示随机种子")
    steps: int = Field(20, ge=1, le=100)

    # 模型选择
    clip_t5_model: str = "t5xxl_fp8_e4m3fn.safetensors"
    clip_model: str = "clip_l.safetensors"
    unet_model: str = "flux1-dev-fp8.safetensors"
    vae_model: str = "ae.safetensors"

    # 采样参数
    sampler: str = "euler"
    scheduler: str = "simple"
    guidance_scale: float = Field(3.5, ge=0.0, le=10.0)
    base_shift: float = Field(0.5, ge=0.0, le=1.0)
    max_shift: float = Field(1.15, ge=1.0, le=2.0)

    # 翻译参数
    translate_source: str = "chinese (simplified)"
    translate_target: str = "english"
    translator: str = "GoogleTranslator [free]"


# ===== 核心工作流处理 =====
class WorkflowManager:
    def __init__(self):
        self.workflow = self._load_template()

    def _load_template(self):
        """加载工作流模板"""
        try:
            with open(CURRENT_WORKFLOW_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载工作流失败: {str(e)}")
            raise

    def update_workflow(self, params: FluxParams) -> dict:
        """动态更新工作流参数"""
        workflow = json.loads(json.dumps(self.workflow))  # 深拷贝

        # 设置节点参数
        node_config = {
            # 翻译节点
            "10": {
                "from_translate": params.translate_source,
                "to_translate": params.translate_target,
                "service": params.translator,
                "text": params.prompt,
            },
            # 双CLIP加载器
            "12": {"clip_name1": params.clip_t5_model, "clip_name2": params.clip_model},
            # UNET加载器
            "14": {"unet_name": params.unet_model},
            # Flux引导
            "15": {"guidance": params.guidance_scale},
            # 随机噪声
            "16": {
                "noise_seed": params.seed
                if params.seed != -1
                else self._generate_seed()
            },
            # Flux采样器
            "17": {
                "max_shift": params.max_shift,
                "base_shift": params.base_shift,
                "width": params.width,
                "height": params.height,
            },
            # 潜在空间
            "19": {"width": params.width, "height": params.height},
            # 采样器选择
            "22": {"sampler_name": params.sampler},
            # 调度器
            "23": {"scheduler": params.scheduler, "steps": params.steps},
            # VAE加载器
            "21": {"vae_name": params.vae_model},
        }

        # 应用参数更新
        for node_id, config in node_config.items():
            if node_id in workflow:
                workflow[node_id]["inputs"].update(config)
            else:
                logger.warning(f"节点 {node_id} 不存在于工作流模板中")

        return workflow

    def _generate_seed(self) -> int:
        """生成随机种子"""
        return int(uuid.uuid4().int % (10**18))


# ===== API端点 =====
@app.post("/generate")
async def generate_image(params: FluxParams):
    """提交生成任务"""
    try:
        # 初始化工作流管理器
        wf_manager = WorkflowManager()
        workflow = wf_manager.update_workflow(params)

        # 提交到ComfyUI
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            try:
                response = await client.post(
                    f"{COMFYUI_HOST}/prompt", json={"prompt": workflow}
                )
                response.raise_for_status()  # 自动处理4xx/5xx状态码
            except httpx.RequestError as e:
                logger.error(f"服务连接失败: {str(e)}")
                raise ServiceConnectionError(f"无法连接到ComfyUI: {str(e)}")
            except httpx.HTTPStatusError as e:
                logger.error(f"生图请求失败[状态码:{response.status_code}]")
                raise ImageGenerationError(f"ComfyUI返回错误: {e.response.text}")

        # 验证响应数据结构
        try:
            comfy_prompt_id = response.json()["prompt_id"]
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"响应数据结构异常: {str(e)}")
            raise DataStructureError("缺少prompt_id字段")

        # 记录任务
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "comfy_id": comfy_prompt_id,
            "status": "processing",
            "outputs": [],
        }
        return {"task_id": task_id}

    except Exception as e:
        # 统一错误处理
        error_type = type(e)
        status_code, detail = ERROR_MAPPING.get(
            error_type, (status.HTTP_500_INTERNAL_SERVER_ERROR, "内部服务错误")
        )
        logger.error(f"[{error_type.__name__}] {detail}: {str(e)}")
        raise HTTPException(status_code=status_code, detail=detail)


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """获取生成结果"""
    # 校验任务存在性
    if task_id not in tasks:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="任务不存在")

    task = tasks[task_id]

    try:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{COMFYUI_HOST}/history/{task['comfy_id']}"
                )
                response.raise_for_status()
            except httpx.RequestError as e:
                raise ServiceConnectionError(f"历史记录查询失败: {str(e)}")
            except httpx.HTTPStatusError as e:
                raise ImageGenerationError(f"历史记录异常: {e.response.text}")

        # 深度校验数据结构
        try:
            history = response.json()
            task_data = history[task["comfy_id"]]
            outputs = task_data["outputs"]

            parsed_outputs = []
            for node_id, node_data in outputs.items():
                if "images" not in node_data:
                    continue
                for img in node_data["images"]:
                    if not all(k in img for k in ("filename", "type", "subfolder")):
                        raise DataStructureError("图片字段缺失")
                    parsed_outputs.append(
                        {
                            "filename": img["filename"],
                            "type": img["type"],
                            "subfolder": img["subfolder"],
                            "url": f"local://output/{img['filename']}",
                        }
                    )

            task["status"] = "completed" if parsed_outputs else "failed"
            task["outputs"] = parsed_outputs
            return task

        except (KeyError, TypeError) as e:
            logger.error(f"历史数据解析失败: {str(e)}")
            raise DataStructureError("无效的历史记录结构")

    except Exception as e:
        # 统一错误处理
        error_type = type(e)
        status_code, detail = ERROR_MAPPING.get(
            error_type, (status.HTTP_500_INTERNAL_SERVER_ERROR, "内部服务错误")
        )
        logger.error(f"[{error_type.__name__}] {detail}: {str(e)}")
        raise HTTPException(status_code=status_code, detail=detail)


# ===== 运行服务 =====
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8088)
