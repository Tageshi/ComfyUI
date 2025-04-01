# comfy_api.py
import json
import uuid
import httpx
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ===== 配置参数 =====
COMFYUI_HOST = "http://127.0.0.1:8188"  # ComfyUI服务地址
WORKFLOW_FILE = "flux_workflow.json"    # 工作流模板
TIMEOUT = 300                           # 超时时间（秒）

# ===== 内存存储任务状态 =====
tasks = {}

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
            with open(WORKFLOW_FILE,encoding='utf-8') as f:
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
                "text": params.prompt
            },
            # 双CLIP加载器
            "12": {
                "clip_name1": params.clip_t5_model,
                "clip_name2": params.clip_model
            },
            # UNET加载器
            "14": {
                "unet_name": params.unet_model
            },
            # Flux引导
            "15": {
                "guidance": params.guidance_scale
            },
            # 随机噪声
            "16": {
                "noise_seed": params.seed if params.seed != -1 else self._generate_seed()
            },
            # Flux采样器
            "17": {
                "max_shift": params.max_shift,
                "base_shift": params.base_shift,
                "width": params.width,
                "height": params.height
            },
            # 潜在空间
            "19": {
                "width": params.width,
                "height": params.height
            },
            # 采样器选择
            "22": {
                "sampler_name": params.sampler
            },
            # 调度器
            "23": {
                "scheduler": params.scheduler,
                "steps": params.steps
            },
            # VAE加载器
            "21": {
                "vae_name": params.vae_model
            }
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
        
        # 生成工作流
        workflow = wf_manager.update_workflow(params)
        
        # 提交到ComfyUI
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{COMFYUI_HOST}/prompt",
                json={"prompt": workflow}
            )
            
        if response.status_code != 200:
            raise HTTPException(502, "ComfyUI服务不可用")
            
        # 记录任务
        task_id = str(uuid.uuid4())
        comfy_prompt_id = response.json()["prompt_id"]
        
        tasks[task_id] = {
            "comfy_id": comfy_prompt_id,
            "status": "processing",
            "outputs": []
        }
        
        return {"task_id": task_id}
        
    except Exception as e:
        logger.error(f"生成失败: {str(e)}")
        raise HTTPException(500, f"生成失败: {str(e)}")

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """获取生成结果"""
    if task_id not in tasks:
        raise HTTPException(404, "任务不存在")
    
    task = tasks[task_id]
    
    try:
        # 查询ComfyUI历史记录
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{COMFYUI_HOST}/history/{task['comfy_id']}"
            )
            
        if response.status_code != 200:
            raise HTTPException(502, "查询ComfyUI失败")
            
        history = response.json()
        
        # 解析输出结果
        outputs = []
        for node_id, node_data in history.get(task['comfy_id'], {}).get("outputs", {}).items():
            if "images" in node_data:
                for img in node_data["images"]:
                    outputs.append({
                        "filename": img["filename"],
                        "type": img["type"],
                        "subfolder": img["subfolder"],
                        # 后续替换为阿里云OSS路径
                        "url": f"local://output/{img['filename']}"
                    })
        
        # 更新任务状态
        task["status"] = "completed" if outputs else "failed"
        task["outputs"] = outputs
        
        return task
    
    except Exception as e:
        logger.error(f"获取结果失败: {str(e)}")
        raise HTTPException(500, f"获取结果失败: {str(e)}")

# ===== 运行服务 =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)