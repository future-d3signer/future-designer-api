from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Optional
import base64
import json
import torch
import gc
import io
from diffusers import MarigoldDepthPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline, DPMSolverMultistepScheduler

app = FastAPI(title="Style Transfer API Server")

# Global variable for LLM to ensure single instance
llm = None
sam = None
dino = None

def get_pipeline():
    global llm
    if llm is None:
        llm = LLM(
            model="Qwen/Qwen2-VL-2B-Instruct",
            dtype=torch.bfloat16,
            gpu_memory_utilization=0.80,  # Reduced from 0.95 to prevent OOM
            max_model_len=4096,
            max_num_seqs=5,
        )
    return llm

def get_depth():
    global depth
    if depth is None:
        depth = MarigoldDepthPipeline.from_pretrained(
            "prs-eth/marigold-depth-lcm-v1-0", torch_dtype=torch.float16, variant="fp16").to("cuda")
    return depth


class MultimodalRequest(BaseModel):
    image_base64: Optional[str] = None
    style: Optional[str] = None

class GenerationResponse(BaseModel):
    imgae_final_base64: str

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")

@app.post("/generate", response_model=GenerationResponse)
async def generate_response(request: MultimodalRequest):
    try:
        if request.image_base64 is None:
            raise HTTPException(status_code=400, detail="Image base64 data is required.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(2024)
        
        pipeline = get_pipeline()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return GenerationResponse(text=)
    
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    get_pipeline()

@app.on_event("shutdown")
async def shutdown_event():
    # Clean up resources
    global llm
    if llm is not None:
        del llm
        llm = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False  # Changed to False to prevent memory issues with reloader
    )