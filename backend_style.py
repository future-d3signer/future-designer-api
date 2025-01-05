import gc
import json
import torch
import uvicorn
import logging

from typing import Dict
from utils import ImageUtils
from contextlib import contextmanager
from pydantic import BaseModel, Field
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionXLControlNetPipeline
)
from fastapi import FastAPI, HTTPException


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StyleRequest(BaseModel):
    depth_image: str = Field(..., description="Base64 encoded depth image")
    style: str = Field(..., description="Style identifier for the transfer")

class StyleResponse(BaseModel):
    generated_image: str = Field(..., description="Base64 encoded generated image")

class ModelManager:
    def __init__(self):
        self._pipeline = None
        self._prompts = None

    def load_prompts(self) -> Dict[str, str]:
        if self._prompts is None:
            logger.info("Loading style prompts")
            try:
                with open("styles.json", "r") as f:
                    self._prompts = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load prompts: {str(e)}")
                raise RuntimeError(f"Failed to load style prompts: {str(e)}")
        return self._prompts

    @property
    def pipeline(self):
        if self._pipeline is None:
            logger.info("Initializing style transfer pipeline")
            try:
                # Initialize ControlNet
                controlnet = ControlNetModel.from_pretrained(
                    "diffusers/controlnet-depth-sdxl-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16"
                ).to("cuda")

                # Initialize Pipeline
                self._pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                    "SG161222/RealVisXL_V4.0",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    controlnet=controlnet
                ).to("cuda")

                # Optimize pipeline
                self._pipeline.unet = torch.compile(
                    self._pipeline.unet,
                    mode="reduce-overhead",
                    fullgraph=True
                )

                # Configure scheduler
                self._pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self._pipeline.scheduler.config,
                    use_karras_sigmas=True
                )

            except Exception as e:
                logger.error(f"Pipeline initialization failed: {str(e)}")
                raise RuntimeError(f"Failed to initialize pipeline: {str(e)}")

        return self._pipeline

    def cleanup(self):
        logger.info("Cleaning up model resources")
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@contextmanager
def cuda_memory_manager():
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

app = FastAPI(
    title="Style Transfer API Server",
    description="API for applying artistic styles to images using depth maps",
    version="1.0.0"
)

model_manager = ModelManager()

@app.post(
    "/generate_style",
    response_model=StyleResponse,
    description="Generate styled image from depth map"
)
async def generate_style(request: StyleRequest):
    with cuda_memory_manager():
        try:
            prompts = model_manager.load_prompts()
            if request.style not in prompts:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid style: {request.style}"
                )
            
            depth_image = ImageUtils.decode_image(request.depth_image)

            generator = torch.Generator(device="cuda").manual_seed(2024)
            
            output = model_manager.pipeline(
                prompt=prompts[request.style],
                negative_prompt=prompts["negative"],
                guidance_scale=6.5,
                num_inference_steps=25,
                image=[depth_image],
                controlnet_conditioning_scale=0.7,
                control_guidance_end=0.7,
                generator=generator,
            )

            generated_image = ImageUtils.encode_image(output.images[0])
            
            return StyleResponse(generated_image=generated_image)

        except Exception as e:
            logger.error(f"Style generation failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Style generation failed: {str(e)}"
            )

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up server...")
    try:
        model_manager.load_prompts()
        _ = model_manager.pipeline
        logger.info("Server startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise RuntimeError(f"Server startup failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down server...")
    model_manager.cleanup()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )