import gc
import json
import torch
import uvicorn
import logging
import numpy as np

from enum import Enum
from typing import Dict
from utils import ImageUtils, CaptionUtils
from contextlib import contextmanager
from pydantic import BaseModel, Field
from diffusers import (
    ControlNetModel,
    MarigoldDepthPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionXLControlNetPipeline
)
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groundingdino.util.inference import load_model
from segmentation import get_segementaion, load_sam_model

# import oneflow as flow
# from onediff.infer_compiler import oneflow_compile


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FurnitureType(str, Enum):
    BED = "bed"
    CHAIR = "chair"
    TABLE = "table"
    SOFA = "sofa"

class FurnitureDescription(BaseModel):
    type: FurnitureType
    style: str
    color: str
    material: str
    shape: str
    details: str
    room_type: str
    price_range: str

class CaptionRequest(BaseModel):
    source_image: str = Field(..., description="Base64 encoded image data")

class StyleRequest(BaseModel):
    style_image: str = Field(..., description="Base64 encoded depth image")
    style: str = Field(..., description="Style identifier for the transfer")

class DepthRequest(BaseModel):
    source_image: str = Field(..., description="Base64 encoded image data")

class StyleResponse(BaseModel):
    generated_image: str = Field(..., description="Base64 encoded generated image")

class DepthResponse(BaseModel):
    depth_image: str

class FurnitureItem(BaseModel):
    caption: FurnitureDescription
    mask: str = Field(..., description="Base64 encoded mask image")

class CaptionResponse(BaseModel):
    furniture: Dict[str, FurnitureItem]

class ModelManager:
    def __init__(self):
        self._pipeline = None
        self._prompts = None
        self._depth = None
        self._llm = None
        self._sam = None
        self._dino = None

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
                    "SG161222/RealVisXL_V5.0_Lightning",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    controlnet=controlnet
                ).to("cuda")

                #self._pipeline.unet = oneflow_compile(self._pipeline.unet)
                
                # self._pipeline.unet = torch.compile(
                #     self._pipeline.unet,
                #     mode="reduce-overhead",
                #     fullgraph=True
                # )

                # Configure scheduler
                self._pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self._pipeline.scheduler.config,
                    use_karras_sigmas=True
                )

            except Exception as e:
                logger.error(f"Pipeline initialization failed: {str(e)}")
                raise RuntimeError(f"Failed to initialize pipeline: {str(e)}")

        return self._pipeline
    
    @property
    def depth(self):
        if self._depth is None:
            logger.info("Initializing Depth model")
            self._depth = MarigoldDepthPipeline.from_pretrained(
                "prs-eth/marigold-depth-lcm-v1-0",
                torch_dtype=torch.float16,
                variant="fp16"
            ).to("cuda")
        return self._depth

    @property
    def llm(self):
        if self._llm is None:
            logger.info("Initializing LLM model")
            self._llm = LLM(
                model="Qwen/Qwen2-VL-2B-Instruct",
                dtype=torch.bfloat16,
                gpu_memory_utilization=0.3,
                max_model_len=1024,
                max_num_seqs=1,
            )
        return self._llm

    @property
    def sam(self):
        if self._sam is None:
            logger.info("Initializing SAM model")
            self._sam = load_sam_model()
        return self._sam

    @property
    def dino(self):
        if self._dino is None:
            logger.info("Initializing DINO model")
            self._dino = load_model(
                "/home/s464915/future-designer/experiments/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                "/home/s464915/future-designer/experiments/GroundingDINO/weights/groundingdino_swint_ogc.pth"
            )
        return self._dino

    def cleanup(self):
        logger.info("Cleaning up models")
        for model_name in ['_llm', '_sam', '_dino', '_pipeline', '_depth']:
            if hasattr(self, model_name) and getattr(self, model_name) is not None:
                delattr(self, model_name)
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()

@app.post(
    "/generate_depth",
    response_model=DepthResponse,
    description="Generate depth map from input image"
)
async def generate_depth(request: DepthRequest):
    with cuda_memory_manager():
        try:
            image = ImageUtils.decode_image(request.source_image).resize((1024, 1024))
            generator = torch.Generator(device="cuda")
            
            # Generate depth map
            depth_output = model_manager.depth(
                image,
                generator=generator
            )
            
            depth_image = model_manager.depth.image_processor.visualize_depth(
                depth_output.prediction,
                color_map="binary" 
            )

            # Convert visualization to base64
            depth_base64 = ImageUtils.encode_image(depth_image[0])
            del depth_output

            return DepthResponse(depth_image=depth_base64)
            
        except Exception as e:
            logger.error(f"Depth generation error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Depth generation failed: {str(e)}"
            )
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
            
            depth_image = ImageUtils.decode_image(request.style_image)

            output = model_manager.pipeline(
                prompt=prompts[request.style],
                negative_prompt=prompts["negative"],
                guidance_scale=6.5,
                num_inference_steps=30,
                image=[depth_image],
                controlnet_conditioning_scale=0.7,
                control_guidance_end=0.7,
                generator=torch.Generator(device="cuda"),
            )

            generated_image = ImageUtils.encode_image(output.images[0])
            del output
            
            return StyleResponse(generated_image=generated_image)

        except Exception as e:
            logger.error(f"Style generation failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Style generation failed: {str(e)}"
            )
        
@app.post(
    "/generate_captions",
    response_model=CaptionResponse,
    description="Generate furniture descriptions from input image"
)
async def generate_response(request: CaptionRequest):
    with cuda_memory_manager():
        try:
            image = ImageUtils.decode_image(request.source_image).resize((1024, 1024))
            images_furniture, masks = get_segementaion(
                image,
                model_manager.sam,
                model_manager.dino
            )

            sampling_params = SamplingParams(
                max_tokens=128,
                temperature=0.0
            )
            
            output_dict = {}
            for i, furniture in enumerate(images_furniture):
                image_base64 = ImageUtils.encode_image(furniture)
                conversation = CaptionUtils.get_conversation_template(image_base64)

                mask_slice = masks[i, 0]
                mask_image = (mask_slice * 255).astype(np.uint8)
                mask_encoded = ImageUtils.encode_image(mask_image)
                
                output = model_manager.llm.chat(
                    conversation,
                    sampling_params=sampling_params
                )
                generated_text = output[0].outputs[0].text
                del output
                caption = CaptionUtils.parse_json_response(generated_text)
                
                output_dict[f"furniture_{i}"] = FurnitureItem(
                    caption=caption,
                    mask=mask_encoded
                )

            return CaptionResponse(furniture=output_dict)

        except Exception as e:
            logger.error(f"Caption generation error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Caption generation failed: {str(e)}"
            )

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up server...")
    try:
        model_manager.load_prompts()
        _ = model_manager.pipeline
        _ = model_manager.depth
        _ = model_manager.llm
        _ = model_manager.sam
        _ = model_manager.dino
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