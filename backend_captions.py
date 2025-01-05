import gc
import torch
import uvicorn
import logging

from enum import Enum
from typing import Dict
from vllm import LLM, SamplingParams
from contextlib import contextmanager
from pydantic import BaseModel, Field
from utils import ImageUtils, CaptionUtils
from fastapi import FastAPI, HTTPException
from diffusers import MarigoldDepthPipeline
from groundingdino.util.inference import load_model
from segmentation import get_segementaion, load_sam_model


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

class DepthRequest(BaseModel):
    source_image: str = Field(..., description="Base64 encoded image data")

class CaptionResponse(BaseModel):
    caption: Dict[str, FurnitureDescription]

class DepthResponse(BaseModel):
    depth_image: str

class ModelManager:
    def __init__(self):
        self._llm = None
        self._sam = None
        self._dino = None
        self._depth = None

    @property
    def llm(self):
        if self._llm is None:
            logger.info("Initializing LLM model")
            self._llm = LLM(
                model="Qwen/Qwen2-VL-2B-Instruct",
                dtype=torch.bfloat16,
                gpu_memory_utilization=0.80,
                max_model_len=4096,
                max_num_seqs=5,
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

    def cleanup(self):
        logger.info("Cleaning up models")
        for model_name in ['_llm', '_sam', '_dino', '_depth']:
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
    title="Multimodal vLLM API Server",
    description="API for furniture analysis and depth estimation",
    version="1.0.0"
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
            image = ImageUtils.decode_image(request.source_image)
            generator = torch.Generator(device="cuda").manual_seed(2024)
            
            # Generate depth map
            depth_output = model_manager.depth(
                image,
                generator=generator
            )
            
            # Convert depth prediction to visualization
            depth_image = model_manager.depth.image_processor.visualize_depth(
                depth_output.prediction,
                color_map="binary"
            )
            # Convert visualization to base64
            depth_base64 = ImageUtils.encode_image(depth_image[0])

            return DepthResponse(depth_image=depth_base64)
            
        except Exception as e:
            logger.error(f"Depth generation error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Depth generation failed: {str(e)}"
            )

@app.post(
    "/generate_captions",
    response_model=CaptionResponse,
    description="Generate furniture descriptions from input image"
)
async def generate_response(request: CaptionRequest):
    with cuda_memory_manager():
        try:
            image = ImageUtils.decode_image(request.source_image)
            masks = get_segementaion(
                image,
                model_manager.sam,
                model_manager.dino
            )

            sampling_params = SamplingParams(
                max_tokens=128,
                temperature=0.0
            )

            output_dict = {}
            for i, mask in enumerate(masks):
                image_base64 = ImageUtils.encode_image(mask)
                conversation = CaptionUtils.get_conversation_template(image_base64)
                
                output = model_manager.llm.chat(
                    conversation,
                    sampling_params=sampling_params
                )
                generated_text = output[0].outputs[0].text
                caption = CaptionUtils.parse_json_response(generated_text)
                
                output_dict[f"furniture_{i}"] = caption

            return CaptionResponse(caption=output_dict)

        except Exception as e:
            logger.error(f"Caption generation error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Caption generation failed: {str(e)}"
            )

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing models...")
    # Warm up models
    _ = model_manager.llm
    _ = model_manager.sam
    _ = model_manager.dino
    _ = model_manager.depth
    logger.info("Server startup complete")

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