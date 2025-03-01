import gc
import json
import torch
import uvicorn
import logging
import numpy as np


from transformers import pipeline
from PIL import Image

from enum import Enum
from typing import Dict
from utils import ImageUtils, CaptionUtils
from contextlib import contextmanager
from pydantic import BaseModel, Field
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    AutoencoderKL,
    LCMScheduler,
    UNet2DConditionModel,
    StableDiffusionXLControlNetInpaintPipeline
)
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groundingdino.util.inference import load_model
from diffusers.utils.logging import set_verbosity
from segmentation import get_segementaion, load_sam_model

from diffusers.image_processor import IPAdapterMaskProcessor

import requests
from PIL import Image
import io
# import oneflow as flow
# from onediff.infer_compiler import oneflow_compile


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
set_verbosity(logging.ERROR) 

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

class ReplaceRequest(BaseModel):
    style: str = Field(..., description="Style identifier for the transfer")
    mask_image: str = Field(..., description="Base64 encoded mask image")
    adapter_image: str = Field(..., description="Base64 encoded furniture image")

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
        self._pipeline_control = None
        self._pipeline_inpaint = None
        self._pipeline_replace = None
        self._prompts = None
        self._depth = None
        self._llm = None
        self._sam = None
        self._dino = None
        self._current_image = None
        self._current_depth = None

    def set_current_image(self, image):
        self._current_image = image
    
    def set_current_depth(self, depth):
        self._current_depth = depth

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
        if self._pipeline_control is None and self._pipeline_inpaint is None:
            logger.info("Initializing style transfer pipeline")
            try:
                controlnet = ControlNetModel.from_pretrained(
                    "diffusers/controlnet-depth-sdxl-1.0",
                    #"destitech/controlnet-inpaint-dreamer-sdxl",
                    torch_dtype=torch.float16,
                    variant="fp16"
                ).to("cuda")

                # unet = UNet2DConditionModel.from_pretrained(
                #     "latent-consistency/lcm-sdxl",
                #     torch_dtype=torch.float16,
                #     variant="fp16",
                # )

                # self._pipeline_control = StableDiffusionXLControlNetPipeline.from_pretrained(
                #     #"SG161222/RealVisXL_V5.0_Lightning",
                #     "RunDiffusion/Juggernaut-XL-v9",
                #     #"SG161222/RealVisXL_V3.0_Turbo",
                #     #"stabilityai/sdxl-turbo",
                #     #"stabilityai/stable-diffusion-xl-base-1.0",
                #     torch_dtype=torch.float16,
                #     variant="fp16",
                #     controlnet=controlnet,
                # ).to("cuda")

                self._pipeline_inpaint = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                    "SG161222/RealVisXL_V5.0_Lightning", controlnet=controlnet, torch_dtype=torch.float16
                    ).to("cuda")
        

                self._pipeline_inpaint.scheduler = LCMScheduler.from_config(
                    self._pipeline_inpaint.scheduler.config
                )

                self._pipeline_inpaint.load_lora_weights("latent-consistency/lcm-lora-sdxl")
                self._pipeline_inpaint.fuse_lora()


                self._pipeline_control = StableDiffusionXLControlNetPipeline.from_pipe(
                    self._pipeline_inpaint,
                    torch_dtype=torch.float16,
                ).to("cuda")

                # self._pipeline_replace.load_ip_adapter(
                #     "h94/IP-Adapter",
                #     subfolder="sdxl_models",
                #     weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                #     image_encoder_folder="models/image_encoder",
                # )

                # self._pipeline_replace.set_ip_adapter_scale(0.4)

                #self._pipeline_control.unet = orginal_unet

                # self._pipeline_inpaint.unload_ip_adapter()
                # self._pipeline_control.unload_ip_adapter()

            except Exception as e:
                logger.error(f"Pipeline initialization failed: {str(e)}")
                raise RuntimeError(f"Failed to initialize pipeline: {str(e)}")

        return self._pipeline_control
    
    @property
    def pipeline_inpaint(self):
        return self._pipeline_inpaint
    
    @property
    def pipeline_control(self):
        return self._pipeline_control
    
    @property
    def pipeline_replace(self):
        return self._pipeline_replace
    
    @property
    def depth(self):
        if self._depth is None:
            logger.info("Initializing Depth model")
            self._depth = pipeline(task="depth-estimation", 
                                   model="depth-anything/Depth-Anything-V2-Small-hf")
        return self._depth

    @property
    def llm(self):
        if self._llm is None:
            logger.info("Initializing LLM model")
            self._llm = LLM(
                # model="Qwen/Qwen2-VL-2B-Instruct",
                model="filnow/qwen-merged-lora",
                dtype=torch.bfloat16,
                gpu_memory_utilization=0.4,
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
        for model_name in ['_llm', '_sam', '_dino', '_pipeline_control', '_depth']:
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
    "/generate_inpaint",
    response_model=StyleResponse,
    description="Generate inpainted image from input image"
)
async def generate_inpaint(request: StyleRequest):
    with cuda_memory_manager():
        try:
            if model_manager._current_image is None or model_manager._current_depth is None:
                raise HTTPException(
                    status_code=400,
                    detail="Original image and depth map must be generated first"
            )
            
            base_prompt = request.style
            enhancement_prompt = "masterpiece, professional lighting, realistic materials, highly detailed"
            full_prompt = f"{base_prompt}, {enhancement_prompt}"
            image = ImageUtils.decode_image(request.style_image)

            seed = torch.randint(0, 100000, (1,)).item()

            padded_mask = ImageUtils.add_mask_padding(image, padding=30)


            blured_image = model_manager.pipeline_inpaint.mask_processor.blur(padded_mask, blur_factor=15)

            negative_prompt = "deformed, low quality, blurry, noise, grainy, duplicate, watermark, text, out of frame"

            output = model_manager.pipeline_inpaint(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                image=model_manager._current_image,
                mask_image=blured_image,
                num_inference_steps=8,
                control_image = model_manager._current_depth,
                #strength=0.99,
                guidance_scale=1.5,
                #ip_adapter_image=model_manager._current_image,
                #cross_attention_scale=1.0,
                generator=torch.Generator(device="cuda"),
                controlnet_conditioning_scale=0.7,
                control_guidance_end=0.7,
                #padding_mask_crop=5,
                # guidance_rescale=0.5,
                # original_inference_steps=50,  # Original model steps
                # denoising_end=1.0
            )

            generated_image = ImageUtils.encode_image(output.images[0])
            del output
            
            return StyleResponse(generated_image=generated_image)

        except Exception as e:
            logger.error(f"Inpaint generation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Inpaint generation failed: {str(e)}"
            )

@app.post(
    "/generate_delete",
    response_model=StyleResponse,
    description="Generate inpainted image from input image"
)
async def generate_delete(request: StyleRequest):
    with cuda_memory_manager():
        try:
            if model_manager._current_image is None or model_manager._current_depth is None:
                raise HTTPException(
                    status_code=400,
                    detail="Original image and depth map must be generated first"
                )
            
            # Use prompts that encourage creating empty space
            base_prompt = "empty space, clean room"
            enhancement_prompt = "continuous floor, clean walls, no furniture, empty area, consistent with surroundings"
            full_prompt = f"{base_prompt}, {enhancement_prompt}"
            image = ImageUtils.decode_image(request.style_image)

            seed = torch.randint(0, 100000, (1,)).item()

            # Add extra padding to ensure proper blending with surroundings
            padded_mask = ImageUtils.add_mask_padding(image, padding=100)
            
            # Use more blur for smoother transitions between original and inpainted areas
            blured_image = model_manager.pipeline_inpaint.mask_processor.blur(padded_mask, blur_factor=20)

            # Stronger negative prompt to avoid generating new objects
            negative_prompt="furniture, decor, objects, items, artifacts, clutter, anything, stuff"
            
            output = model_manager.pipeline_inpaint(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                image=model_manager._current_image,
                mask_image=blured_image,
                num_inference_steps=10,  # Increased for better quality
                control_image=model_manager._current_depth,
                guidance_scale=3.5,  # Increased to better follow the prompt
                generator=torch.Generator(device="cuda").manual_seed(seed),  # Use the seed for reproducibility
            )

            generated_image = ImageUtils.encode_image(output.images[0])
            del output
            
            return StyleResponse(generated_image=generated_image)

        except Exception as e:
            logger.error(f"Delete furniture generation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Delete furniture generation failed: {str(e)}"
            )
        
@app.post(
    "/generate_replace",
    response_model=StyleResponse,
    description="Generate inpainted image from input image"
)
async def generate_replace(request: ReplaceRequest):
    with cuda_memory_manager():
        try:
            if model_manager._current_image is None or model_manager._current_depth is None:
                raise HTTPException(
                    status_code=400,
                    detail="Original image and depth map must be generated first"
            )
            
            base_prompt = request.style
            enhancement_prompt = "masterpiece, professional lighting, realistic materials, highly detailed"
            full_prompt = f"{base_prompt}, {enhancement_prompt}"

            mask_image = ImageUtils.decode_image(request.mask_image)

            processor = IPAdapterMaskProcessor()
            ip_masks = processor.preprocess(mask_image, height=1024, width=1024)

            seed = torch.randint(0, 100000, (1,)).item()

            padded_mask = ImageUtils.add_mask_padding(mask_image, padding=30)

            adapter_image_path = f"https://similarimages.blob.core.windows.net/fd1new/{request.adapter_image}"

            response = requests.get(adapter_image_path)
            load_adapter_image = Image.open(io.BytesIO(response.content))

            blured_image = model_manager.pipeline_inpaint.mask_processor.blur(padded_mask, blur_factor=15)

            negative_prompt = "deformed, low quality, blurry, noise, grainy, duplicate, watermark, text, out of frame"

            output = model_manager.pipeline_replace(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                image=model_manager._current_image,
                mask_image=blured_image,
                num_inference_steps=8,
                strength=0.99,
                guidance_scale=2.5,
                ip_adapter_image=load_adapter_image,
                #cross_attention_scale=1.0,
                generator=torch.Generator(device="cuda"),
                #padding_mask_crop=5,
                guidance_rescale=0.5,
                original_inference_steps=50,  # Original model steps
                denoising_end=1.0,
                cross_attention_kwargs={"ip_adapter_masks": ip_masks},
            )

            generated_image = ImageUtils.encode_image(output.images[0])
            del output
            
            return StyleResponse(generated_image=generated_image)

        except Exception as e:
            logger.error(f"Inpaint generation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Inpaint generation failed: {str(e)}"
            )

@app.post(
    "/generate_depth",
    response_model=DepthResponse,
    description="Generate depth map from input image"
)
async def generate_depth(request: DepthRequest):
    with cuda_memory_manager():
        try:
            image = ImageUtils.decode_image(request.source_image).resize((1024, 1024))

            model_manager.set_current_image(image)

            depth = model_manager.depth(image)["depth"]

            model_manager.set_current_depth(depth)

            depth_base64 = ImageUtils.encode_image(depth)
            del depth
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
            
            #depth_image = ImageUtils.decode_image(request.style_image)

            output = model_manager.pipeline_control(
                prompt=prompts[request.style],
                negative_prompt=prompts["negative"],
                guidance_scale=1.5,
                num_inference_steps=10,
                image=[model_manager._current_depth],
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