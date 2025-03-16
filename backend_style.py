import gc
import os
import io
import json
import torch
import uvicorn
import logging
import requests
import numpy as np


from transformers import pipeline
from PIL import Image, ImageStat, ImageFilter, ImageOps
from enum import Enum
from typing import Dict
from utils import ImageUtils, CaptionUtils
from contextlib import contextmanager
from pydantic import BaseModel, Field
from diffusers import (
    AutoencoderKL,
    LCMScheduler,
    ControlNetUnionModel,
    StableDiffusionXLControlNetUnionPipeline,
    StableDiffusionXLControlNetUnionInpaintPipeline,
    TCDScheduler
)
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groundingdino.util.inference import load_model
from diffusers.utils.logging import set_verbosity
from segmentation import get_segementaion, load_sam_model
from bs4 import BeautifulSoup
from diffusers.image_processor import IPAdapterMaskProcessor
from pymilvus import MilvusClient, model
from dotenv import load_dotenv
import base64

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
set_verbosity(logging.ERROR) 


class SearchRequest(BaseModel):
    type: str = ""
    style: str = ""
    color: str = ""
    material: str = ""
    shape: str = ""
    details: str = ""
    room_type: str = ""
    price_range: str = ""

class URLRequest(BaseModel):
    url: str

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
    box: str = Field(..., description="Base64 encoded mask image")
    furniture_image: str = Field(..., description="Base64 encoded furniture image")

class CaptionResponse(BaseModel):
    furniture: Dict[str, FurnitureItem]

class ModelManager:
    def __init__(self):
        self._pipeline_control = None
        self._pipeline_inpaint = None
        self._prompts = None
        self._depth = None
        self._llm = None
        self._sam = None
        self._dino = None
        self._current_image = None
        self._current_depth = None
        self._black_image = Image.new("RGB", (1024, 1024), (0, 0, 0))
        self._enchancment_prompt = "masterpiece, professional lighting, realistic materials, highly detailed"

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
                controlnet = ControlNetUnionModel.from_pretrained(
                    "OzzyGT/controlnet-union-promax-sdxl-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16",
                ).to("cuda")

                vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", 
                                                    torch_dtype=torch.float16).to("cuda")

                self._pipeline_inpaint = StableDiffusionXLControlNetUnionInpaintPipeline.from_pretrained(
                    "SG161222/RealVisXL_V5.0", 
                    controlnet=controlnet,
                    vae=vae, 
                    torch_dtype=torch.float16,
                    variant="fp16"
                ).to("cuda")

                self._pipeline_inpaint.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="sdxl_models",
                    weight_name="ip-adapter_sdxl_vit-h.safetensors",
                    image_encoder_folder="models/image_encoder",
                )

                self._pipeline_inpaint.set_ip_adapter_scale(0.4)
        
                self._pipeline_inpaint.scheduler = TCDScheduler.from_config(
                    self._pipeline_inpaint.scheduler.config
                )
                self._pipeline_inpaint.load_lora_weights("h1t/TCD-SDXL-LoRA")

                self._pipeline_inpaint.fuse_lora()

                self._pipeline_control = StableDiffusionXLControlNetUnionPipeline.from_pipe(
                    self._pipeline_inpaint,
                    torch_dtype=torch.float16
                ).to("cuda")

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
                model="filnow/qwen-merged-lora",
                dtype=torch.bfloat16,
                gpu_memory_utilization=0.35,
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

#MILUV CONFIG
milvus_uri = os.getenv("MILVUS_URL")
token = os.getenv("MILVUS_TOKEN")
collection_name = "furniture_synthetic_dataset_v2"
milvus_client = MilvusClient(uri=milvus_uri, token=token)

embedding_fn = model.DefaultEmbeddingFunction()

relevant_columns = ['type', 
                    'style', 
                    'color', 
                    'material', 
                    'shape', 
                    'details', 
                    'room_type', 
                    'price_range'
                ]

@app.post("/search")
def search_similar_items(req: SearchRequest, top_k: int = 5):
    milvus_client.load_collection(collection_name)

    vector_style = embedding_fn.encode_queries([req.style])[0].tolist()
    vector_color = embedding_fn.encode_queries([req.color])[0].tolist()
    vector_material = embedding_fn.encode_queries([req.material])[0].tolist()
    vector_details = embedding_fn.encode_queries([req.details])[0].tolist()

    search_results_style = milvus_client.search(
        collection_name=collection_name,
        data=[vector_style],
        anns_field="vector_style",
        limit=top_k,
        filter=f'type == "{req.type}"',
        output_fields=["columns", "image_name"],
    )

    search_results_color = milvus_client.search(
        collection_name=collection_name,
        data=[vector_color],
        anns_field="vector_color",
        limit=top_k,
        filter=f'type == "{req.type}"',
        output_fields=["columns", "image_name"],
    )

    search_results_material = milvus_client.search(
        collection_name=collection_name,
        data=[vector_material],
        anns_field="vector_material",
        limit=top_k,
        filter=f'type == "{req.type}"',
        output_fields=["columns", "image_name"],
    )

    search_results_details = milvus_client.search(
        collection_name=collection_name,
        data=[vector_details],
        anns_field="vector_details",
        limit=top_k,
        filter=f'type == "{req.type}"',
        output_fields=["columns", "image_name"],
    )

    # Combine results
    combined_results = []
    for result in [
        search_results_style,
        search_results_color,
        search_results_material,
        search_results_details,
    ]:
        for hit in result[0]:
            combined_results.append(
                {"id": hit["id"], "distance": hit["distance"], "entity": hit["entity"]}
            )

    # Sort combined results by distance
    combined_results.sort(key=lambda x: x["distance"], reverse=True)
    return {"results": combined_results[:top_k]}

@app.post("/proxy-image")
async def proxy_image(request: dict):
    url = request.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
        
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.otodom.pl/"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        return {"image": image_base64}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image: {str(e)}")


@app.post("/scrape-images")
async def scrape_images(request: URLRequest):
    url = request.url
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        gallery_div = soup.find('div', class_='css-bbh9aa elvndys0')
        if gallery_div:
            image_tags = gallery_div.find_all('img')
            image_links = [img['src'] for img in image_tags if 'src' in img.attrs]
            return {"image_links": image_links}
        else:
            raise HTTPException(status_code=404, detail="No image gallery thumbnails found.")
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to retrieve the webpage.")

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
            full_prompt = f"{base_prompt}, {model_manager._enchancment_prompt}"
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
                num_inference_steps=5,
                control_image=[model_manager._current_depth],
                guidance_scale=1.5,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                controlnet_conditioning_scale=0.7,
                control_guidance_end=0.7,
                control_mode=[1],
                eta=0.3,
                strength=0.99,
                ip_adapter_image=model_manager._black_image
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
                raise HTTPException(status_code=400, detail="Original image and depth map must be generated first")

            mask_image = ImageUtils.decode_image(request.style_image)
            padded_mask = ImageUtils.add_mask_padding(mask_image, padding=64)

            blurred_mask = padded_mask.filter(ImageFilter.GaussianBlur(radius=15))  
            binary_mask = blurred_mask.point(lambda x: 0 if x > 127 else 255) 

            original_array = np.array(model_manager._current_image)
            mask_array = np.array(binary_mask) == 0
            original_array[mask_array] = [0, 0, 0]
            result_image = Image.fromarray(original_array)

            mask_for_stats = padded_mask.convert("L")  # Ensure single-channel
            inverted_mask = ImageOps.invert(mask_for_stats)  # Black (masked) -> white, white (unmasked) -> black
            stats = ImageStat.Stat(model_manager._current_image, mask=inverted_mask)
            avg_color = tuple(int(c) for c in stats.mean[:3])  # RGB average of unmasked area
            neutral_fill = Image.new("RGB", model_manager._current_image.size, avg_color)
            neutral_image = Image.composite(model_manager._current_image, neutral_fill, binary_mask)

            negative_prompt = "furniture, objects, decorations, plants, clutter, people"
            generator = torch.Generator(device="cuda").manual_seed(torch.randint(0, 100000, (1,)).item())

            output = model_manager.pipeline_inpaint(
                prompt=model_manager._enchancment_prompt,
                negative_prompt=negative_prompt,
                image=neutral_image,  
                mask_image=blurred_mask,
                control_image=[result_image],  
                control_mode=[7],
                num_inference_steps=8,  
                guidance_scale=1.5,    
                generator=generator,
                eta=0.3,               
                strength=0.99,          
                controlnet_conditioning_scale=1.0,  
                ip_adapter_image=model_manager._black_image
            )

            generated_image = ImageUtils.encode_image(output.images[0])
            del output

            return StyleResponse(generated_image=generated_image)

        except Exception as e:
            logger.error(f"Delete generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Delete generation failed: {str(e)}")

        
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
            response = requests.get(f"https://futuredesigner.blob.core.windows.net/futuredesigner1/{request.adapter_image}")
            load_adapter_image = Image.open(io.BytesIO(response.content))
            
            full_prompt = f"{request.style}, {model_manager._enchancment_prompt}"

            mask_image = ImageUtils.decode_image(request.mask_image)
            padded_mask = ImageUtils.add_mask_padding(mask_image, padding=30)

            processor = IPAdapterMaskProcessor()
            ip_masks = processor.preprocess(padded_mask, height=1024, width=1024)

            binary_mask = padded_mask.point(lambda x: 0 if x > 127 else 255)
            
            original_array = np.array(model_manager._current_image)
            mask_array = np.array(binary_mask) == 0
            original_array[mask_array] = [0, 0, 0]
            result_image = Image.fromarray(original_array)

            negative_prompt = "deformed, low quality, blurry, noise, grainy, duplicate, watermark, text, out of frame"

            output = model_manager.pipeline_inpaint(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                image=model_manager._current_image,
                mask_image=padded_mask,
                num_inference_steps=5,
                guidance_scale=2.0,
                ip_adapter_image=load_adapter_image,
                generator=torch.Generator(device="cuda").manual_seed(torch.randint(0, 100000, (1,)).item()),
                strength=0.99,
                cross_attention_kwargs={"ip_adapter_masks": ip_masks},
                control_image=[result_image],
                controlnet_conditioning_scale=0.3,
                control_guidance_end=0.3,
                eta=0.3,
                control_mode=[6]
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

            output = model_manager.pipeline_control(
                prompt=prompts[request.style],
                negative_prompt=prompts["negative"],
                width=1024,
                height=1024,
                guidance_scale=1.5,
                num_inference_steps=7,
                control_image=[model_manager._current_depth],
                controlnet_conditioning_scale=0.9,
                control_guidance_end=0.9,
                control_mode=[1],
                generator=torch.Generator(device="cuda"),
                eta=0.3,
                ip_adapter_image=model_manager._black_image,
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
            images_furniture, masks, boxes = get_segementaion(
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
                
                x0, y0, x1, y1 = boxes[i].int().cpu().numpy()

                box_image = np.zeros((1024, 1024), dtype=np.uint8)
                box_image[y0:y1, x0:x1] = 255  # Fill bounding box with white

                mask_slice = masks[i, 0]
                mask_image = (mask_slice * 255).astype(np.uint8)
                
                mask_encoded = ImageUtils.encode_image(mask_image)
                box_encoded = ImageUtils.encode_image(box_image)
                
                output = model_manager.llm.chat(
                    conversation,
                    sampling_params=sampling_params
                )
                generated_text = output[0].outputs[0].text
                del output
                caption = CaptionUtils.parse_json_response(generated_text)
                
                output_dict[f"furniture_{i}"] = FurnitureItem(
                    caption=caption,
                    mask=mask_encoded,
                    box=box_encoded,
                    furniture_image=image_base64
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