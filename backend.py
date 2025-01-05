from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn
from typing import Optional
import base64
import json
import torch
import gc
from segmentation import get_segementaion, load_sam_model
import io
from groundingdino.util.inference import load_model
from diffusers import MarigoldDepthPipeline
from PIL import Image


app = FastAPI(title="Multimodal vLLM API Server")

llm = None
sam = None
dino = None
depth = None

def get_llm():
    global llm
    if llm is None:
        llm = LLM(
            model="Qwen/Qwen2-VL-2B-Instruct",
            dtype=torch.bfloat16,
            gpu_memory_utilization=0.80, 
            max_model_len=4096,
            max_num_seqs=5,
        )
    return llm

def get_sam():
    global sam
    if sam is None:
        sam = load_sam_model()
    return sam

def get_dino():
    global dino
    if dino is None:
        dino = load_model("/home/s464915/future-designer/experiments/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/s464915/future-designer/experiments/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    return dino

def get_depth():
    global depth
    if depth is None:
        depth = MarigoldDepthPipeline.from_pretrained(
            "prs-eth/marigold-depth-lcm-v1-0", torch_dtype=torch.float16, variant="fp16").to("cuda")
    return depth

class MultimodalRequest(BaseModel):
    image_base64: Optional[str] = None

class DepthRequest(BaseModel):
    image_base64: Optional[str] = None

class GenerationResponse(BaseModel):
    text: dict

class DepthResponse(BaseModel):
    image_final_base64: str

def convert_string_to_json(input_string: str) -> dict:
    try:
        if input_string.startswith("```json\n"):
            json_string = input_string.split("```json\n")[1].split("```")[0]
        else:
            json_string = input_string.replace('\n', '')
        return json.loads(json_string)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON parsing error: {str(e)}")
    
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")

def decode_image(image_base64):
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes))

@app.post("/generate_depth", response_model=DepthResponse)
async def generate_depth(request: DepthRequest):
    try:
        if request.image_base64 is None:
            raise HTTPException(status_code=400, detail="Image base64 data is required.")
        
        generator = torch.Generator(device="cuda").manual_seed(2024)
        image = decode_image(request.image_base64)
        depth = get_depth()

        depth_image = depth(image, generator=generator).prediction
        depth_image = depth.image_processor.visualize_depth(depth_image, color_map="binary")
        depth_image = encode_image(depth_image)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return DepthResponse(image_final_base64=depth_image)
    
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_captions", response_model=GenerationResponse)
async def generate_response(request: MultimodalRequest):
    try:
        if request.image_base64 is None:
            raise HTTPException(status_code=400, detail="Image base64 data is required.")
        
        image = decode_image(request.image_base64)

        model = get_llm()
        sam_model = get_sam()
        dino_model = get_dino()

        masks = get_segementaion(image, sam_model, dino_model)

        sampling_params = SamplingParams(
            max_tokens=128,
            temperature=0.0
        )

        output_dict = {}

        for i in range(0, len(masks)):
            image_base64 = encode_image(masks[i])

            conversation = [
                {
                    "role": "system",
                    "content": """You are a furniture expert. Analyze images and provide descriptions in this exact JSON format:
                    {
                        "type": "bed, chair, table, sofa",
                        "style": "overall style",
                        "color": "main color",
                        "material": "primary material",
                        "shape": "general shape",
                        "details": "one decorative feature",
                        "room_type": "room type",
                        "price_range": "price range in one word"
                    }
                    Only use the specified furniture types. Keep descriptions concise and factual."""
                },
                {
                    "role": "assistant",
                    "content": "I will analyze the image and respond with a valid JSON object following the exact schema."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        },
                        {
                            "type": "text",
                            "text": "Describe this furniture piece in JSON format."
                        }
                    ]
                }
            ]

            # Generate response
            output = model.chat(conversation, sampling_params=sampling_params)
            generated_text = output[0].outputs[0].text
            caption = convert_string_to_json(generated_text)

            output_dict[f"furniture_{i}"] = caption

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return GenerationResponse(text=output_dict)
    
    except Exception as e:
        # Clear memory on error too
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    get_llm()
    get_sam()
    get_dino()
    get_depth()

@app.on_event("shutdown")
async def shutdown_event():
    global llm
    global sam
    global dino
    global depth
    if llm is not None:
        del llm
        llm = None
    if sam is not None:
        del sam
        sam = None
    if dino is not None:
        del dino
        dino = None
    if depth is not None:
        del depth
        depth = None
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