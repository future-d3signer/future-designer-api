import gc
import json
import torch
import logging

from vllm import LLM
from PIL import Image
from app.core.config import settings 
from transformers import (
    pipeline,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
)
from diffusers import (
    TCDScheduler,
    AutoencoderKL,
    ControlNetUnionModel,
    StableDiffusionXLControlNetUnionPipeline,
    StableDiffusionXLControlNetUnionInpaintPipeline
)
from app.utils.segmentation_utils import load_sam_predictor 


logger = logging.getLogger(__name__)

class ModelProvider:
    def __init__(self):
        self._prompts: dict | None = None
        self._pipeline_control: StableDiffusionXLControlNetUnionPipeline | None = None
        self._pipeline_inpaint: StableDiffusionXLControlNetUnionInpaintPipeline | None = None
        self._depth_estimator = None
        self._vlm_model = None
        self._sam_predictor = None
        self._dino_model = None
        
        self.black_image = Image.new("RGB", (1024, 1024), (0, 0, 0))
        self.enhancement_prompt = "masterpiece, professional lighting, realistic materials, highly detailed"

    def load_all_models(self):
        logger.info("Loading all ML models...")
        self.get_prompts()
        self.get_diffusion_pipelines()
        self.get_depth_estimator()
        self.get_vlm_model()
        self.get_sam_predictor()
        self.get_dino_model()
        logger.info("All ML models loaded.")

    def get_prompts(self) -> dict:
        if self._prompts is None:
            logger.info("Loading style prompts...")
            try:
                with open(settings.PROMPTS_FILE_PATH, "r") as f:
                    self._prompts = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load prompts: {e}")
                raise RuntimeError(f"Failed to load style prompts: {e}")
        return self._prompts

    def get_diffusion_pipelines(self) -> tuple[StableDiffusionXLControlNetUnionPipeline, StableDiffusionXLControlNetUnionInpaintPipeline]:
        if self._pipeline_control is None or self._pipeline_inpaint is None:
            logger.info("Initializing diffusion pipelines...")

            controlnet = ControlNetUnionModel.from_pretrained(
                "OzzyGT/controlnet-union-promax-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
            ).to("cuda")
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
            self._pipeline_inpaint = StableDiffusionXLControlNetUnionInpaintPipeline.from_pretrained(
                "SG161222/RealVisXL_V5.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16, variant="fp16"
            ).to("cuda")
           
            self._pipeline_inpaint.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="sdxl_models",
                    weight_name="ip-adapter_sdxl_vit-h.safetensors",
                    image_encoder_folder="models/image_encoder",
                )
            self._pipeline_inpaint.set_ip_adapter_scale(0.6)
            self._pipeline_inpaint.scheduler = TCDScheduler.from_config(self._pipeline_inpaint.scheduler.config)
            self._pipeline_inpaint.load_lora_weights("h1t/TCD-SDXL-LoRA")
            self._pipeline_inpaint.fuse_lora()

            self._pipeline_control = StableDiffusionXLControlNetUnionPipeline.from_pipe(
                self._pipeline_inpaint, torch_dtype=torch.float16
            ).to("cuda")
            logger.info("Diffusion pipelines initialized.")
        return self._pipeline_control, self._pipeline_inpaint

    def get_depth_estimator(self):
        if self._depth_estimator is None:
            logger.info("Initializing Depth model...")
            self._depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")
        return self._depth_estimator

    def get_vlm_model(self):
        if self._vlm_model is None:
            logger.info("Initializing LLM model...")
            self._vlm_model = LLM(
                model="filnow/qwen-merged-lora",
                dtype=torch.bfloat16,
                gpu_memory_utilization=0.35, 
                max_model_len=1024,
                max_num_seqs=1
            )
        return self._vlm_model

    def get_sam_predictor(self):
        if self._sam_predictor is None:
            logger.info("Initializing SAM predictor...")
            self._sam_predictor = load_sam_predictor() 
        return self._sam_predictor

    def get_dino_model(self):
        if self._dino_model is None:
            logger.info("Initializing DINO model...")
            self._dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to("cuda")
            self._dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        return self._dino_model, self._dino_processor

    def cleanup(self):
        logger.info("Cleaning up models...")
        del self._pipeline_control
        del self._pipeline_inpaint
        del self._depth_estimator
        del self._vlm_model
        del self._dino_model
        del self._dino_processor
        del self._sam_predictor
        self._sam_predictor = None
        self._pipeline_control = None
        self._pipeline_inpaint = None
        self._depth_estimator = None
        self._vlm_model = None
        self._dino_model = None
        self._dino_processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Model cleanup complete.")
