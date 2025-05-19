import torch
import gc
from contextlib import contextmanager
from typing import Generator
from app.models.model_provider import ModelProvider
from app.services.image_service import ImageService
from app.services.milvus_service import MilvusService
from app.services.web_service import WebService
from fastapi import Depends

# Create a single instance of ModelProvider to be shared
# This assumes models are loaded once and are thread-safe for inference.
# VLLM is designed for concurrent requests. Diffusers pipelines might need careful handling
# if not inherently thread-safe or if internal state is modified during inference.
# For diffusers, it's generally safe if each call gets its own generator and doesn't modify pipe state.
_model_provider_instance = None

def get_model_provider() -> ModelProvider:
    global _model_provider_instance
    if _model_provider_instance is None:
        # This will only run once when the first request hits a route needing it,
        # or you can initialize it at app startup.
        _model_provider_instance = ModelProvider()
        # Eagerly load models if desired, otherwise they lazy load
        # _model_provider_instance.load_all_models() # Better to do this at app startup
    return _model_provider_instance

def get_image_service(model_provider: ModelProvider = Depends(get_model_provider)) -> ImageService:
    return ImageService(model_provider)

def get_milvus_service() -> MilvusService:
    # MilvusClient might be better as a singleton too if connection pooling is handled.
    # For simplicity, creating per request or per service instance is fine.
    return MilvusService()

def get_web_service() -> WebService:
    return WebService()

@contextmanager
def cuda_memory_manager_context() -> Generator[None, None, None]:
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# This can be used as a dependency
def CudaMemoryManagerDep() -> Generator[None, None, None]:
    with cuda_memory_manager_context():
        yield