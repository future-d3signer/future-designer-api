import gc
import torch

from fastapi import Depends
from typing import Generator
from contextlib import contextmanager
from app.services.web_service import WebService
from app.models.model_provider import ModelProvider
from app.services.image_service import ImageService
from app.services.milvus_service import MilvusService


_model_provider_instance = None

def get_model_provider() -> ModelProvider:
    global _model_provider_instance
    if _model_provider_instance is None:
        _model_provider_instance = ModelProvider()
        
        _model_provider_instance.load_all_models() 
    return _model_provider_instance

def get_image_service(model_provider: ModelProvider = Depends(get_model_provider)) -> ImageService:
    return ImageService(model_provider)

def get_milvus_service() -> MilvusService:
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


def CudaMemoryManagerDep() -> Generator[None, None, None]:
    with cuda_memory_manager_context():
        yield