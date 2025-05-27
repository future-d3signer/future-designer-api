import logging

from app.schemas.image_processing import (
    DepthRequest,
    DepthResponse,
    CaptionRequest, 
    CaptionResponse,
    TransparencyRequest, 
    TransparencyResponse
)
from app.services.image_service import ImageService
from fastapi import APIRouter, HTTPException, Depends
from app.api.deps import get_image_service, CudaMemoryManagerDep


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/image-analysis", tags=["Image Analysis"])


@router.post("/generate_depth", response_model=DepthResponse)
async def generate_depth_endpoint(
    request: DepthRequest, 
    image_svc: ImageService = Depends(get_image_service),
    _cm: None = Depends(CudaMemoryManagerDep)
):
    try:
        _, _, depth_b64 = image_svc.generate_depth_map(request.source_image)
        return DepthResponse(depth_image=depth_b64)
    except Exception as e:
        logger.error(f"Depth generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Depth generation failed: {str(e)}")


@router.post("/generate_captions", response_model=CaptionResponse)
async def generate_captions_endpoint(
    request: CaptionRequest,
    image_svc: ImageService = Depends(get_image_service),
    _cm: None = Depends(CudaMemoryManagerDep)
):
    try:
        captions_data = image_svc.generate_captions_for_image(request.source_image)
        return CaptionResponse(furniture=captions_data)
    except Exception as e:
        logger.error(f"Caption generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {str(e)}")


@router.post("/generate_transparency", response_model=TransparencyResponse)
async def generate_transparency_endpoint(
    request: TransparencyRequest,
    image_svc: ImageService = Depends(get_image_service),
    _cm: None = Depends(CudaMemoryManagerDep)
):
    try:
        transparent_image_b64 = image_svc.make_furniture_transparent(request.furniture_image)
        return TransparencyResponse(transparent_image=transparent_image_b64)
    except ValueError as e: 
        logger.warning(f"Transparency generation validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Transparency generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transparency generation failed: {str(e)}")