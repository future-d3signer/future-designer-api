import logging
from fastapi import APIRouter, HTTPException, Depends

# Assuming these are defined in app.schemas.image_processing
from app.schemas.image_processing import (
    DepthRequest as BaseDepthRequest, # Original schema
    DepthResponse,
    CaptionRequest as BaseCaptionRequest, # Original schema
    CaptionResponse,
    TransparencyRequest as BaseTransparencyRequest, # Original schema
    TransparencyResponse
)
# If not, define them:
# class BaseDepthRequest(BaseModel): source_image: str
# class BaseCaptionRequest(BaseModel): source_image: str
# class BaseTransparencyRequest(BaseModel): furniture_image: str # This is a URL suffix

from app.services.image_service import ImageService
from app.api.deps import get_image_service, CudaMemoryManagerDep
from app.utils.image_utils import ImageUtils # For potential decoding if not done in service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/image-analysis", tags=["Image Analysis"])


@router.post("/generate_depth", response_model=DepthResponse)
async def generate_depth_endpoint(
    request: BaseDepthRequest, # Client sends original image
    image_svc: ImageService = Depends(get_image_service),
    _cm: None = Depends(CudaMemoryManagerDep)
):
    """
    Generate depth map from an input image.
    The client must store the original source_image_b64 if it's needed for subsequent calls,
    as this endpoint only returns the depth_image_b64.
    """
    try:
        # ImageService.generate_depth_map returns (original_pil, depth_pil, depth_b64)
        _, _, depth_b64 = image_svc.generate_depth_map(request.source_image)
        return DepthResponse(depth_image=depth_b64)
    except Exception as e:
        logger.error(f"Depth generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Depth generation failed: {str(e)}")


@router.post("/generate_captions", response_model=CaptionResponse)
async def generate_captions_endpoint(
    request: BaseCaptionRequest,
    image_svc: ImageService = Depends(get_image_service),
    _cm: None = Depends(CudaMemoryManagerDep)
):
    """
    Generate furniture descriptions from an input image.
    """
    try:
        # ImageService.generate_captions_for_image expects source_image_b64
        captions_data = image_svc.generate_captions_for_image(request.source_image)
        return CaptionResponse(furniture=captions_data)
    except Exception as e:
        logger.error(f"Caption generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {str(e)}")


@router.post("/generate_transparency", response_model=TransparencyResponse)
async def generate_transparency_endpoint(
    request: BaseTransparencyRequest,
    image_svc: ImageService = Depends(get_image_service),
    _cm: None = Depends(CudaMemoryManagerDep)
):
    """
    Generate transparent background for a furniture image (fetched by URL suffix).
    """
    try:
        # ImageService.make_furniture_transparent expects furniture_image URL suffix
        transparent_image_b64 = image_svc.make_furniture_transparent(request.furniture_image)
        return TransparencyResponse(transparent_image=transparent_image_b64)
    except ValueError as e: # Specific error for "no furniture detected"
        logger.warning(f"Transparency generation validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Transparency generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transparency generation failed: {str(e)}")