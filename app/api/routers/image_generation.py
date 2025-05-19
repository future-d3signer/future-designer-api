import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

# Assuming these are defined in app.schemas.image_processing
from app.schemas.image_processing import (
    StyleResponse,
    # Base StyleRequest and ReplaceRequest might need to be redefined or extended for state
)
# For clarity, let's define the stateful request schemas here:
# These should ideally be in app.schemas.image_processing.py

class GenerateStyleRequest(BaseModel):
    style: str = Field(..., description="Style identifier for the transfer (e.g., key from styles.json)")
    depth_image_b64: str = Field(..., description="Base64 encoded depth map image")

class GenerateInpaintRequest(BaseModel):
    style_prompt: str = Field(..., description="Prompt describing the desired inpaint content or style")
    orginal_image_b64: str = Field(..., description="Base64 encoded original source image")
    depth_image_b64: str = Field(..., description="Base64 encoded depth map image")
    mask_image_b64: str = Field(..., description="Base64 encoded mask image (white is masked region)")

class GenerateDeleteRequest(BaseModel):
    orginal_image_b64: str = Field(..., description="Base64 encoded original source image")
    box_image_b64: str = Field(..., description="Base64 encoded mask image (object to delete)")

class GenerateReplaceRequest(BaseModel):
    style_prompt: str = Field(..., description="Prompt describing the desired replacement content or style")
    orginal_image_b64: str = Field(..., description="Base64 encoded original source image")
    mask_image_b64: str = Field(..., description="Base64 encoded mask image (region to replace)")
    adapter_image_name: str = Field(..., description="Identifier/URL suffix for the furniture image to adapt style from")
 

from app.services.image_service import ImageService
from app.api.deps import get_image_service, CudaMemoryManagerDep
from app.utils.image_utils import ImageUtils # For decoding b64 to PIL

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/image-generation", tags=["Image Generation"])


@router.post("/generate_style", response_model=StyleResponse)
async def generate_style_endpoint(
    request: GenerateStyleRequest,
    image_svc: ImageService = Depends(get_image_service),
    _cm: None = Depends(CudaMemoryManagerDep)
):
    """
    Apply an artistic style to an image using its depth map.
    Client must provide original image, depth map, and style key.
    """
    try:
        depth_pil = ImageUtils.decode_image(request.depth_image_b64)

        # ImageService.generate_styled_image expects style_key and depth_image_pil
        generated_b64 = image_svc.generate_styled_image(
            style_key=request.style,
            depth_image_pil=depth_pil
            # Pass original_pil if your service method was updated to use it
        )
        return StyleResponse(generated_image=generated_b64)
    except ValueError as e: # For specific errors like invalid style
        logger.warning(f"Style generation validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Style generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Style generation failed: {str(e)}")


@router.post("/generate_inpaint", response_model=StyleResponse)
async def generate_inpaint_endpoint(
    request: GenerateInpaintRequest,
    image_svc: ImageService = Depends(get_image_service),
    _cm: None = Depends(CudaMemoryManagerDep)
):
    """
    Inpaint a masked region of an image based on a prompt, guided by depth.
    """
    try:
        mask_image_pil = ImageUtils.decode_image(request.mask_image_b64) # style_image is the mask here
        depth_image_pil = ImageUtils.decode_image(request.depth_image_b64)
        orginal_image_pil = ImageUtils.decode_image(request.orginal_image_b64)
        # ImageService.generate_inpaint expects base_prompt, mask_b64, original_pil, depth_pil
        generated_b64 = image_svc.generate_inpaint(
            base_prompt=request.style_prompt,
            orginal_image_pil=orginal_image_pil,
            mask_image_pil=mask_image_pil,
            depth_image_pil=depth_image_pil
        )
        return StyleResponse(generated_image=generated_b64)
    except Exception as e:
        logger.error(f"Inpaint generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inpaint generation failed: {str(e)}")


@router.post("/generate_delete", response_model=StyleResponse)
async def generate_delete_endpoint(
    request: GenerateDeleteRequest,
    image_svc: ImageService = Depends(get_image_service),
    _cm: None = Depends(CudaMemoryManagerDep)
):
    """
    Remove/delete a masked object from an image, guided by depth.
    """
    try:
        orginal_pil = ImageUtils.decode_image(request.orginal_image_b64)
        box_image_pil = ImageUtils.decode_image(request.box_image_b64) # style_image is the mask here

        # ImageService.generate_delete expects mask_b64, original_pil, depth_pil
        generated_b64 = image_svc.generate_delete(
            box_image_pil=box_image_pil,
            orginal_image_pil=orginal_pil,
        )
        return StyleResponse(generated_image=generated_b64)
    except Exception as e:
        logger.error(f"Delete generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Delete generation failed: {str(e)}")


@router.post("/generate_replace", response_model=StyleResponse)
async def generate_replace_endpoint(
    request: GenerateReplaceRequest,
    image_svc: ImageService = Depends(get_image_service),
    _cm: None = Depends(CudaMemoryManagerDep)
):
    """
    Replace a masked region of an image with content influenced by an adapter image and prompt.
    """
    try:
        orginal_image_pil = ImageUtils.decode_image(request.orginal_image_b64)
        mask_image_pil = ImageUtils.decode_image(request.mask_image_b64) # style_image is the mask here

        # ImageService.generate_replace expects:
        # style_prompt, mask_image_b64, adapter_image_url_suffix, original_image_pil, depth_image_pil (optional)
        generated_b64 = image_svc.generate_replace(
            style_prompt=request.style_prompt,
            orginal_image_pil=orginal_image_pil,
            mask_image_pil=mask_image_pil,
            adapter_image_name=request.adapter_image_name,
        )
        return StyleResponse(generated_image=generated_b64)
    except Exception as e:
        logger.error(f"Replace generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Replace generation failed: {str(e)}")