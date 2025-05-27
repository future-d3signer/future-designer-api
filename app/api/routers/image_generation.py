import logging

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from app.schemas.image_processing import (
    StyleResponse,
)


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
from app.utils.image_utils import ImageUtils 

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/image-generation", tags=["Image Generation"])


@router.post("/generate_style", response_model=StyleResponse)
async def generate_style_endpoint(
    request: GenerateStyleRequest,
    image_svc: ImageService = Depends(get_image_service),
    _cm: None = Depends(CudaMemoryManagerDep)
):
    try:
        depth_pil = ImageUtils.decode_image(request.depth_image_b64)

        generated_b64 = image_svc.generate_styled_image(
            style_key=request.style,
            depth_image_pil=depth_pil
        )
        return StyleResponse(generated_image=generated_b64)
    except ValueError as e: 
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
    try:
        mask_image_pil = ImageUtils.decode_image(request.mask_image_b64) 
        depth_image_pil = ImageUtils.decode_image(request.depth_image_b64)
        orginal_image_pil = ImageUtils.decode_image(request.orginal_image_b64)
     
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
    try:
        orginal_pil = ImageUtils.decode_image(request.orginal_image_b64)
        box_image_pil = ImageUtils.decode_image(request.box_image_b64) 

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
    try:
        orginal_image_pil = ImageUtils.decode_image(request.orginal_image_b64)
        mask_image_pil = ImageUtils.decode_image(request.mask_image_b64) 

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