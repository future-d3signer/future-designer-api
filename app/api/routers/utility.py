import logging
import requests
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from app.schemas.search import URLRequest 
from app.schemas.image_processing import CompositeRequest 


from app.services.web_service import WebService
from app.services.image_service import ImageService 
from app.api.deps import get_web_service, get_image_service, CudaMemoryManagerDep

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/utility", tags=["Utility"])


class ProxyImageRequest(BaseModel): 
    url: str

class ProxyImageResponse(BaseModel):
    image: str 

class ScrapeResponse(BaseModel):
    image_links: List[str]

class CompositeResponse(BaseModel):
    composited_image: str 


@router.post("/proxy-image", response_model=ProxyImageResponse)
async def proxy_image_endpoint(
    request: ProxyImageRequest, 
    web_svc: WebService = Depends(get_web_service)
):
    if not request.url:
        raise HTTPException(status_code=400, detail="URL is required")
    try:
        image_base64 = web_svc.proxy_image(request.url)
        return ProxyImageResponse(image=image_base64)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching image for proxy: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Error fetching image from external URL: {str(e)}")
    except Exception as e:
        logger.error(f"Proxy image error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Proxy image failed: {str(e)}")


@router.post("/scrape-images", response_model=ScrapeResponse)
async def scrape_images_endpoint(
    request: URLRequest,
    web_svc: WebService = Depends(get_web_service)
):
    try:
        image_links = web_svc.scrape_image_links(request.url)
        if not image_links: 
             raise HTTPException(status_code=404, detail="No image gallery thumbnails found.")
        return ScrapeResponse(image_links=image_links)
    except ValueError as e: 
        logger.warning(f"Scraping validation error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching page for scraping: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Failed to retrieve the webpage: {str(e)}")
    except Exception as e:
        logger.error(f"Scrape images error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scrape images failed: {str(e)}")


@router.post("/composite_furniture", response_model=CompositeResponse)
async def composite_furniture_endpoint(
    request: CompositeRequest,
    image_svc: ImageService = Depends(get_image_service),
    _cm: None = Depends(CudaMemoryManagerDep)
):
    try:
        composited_image_data_url = image_svc.composite_and_blend_furniture(
            room_image_b64=request.room_image,
            furniture_image_b64=request.furniture_image,
            position=request.position,
            size=request.size
        )
        return CompositeResponse(composited_image=composited_image_data_url)
    except Exception as e:
        logger.error(f"Furniture composition failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Furniture composition failed: {str(e)}")