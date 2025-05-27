import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from app.schemas.search import SearchRequest 
from app.services.milvus_service import MilvusService
from app.api.deps import get_milvus_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])


class SearchAPIResponse(BaseModel): 
    results: list 


@router.post("", response_model=SearchAPIResponse) 
async def search_similar_items_endpoint(
    req: SearchRequest,
    top_k: int = Query(5, ge=1, le=50, description="Number of top similar items to retrieve"),
    milvus_svc: MilvusService = Depends(get_milvus_service)
):
    try:
        search_results = milvus_svc.search_similar(req, top_k)
        return SearchAPIResponse(results=search_results["results"]) 
    except Exception as e:
        logger.error(f"Milvus search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search operation failed: {str(e)}")