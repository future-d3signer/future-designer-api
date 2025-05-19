import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

# Assuming SearchRequest and a potential SearchResponse are in app.schemas.search
from app.schemas.search import SearchRequest # Original schema
# If not, define it:
# class SearchRequest(BaseModel): type: str = ""; style: str = ""; ...

from app.services.milvus_service import MilvusService
from app.api.deps import get_milvus_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])


class SearchAPIResponse(BaseModel): # Define a response model for search results
    results: list # Or a more specific type like List[Dict[str, Any]]


@router.post("", response_model=SearchAPIResponse) # Changed path to align with original /search
async def search_similar_items_endpoint(
    req: SearchRequest,
    top_k: int = Query(5, ge=1, le=50, description="Number of top similar items to retrieve"),
    milvus_svc: MilvusService = Depends(get_milvus_service)
):
    """
    Search for similar items in the Milvus database based on provided criteria.
    """
    try:
        # MilvusService.search_similar expects SearchRequest and top_k
        search_results = milvus_svc.search_similar(req, top_k)
        return SearchAPIResponse(results=search_results["results"]) # Adapt to match SearchAPIResponse structure
    except Exception as e:
        logger.error(f"Milvus search error: {e}", exc_info=True)
        # Consider specific error codes for Milvus connection issues vs. bad queries
        raise HTTPException(status_code=500, detail=f"Search operation failed: {str(e)}")