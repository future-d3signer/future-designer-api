import torch

from app.core.config import settings
from app.schemas.search import SearchRequest
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker


class MilvusService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.client = MilvusClient(uri=settings.MILVUS_URL, token=settings.MILVUS_TOKEN)
        self.embedding_fn = SentenceTransformer(settings.EMBEDING_MODEL_NAME, device=self.device)
        self.client.load_collection(settings.MILVUS_COLLECTION_NAME)
    
    def normalize_vector(self, vector):
        norm = sum(x*x for x in vector) ** 0.5
        return [x/norm for x in vector] if norm > 0 else vector

    def search_similar(self, req: SearchRequest, top_k: int = 5) -> dict:
        query_vectors = {}
        weights = []
        field_weights = {
            "style": 1.0,
            "color": 1.0,
            "material": 0.8,
            "details": 0.6,
        }

        # Only process non-empty fields
        if req.style:
            vector = self.embedding_fn.encode([req.style])[0].tolist()
            query_vectors["style"] = (
                self.normalize_vector(vector),
                "vector_style",
            )
            weights.append(field_weights["style"])

        if req.color:
            query_vectors["color"] = (
                self.embedding_fn.encode([req.color])[0].tolist(),
                "vector_color",
            )
            weights.append(field_weights["color"])

        if req.material:
            query_vectors["material"] = (
                self.embedding_fn.encode([req.material])[0].tolist(),
                "vector_material",
            )
            weights.append(field_weights["material"])

        if req.details:
            query_vectors["details"] = (
                self.embedding_fn.encode([req.details])[0].tolist(),
                "vector_details",
            )
            weights.append(field_weights["details"])

        if not query_vectors:
            return {"ids": [], "results": []}

        base_param = {
            "param": {
                "metric_type": "IP",  
                "params": {
                    "nprobe": 32,  
                    "ef": top_k * 8,  
                },
            },
            "limit": 20,
        }

        if req.type:
            base_param["expr"] = f'type == "{req.type}"'

        search_requests = []
        for vector_data in query_vectors.values():
            vector, field_name = vector_data
            search_param = {"data": [vector], "anns_field": field_name, **base_param}
            search_requests.append(AnnSearchRequest(**search_param))

        if len(weights) > 0:
            weight_sum = sum(weights)
            normalized_weights = [w / weight_sum for w in weights]
            ranker = WeightedRanker(*normalized_weights)
        else:
            ranker = WeightedRanker(1.0)  

        merged_results = self.client.hybrid_search(
            collection_name=settings.MILVUS_COLLECTION_NAME,
            reqs=search_requests,
            ranker=ranker,
            limit=top_k,
            output_fields=["columns", "image_name"],
            timeout=10,  
        )

        results_list = []

        if merged_results and len(merged_results) > 0:
            for hit in merged_results[0]:
                results_list.append(
                    {
                        "id": hit["id"],
                        "distance": hit["distance"],
                        "entity": {
                            "columns": hit["entity"]["columns"],
                            "image_name": hit["entity"]["image_name"],
                        },
                    }
                )

        return {"results": results_list[:top_k]}
