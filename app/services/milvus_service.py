from pymilvus import MilvusClient, model
from app.core.config import settings
from app.schemas.search import SearchRequest # Assuming you moved SearchRequest here


class MilvusService:
    def __init__(self):
        self.client = MilvusClient(uri=settings.MILVUS_URL, token=settings.MILVUS_TOKEN)
        self.collection_name = "furniture_synthetic_dataset_10k" # Or make this configurable
        self.embedding_fn = model.DefaultEmbeddingFunction()
        # Consider loading the collection once if it's always the same
        self.client.load_collection(self.collection_name)

    def search_similar(self, req: SearchRequest, top_k: int = 5) -> dict:
        #self.client.load_collection(self.collection_name) # Or load in __init__ if always used

        vector_style = self.embedding_fn.encode_queries([req.style])[0].tolist()
        vector_details = self.embedding_fn.encode_queries([req.details])[0].tolist()
        vector_material = self.embedding_fn.encode_queries([req.material])[0].tolist()
        vector_color = self.embedding_fn.encode_queries([req.color])[0].tolist()
    

        search_results_style = self.client.search(
            collection_name=self.collection_name,
            data=[vector_style],
            anns_field="vector_style",
            limit=top_k,
            filter=f'type == "{req.type}"',
            output_fields=["columns", "image_name"],
        )

        search_results_color = self.client.search(
            collection_name=self.collection_name,
            data=[vector_color],
            anns_field="vector_color",
            limit=top_k,
            filter=f'type == "{req.type}"',
            output_fields=["columns", "image_name"],
        )

        search_results_material = self.client.search(
            collection_name=self.collection_name,
            data=[vector_material],
            anns_field="vector_material",
            limit=top_k,
            filter=f'type == "{req.type}"',
            output_fields=["columns", "image_name"],
        )

        search_results_details = self.client.search(
            collection_name=self.collection_name,
            data=[vector_details],
            anns_field="vector_details",
            limit=top_k,
            filter=f'type == "{req.type}"',
            output_fields=["columns", "image_name"],
        )

        # Combine results
        combined_results = []
        for result in [
            search_results_style,
            search_results_color,
            search_results_material,
            search_results_details,
        ]:
            for hit in result[0]:
                combined_results.append(
                    {"id": hit["id"], "distance": hit["distance"], "entity": hit["entity"]}
                )
        
        combined_results.sort(key=lambda x: x["distance"], reverse=True) # Or False for closer is better
        self.client.release_collection(self.collection_name) # Good practice
        return {"results": combined_results[:top_k]}
