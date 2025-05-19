import os
from pymilvus import MilvusClient, model
from app.core.config import settings
from app.schemas.search import SearchRequest # Assuming you moved SearchRequest here

class MilvusService:
    def __init__(self):
        self.client = MilvusClient(uri=settings.MILVUS_URL, token=settings.MILVUS_TOKEN)
        self.collection_name = "furniture_synthetic_dataset_v2" # Or make this configurable
        self.embedding_fn = model.DefaultEmbeddingFunction()
        # Consider loading the collection once if it's always the same
        self.client.load_collection(self.collection_name)

    def search_similar(self, req: SearchRequest, top_k: int = 5) -> dict:
        #self.client.load_collection(self.collection_name) # Or load in __init__ if always used

        vector_style = self.embedding_fn.encode_queries([req.style])[0].tolist()
        vector_details = self.embedding_fn.encode_queries([req.details])[0].tolist()
        vector_material = self.embedding_fn.encode_queries([req.material])[0].tolist()
        vector_color = self.embedding_fn.encode_queries([req.color])[0].tolist()
    

        search_params = [
            ("vector_style", vector_style),
            ("vector_color", vector_color), # Assuming other fields might be direct filters or need encoding
            ("vector_material", vector_material),
            ("vector_details", vector_details),
        ]
        
        combined_results = []
        # Simplified search loop example
        for anns_field, data_vector in search_params:
            if not data_vector: continue # Skip if empty query for a field
            
            # Ensure data_vector is list of lists for Milvus search
            search_data = [data_vector] if isinstance(data_vector[0], float) else data_vector

            results = self.client.search(
                collection_name=self.collection_name,
                data=search_data, # Milvus expects list of query vectors
                anns_field=anns_field,
                limit=top_k,
                filter=f'type == "{req.type}"',
                output_fields=["columns", "image_name"], # Adjust as needed
            )
            for hit_list in results: # results is a list of lists of hits
                for hit in hit_list:
                    combined_results.append(
                        {"id": hit["id"], "distance": hit["distance"], "entity": hit["entity"]}
                    )
        
        combined_results.sort(key=lambda x: x["distance"], reverse=True) # Or False for closer is better
        self.client.release_collection(self.collection_name) # Good practice
        return {"results": combined_results[:top_k]}