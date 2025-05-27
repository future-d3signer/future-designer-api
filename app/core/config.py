from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MILVUS_URL: str
    MILVUS_TOKEN: str
    MILVUS_COLLECTION_NAME: str
    PROMPTS_FILE_PATH: str = "styles.json"
    EMBEDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    DINO_PROMPT: str = "chair. sofa. table. bed." 

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'

settings = Settings()