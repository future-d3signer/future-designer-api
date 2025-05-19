from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MILVUS_URL: str
    MILVUS_TOKEN: str
    DINO_CONFIG_PATH: str = "/home/s464915/future-designer/experiments/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    DINO_WEIGHTS_PATH: str = "/home/s464915/future-designer/experiments/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    PROMPTS_FILE_PATH: str = "styles.json"
    SAM2_CHECKPOINT_PATH: str = "/home/s464915/future-designer/experiments/segment-anything-2/checkpoints/sam2.1_hiera_large.pt" 
    SAM2_MODEL_CONFIG: str = "configs/sam2.1/sam2.1_hiera_l.yaml" 

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'

settings = Settings()