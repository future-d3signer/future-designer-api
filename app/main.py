import logging

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.deps import get_model_provider
from fastapi.middleware.cors import CORSMiddleware 
from diffusers.utils.logging import set_verbosity as set_diffusers_verbosity
from app.api.routers import image_generation, image_analysis, search, utility


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
set_diffusers_verbosity(logging.ERROR) 

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up server and loading models...")
    model_provider = get_model_provider() 
    try:
        model_provider.load_all_models() 
        logger.info("Server startup complete. Models loaded.")
    except Exception as e:
        logger.error(f"Model loading during startup failed: {e}", exc_info=True)
        raise RuntimeError(f"Server startup failed due to model loading error: {e}")
    
    yield 

    # Shutdown
    logger.info("Shutting down server and cleaning up models...")
    if model_provider:
        model_provider.cleanup()
    logger.info("Server shutdown complete.")

app = FastAPI(
    title="Style Transfer API Server",
    description="API for applying artistic styles to images using depth maps",
    version="1.0.0",
    lifespan=lifespan 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(image_generation.router)
app.include_router(image_analysis.router) 
app.include_router(search.router)        
app.include_router(utility.router)       


@app.get("/")
async def root():
    return {"message": "Style Transfer API is running"}