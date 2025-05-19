import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings # Ensure settings are loaded early
from app.api.routers import image_generation, image_analysis, search, utility
from app.api.deps import get_model_provider # For startup/shutdown
from diffusers.utils.logging import set_verbosity as set_diffusers_verbosity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
set_diffusers_verbosity(logging.ERROR) # Set diffusers verbosity

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up server and loading models...")
    model_provider = get_model_provider() # Get or create instance
    try:
        model_provider.load_all_models() # Eagerly load all models
        logger.info("Server startup complete. Models loaded.")
    except Exception as e:
        logger.error(f"Model loading during startup failed: {e}", exc_info=True)
        # Depending on severity, you might want to raise an error to stop the app
        # For now, it will log and continue, potentially failing on first request.
        raise RuntimeError(f"Server startup failed due to model loading error: {e}")
    
    yield # Application runs here

    # Shutdown
    logger.info("Shutting down server and cleaning up models...")
    if model_provider:
        model_provider.cleanup()
    logger.info("Server shutdown complete.")

app = FastAPI(
    title="Style Transfer API Server",
    description="API for applying artistic styles to images using depth maps",
    version="1.0.0",
    lifespan=lifespan # Use the new lifespan context manager
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
app.include_router(image_analysis.router) # Assuming you create this
app.include_router(search.router)         # Assuming you create this
app.include_router(utility.router)        # Assuming you create this

# Optional: Add a root path for health check or basic info
@app.get("/")
async def root():
    return {"message": "Style Transfer API is running"}