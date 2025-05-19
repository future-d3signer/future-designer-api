import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",  # Path to your FastAPI app instance
        host="0.0.0.0",
        port=8000,
        reload=False,    # Set to True for development, False for production
        workers=1        # For GPU tasks, typically 1 worker unless models are on different GPUs or VLLM handles its own concurrency well.
    )