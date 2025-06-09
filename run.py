import uvicorn
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API on")

    args = parser.parse_args()
    uvicorn.run(
        "app.main:app",  
        host="0.0.0.0",
        port=8000,
        reload=False,    
        workers=1        
    )
