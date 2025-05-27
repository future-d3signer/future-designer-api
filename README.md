# Future Designer API 🎨

An AI-powered FastAPI service for furniture design and image processing, featuring advanced computer vision capabilities for interior design and room composition.

## ✨ Features

- **🪑 Furniture Detection & Captioning**: Automatically detect and describe furniture in images using state-of-the-art AI models
- **🎨 Style Transfer**: Apply artistic styles to furniture and rooms for creative design variations  
- **📏 Depth Estimation**: Generate accurate depth maps for 3D scene understanding
- **🖼️ Background Removal**: Extract furniture with clean transparent backgrounds
- **🏠 Room Composition**: Intelligently composite furniture into room scenes
- **🔍 Vector Search**: Find similar furniture styles using AI-powered embeddings with Milvus

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- CUDA-compatible GPU (recommended for optimal performance)
- Milvus vector database access

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd future-designer-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env_example .env
   # Edit .env with your configuration
   ```

5. **Run the API**
   ```bash
   uvicorn app.main:app --reload
   ```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs  

## 📖 Documentation

Comprehensive documentation is available in Sphinx format:

```bash
# Build documentation
cd docs
sphinx-build -b html source build

# View documentation
cd build && python -m http.server 8080
# Open http://localhost:8080
```

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/caption` | POST | Analyze images to detect and describe furniture |
| `/style` | POST | Apply style transfer to create design variations |
| `/depth` | POST | Generate depth maps from room images |
| `/transparency` | POST | Remove backgrounds from furniture images |
| `/composite` | POST | Compose furniture into room scenes |

## 💡 Usage Examples

### Furniture Detection
```python
import requests
import base64

# Encode image
with open("room.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Detect furniture
response = requests.post("http://localhost:8000/caption", json={
    "source_image": image_data
})

furniture = response.json()["furniture"]
```

### Style Transfer
```python
# Apply modern style
response = requests.post("http://localhost:8000/style", json={
    "style_image": depth_image_b64,
    "style": "modern_minimalist" 
})

styled_result = response.json()["generated_image"]
```

## 🏗️ Architecture

```
future-designer-api/
├── app/
│   ├── api/                 # API endpoints and dependencies
│   ├── core/               # Configuration and settings
│   ├── models/             # AI model management
│   ├── schemas/            # Pydantic data models
│   ├── services/           # Business logic services
│   └── main.py            # FastAPI application
├── docs/                  # Sphinx documentation
├── tests/                 # Test suite
└── requirements.txt       # Dependencies
```


### Docker
```bash
# Build image
docker build -t future-designer-api .

# Run container
docker run -p 8000:8000 --env-file .env future-designer-api
```
