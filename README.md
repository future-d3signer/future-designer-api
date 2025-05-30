# Future Designer API 🎨

An AI-powered FastAPI service for furniture design and image processing, featuring advanced generative AI capabilities for interior design and room composition.

## 🎥 Demo

[![Future Designer API Demo](https://img.youtube.com/vi/NOlGHFNzzrM/0.jpg)](https://youtu.be/NOlGHFNzzrM)

*Click to watch the demo video showcasing furniture detection, replacement, and room composition features*

## 🌐 Live Demo

**Coming Soon!** Application online: **[Future Designer Web App](https://futuredesignerai.com/)**

*Interactive web interface with a user-friendly frontend*

## ✨ Features

- **🪑 Furniture Detection & Captioning**: Automatically detect and describe furniture in images using fine-tuned vision-language models with structured attribute extraction
- **🎨 Style Transfer**: Apply artistic styles to furniture and rooms using depth-guided generation techniques
- **📏 Depth Estimation**: Generate accurate depth maps for 3D scene understanding and spatial reasoning
- **🖼️ Background Removal**: Extract furniture with clean transparent backgrounds using advanced segmentation
- **🔄 Furniture Replacement**: Intelligently replace existing furniture with new items while maintaining scene coherence
- **🗑️ Object Deletion**: Remove unwanted furniture from images with context-aware inpainting
- **🎨 Image Inpainting**: Fill masked regions with contextually appropriate content
- **🔍 Vector Search**: Find similar furniture styles using AI-powered embeddings with weighted multi-field search
- **🌐 Web Scraping**: Extract and proxy images from web pages for processing
- **🏠 Room Composition**: Composite furniture into room scenes with precise positioning and scaling

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- CUDA-compatible GPU with atleast 24GB of VRAM (recommended for optimal performance)
- Milvus vector database access

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd future-designer-api
   ```

2. **Create virtual environment**
   ```bash
   conda create -n future-designer python=3.11
   conda activate future-designer
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install flash-attn==2.7.3
   ```

4. **Install SAM2 outside repo directory**
   ```bash
   cd .. 
   git clone https://github.com/facebookresearch/sam2.git && cd sam2
   pip install -e . && cd ..
   cd future-designer-api
   ```

5. **Configure environment**
   ```bash
   cp .env_example .env
   # Edit .env with your Milvus configuration:
   # MILVUS_URL=your_milvus_endpoint
   # MILVUS_TOKEN=your_milvus_token
   # MILVUS_COLLECTION_NAME=furniture_collection
   ```

6. **Run the API**
   ```bash
   python run.py
   ```

The API will be available at:
- **API**: http://localhost:8000

## 📖 Documentation

Comprehensive documentation is available in Sphinx format:

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs
sphinx-build -b html source build

# View documentation
cd build && python -m http.server 8080
# Open http://localhost:8080
```
## 🧪 Testing

Run the test suite using pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api/test_api_routers.py

```

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/image-analysis/generate_captions` | POST | Analyze images to detect and describe furniture |
| `/image-analysis/generate_depth` | POST | Generate depth maps from room images |
| `/image-analysis/generate_transparency` | POST | Remove backgrounds from furniture images |
| `/image-generation/generate_style` | POST | Apply style transfer to create design variations |
| `/image-generation/generate_inpaint` | POST | Inpaint areas of images with new content |
| `/image-generation/generate_delete` | POST | Remove furniture from images |
| `/image-generation/generate_replace` | POST | Replace furniture with new items |
| `/search` | POST | Find similar furniture using vector search |
| `/utility/proxy-image` | POST | Proxy external image URLs |
| `/utility/scrape-images` | POST | Scrape image links from web pages |
| `/utility/composite_furniture` | POST | Compose furniture into room scenes |
## 💡 Usage Examples

### Furniture Detection
```python
import requests
import base64

# Encode image
with open("room.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Detect furniture
response = requests.post("http://localhost:8000/image-analysis/generate_captions", json={
    "source_image": image_data
})

furniture = response.json()["furniture"]
```

### Style Transfer
```python
# Encode depth image
with open("room_depth.jpg", "rb") as f:
    depth_image_data = base64.b64encode(f.read()).decode()

# Apply modern style
response = requests.post("http://localhost:8000/image-generation/generate_style", json={
    "style": "modern",
    "depth_image_b64": depth_image_data 
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
├── experiments/           # Research experiments and model training
│   ├── notebooks/         # Jupyter notebooks for experiments
│   ├── vllm_labelling/    # Vision-Language Model experiments
│   └── README.md          # Detailed experiments documentation
├── docs/                  # Sphinx documentation
├── tests/                 # Test suite
└── requirements.txt       # Dependencies
```

### Experiments Directory

The [`experiments/`](experiments/) directory contains research code, model training scripts, and evaluation tools used to develop and improve the AI models powering the API. This includes:

- **Model Fine-tuning**: LoRA training scripts for PaliGemma and Qwen2-VL models
- **Data Generation**: Synthetic furniture dataset creation and labeling
- **Evaluation**: CLIP model performance testing and benchmarking
- **Research Notebooks**: Interactive experiments for furniture detection and captioning

See the [experiments README](experiments/README.md) for detailed information about the synthetic furniture dataset and pre-trained models.

### Apptainer (Singularity)
```bash
# Build SIF container
apptainer build future-designer-api.sif future-designer-api.def

# Run container
apptainer run --nv \
  --env MILVUS_URL="your_actual_milvus_endpoint" \
  --env MILVUS_TOKEN="your_actual_milvus_token" \
  --env MILVUS_COLLECTION_NAME="your_collection_name" \
  future-designer-api.sif
```
