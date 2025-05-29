Future Designer API Documentation
=================================

Future Designer API is an AI-powered FastAPI service for furniture design and image processing, featuring advanced generative AI capabilities for interior design and room composition.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api_reference
   schemas
   configuration

Features
--------

* 🪑 **Furniture Detection & Captioning**: Automatically detect and describe furniture in images using fine-tuned vision-language models
* 🎨 **Style Transfer**: Apply artistic styles to furniture and rooms using depth-guided generation techniques  
* 📏 **Depth Estimation**: Generate accurate depth maps for 3D scene understanding
* 🖼️ **Background Removal**: Extract furniture with clean transparent backgrounds
* 🔄 **Furniture Replacement**: Intelligently replace existing furniture with new items
* 🗑️ **Object Deletion**: Remove unwanted furniture from images with context-aware inpainting
* 🎨 **Image Inpainting**: Fill masked regions with contextually appropriate content
* 🔍 **Vector Search**: Find similar furniture styles using AI-powered embeddings
* 🌐 **Web Scraping**: Extract and proxy images from web pages for processing
* 🏠 **Room Composition**: Composite furniture into room scenes with precise positioning

Prerequisites
-------------

* Python 3.11
* CUDA-compatible GPU with at least 24GB of VRAM (recommended for optimal performance)
* Milvus vector database access

Installation
------------

1. Clone the repository::

    git clone <repository-url>
    cd future-designer-api

2. Create virtual environment::

    conda create -n future-designer python=3.11
    conda activate future-designer

3. Install dependencies::

    pip install -r requirements.txt
    pip install flash-attn==2.7.3

4. Install SAM2 outside repo directory::

    cd .. 
    git clone https://github.com/facebookresearch/sam2.git && cd sam2
    pip install -e . && cd ..
    cd future-designer-api

Configuration
-------------

Copy the environment template::

    cp .env_example .env

Edit ``.env`` with your settings::

    MILVUS_URL=your_milvus_endpoint
    MILVUS_TOKEN=your_milvus_token
    MILVUS_COLLECTION_NAME=furniture_collection

Running
-------

Start the server::

    python run.py

The API will be available at http://localhost:8000

API Documentation will be at http://localhost:8000/docs

Testing
-------

Run the test suite using pytest::

    # Run all tests
    pytest

    # Run specific test file
    pytest tests/test_api/test_api_routers.py

Documentation
-------------

Build and view the documentation::

    # Install documentation dependencies
    pip install sphinx sphinx-rtd-theme

    # Build documentation
    cd docs
    sphinx-build -b html source build

    # View documentation
    cd build && python -m http.server 8080

Open http://localhost:8080 to view the documentation.

API Endpoints
-------------

The API provides the following main endpoints:

Image Analysis
~~~~~~~~~~~~~~

* ``/image-analysis/generate_captions`` - Analyze images to detect and describe furniture
* ``/image-analysis/generate_depth`` - Generate depth maps from room images  
* ``/image-analysis/generate_transparency`` - Remove backgrounds from furniture images

Image Generation
~~~~~~~~~~~~~~~~

* ``/image-generation/generate_style`` - Apply style transfer to create design variations
* ``/image-generation/generate_inpaint`` - Inpaint areas of images with new content
* ``/image-generation/generate_delete`` - Remove furniture from images
* ``/image-generation/generate_replace`` - Replace furniture with new items

Utility & Search
~~~~~~~~~~~~~~~~

* ``/search`` - Find similar furniture using vector search
* ``/utility/proxy-image`` - Proxy external image URLs
* ``/utility/scrape-images`` - Scrape image links from web pages
* ``/utility/composite_furniture`` - Compose furniture into room scenes

Usage Examples
--------------

Furniture Detection::

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

Style Transfer::

    # Apply modern style
    response = requests.post("http://localhost:8000/image-generation/generate_style", json={
        "depth_image_b64": depth_image_b64,
        "style": "modern" 
    })

    styled_result = response.json()["generated_image"]

Container Deployment
--------------------

Using Apptainer (Singularity)::

    # Build SIF container
    apptainer build future-designer-api.sif future-designer-api.def

    # Run container
    apptainer run --nv \
      --env MILVUS_URL="your_actual_milvus_endpoint" \
      --env MILVUS_TOKEN="your_actual_milvus_token" \
      --env MILVUS_COLLECTION_NAME="your_collection_name" \
      future-designer-api.sif

Architecture
------------

The project follows a modular architecture::

    future-designer-api/
    ├── app/
    │   ├── api/                 # API endpoints and dependencies
    │   ├── core/               # Configuration and settings
    │   ├── models/             # AI model management
    │   ├── schemas/            # Pydantic data models
    │   ├── services/           # Business logic services
    │   └── main.py            # FastAPI application
    ├── experiments/           # Research experiments and model training
    ├── docs/                  # Sphinx documentation
    ├── tests/                 # Test suite
    └── requirements.txt       # Dependencies

The ``experiments/`` directory contains research code, model training scripts, and evaluation tools used to develop and improve the AI models powering the API.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`