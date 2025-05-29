API Reference
=============

The Future Designer API provides REST endpoints for furniture design and image processing using advanced AI models.

Base URL: ``http://localhost:8000``

Interactive Documentation: ``http://localhost:8000/docs``

Endpoints Overview
------------------

Image Analysis
~~~~~~~~~~~~~~

.. raw:: html

   <div class="http-method post">POST</div>

**/image-analysis/generate_captions** - Analyze images to detect and describe furniture

.. raw:: html

   <div class="http-method post">POST</div>

**/image-analysis/generate_depth** - Generate depth maps from room images

.. raw:: html

   <div class="http-method post">POST</div>

**/image-analysis/generate_transparency** - Remove backgrounds from furniture images

Image Generation
~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="http-method post">POST</div>

**/image-generation/generate_style** - Apply style transfer to create design variations

.. raw:: html

   <div class="http-method post">POST</div>

**/image-generation/generate_inpaint** - Inpaint areas of images with new content

.. raw:: html

   <div class="http-method post">POST</div>

**/image-generation/generate_delete** - Remove furniture from images

.. raw:: html

   <div class="http-method post">POST</div>

**/image-generation/generate_replace** - Replace furniture with new items

Search & Utility
~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="http-method post">POST</div>

**/search** - Find similar furniture using vector search

.. raw:: html

   <div class="http-method post">POST</div>

**/utility/proxy-image** - Proxy external image URLs

.. raw:: html

   <div class="http-method post">POST</div>

**/utility/scrape-images** - Scrape image links from web pages

.. raw:: html

   <div class="http-method post">POST</div>

**/utility/composite_furniture** - Compose furniture into room scenes

Detailed Endpoints
------------------

Image Analysis Endpoints
~~~~~~~~~~~~~~~~~~~~~~~~~

Furniture Caption Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <div class="http-method post">POST</div>

``/image-analysis/generate_captions``

Analyzes room images to detect and describe furniture items using fine-tuned vision-language models with structured attribute extraction.

**Request Body:**

.. code-block:: json

   {
     "source_image": "base64_encoded_image_data"
   }

**Response:**

.. code-block:: json

   {
     "furniture": {
       "item_1": {
         "caption": {
           "type": "chair",
           "style": "modern",
           "color": "blue",
           "material": "fabric",
           "details": "Comfortable armchair with cushions",
           "room_type": "living_room"
         },
         "mask": "base64_encoded_segmentation_mask",
         "box": "base64_encoded_bounding_box_image",
         "furniture_image": "base64_encoded_extracted_furniture"
       }
     }
   }

.. note::
   Uses fine-tuned vision-language models for accurate furniture detection with structured attribute extraction.

Depth Map Generation
^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <div class="http-method post">POST</div>

``/image-analysis/generate_depth``

Generates accurate depth maps for 3D scene understanding and spatial reasoning.

**Request Body:**

.. code-block:: json

   {
     "source_image": "base64_encoded_image_data"
   }

**Response:**

.. code-block:: json

   {
     "depth_image": "base64_encoded_depth_map"
   }

.. tip::
   Depth maps enable precise spatial understanding for furniture placement and room composition.

Background Removal
^^^^^^^^^^^^^^^^^^

.. raw:: html

   <div class="http-method post">POST</div>

``/image-analysis/generate_transparency``

Extract furniture with clean transparent backgrounds using advanced segmentation.

**Request Body:**

.. code-block:: json

   {
     "furniture_image": "base64_encoded_furniture_with_background"
   }

**Response:**

.. code-block:: json

   {
     "transparent_image": "base64_encoded_transparent_furniture"
   }

.. warning::
   Works best with furniture images that have clear subject-background separation.

Image Generation Endpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Style Transfer
^^^^^^^^^^^^^^

.. raw:: html

   <div class="http-method post">POST</div>

``/image-generation/generate_style``

Apply artistic styles to furniture and rooms using depth-guided generation techniques.

**Request Body:**

.. code-block:: json

   {
     "depth_image_b64": "base64_encoded_depth_image",
     "style": "modern"
   }

**Response:**

.. code-block:: json

   {
     "generated_image": "base64_encoded_styled_image"
   }

**Available Styles:**

- ``modern``
- ``scandinavian``
- ``industrial``
- ``victorian"``
- ``artdeco``
- ``bohemian``

Image Inpainting
^^^^^^^^^^^^^^^^

.. raw:: html

   <div class="http-method post">POST</div>

``/image-generation/generate_inpaint``

Fill masked regions with contextually appropriate content.

**Request Body:**

.. code-block:: json

   {
     "source_image": "base64_encoded_image",
     "mask": "base64_encoded_mask_image",
     "prompt": "modern sofa in living room"
   }

**Response:**

.. code-block:: json

   {
     "inpainted_image": "base64_encoded_result"
   }

Object Deletion
^^^^^^^^^^^^^^^

.. raw:: html

   <div class="http-method post">POST</div>

``/image-generation/generate_delete``

Remove unwanted furniture from images with context-aware inpainting.

**Request Body:**

.. code-block:: json

   {
     "source_image": "base64_encoded_image",
     "object_mask": "base64_encoded_deletion_mask"
   }

**Response:**

.. code-block:: json

   {
     "result_image": "base64_encoded_image_without_object"
   }

Furniture Replacement
^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <div class="http-method post">POST</div>

``/image-generation/generate_replace``

Intelligently replace existing furniture with new items while maintaining scene coherence.

**Request Body:**

.. code-block:: json

   {
     "room_image": "base64_encoded_room",
     "target_mask": "base64_encoded_furniture_mask",
     "replacement_prompt": "modern leather armchair"
   }

**Response:**

.. code-block:: json

   {
     "replaced_image": "base64_encoded_result"
   }

Search & Utility Endpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vector Search
^^^^^^^^^^^^^

.. raw:: html

   <div class="http-method post">POST</div>

``/search``

Find similar furniture styles using AI-powered embeddings with weighted multi-field search.

**Request Body:**

.. code-block:: json

   {
     "query": "modern blue chair",
     "limit": 10,
     "filters": {
       "category": "seating",
       "style": "modern"
     }
   }

**Response:**

.. code-block:: json

   {
     "results": [
       {
         "id": "furniture_123",
         "score": 0.95,
         "metadata": {
           "type": "chair",
           "style": "modern",
           "color": "blue"
         }
       }
     ]
   }

Image Proxy
^^^^^^^^^^^

.. raw:: html

   <div class="http-method post">POST</div>

``/utility/proxy-image``

Proxy external image URLs for processing.

**Request Body:**

.. code-block:: json

   {
     "image_url": "https://example.com/furniture.jpg"
   }

**Response:**

.. code-block:: json

   {
     "image_data": "base64_encoded_proxied_image"
   }

Web Scraping
^^^^^^^^^^^^

.. raw:: html

   <div class="http-method post">POST</div>

``/utility/scrape-images``

Extract image links from web pages.

**Request Body:**

.. code-block:: json

   {
     "url": "https://example.com/furniture-gallery",
     "selector": ".product-image"
   }

**Response:**

.. code-block:: json

   {
     "images": [
       "https://example.com/image1.jpg",
       "https://example.com/image2.jpg"
     ]
   }

Room Composition
^^^^^^^^^^^^^^^^

.. raw:: html

   <div class="http-method post">POST</div>

``/utility/composite_furniture``

Composite furniture into room scenes with precise positioning and scaling.

**Request Body:**

.. code-block:: json

   {
     "room_image": "base64_encoded_room_image",
     "furniture_image": "base64_encoded_furniture_image",
     "position": {"x": 100, "y": 150},
     "scale": 0.8,
     "blend_mode": "normal"
   }

**Response:**

.. code-block:: json

   {
     "composite_image": "base64_encoded_result"
   }

**Position Parameters:**

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - x
     - integer
     - Horizontal position in pixels from left edge
   * - y
     - integer
     - Vertical position in pixels from top edge
   * - scale
     - float
     - Scaling factor (0.1 to 2.0)
   * - blend_mode
     - string
     - Blending mode: normal, multiply, overlay

Error Responses
---------------

All endpoints return standardized error responses:

**400 Bad Request**

.. code-block:: json

   {
     "detail": "Invalid image format or missing required fields"
   }

**422 Unprocessable Entity**

.. code-block:: json

   {
     "detail": [
       {
         "loc": ["body", "source_image"],
         "msg": "field required",
         "type": "value_error.missing"
       }
     ]
   }

**500 Internal Server Error**

.. code-block:: json

   {
     "detail": "Internal processing error occurred"
   }

**503 Service Unavailable**

.. code-block:: json

   {
     "detail": "AI model temporarily unavailable"
   }

Rate Limiting
-------------

The API implements rate limiting to ensure fair usage:

- **Standard tier**: 100 requests per minute
- **Analysis endpoints**: 10 requests per minute (due to GPU processing)
- **Generation endpoints**: 5 requests per minute (due to high computational cost)

Authentication
--------------

Currently, the API runs without authentication in development mode. Production deployments should implement proper API key authentication.

SDK Examples
------------

**Python SDK Usage:**

.. code-block:: python

   import requests
   import base64

   # Initialize client
   BASE_URL = "http://localhost:8000"

   # Furniture detection
   with open("room.jpg", "rb") as f:
       image_data = base64.b64encode(f.read()).decode()

   response = requests.post(f"{BASE_URL}/image-analysis/generate_captions", json={
       "source_image": image_data
   })

   furniture_data = response.json()

   # Style transfer
   response = requests.post(f"{BASE_URL}/image-generation/generate_style", json={
       "depth_image_b64": depth_image_b64,
       "style": "modern"
   })

   styled_image = response.json()["generated_image"]