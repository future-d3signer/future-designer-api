API Reference
=============

The Future Designer API provides REST endpoints for furniture design and image processing.

Base URL: ``http://localhost:8000``

Endpoints Overview
------------------

.. raw:: html

   <div class="http-method post">POST</div>

**/caption** - Generate furniture descriptions from images

.. raw:: html

   <div class="http-method post">POST</div>

**/style** - Apply style transfer to images

.. raw:: html

   <div class="http-method post">POST</div>

**/depth** - Generate depth maps from images

.. raw:: html

   <div class="http-method post">POST</div>

**/transparency** - Remove backgrounds from furniture images

.. raw:: html

   <div class="http-method post">POST</div>

**/composite** - Composite furniture into room scenes

Detailed Endpoints
------------------

Furniture Caption Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="http-method post">POST</div>

``/caption``

Analyzes room images to detect and describe furniture items using AI.

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
   The API can detect multiple furniture items in a single image. Each item gets a unique identifier.

Style Transfer
~~~~~~~~~~~~~~

.. raw:: html

   <div class="http-method post">POST</div>

``/style``

Applies artistic style transfer to create design variations.

**Request Body:**

.. code-block:: json

   {
     "style_image": "base64_encoded_depth_image",
     "style": "modern_minimalist"
   }

**Response:**

.. code-block:: json

   {
     "generated_image": "base64_encoded_styled_image"
   }

**Available Styles:**

- ``modern_minimalist``
- ``scandinavian``
- ``industrial``
- ``vintage``
- ``contemporary``

Depth Map Generation
~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="http-method post">POST</div>

``/depth``

Generates depth maps for 3D scene understanding.

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
   Depth maps are useful for understanding spatial relationships in room layouts.

Background Removal
~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="http-method post">POST</div>

``/transparency``

Removes backgrounds from furniture images to create transparent PNGs.

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
   Works best with furniture images that have clear, contrasting backgrounds.

Room Composition
~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="http-method post">POST</div>

``/composite``

Composites furniture items into room scenes with precise positioning.

**Request Body:**

.. code-block:: json

   {
     "room_image": "base64_encoded_room_image",
     "furniture_image": "base64_encoded_furniture_image",
     "position": {"x": 100, "y": 150},
     "size": {"width": 200, "height": 180}
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
   * - width
     - integer
     - Desired width of furniture in pixels
   * - height
     - integer
     - Desired height of furniture in pixels

Error Responses
---------------

All endpoints may return the following error responses:

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
     "detail": "Internal processing error"
   }