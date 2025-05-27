Future Designer API Documentation
=================================

A FastAPI-based service for furniture design and image processing.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api_reference
   schemas
   configuration

Overview
--------

The Future Designer API provides:

* Furniture detection and captioning
* Style transfer for design variations
* Depth estimation from images
* Background removal for furniture
* Room composition capabilities

Quick Start
-----------

1. Install dependencies::

   pip install -r requirements.txt

2. Configure environment::

   cp .env_example .env

3. Run the API::

   uvicorn app.main:app --reload

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`