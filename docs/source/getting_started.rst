Getting Started
===============

Installation
------------

Prerequisites:

* Python 3.8+
* Milvus vector database access

Install the application::

    git clone <repository-url>
    cd future-designer-api
    pip install -r requirements.txt

Configuration
-------------

Copy the environment template::

    cp .env_example .env

Edit ``.env`` with your settings::

    MILVUS_URL=your_milvus_url
    MILVUS_TOKEN=your_milvus_token
    MILVUS_COLLECTION_NAME=your_collection_name

Running
-------

Start the server::

    uvicorn app.main:app --reload

The API will be available at http://localhost:8000

API Documentation will be at http://localhost:8000/docs