Configuration
=============

.. automodule:: app.core.config
   :members:
   :undoc-members:
   :show-inheritance:

Settings
--------

.. autoclass:: app.core.config.Settings
   :members:

Environment Variables
---------------------

Required Variables
~~~~~~~~~~~~~~~~~~

MILVUS_URL
    URL of the Milvus vector database instance

MILVUS_TOKEN
    Authentication token for Milvus access

MILVUS_COLLECTION_NAME
    Name of the collection for vector storage

Optional Variables
~~~~~~~~~~~~~~~~~~

PROMPTS_FILE_PATH
    Path to styles configuration file (default: "styles.json")

EMBEDING_MODEL_NAME
    Sentence transformer model name (default: "all-MiniLM-L6-v2")

DINO_PROMPT
    Object detection prompt (default: "chair. sofa. table. bed.")