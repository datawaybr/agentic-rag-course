## Workflow

 - Search for documentation
 - Download the docs and store search docs on vectorDB


docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant# agentic-rag-course
