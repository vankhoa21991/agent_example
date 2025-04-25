# FastEmbed + Qdrant RAG

This example demonstrates how to implement Retrieval Augmented Generation (RAG) using FastEmbed for embeddings and Qdrant for vector storage.

## Features

- Uses FastEmbed embeddings (lightweight, fast, open-source)
- Supports both local Qdrant and Qdrant Cloud
- Document loading and chunking with LangChain text splitters
- Simple RAG implementation with retrieval and generation

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables:

```bash
# Create a .env file with your API keys
# For OpenAI (for the LLM part)
OPENAI_API_KEY=your_openai_api_key

# For Qdrant Cloud (optional)
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_cloud_api_key

# For local Qdrant (optional, used if cloud credentials not provided)
QDRANT_LOCAL_PATH=./qdrant_storage
```

## Usage

### Basic Example

Run the basic example with sample documents:

```bash
python fastemb_qdrant_rag.py
```

### Document Loading Example

Load your own documents and query them:

```bash
# Index documents from a directory
python fastemb_qdrant_rag_example.py --docs_dir /path/to/your/documents

# Run a single query
python fastemb_qdrant_rag_example.py --query "Your question here?"

# Interactive mode
python fastemb_qdrant_rag_example.py --interactive

# Specify file patterns to load
python fastemb_qdrant_rag_example.py --docs_dir /path/to/docs --file_pattern "*.pdf,*.txt"

# Specify number of documents to retrieve
python fastemb_qdrant_rag_example.py --query "Your question?" --top_k 5
```

## Using in Your Own Projects

You can use the `FastEmbedQdrantRAG` class in your own projects:

```python
from fastemb_qdrant_rag import FastEmbedQdrantRAG
from langchain_core.documents import Document

# Initialize the RAG system
rag = FastEmbedQdrantRAG()

# Add your documents
docs = [
    Document(page_content="Your document text here", metadata={"source": "example.txt"})
]
rag.add_documents(docs)

# Query the system
response = rag.query("Your question here?")
print(response)
```

## Customization

You can customize the system by modifying the constants in `fastemb_qdrant_rag.py`:

- `EMBEDDING_MODEL`: Change the FastEmbed model (default: "BAAI/bge-small-en")
- `EMBEDDING_DIM`: Update the dimension to match your chosen model
- `COLLECTION_NAME`: Change the Qdrant collection name
- `CHUNK_SIZE` and `CHUNK_OVERLAP`: Adjust document chunking parameters 