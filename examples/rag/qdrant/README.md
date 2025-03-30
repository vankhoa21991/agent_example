# Qdrant RAG Examples

This folder contains examples demonstrating how to implement Retrieval Augmented Generation (RAG) using Qdrant vector database.

## Files

- `ex25_qrant_rag_langchain.py`: Main RAG implementation using Qdrant and LangChain
- `ex26_qdrant_agentic_rag.py`: Agentic RAG implementation with Qdrant
- Supporting modules:
  - `db_utils.py`: Database utilities for Qdrant
  - `langchain_utils.py`: LangChain integration utilities
  - `prompts.py`: Prompt templates for RAG
  - `qdrant_utils.py`: Qdrant-specific utilities
  - `utils.py`: General utilities

## Key Concepts

- Setting up Qdrant for vector search
- Creating and managing vector embeddings in Qdrant
- Implementing semantic search with Qdrant
- Building RAG pipelines with Qdrant as the vector store
- Integrating Qdrant with LangChain
- Advanced RAG techniques with Qdrant

## Prerequisites

To run these examples, you'll need:

- Qdrant installed locally or a Qdrant Cloud account
- API keys for the language models used
- Python packages: qdrant-client, langchain, etc.

## Usage

```bash
python ex25_qrant_rag_langchain.py
```

Qdrant is a vector database designed specifically for vector similarity search, making it an excellent choice for RAG implementations. It offers features like filtering during search, efficient vector storage, and high performance.
