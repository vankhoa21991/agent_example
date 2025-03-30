# Qdrant RAG with LangChain

This directory contains a comprehensive implementation of Retrieval Augmented Generation (RAG) using Qdrant vector database and LangChain.

## Files

- `ex25_qrant_rag_langchain.py`: Main implementation file that orchestrates the RAG pipeline
- `db_utils.py`: Database utilities for managing connections and operations
- `langchain_utils.py`: Utilities for integrating with LangChain components
- `prompts.py`: Prompt templates used in the RAG system
- `qdrant_utils.py`: Qdrant-specific utilities for vector operations
- `utils.py`: General utility functions

## Architecture

This implementation follows a modular architecture:

1. **Data Ingestion**: Process and embed documents
2. **Vector Storage**: Store embeddings in Qdrant
3. **Query Processing**: Transform user queries into vector representations
4. **Retrieval**: Find relevant documents using vector similarity
5. **Generation**: Use retrieved context to generate responses

## Key Features

- Modular design with separation of concerns
- Efficient vector search with Qdrant
- Integration with LangChain for prompt management and LLM interaction
- Customizable prompt templates
- Database utilities for persistent storage

## Usage

To run this example:

```bash
python ex25_qrant_rag_langchain.py
```

## Requirements

- Qdrant (local or cloud instance)
- LangChain
- OpenAI API key or other compatible LLM
- Python 3.8+

This implementation demonstrates best practices for building a production-ready RAG system with Qdrant and LangChain.
