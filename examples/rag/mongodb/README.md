# MongoDB RAG Examples

This folder contains examples demonstrating how to implement Retrieval Augmented Generation (RAG) using MongoDB Atlas Vector Search.

## Files

- `ex22_mongodb_rag.py`: Basic RAG implementation using MongoDB Atlas Vector Search
- `ex23_mongodb_agentic_rag.py`: Agentic RAG implementation with MongoDB
- `ex24_mongodb_summary_langchain.py`: RAG with summarization using MongoDB and LangChain
- `mongodb_atlas.py`: Utility functions for working with MongoDB Atlas

## Key Concepts

- Setting up MongoDB Atlas Vector Search
- Creating and managing vector embeddings in MongoDB
- Implementing semantic search with MongoDB
- Building RAG pipelines with MongoDB as the vector store
- Integrating MongoDB with LangChain
- Advanced RAG techniques with MongoDB

## Prerequisites

To run these examples, you'll need:

- A MongoDB Atlas account
- MongoDB Atlas Vector Search enabled
- MongoDB connection string in your environment variables

## Usage

```bash
export MONGO_AUTH="your_mongodb_connection_string"
python ex22_mongodb_rag.py
```

MongoDB Atlas Vector Search provides a powerful and scalable solution for implementing RAG systems, with features like semantic caching, hybrid search, and more.
