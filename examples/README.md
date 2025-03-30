# Organized Examples

This directory contains a structured collection of examples demonstrating various techniques and applications of language models, agents, and AI systems. The examples have been organized into logical categories to make it easier to find and understand related concepts.

## Directory Structure

- **basic/**: Fundamental examples demonstrating core concepts
- **agent_tools/**: Examples of agents using custom tools
- **groq/**: Examples using Groq's API and models
- **langgraph/**: Examples using LangGraph for structured agent workflows
- **mcp/**: Model Context Protocol examples for connecting models to external tools
- **rag/**: Retrieval Augmented Generation examples
  - **mongodb/**: RAG implementations using MongoDB
  - **qdrant/**: RAG implementations using Qdrant
  - **advanced/**: Advanced RAG techniques
- **documentation/**: Examples related to documentation generation and processing
- **utilities/**: Utility scripts and helper functions

## Getting Started

If you're new to these examples, we recommend starting with the basic examples to understand the fundamental concepts, then exploring the more specialized categories based on your interests.

Each subdirectory contains its own README with more detailed information about the examples in that category.

## Prerequisites

Most examples require:

1. Python 3.8+
2. Required packages (see requirements.txt in the root directory)
3. API keys for various services (OpenAI, Groq, etc.)

## Common Environment Variables

Many examples use environment variables for configuration. The most common ones include:

- `OPENAI_API_KEY`: For OpenAI API access
- `GROQ_API_KEY`: For Groq API access
- `MONGO_AUTH`: For MongoDB Atlas connection
- `QDRANT_URL`: For Qdrant connection

You can set these in a `.env` file in the root directory.
