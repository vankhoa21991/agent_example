# Document Processing and RAG System with Groq LLM

This script provides a powerful document processing and retrieval-augmented generation (RAG) system using Groq LLM. It can process PDF files and URLs, save them as markdown files, and add them to a Qdrant vector database for efficient retrieval.

## Features

1. **Document Processing**:
   - Convert PDF files and URLs to markdown using docling
   - Save processed documents to the output directory
   - Split documents into chunks and index them in Qdrant

2. **Query Processing**:
   - Automatically classify queries as Q/A or summary tasks
   - For Q/A tasks: Retrieve relevant document chunks and generate answers
   - For summary tasks: Generate summaries for each document and create a final synthesized summary

## Requirements

- Python 3.8+
- Groq API key (set as environment variable `GROQ_API_KEY`)
- Required packages:
  - docling
  - langchain
  - qdrant_client
  - groq
  - langchain_community
  - langchain_qdrant
  - fastembed

## Installation

1. Ensure you have all required packages installed:

```bash
pip install docling langchain qdrant-client groq langchain-groq langchain-qdrant
```

2. Set up your environment variables:

```bash
export GROQ_API_KEY="your-groq-api-key"
export QDRANT_DB_PATH="./qdrant_storage"  # Optional, defaults to "./qdrant_storage"
export OUTPUT_DIR="./output"  # Optional, defaults to "./output"
```

Alternatively, create a `.env` file with these variables.

## Usage

### Processing Documents

To process PDF files or URLs and add them to the RAG system:

```bash
python ex29.py --process_docs --sources source1.pdf source2.pdf https://example.com/document.pdf
```

This will:
1. Convert each source to markdown
2. Save the markdown files to the output directory
3. Split the documents into chunks
4. Create embeddings for each chunk
5. Store the chunks in Qdrant

### Querying the System

To query the system:

```bash
python ex29.py --query "What is the main topic discussed in these documents?"
```

The system will:
1. Automatically determine if this is a Q/A or summary task
2. For Q/A tasks: Retrieve relevant document chunks and generate an answer
3. For summary tasks: Generate summaries for each document and create a final synthesized summary

## Examples

### Q/A Example

```bash
python ex29.py --query "What are the key findings in the research paper?"
```

### Summary Example

```bash
python ex29.py --query "Summarize the content of all documents"
```

## How It Works

1. **Document Processing**:
   - The system uses docling to convert documents to markdown
   - Documents are split into chunks using RecursiveCharacterTextSplitter
   - Groq embeddings are created for each chunk
   - Chunks are stored in Qdrant with their embeddings and metadata

2. **Query Classification**:
   - The system uses Groq LLM to classify queries as either Q/A or summary tasks
   - Classification is based on the query's intent and structure

3. **Q/A Processing**:
   - For Q/A tasks, the system creates an embedding for the query
   - It retrieves the most similar document chunks from Qdrant
   - The retrieved chunks are used as context to generate an answer

4. **Summary Processing**:
   - For summary tasks, the system summarizes each document individually
   - It then creates a final summary that synthesizes information from all document summaries

## Customization

You can customize the system by modifying the following constants in the script:

- `EMBEDDING_DIMENSION`: Dimension for embeddings (default: 4096)
- `CHUNK_SIZE`: Size of document chunks (default: 1500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K_RESULTS`: Number of results to retrieve for Q/A (default: 5)
- `GROQ_MODEL`: Groq model to use (default: "llama-3.3-70b-versatile")

## Troubleshooting

- If you encounter issues with embedding creation, the system will fall back to random vectors
- Check the log file `ex29.log` for detailed error messages and debugging information
- Ensure your Groq API key is valid and has sufficient quota
