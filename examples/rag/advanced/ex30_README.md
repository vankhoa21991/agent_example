# Document Processing and RAG System with OpenAI LLM using LangGraph

This script provides a powerful document processing and retrieval-augmented generation (RAG) system using OpenAI LLM and LangGraph. It can process PDF files and URLs, save them as markdown files, and add them to a ChromaDB vector database for efficient retrieval.

## Features

1. **Document Processing**:
   - Convert PDF files and URLs to markdown using docling
   - Save processed documents to the output directory
   - Split documents into chunks and index them in ChromaDB

2. **Query Processing with LangGraph**:
   - Uses a graph-based workflow to process queries
   - Automatically classifies queries as Q/A or summary tasks
   - For Q/A tasks: Retrieves relevant document chunks and generates answers
   - For summary tasks: Generates summaries for each document and creates a final synthesized summary

## Key Improvements over ex29.py

1. **LangGraph Integration**:
   - Implements a structured workflow using LangGraph's StateGraph
   - Defines clear nodes for each step of the process
   - Uses conditional edges to route queries based on classification
   - Provides better separation of concerns and more maintainable code

2. **OpenAI Integration**:
   - Uses the LangChain ChatOpenAI integration for high-quality LLM interactions
   - Implements OpenAI embeddings for state-of-the-art vector representations
   - Implements prompt templates and chains for each step of the process

3. **ChromaDB Instead of Qdrant**:
   - Uses ChromaDB as the vector database for storing and retrieving document embeddings
   - Provides a simpler and more lightweight vector database solution

## Requirements

- Python 3.8+
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)
- Required packages:
  - docling
  - langchain
  - chromadb
  - openai
  - langchain_openai
  - langchain_community
  - langgraph

## Installation

1. Ensure you have all required packages installed:

```bash
pip install docling langchain chromadb openai langchain-openai langchain-community langgraph
```

2. Set up your environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export CHROMA_DB_PATH="./chroma_storage"  # Optional, defaults to "./chroma_storage"
export OUTPUT_DIR="./output"  # Optional, defaults to "./output"
```

Alternatively, create a `.env` file with these variables.

## Usage

### Processing Documents

To process PDF files or URLs and add them to the RAG system:

```bash
python ex30.py --process_docs --sources source1.pdf source2.pdf https://example.com/document.pdf
```

This will:
1. Convert each source to markdown
2. Save the markdown files to the output directory
3. Split the documents into chunks
4. Create embeddings for each chunk
5. Store the chunks in ChromaDB

### Querying the System

To query the system:

```bash
python ex30.py --query "What is the main topic discussed in these documents?"
```

The system will:
1. Use LangGraph to process the query through a defined workflow
2. Classify the query as Q/A or summary task
3. Route the query to the appropriate processing path
4. Return a comprehensive answer or summary

## LangGraph Workflow

The LangGraph workflow consists of the following nodes:

1. **classify**: Determines if the query is a Q/A or summary task
2. **retrieve**: Retrieves relevant documents from ChromaDB
3. **answer**: For Q/A tasks, generates an answer based on retrieved documents
4. **summarize**: For summary tasks, generates summaries for each document
5. **final_summary**: Creates a synthesized summary from individual document summaries

The workflow uses conditional edges to route queries based on their classification, ensuring each query is processed appropriately.

## Examples

### Q/A Example

```bash
python ex30.py --query "What are the key findings in the research paper?"
```

### Summary Example

```bash
python ex30.py --query "Summarize the content of all documents"
```

## How It Works

1. **Document Processing**:
   - The system uses docling to convert documents to markdown
   - Documents are split into chunks using RecursiveCharacterTextSplitter
   - OpenAI embeddings are created for each chunk
   - Chunks are stored in ChromaDB with their embeddings and metadata

2. **Query Processing with LangGraph**:
   - The query enters the graph at the "classify" node
   - Based on classification, it follows different paths through the graph
   - Each node in the graph updates the state with new information
   - The final state contains the answer to the query

## Customization

You can customize the system by modifying the following constants in the script:

- `EMBEDDING_DIMENSION`: Dimension for embeddings (default: 768)
- `CHUNK_SIZE`: Size of document chunks (default: 5000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K_RESULTS`: Number of results to retrieve for Q/A (default: 5)
- `OPENAI_MODEL`: OpenAI model to use (default: "gpt-4o")
- `OPENAI_EMBEDDING_MODEL`: OpenAI embedding model to use (default: "text-embedding-3-small")

## Troubleshooting

- Check the log file `ex30.log` for detailed error messages and debugging information
- Ensure your OpenAI API key is valid and has sufficient quota
- If you encounter issues with document processing, try different file formats or URLs
