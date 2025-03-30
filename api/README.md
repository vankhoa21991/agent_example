# Document Processing API

A FastAPI-based API for processing documents (URLs and PDFs) with docling and storing the results in MongoDB.

## Features

- Process URLs with docling and convert to markdown
- Process PDF files with docling and convert to markdown
- Save processed documents to MongoDB
- Retrieve documents from MongoDB
- Search documents in MongoDB
- Chat with documents using RAG (Retrieval-Augmented Generation)
- Automatic query classification (Q/A or summary) using LangGraph
- Document indexing for RAG

## Requirements

- Python 3.8+
- MongoDB database
- Docling library

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Set up environment variables:

Copy the `.env.example` file to `.env` and update the values:

```bash
cp api/.env.example .env
```

Then edit the `.env` file with your specific configuration:

```
MONGO_AUTH=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DB_NAME=document_processing
OUTPUT_DIR=./output
API_PORT=8000  # Optional, defaults to 8000
GROQ_API_KEY=your_groq_api_key_here  # Required for RAG system
CHROMA_DB_PATH=./chroma_storage  # Optional, defaults to ./chroma_storage
```

### RAG System Configuration

The RAG system uses LangGraph and vector databases for document retrieval and question answering. You can configure the following options in your `.env` file:

#### Required Configuration

- `GROQ_API_KEY`: API key for Groq's LLM API (used for query classification, document retrieval, and answer generation)
  1. Create an account at [Groq](https://console.groq.com/signup)
  2. Generate an API key in the Groq console
  3. Add the API key to your `.env` file

#### Optional Configuration

- `RAG_PROVIDER_TYPE`: Vector database provider (default: "chromadb")
  - Options: "chromadb", "qdrant", "faiss"
- `EMBEDDING_MODEL_TYPE`: Embedding model provider (default: "groq")
  - Options: "groq", "openai"
- `GROQ_MODEL`: Groq model to use (default: "llama-3.1-70b-versatile")
- `TOP_K_RESULTS`: Number of results to return from vector search (default: 5)

#### Vector Storage Paths

- `CHROMA_DB_PATH`: Path to ChromaDB storage (default: "./chroma_storage")
- `QDRANT_URL`: URL for Qdrant server (optional, for Qdrant)
- `QDRANT_PATH`: Path to local Qdrant storage (optional, for Qdrant)
- `FAISS_PATH`: Path to FAISS index (optional, for FAISS)

### MongoDB Setup

The API requires a MongoDB database to store processed documents. You have several options:

#### Option 1: MongoDB Atlas (Recommended)

1. Create a free account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
2. Create a new cluster
3. In the Security section, create a database user with read/write permissions
4. In the Network Access section, add your IP address to the IP Access List
5. Click "Connect" on your cluster, select "Connect your application", and copy the connection string
6. Replace `<username>`, `<password>`, and `<clustername>` in the connection string with your actual values
7. Add the connection string to your `.env` file as `MONGO_AUTH`

Example connection string:
```
MONGO_AUTH=mongodb+srv://myuser:mypassword@mycluster.mongodb.net/?retryWrites=true&w=majority
```

#### Option 2: Local MongoDB

1. [Install MongoDB Community Edition](https://www.mongodb.com/docs/manual/administration/install-community/) on your local machine
2. Start the MongoDB service
3. Add the connection string to your `.env` file as `MONGO_AUTH`

Example connection string for local MongoDB:
```
MONGO_AUTH=mongodb://localhost:27017/
```

#### Troubleshooting MongoDB Connection

If you encounter authentication errors:
- Double-check your username and password in the connection string
- Ensure your IP address is in the MongoDB Atlas IP Access List
- Verify that the database user has the correct permissions
- Try using the MongoDB Compass application to test your connection string

## Running the API

There are two ways to start the API server:

### Option 1: Using the run.py script (Recommended)

```bash
python api/run.py
```

Or directly:

```bash
./api/run.py
```

The run.py script handles Python path issues automatically, making it the easiest way to run the API.

### Option 2: Using uvicorn directly

```bash
cd api
uvicorn main:app --reload
```

Note: When using uvicorn directly, you need to be in the api directory for the imports to work correctly.

The API will be available at http://localhost:8000 (or the port specified in the API_PORT environment variable).

## API Documentation

Once the API is running, you can access the interactive API documentation at http://localhost:8000/docs.

### Endpoints

#### Process URL

```
POST /process/url
```

Process a URL with docling and save to MongoDB.

**Request Body:**

```json
{
  "url": "https://example.com",
  "collection_name": "documents",
  "tags": ["example", "documentation"],
  "session_id": "optional-session-id"
}
```

#### Process PDF

```
POST /process/pdf
```

Process a PDF file with docling and save to MongoDB.

**Form Data:**

- `file`: PDF file
- `collection_name`: MongoDB collection name (default: "documents")
- `tags`: List of tags for the document
- `session_id`: (Optional) Session ID for grouping documents

#### Get Document

```
GET /documents/{document_id}
```

Get a document by ID.

**Query Parameters:**

- `collection_name`: MongoDB collection name (default: "documents")

#### List Documents

```
GET /documents
```

List documents.

**Query Parameters:**

- `collection_name`: MongoDB collection name (default: "documents")
- `limit`: Maximum number of documents to return (default: 10)
- `skip`: Number of documents to skip (default: 0)

#### Search Documents

```
POST /search
```

Search documents.

**Request Body:**

```json
{
  "query": "search query",
  "collection_name": "documents",
  "limit": 10
}
```

#### Get Documents by Session

```
GET /documents/session/{session_id}
```

Get documents by session ID.

**Path Parameters:**

- `session_id`: Session ID

**Query Parameters:**

- `collection_name`: MongoDB collection name (default: "documents")
- `limit`: Maximum number of documents to return (default: 100)

#### Chat with RAG

```
POST /chat
```

Chat with the RAG system. The system will automatically classify the query as either a Q/A or summary task, retrieve relevant documents, and generate an answer.

**Request Body:**

```json
{
  "query": "What is artificial intelligence?",
  "index_document": false,
  "document_id": "65f1a2b3c4d5e6f7a8b9c0d1",
  "session_id": "optional-session-id"
}
```

- `query`: The user's question or request
- `index_document`: (Optional) Whether to index the document specified by document_id in the RAG system
- `document_id`: (Optional) The ID of a document to index in the RAG system
- `session_id`: (Optional) Session ID to retrieve documents from. If provided, the system will use all documents with this session_id as context for answering the query, instead of using the vector search.

**Response:**

```json
{
  "answer": "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems...",
  "query_type": "qa"
}
```

- `answer`: The generated answer to the query
- `query_type`: The type of query, either "qa" (question answering) or "summary"

## Example Usage

### Using the Test Script

A test script is provided to demonstrate how to use the API programmatically:

```bash
python api/test_api.py
```

This script will:
1. Check if the API is running
2. Process a URL (Wikipedia page on Artificial Intelligence)
3. Retrieve the processed document
4. List documents in the database
5. Search for documents matching a query

### Using curl

#### Process a URL

```bash
curl -X POST "http://localhost:8000/process/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "collection_name": "documents",
    "tags": ["AI", "Wikipedia"]
  }'
```

### Process a PDF

```bash
curl -X POST "http://localhost:8000/process/pdf" \
  -F "file=@document.pdf" \
  -F "collection_name=documents" \
  -F "tags=report,2023"
```

### Get a Document

```bash
curl -X GET "http://localhost:8000/documents/65f1a2b3c4d5e6f7a8b9c0d1?collection_name=documents"
```

### List Documents

```bash
curl -X GET "http://localhost:8000/documents?collection_name=documents&limit=5&skip=0"
```

### Search Documents

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence",
    "collection_name": "documents",
    "limit": 5
  }'
```

### Get Documents by Session

```bash
curl -X GET "http://localhost:8000/documents/session/my-session-id?collection_name=documents&limit=10"
```

### Chat with RAG

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "index_document": false,
    "document_id": "65f1a2b3c4d5e6f7a8b9c0d1"
  }'
```

### Chat with RAG and Index a Document

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the document",
    "index_document": true,
    "document_id": "65f1a2b3c4d5e6f7a8b9c0d1"
  }'
```

### Chat with RAG using Session ID

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize all the documents in this session",
    "session_id": "my-session-id"
  }'
```
