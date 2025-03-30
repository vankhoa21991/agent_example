#!/usr/bin/env python3
"""
main.py - FastAPI endpoints for processing documents with docling and storing in MongoDB

This API provides endpoints to:
1. Process URLs with docling
2. Process PDF files with docling
3. Save processed documents to MongoDB
4. Retrieve documents from MongoDB
"""

import os
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

# Import local modules
from document_processor import DocumentProcessor
from database import MongoDBHandler
from rag_processor import RAGProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Document Processing API",
    description="API for processing documents with docling and storing in MongoDB",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document processor
document_processor = DocumentProcessor()

# Initialize database handler with error handling
try:
    db_handler = MongoDBHandler()
except ValueError as e:
    logger.error(f"MongoDB connection error: {e}")
    # We'll initialize the app anyway, but endpoints that require MongoDB will return errors
    db_handler = None

# Initialize RAG processor with error handling
try:
    rag_processor = RAGProcessor()
except Exception as e:
    logger.error(f"RAG processor initialization error: {e}")
    # We'll initialize the app anyway, but endpoints that require RAG will return errors
    rag_processor = None

# Define request and response models
class UrlRequest(BaseModel):
    url: HttpUrl
    collection_name: Optional[str] = "documents"
    tags: Optional[List[str]] = []
    session_id: Optional[str] = None

class DocumentResponse(BaseModel):
    id: str
    content: str
    metadata: Dict
    created_at: str
    session_id: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    collection_name: Optional[str] = "documents"
    limit: Optional[int] = 10

class ChatRequest(BaseModel):
    query: str
    index_document: Optional[bool] = False
    document_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    query_type: str

@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    mongodb_status = "connected" if db_handler is not None else "disconnected"
    rag_status = "connected" if rag_processor is not None else "disconnected"
    return {
        "message": "Document Processing API is running",
        "mongodb_status": mongodb_status,
        "rag_status": rag_status,
        "version": "1.0.0"
    }

@app.post("/process/url", response_model=DocumentResponse)
async def process_url(request: UrlRequest):
    """
    Process a URL with docling and save to MongoDB
    
    Args:
        request: UrlRequest object containing URL and optional parameters
        
    Returns:
        DocumentResponse object with processed document details
    """
    # Check if MongoDB is connected
    if db_handler is None:
        raise HTTPException(
            status_code=500, 
            detail="MongoDB is not connected. Please check your MongoDB connection settings in the .env file."
        )
        
    try:
        # Process URL with docling
        logger.info(f"Processing URL: {request.url}")
        markdown_content = document_processor.process_url(str(request.url))
        
        # Create metadata
        metadata = {
            "source": str(request.url),
            "source_type": "url",
            "processed_time": datetime.now().isoformat(),
            "tags": request.tags
        }
        
        # Save to MongoDB
        doc_id = db_handler.save_document(
            content=markdown_content,
            metadata=metadata,
            session_id=request.session_id,
            collection_name=request.collection_name
        )
        
        return DocumentResponse(
            id=str(doc_id),
            content=markdown_content,
            metadata=metadata,
            created_at=datetime.now().isoformat(),
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

@app.post("/process/pdf", response_model=DocumentResponse)
async def process_pdf(
    file: UploadFile = File(...),
    collection_name: str = Form("documents"),
    tags: List[str] = Form([]),
    session_id: Optional[str] = Form(None)
):
    """
    Process a PDF file with docling and save to MongoDB
    
    Args:
        file: Uploaded PDF file
        collection_name: MongoDB collection name
        tags: List of tags for the document
        
    Returns:
        DocumentResponse object with processed document details
    """
    # Check if MongoDB is connected
    if db_handler is None:
        raise HTTPException(
            status_code=500, 
            detail="MongoDB is not connected. Please check your MongoDB connection settings in the .env file."
        )
        
    try:
        # Check if file is a PDF
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read file content
        file_content = await file.read()
        
        # Process PDF with docling
        logger.info(f"Processing PDF: {file.filename}")
        markdown_content = document_processor.process_pdf_bytes(file_content)
        
        # Create metadata
        metadata = {
            "filename": file.filename,
            "source_type": "pdf",
            "processed_time": datetime.now().isoformat(),
            "tags": tags
        }
        
        # Save to MongoDB
        doc_id = db_handler.save_document(
            content=markdown_content,
            metadata=metadata,
            session_id=session_id,
            collection_name=collection_name
        )
        
        return DocumentResponse(
            id=str(doc_id),
            content=markdown_content,
            metadata=metadata,
            created_at=datetime.now().isoformat(),
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, collection_name: str = "documents"):
    """
    Get a document by ID
    
    Args:
        document_id: Document ID
        collection_name: MongoDB collection name
        
    Returns:
        DocumentResponse object with document details
    """
    # Check if MongoDB is connected
    if db_handler is None:
        raise HTTPException(
            status_code=500, 
            detail="MongoDB is not connected. Please check your MongoDB connection settings in the .env file."
        )
        
    try:
        # Get document from MongoDB
        document = db_handler.get_document(document_id, collection_name)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentResponse(
            id=str(document["_id"]),
            content=document["content"],
            metadata=document["metadata"],
            created_at=document["created_at"],
            session_id=document.get("session_id")
        )
        
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")

@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents(collection_name: str = "documents", limit: int = 10, skip: int = 0):
    """
    List documents
    
    Args:
        collection_name: MongoDB collection name
        limit: Maximum number of documents to return
        skip: Number of documents to skip
        
    Returns:
        List of DocumentResponse objects
    """
    # Check if MongoDB is connected
    if db_handler is None:
        raise HTTPException(
            status_code=500, 
            detail="MongoDB is not connected. Please check your MongoDB connection settings in the .env file."
        )
        
    try:
        # Get documents from MongoDB
        documents = db_handler.list_documents(collection_name, limit, skip)
        
        return [
            DocumentResponse(
                id=str(doc["_id"]),
                content=doc["content"],
                metadata=doc["metadata"],
                created_at=doc["created_at"],
                session_id=doc.get("session_id")
            )
            for doc in documents
        ]
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.post("/search", response_model=List[DocumentResponse])
async def search_documents(request: QueryRequest):
    """
    Search documents
    
    Args:
        request: QueryRequest object containing query and optional parameters
        
    Returns:
        List of DocumentResponse objects matching the query
    """
    # Check if MongoDB is connected
    if db_handler is None:
        raise HTTPException(
            status_code=500, 
            detail="MongoDB is not connected. Please check your MongoDB connection settings in the .env file."
        )
        
    try:
        # Search documents in MongoDB
        documents = db_handler.search_documents(
            query=request.query,
            collection_name=request.collection_name,
            limit=request.limit
        )
        
        return [
            DocumentResponse(
                id=str(doc["_id"]),
                content=doc["content"],
                metadata=doc["metadata"],
                created_at=doc["created_at"],
                session_id=doc.get("session_id")
            )
            for doc in documents
        ]
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app.get("/documents/session/{session_id}", response_model=List[DocumentResponse])
async def get_documents_by_session(session_id: str, collection_name: str = "documents", limit: int = 100):
    """
    Get documents by session ID
    
    Args:
        session_id: Session ID
        collection_name: MongoDB collection name
        limit: Maximum number of documents to return
        
    Returns:
        List of DocumentResponse objects
    """
    # Check if MongoDB is connected
    if db_handler is None:
        raise HTTPException(
            status_code=500, 
            detail="MongoDB is not connected. Please check your MongoDB connection settings in the .env file."
        )
        
    try:
        # Get documents from MongoDB
        documents = db_handler.get_documents_by_session(
            session_id=session_id,
            collection_name=collection_name,
            limit=limit
        )
        
        return [
            DocumentResponse(
                id=str(doc["_id"]),
                content=doc["content"],
                metadata=doc["metadata"],
                created_at=doc["created_at"],
                session_id=doc.get("session_id")
            )
            for doc in documents
        ]
        
    except Exception as e:
        logger.error(f"Error getting documents by session: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting documents by session: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the RAG system
    
    The system will:
    1. Use LangGraph to determine if the query is a Q/A or summary task
    2. Retrieve relevant documents for Q/A tasks
    3. Generate summaries for summary tasks
    4. Optionally index a document if requested
    
    Args:
        request: ChatRequest object containing query and optional parameters
        
    Returns:
        ChatResponse object with answer and query type
    """
    # Check if RAG processor is initialized
    if rag_processor is None:
        raise HTTPException(
            status_code=500, 
            detail="RAG processor is not initialized. Please check your Groq API key in the .env file."
        )
    
    try:
        # If document_id is provided and index_document is True, index the document
        if request.index_document and request.document_id and db_handler is not None:
            # Get document from MongoDB
            document = db_handler.get_document(request.document_id)
            
            if document:
                # Index document in RAG processor
                logger.info(f"Indexing document with ID: {request.document_id}")
                rag_processor.index_document(
                    content=document["content"],
                    metadata=document["metadata"]
                )
        
        # Process query
        logger.info(f"Processing chat query: {request.query}")
        
        # Get initial state to determine query type
        initial_state = {
            "query": request.query,
            "query_type": None,
            "documents": None,
            "document_summaries": None,
            "generation": None,
            "final_answer": None,
            "session_id": request.session_id
        }
        
        # Classify query
        classified_state = rag_processor._classify_query(initial_state)
        query_type = classified_state["query_type"]
        
        # Process query with RAG processor
        answer = rag_processor.process_query(request.query, session_id=request.session_id)
        
        return ChatResponse(
            answer=answer,
            query_type=query_type
        )
        
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
