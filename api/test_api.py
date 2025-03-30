#!/usr/bin/env python3
"""
test_api.py - Test script for the Document Processing API

This script demonstrates how to use the Document Processing API programmatically.
"""

import os
import sys
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# API base URL
API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://localhost:{API_PORT}"

def test_api_status():
    """Test if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print(f"❌ API returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API is not running. Please start the API server first.")
        return False

def process_url(url, collection_name="documents", tags=None, session_id=None):
    """
    Process a URL with the API
    
    Args:
        url: URL to process
        collection_name: MongoDB collection name
        tags: List of tags for the document
        session_id: Session ID for grouping documents
        
    Returns:
        Document ID if successful, None otherwise
    """
    if tags is None:
        tags = []
        
    data = {
        "url": url,
        "collection_name": collection_name,
        "tags": tags,
        "session_id": session_id
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/process/url",
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ URL processed successfully. Document ID: {result['id']}")
            return result['id']
        else:
            print(f"❌ Error processing URL: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error processing URL: {str(e)}")
        return None

def process_pdf(pdf_path, collection_name="documents", tags=None, session_id=None):
    """
    Process a PDF file with the API
    
    Args:
        pdf_path: Path to PDF file
        collection_name: MongoDB collection name
        tags: List of tags for the document
        session_id: Session ID for grouping documents
        
    Returns:
        Document ID if successful, None otherwise
    """
    if tags is None:
        tags = []
        
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return None
        
    try:
        with open(pdf_path, "rb") as f:
            files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
            data = {
                "collection_name": collection_name,
                "tags": ",".join(tags),
                "session_id": session_id
            }
            
            response = requests.post(
                f"{BASE_URL}/process/pdf",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ PDF processed successfully. Document ID: {result['id']}")
                return result['id']
            else:
                print(f"❌ Error processing PDF: {response.text}")
                return None
    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")
        return None

def get_document(document_id, collection_name="documents"):
    """
    Get a document by ID
    
    Args:
        document_id: Document ID
        collection_name: MongoDB collection name
        
    Returns:
        Document if successful, None otherwise
    """
    try:
        response = requests.get(
            f"{BASE_URL}/documents/{document_id}",
            params={"collection_name": collection_name}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Document retrieved successfully")
            return result
        else:
            print(f"❌ Error getting document: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error getting document: {str(e)}")
        return None

def list_documents(collection_name="documents", limit=5, skip=0):
    """
    List documents
    
    Args:
        collection_name: MongoDB collection name
        limit: Maximum number of documents to return
        skip: Number of documents to skip
        
    Returns:
        List of documents if successful, None otherwise
    """
    try:
        response = requests.get(
            f"{BASE_URL}/documents",
            params={
                "collection_name": collection_name,
                "limit": limit,
                "skip": skip
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Retrieved {len(result)} documents")
            return result
        else:
            print(f"❌ Error listing documents: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error listing documents: {str(e)}")
        return None

def search_documents(query, collection_name="documents", limit=5):
    """
    Search documents
    
    Args:
        query: Search query
        collection_name: MongoDB collection name
        limit: Maximum number of documents to return
        
    Returns:
        List of documents matching the query if successful, None otherwise
    """
    try:
        data = {
            "query": query,
            "collection_name": collection_name,
            "limit": limit
        }
        
        response = requests.post(
            f"{BASE_URL}/search",
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {len(result)} documents matching query: {query}")
            return result
        else:
            print(f"❌ Error searching documents: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error searching documents: {str(e)}")
        return None

def chat_with_rag(query, index_document=False, document_id=None, session_id=None):
    """
    Chat with the RAG system
    
    Args:
        query: User query
        index_document: Whether to index the document
        document_id: Document ID to index
        session_id: Session ID for retrieving documents
        
    Returns:
        Chat response if successful, None otherwise
    """
    try:
        data = {
            "query": query,
            "index_document": index_document,
            "document_id": document_id,
            "session_id": session_id
        }
        
        response = requests.post(
            f"{BASE_URL}/chat",
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat response received. Query type: {result['query_type']}")
            print(f"Answer: {result['answer']}...")  # Print first 100 chars of answer
            return result
        else:
            print(f"❌ Error chatting with RAG: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error chatting with RAG: {str(e)}")
        return None

def get_documents_by_session(session_id, collection_name="documents", limit=100):
    """
    Get documents by session ID
    
    Args:
        session_id: Session ID
        collection_name: MongoDB collection name
        limit: Maximum number of documents to return
        
    Returns:
        List of documents if successful, None otherwise
    """
    try:
        response = requests.get(
            f"{BASE_URL}/documents/session/{session_id}",
            params={
                "collection_name": collection_name,
                "limit": limit
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Retrieved {len(result)} documents with session ID: {session_id}")
            return result
        else:
            print(f"❌ Error getting documents by session: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error getting documents by session: {str(e)}")
        return None

def main():
    """Main function"""
    # Check if API is running
    if not test_api_status():
        print("Please start the API server first with: python api/run.py")
        sys.exit(1)
    
    # Create a session ID
    session_id = f"test_session_{int(datetime.now().timestamp())}"
    print(f"Using session ID: {session_id}")
    
    # Process URLs with the same session ID
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Natural_language_processing"
    ]
    
    doc_ids = []
    for i, url in enumerate(urls):
        print(f"\nProcessing URL {i+1}/{len(urls)}: {url}")
        doc_id = process_url(url, tags=["AI", "Wikipedia"], session_id=session_id)
        if doc_id:
            doc_ids.append(doc_id)
    
    if doc_ids:
        # Get documents by session ID
        print("\n--- Getting documents by session ID ---")
        session_documents = get_documents_by_session(session_id)
        if session_documents:
            print(f"Found {len(session_documents)} documents in session {session_id}")
            for i, doc in enumerate(session_documents):
                print(f"Document {i+1}: {doc['metadata'].get('source', 'Unknown source')}")
        
        # Get the first document
        # document = get_document(doc_ids[0])
        # if document:
        #     print(f"\nDocument content length: {len(document['content'])} characters")
        #     print(f"Document metadata: {json.dumps(document['metadata'], indent=2)}")
        
        # List documents
        # documents = list_documents(limit=5)
        # if documents:
        #     print(f"\nDocument count: {len(documents)}")
            
        # # Search documents
        # search_results = search_documents("artificial intelligence")
        # if search_results:
        #     print(f"\nSearch results count: {len(search_results)}")
            
        # Chat with RAG - Question 1: "what is AI" (without session_id)
        print("\n--- Chat with RAG: Question 1 (without session_id) ---")
        print("Query: what is AI")
        chat_response1 = chat_with_rag("what is AI")
        
        # Chat with RAG - Question 2: "what is the documents about?" (with session_id)
        print("\n--- Chat with RAG: Question 2 (with session_id) ---")
        print("Query: what is the documents about?")
        print(f"Using session ID: {session_id}")
        chat_response2 = chat_with_rag(
            "what is the documents about?",
            session_id=session_id
        )
        
        # Chat with RAG - Question 3: "summarize the documents" (with session_id)
        print("\n--- Chat with RAG: Question 3 (with session_id) ---")
        print("Query: summarize the documents")
        print(f"Using session ID: {session_id}")
        chat_response3 = chat_with_rag(
            "summarize the documents",
            session_id=session_id
        )

if __name__ == "__main__":
    main()
