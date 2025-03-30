#!/usr/bin/env python3
"""
database.py - MongoDB operations for document storage and retrieval

This module provides functionality to store and retrieve documents from MongoDB.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("database.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
MONGODB_URI = os.getenv("MONGO_AUTH")
DB_NAME = os.getenv("MONGODB_DB_NAME", "document_processing")

class MongoDBHandler:
    """Handles MongoDB operations for document storage and retrieval"""
    
    def __init__(self):
        """Initialize the MongoDB handler"""
        if not MONGODB_URI:
            raise ValueError("MONGO_AUTH environment variable is not set. Please set it in your .env file.")
        
        try:
            # Initialize MongoDB client
            self.client = MongoClient(MONGODB_URI)
            print(MONGODB_URI)
            
            self.db = self.client[DB_NAME]
            logger.info(f"Connected to MongoDB database: {DB_NAME}")
        except Exception as e:
            error_msg = str(e)
            if "authentication failed" in error_msg.lower():
                raise ValueError(
                    "MongoDB authentication failed. Please check your MONGO_AUTH connection string in the .env file. "
                    "Make sure the username, password, and cluster information are correct."
                )
            else:
                raise ValueError(f"Failed to connect to MongoDB: {error_msg}")
    
    def save_document(self, content: str, metadata: Dict, session_id: Optional[str] = None, collection_name: str = "documents") -> str:
        """
        Save a document to MongoDB
        
        Args:
            content: Document content
            metadata: Document metadata
            session_id: Session ID for grouping documents
            collection_name: MongoDB collection name
            
        Returns:
            Document ID
        """
        logger.info(f"Saving document to collection: {collection_name}")
        
        try:
            # Get collection
            collection = self.db[collection_name]
            
            # Create document
            document = {
                "content": content,
                "metadata": metadata,
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Insert document
            result = collection.insert_one(document)
            
            logger.info(f"Document saved with ID: {result.inserted_id}")
            return result.inserted_id
            
        except Exception as e:
            logger.error(f"Error saving document: {e}")
            raise Exception(f"Error saving document: {str(e)}")
    
    def get_document(self, document_id: str, collection_name: str = "documents") -> Optional[Dict]:
        """
        Get a document by ID
        
        Args:
            document_id: Document ID
            collection_name: MongoDB collection name
            
        Returns:
            Document or None if not found
        """
        logger.info(f"Getting document with ID: {document_id} from collection: {collection_name}")
        
        try:
            # Get collection
            collection = self.db[collection_name]
            
            # Get document
            document = collection.find_one({"_id": ObjectId(document_id)})
            
            if document:
                logger.info(f"Document found with ID: {document_id}")
            else:
                logger.info(f"Document not found with ID: {document_id}")
                
            return document
            
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            raise Exception(f"Error getting document: {str(e)}")
    
    def list_documents(self, collection_name: str = "documents", limit: int = 10, skip: int = 0) -> List[Dict]:
        """
        List documents
        
        Args:
            collection_name: MongoDB collection name
            limit: Maximum number of documents to return
            skip: Number of documents to skip
            
        Returns:
            List of documents
        """
        logger.info(f"Listing documents from collection: {collection_name}")
        
        try:
            # Get collection
            collection = self.db[collection_name]
            
            # Get documents
            documents = list(collection.find().sort("created_at", -1).skip(skip).limit(limit))
            
            logger.info(f"Found {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise Exception(f"Error listing documents: {str(e)}")
    
    def search_documents(self, query: str, collection_name: str = "documents", limit: int = 10) -> List[Dict]:
        """
        Search documents
        
        Args:
            query: Search query
            collection_name: MongoDB collection name
            limit: Maximum number of documents to return
            
        Returns:
            List of documents matching the query
        """
        logger.info(f"Searching documents with query: {query} in collection: {collection_name}")
        
        try:
            # Get collection
            collection = self.db[collection_name]
            
            # Create text index if it doesn't exist
            if "content_text" not in collection.index_information():
                collection.create_index([("content", "text")])
                logger.info("Created text index on content field")
            
            # Search documents
            documents = list(collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit))
            
            logger.info(f"Found {len(documents)} documents matching query: {query}")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise Exception(f"Error searching documents: {str(e)}")
    
    def delete_document(self, document_id: str, collection_name: str = "documents") -> bool:
        """
        Delete a document by ID
        
        Args:
            document_id: Document ID
            collection_name: MongoDB collection name
            
        Returns:
            True if document was deleted, False otherwise
        """
        logger.info(f"Deleting document with ID: {document_id} from collection: {collection_name}")
        
        try:
            # Get collection
            collection = self.db[collection_name]
            
            # Delete document
            result = collection.delete_one({"_id": ObjectId(document_id)})
            
            if result.deleted_count > 0:
                logger.info(f"Document deleted with ID: {document_id}")
                return True
            else:
                logger.info(f"Document not found with ID: {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise Exception(f"Error deleting document: {str(e)}")
    
    def update_document(self, document_id: str, content: Optional[str] = None, 
                       metadata: Optional[Dict] = None, collection_name: str = "documents") -> bool:
        """
        Update a document by ID
        
        Args:
            document_id: Document ID
            content: New document content (optional)
            metadata: New document metadata (optional)
            collection_name: MongoDB collection name
            
        Returns:
            True if document was updated, False otherwise
        """
        logger.info(f"Updating document with ID: {document_id} in collection: {collection_name}")
        
        try:
            # Get collection
            collection = self.db[collection_name]
            
            # Create update document
            update_doc = {"updated_at": datetime.now().isoformat()}
            
            if content is not None:
                update_doc["content"] = content
                
            if metadata is not None:
                update_doc["metadata"] = metadata
            
            # Update document
            result = collection.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": update_doc}
            )
            
            if result.modified_count > 0:
                logger.info(f"Document updated with ID: {document_id}")
                return True
            else:
                logger.info(f"Document not found or not modified with ID: {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise Exception(f"Error updating document: {str(e)}")
    
    def get_documents_by_session(self, session_id: str, collection_name: str = "documents", limit: int = 100) -> List[Dict]:
        """
        Get documents by session ID
        
        Args:
            session_id: Session ID
            collection_name: MongoDB collection name
            limit: Maximum number of documents to return
            
        Returns:
            List of documents
        """
        logger.info(f"Getting documents with session ID: {session_id} from collection: {collection_name}")
        
        try:
            # Get collection
            collection = self.db[collection_name]
            
            # Get documents
            documents = list(collection.find({"session_id": session_id}).sort("created_at", -1).limit(limit))
            
            logger.info(f"Found {len(documents)} documents with session ID: {session_id}")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting documents by session: {e}")
            raise Exception(f"Error getting documents by session: {str(e)}")
