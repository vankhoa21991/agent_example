#!/usr/bin/env python3
"""
rag.py - RAG providers for document retrieval

This module provides different RAG providers for document retrieval:
- ChromaDB
- Qdrant
- FAISS
"""

import os
import logging
from typing import Dict, List, Optional, Union, Protocol, Any
from pathlib import Path
from uuid import uuid4
from abc import ABC, abstractmethod

# LangChain imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

# Vector store imports
from langchain_community.vectorstores import Chroma, Qdrant, FAISS

# Embedding imports
# from langchain_groq import GroqEmbeddings
from langchain_openai import OpenAIEmbeddings

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5

class RAGProvider(ABC):
    """Abstract base class for RAG providers"""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the RAG provider"""
        pass
    
    @abstractmethod
    def index_document(self, content: str, metadata: Dict) -> List[str]:
        """
        Index a document
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Document]:
        """
        Search for documents
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of document chunks
        """
        pass
    
    @abstractmethod
    def get_retriever(self, top_k: int = TOP_K_RESULTS) -> Any:
        """
        Get a retriever
        
        Args:
            top_k: Number of results to return
            
        Returns:
            Retriever
        """
        pass

class ChromaDBProvider(RAGProvider):
    """ChromaDB RAG provider"""
    
    def __init__(self, embedding_model: Embeddings, collection_name: str = "document_processing_collection", persist_directory: str = "./chroma_storage"):
        """
        Initialize the ChromaDB provider
        
        Args:
            embedding_model: Embedding model
            collection_name: Collection name
            persist_directory: Directory to persist the database
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.vector_store = None
        
    def initialize(self) -> None:
        """Initialize the ChromaDB provider"""
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model
        )
        
        logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
    
    def index_document(self, content: str, metadata: Dict) -> List[str]:
        """
        Index a document in ChromaDB
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            List of document IDs
        """
        logger.info(f"Indexing document: {metadata.get('source', 'unknown')}")
        
        try:
            # Create a Document object
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            
            # Split the document
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', ','],
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents([doc])
            
            # Generate UUIDs for all chunks
            doc_ids = [str(uuid4()) for _ in range(len(chunks))]
            
            # Add documents to vector store
            self.vector_store.add_documents(documents=chunks, ids=doc_ids)
            
            # Persist the changes
            if hasattr(self.vector_store, '_collection'):
                self.vector_store.persist()
            
            logger.info(f"Indexed {len(chunks)} chunks")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            # Return empty list in case of error
            return []
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Document]:
        """
        Search for documents in ChromaDB
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of document chunks
        """
        logger.info(f"Searching for: {query}")
        
        try:
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k}
            )
            
            # Search for documents
            docs = retriever.invoke(query)
            return docs
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            # Return empty list in case of error
            return []
    
    def get_retriever(self, top_k: int = TOP_K_RESULTS) -> Any:
        """
        Get a retriever
        
        Args:
            top_k: Number of results to return
            
        Returns:
            Retriever
        """
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

class QdrantProvider(RAGProvider):
    """Qdrant RAG provider"""
    
    def __init__(self, embedding_model: Embeddings, collection_name: str = "document_processing_collection", url: Optional[str] = None, path: Optional[str] = None):
        """
        Initialize the Qdrant provider
        
        Args:
            embedding_model: Embedding model
            collection_name: Collection name
            url: Qdrant server URL
            path: Path to local Qdrant database
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.url = url
        self.path = path
        self.vector_store = None
        
    def initialize(self) -> None:
        """Initialize the Qdrant provider"""
        # Initialize vector store
        if self.url:
            self.vector_store = Qdrant(
                client=None,  # Will be initialized with the URL
                collection_name=self.collection_name,
                embeddings=self.embedding_model,
                url=self.url
            )
        elif self.path:
            # Ensure path exists
            Path(self.path).mkdir(parents=True, exist_ok=True)
            
            self.vector_store = Qdrant(
                client=None,  # Will be initialized with the path
                collection_name=self.collection_name,
                embeddings=self.embedding_model,
                path=self.path
            )
        else:
            raise ValueError("Either url or path must be provided")
        
        logger.info(f"Qdrant initialized with collection: {self.collection_name}")
    
    def index_document(self, content: str, metadata: Dict) -> List[str]:
        """
        Index a document in Qdrant
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            List of document IDs
        """
        logger.info(f"Indexing document: {metadata.get('source', 'unknown')}")
        
        try:
            # Create a Document object
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            
            # Split the document
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', ','],
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents([doc])
            
            # Generate UUIDs for all chunks
            doc_ids = [str(uuid4()) for _ in range(len(chunks))]
            
            # Add documents to vector store
            self.vector_store.add_documents(documents=chunks, ids=doc_ids)
            
            logger.info(f"Indexed {len(chunks)} chunks")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            # Return empty list in case of error
            return []
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Document]:
        """
        Search for documents in Qdrant
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of document chunks
        """
        logger.info(f"Searching for: {query}")
        
        try:
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k}
            )
            
            # Search for documents
            docs = retriever.invoke(query)
            return docs
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            # Return empty list in case of error
            return []
    
    def get_retriever(self, top_k: int = TOP_K_RESULTS) -> Any:
        """
        Get a retriever
        
        Args:
            top_k: Number of results to return
            
        Returns:
            Retriever
        """
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

class FAISSProvider(RAGProvider):
    """FAISS RAG provider"""
    
    def __init__(self, embedding_model: Embeddings, index_name: str = "document_processing_index", save_path: Optional[str] = "./faiss_index"):
        """
        Initialize the FAISS provider
        
        Args:
            embedding_model: Embedding model
            index_name: Index name
            save_path: Path to save the index
        """
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.save_path = save_path
        self.vector_store = None
        self.documents = []  # Store documents for indexing
        
    def initialize(self) -> None:
        """Initialize the FAISS provider"""
        # Check if index exists
        if self.save_path and Path(f"{self.save_path}/{self.index_name}").exists():
            # Load existing index
            self.vector_store = FAISS.load_local(
                folder_path=self.save_path,
                index_name=self.index_name,
                embeddings=self.embedding_model
            )
        else:
            # Create new index
            self.vector_store = FAISS(
                embedding_function=self.embedding_model,
                index_name=self.index_name
            )
        
        logger.info(f"FAISS initialized with index: {self.index_name}")
    
    def index_document(self, content: str, metadata: Dict) -> List[str]:
        """
        Index a document in FAISS
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            List of document IDs
        """
        logger.info(f"Indexing document: {metadata.get('source', 'unknown')}")
        
        try:
            # Create a Document object
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            
            # Split the document
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', ','],
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents([doc])
            
            # Generate UUIDs for all chunks
            doc_ids = [str(uuid4()) for _ in range(len(chunks))]
            
            # Add documents to vector store
            self.vector_store.add_documents(documents=chunks)
            
            # Store documents for reference
            self.documents.extend(chunks)
            
            # Save index
            if self.save_path:
                Path(self.save_path).mkdir(parents=True, exist_ok=True)
                self.vector_store.save_local(self.save_path)
            
            logger.info(f"Indexed {len(chunks)} chunks")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            # Return empty list in case of error
            return []
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Document]:
        """
        Search for documents in FAISS
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of document chunks
        """
        logger.info(f"Searching for: {query}")
        
        try:
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k}
            )
            
            # Search for documents
            docs = retriever.invoke(query)
            return docs
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            # Return empty list in case of error
            return []
    
    def get_retriever(self, top_k: int = TOP_K_RESULTS) -> Any:
        """
        Get a retriever
        
        Args:
            top_k: Number of results to return
            
        Returns:
            Retriever
        """
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

def create_embedding_model(model_type: str, api_key: Optional[str] = None) -> Embeddings:
    """
    Create an embedding model
    
    Args:
        model_type: Model type (groq, openai)
        api_key: API key
        
    Returns:
        Embedding model
    """
    if model_type.lower() == "groq":
        # return GroqEmbeddings(
        #     model="llama-3.1-70b-versatile",
        #     api_key=api_key or os.getenv("GROQ_API_KEY")
        # )
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
    elif model_type.lower() == "openai":
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported embedding model type: {model_type}")

def create_rag_provider(provider_type: str, embedding_model: Embeddings, **kwargs) -> RAGProvider:
    """
    Create a RAG provider
    
    Args:
        provider_type: Provider type (chromadb, qdrant, faiss)
        embedding_model: Embedding model
        **kwargs: Additional arguments for the provider
        
    Returns:
        RAG provider
    """
    if provider_type.lower() == "chromadb":
        return ChromaDBProvider(
            embedding_model=embedding_model,
            collection_name=kwargs.get("collection_name", "document_processing_collection"),
            persist_directory=kwargs.get("persist_directory", "./chroma_storage")
        )
    elif provider_type.lower() == "qdrant":
        return QdrantProvider(
            embedding_model=embedding_model,
            collection_name=kwargs.get("collection_name", "document_processing_collection"),
            url=kwargs.get("url"),
            path=kwargs.get("path")
        )
    elif provider_type.lower() == "faiss":
        return FAISSProvider(
            embedding_model=embedding_model,
            index_name=kwargs.get("index_name", "document_processing_index"),
            save_path=kwargs.get("save_path", "./faiss_index")
        )
    else:
        raise ValueError(f"Unsupported RAG provider type: {provider_type}")
