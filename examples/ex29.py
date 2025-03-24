#!/usr/bin/env python3
"""
ex29.py - Document Processing and RAG System with Groq LLM

This script provides two main features:
1. Read list of PDF files or URLs with docling, save as markdown files, split and add to Qdrant RAG
2. Process user queries by determining if they're Q/A or summary tasks, then either retrieve answers 
   from RAG or generate summaries of the documents

Usage:
    - To process documents: python ex29.py --process_docs --sources source1.pdf source2.pdf
    - To query the system: python ex29.py --query "your query here"
"""

import os
import sys
import time
import argparse
import logging
import asyncio
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional, Literal
from uuid import uuid4
from datetime import datetime

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TesseractOcrOptions,
)

# Langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Qdrant imports
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Groq imports
from groq import Groq
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ex29.log")
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_DB_PATH = os.getenv("QDRANT_DB_PATH", "./qdrant_storage")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
COLLECTION_NAME = "ex29_collection2"
EMBEDDING_DIMENSION = 768  # Dimension for FastEmbedEmbeddings
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5
GROQ_MODEL = "llama-3.3-70b-versatile"  # Default model

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def url_to_markdown(url: str) -> str:
    """
    Convert URL content to markdown using docling
    
    Args:
        url: URL to process
        
    Returns:
        Markdown text
    """
    logger.info(f"Processing URL: {url}")
    
    try:
        # Fetch the URL content
        # text = urllib.request.urlopen(url).read()
        
        # # Create input document
        # in_doc = InputDocument(
        #     path_or_stream=BytesIO(text),
        #     format=InputFormat.HTML,
        #     backend=HTMLDocumentBackend,
        #     filename=f"{url.split('/')[-1]}.html",
        # )
        
        # # Convert to document
        # backend = HTMLDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(text))
        # dl_doc = backend.convert()

        accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.MPS
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        converter = DocumentConverter(format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
                InputFormat.IMAGE: TesseractOcrOptions(),
                InputFormat.DOCX: None,
                InputFormat.HTML: None,
            })
        result = converter.convert(url)
        
        # Export to markdown
        markdown_text = result.document.export_to_markdown()
        return markdown_text
    
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return f"# Error Processing URL\n\nFailed to process {url}. Error: {str(e)}"

def pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert PDF content to markdown using docling
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Markdown text
    """
    logger.info(f"Processing PDF: {pdf_path}")
    
    try:
        # Configure accelerator options
        accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.CUDA
        )
        
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        
        # Create document converter
        converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
            InputFormat.IMAGE: TesseractOcrOptions(),
            InputFormat.DOCX: None,
            InputFormat.HTML: None,
        })
        
        # Convert document
        result = converter.convert(pdf_path)
        
        # Export to markdown
        markdown_text = result.document.export_to_markdown()
        return markdown_text
    
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        return f"# Error Processing PDF\n\nFailed to process {pdf_path}. Error: {str(e)}"

class DocumentProcessor:
    """Handles document processing with docling"""
    
    def __init__(self):
        """Initialize the document processor"""
        pass
        
    def process_document(self, source: str) -> Tuple[str, str]:
        """
        Process a document from a file path or URL
        
        Args:
            source: Path to a file or URL
            
        Returns:
            Tuple of (markdown_content, output_path)
        """
        logger.info(f"Processing document: {source}")
        
        try:
            # Determine if source is a URL or a file
            is_url = source.startswith(('http://', 'https://'))
            
            # Convert to markdown based on source type
            if is_url:
                markdown_content = url_to_markdown(source)
            else:
                markdown_content = pdf_to_markdown(source)
            
            # Generate output filename
            source_name = os.path.basename(source) if os.path.exists(source) else source.split('/')[-1]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{source_name}_{timestamp}.md"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # Save markdown to file
            with open(output_path, "w") as f:
                f.write(markdown_content)
                
            logger.info(f"Saved markdown to: {output_path}")
            
            return markdown_content, output_path
            
        except Exception as e:
            logger.error(f"Error processing document {source}: {str(e)}")
            raise
            
class QdrantIndexer:
    """Handles document indexing in Qdrant"""
    
    def __init__(self):
        """Initialize the Qdrant indexer"""
        self.db_path = QDRANT_DB_PATH
        self.collection_name = COLLECTION_NAME
        
        # Initialize embeddings
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        
        # Initialize Qdrant client
        self.client = QdrantClient(path=self.db_path)
        
        # Check if collection exists, if not create it
        collections = self.client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if self.collection_name not in collection_names:
            # Create collection with appropriate dimensions for the embedding model
            # For OpenAI text-embedding-3-small, the dimension is 1536
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE)
            )
            print(f"Created new collection: {self.collection_name}")

        # Initialize vector store with force_recreate=True to handle dimension mismatch
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
              )
        
        logger.info("Vector store initialized successfully")
    
    def index_document(self, content: str, metadata: Dict) -> List[str]:
        """
        Index a document in Qdrant
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            List of document IDs
        """
        logger.info(f"Indexing document: {metadata.get('file_name', 'unknown')}")
        
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
        
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
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
            
            # Convert to dictionary format
            results = []
            for doc in docs:
                results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": 0.0  # Score not available in this format
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            # Return empty list in case of error
            return []

class GroqLLM:
    """Handles interactions with the Groq LLM"""
    
    def __init__(self, model: str = GROQ_MODEL):
        """Initialize the Groq LLM client"""
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = model
        
    def classify_query(self, query: str) -> str:
        """
        Classify a query as either 'qa' or 'summary'
        
        Args:
            query: User query
            
        Returns:
            Classification as 'qa' or 'summary'
        """
        logger.info(f"Classifying query: {query}")
        
        messages = [
            {
                "role": "system",
                "content": """You are a query classifier. Your task is to determine if a user query is asking for:
                1. A question-answering task (labeled as 'qa') - where the user is seeking specific information or answers
                2. A summarization task (labeled as 'summary') - where the user is asking for a summary or overview
                
                Respond with ONLY 'qa' or 'summary'.
                """
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0.0,
            max_tokens=10
        )
        
        classification = response.choices[0].message.content.strip().lower()
        
        # Ensure we get a valid classification
        if classification not in ['qa', 'summary']:
            # Default to qa if classification is unclear
            classification = 'qa'
            
        logger.info(f"Query classified as: {classification}")
        return classification
        
    def answer_question(self, query: str, context: List[Dict]) -> str:
        """
        Answer a question using RAG
        
        Args:
            query: User query
            context: List of document chunks
            
        Returns:
            Answer to the question
        """
        logger.info(f"Answering question: {query}")
        
        # Format context
        formatted_context = ""
        for i, doc in enumerate(context):
            formatted_context += f"\n--- Document {i+1} ---\n"
            formatted_context += doc["text"]
            formatted_context += f"\n--- End Document {i+1} ---\n"
            
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question.
                If you don't find the answer in the context, just say that you don't have that information.
                
                Context:
                {formatted_context}
                """
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    def summarize_document(self, content: str, metadata: Dict) -> str:
        """
        Summarize a document
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            Summary of the document
        """
        logger.info(f"Summarizing document: {metadata.get('file_name', 'unknown')}")
        
        messages = [
            {
                "role": "system",
                "content": """You are a document summarizer. Your task is to create a concise yet comprehensive summary of the document.
                Focus on the main points, key findings, and important details.
                Make the summary clear, well-structured, and informative.
                """
            },
            {
                "role": "user",
                "content": f"Please summarize the following document:\n\n{content}"
            }
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    def create_final_summary(self, summaries: List[Dict]) -> str:
        """
        Create a final summary from individual document summaries
        
        Args:
            summaries: List of document summaries with metadata
            
        Returns:
            Final summary
        """
        logger.info("Creating final summary")
        
        # Format summaries
        formatted_summaries = ""
        for i, summary in enumerate(summaries):
            formatted_summaries += f"\n--- Document {i+1}: {summary['metadata'].get('file_name', 'unknown')} ---\n"
            formatted_summaries += summary["summary"]
            formatted_summaries += f"\n--- End Document {i+1} ---\n"
            
        messages = [
            {
                "role": "system",
                "content": """You are a document synthesizer. Your task is to create a comprehensive final summary that reorganizes and integrates information from multiple document summaries.
                Identify common themes, highlight key points, and present a coherent overview that connects information across all documents.
                Make the final summary well-structured, informative, and easy to understand.
                """
            },
            {
                "role": "user",
                "content": f"Please create a final summary that synthesizes the following document summaries:\n\n{formatted_summaries}"
            }
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0.3
        )
        
        return response.choices[0].message.content

class DocumentSystem:
    """Main system that integrates all components"""
    
    def __init__(self):
        """Initialize the document system"""
        self.processor = DocumentProcessor()
        self.indexer = QdrantIndexer()
        self.llm = GroqLLM()
        self.documents = {}  # Store processed documents
        
        # Create output directory if it doesn't exist
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        # Load previously processed documents
        self._load_processed_documents()
        
    def _load_processed_documents(self):
        """Load previously processed documents from the output directory and Qdrant"""
        logger.info("Loading previously processed documents")
        
        try:
            # Get all markdown files in the output directory
            md_files = list(Path(OUTPUT_DIR).glob("*.md"))
            
            if not md_files:
                logger.info("No previously processed documents found")
                return
                
            # Load each markdown file
            for md_file in md_files:
                try:
                    # Read the markdown file
                    with open(md_file, "r") as f:
                        content = f.read()
                        
                    # Create metadata
                    metadata = {
                        "file_name": md_file.name,
                        "source": "loaded_from_disk",
                        "processed_time": md_file.stat().st_mtime
                    }
                    
                    # Store document
                    self.documents[str(md_file)] = {
                        "content": content,
                        "metadata": metadata,
                        "doc_ids": []  # We don't have the doc_ids, but that's okay for summary tasks
                    }
                    
                    logger.info(f"Loaded document: {md_file.name}")
                    
                except Exception as e:
                    logger.error(f"Error loading document {md_file}: {str(e)}")
            
            logger.info(f"Loaded {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
        
    def process_documents(self, sources: List[Union[str, Dict]]) -> None:
        """
        Process a list of documents
        
        Args:
            sources: List of file paths, URLs, or dictionaries with source and type
        """
        logger.info(f"Processing {len(sources)} documents")
        
        for source_item in sources:
            try:
                # Handle both string sources and dictionary sources
                if isinstance(source_item, dict):
                    source = source_item["source"]
                else:
                    source = source_item
                
                # Process document
                content, path = self.processor.process_document(source)
                
                # Create metadata
                metadata = {
                    "file_name": os.path.basename(path),
                    "source": source,
                    "processed_time": datetime.now().isoformat()
                }
                
                # Index document
                doc_ids = self.indexer.index_document(content, metadata)
                
                # Store document
                self.documents[path] = {
                    "content": content,
                    "metadata": metadata,
                    "doc_ids": doc_ids
                }
                
                logger.info(f"Successfully processed and indexed: {source}")
                
            except Exception as e:
                if isinstance(source_item, dict):
                    logger.error(f"Error processing document {source_item['source']}: {str(e)}")
                else:
                    logger.error(f"Error processing document {source_item}: {str(e)}")
                
    def process_query(self, query: str) -> str:
        """
        Process a user query
        
        Args:
            query: User query
            
        Returns:
            Response to the query
        """
        logger.info(f"Processing query: {query}")
        
        # Classify query
        query_type = self.llm.classify_query(query)
        
        if query_type == 'qa':
            # Q/A task
            logger.info("Handling as Q/A task")
            
            # Search for relevant documents
            results = self.indexer.search(query)
            
            # Answer question
            answer = self.llm.answer_question(query, results)
            return answer
            
        else:
            # Summary task
            logger.info("Handling as summary task")
            
            # Get all documents
            document_summaries = []
            
            for path, doc in self.documents.items():
                # Summarize document
                summary = self.llm.summarize_document(doc["content"], doc["metadata"])
                
                document_summaries.append({
                    "summary": summary,
                    "metadata": doc["metadata"]
                })
                
            # Create final summary
            final_summary = self.llm.create_final_summary(document_summaries)
            return final_summary

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Document Processing and RAG System")
    
    # Create mutually exclusive group for operation mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--process_docs", action="store_true", help="Process documents")
    group.add_argument("--query", type=str, help="Query the system")
    
    # Add sources argument
    parser.add_argument("--sources", nargs="+", help="List of file paths or URLs")
    
    args = parser.parse_args()
    
    
    # Initialize system
    system = DocumentSystem()
    
    if args.process_docs:
        # Check if sources are provided
        if not args.sources:
            logger.error("No sources provided. Use --sources to specify file paths or URLs.")
            sys.exit(1)
            
        # Convert sources to the expected format
        formatted_sources = []
        for source in args.sources:
            # Determine if source is a URL or a file
            if source.startswith(('http://', 'https://')):
                formatted_sources.append({"source": source, "type": "url"})
            else:
                formatted_sources.append({"source": source, "type": "pdf"})
        
        # Process documents
        system.process_documents(formatted_sources)
        print(f"Successfully processed {len(formatted_sources)} documents.")
        
    elif args.query:
        # Check if documents have been processed
        if not system.documents:
            logger.warning("No documents have been processed. Results may be limited.")
            
        # Process query
        response = system.process_query(args.query)
        print("\nResponse:")
        print(response)

if __name__ == "__main__":
    # Example usage:
    # python ex29.py --process_docs --sources https://en.wikipedia.org/wiki/Artificial_intelligence https://en.wikipedia.org/wiki/Machine_learning
    # python ex29.py --query "What is artificial intelligence?"
    main()
