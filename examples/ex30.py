#!/usr/bin/env python3
"""
ex30.py - Document Processing and RAG System with OpenAI LLM using LangGraph

This script provides two main features:
1. Read list of PDF files or URLs with docling, save as markdown files, split and add to ChromaDB RAG
2. Process user queries using a LangGraph workflow that determines if they're Q/A or summary tasks,
   then either retrieves answers from RAG or generates summaries of the documents

Usage:
    - To process documents: python ex30.py --process_docs --sources source1.pdf source2.pdf
    - To query the system: python ex30.py --query "your query here"
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
from typing import List, Dict, Tuple, Union, Optional, Literal, TypedDict, Annotated
from uuid import uuid4
from datetime import datetime
import operator
import json

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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# ChromaDB imports
from langchain_community.vectorstores import Chroma

# LangGraph imports
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# OpenAI imports
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ex30.log")
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_storage")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
COLLECTION_NAME = "ex30_collection"
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5
OPENAI_MODEL = "gpt-4o"  # Default model
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # Default embedding model

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
        accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.CPU
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
            num_threads=8, device=AcceleratorDevice.CPU
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
            
class ChromaIndexer:
    """Handles document indexing in ChromaDB"""
    
    def __init__(self):
        """Initialize the ChromaDB indexer"""
        self.db_path = CHROMA_DB_PATH
        self.collection_name = COLLECTION_NAME
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=self.db_path,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        
        logger.info("Vector store initialized successfully")
    
    def index_document(self, content: str, metadata: Dict) -> List[str]:
        """
        Index a document in ChromaDB
        
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

# Initialize LLM
llm = ChatOpenAI(
    temperature=0,
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY
)

# Define the state for our LangGraph
class RAGState(TypedDict):
    query: str
    query_type: Optional[str]
    documents: Optional[List[Document]]
    document_summaries: Optional[List[Dict]]
    generation: Optional[str]
    final_answer: Optional[str]

# Define the LangGraph nodes
def classify_query(state: RAGState) -> RAGState:
    """
    Classify a query as either 'qa' or 'summary'
    """
    logger.info(f"Classifying query: {state['query']}")
    
    # Create prompt for classification
    prompt = PromptTemplate(
        template="""You are a query classifier. Your task is to determine if a user query is asking for:
        1. A question-answering task (labeled as 'qa') - where the user is seeking specific information or answers
        2. A summarization task (labeled as 'summary') - where the user is asking for a summary or overview
        
        Respond with ONLY 'qa' or 'summary'.
        
        User query: {query}
        Classification:""",
        input_variables=["query"],
    )
    
    # Create chain
    classification_chain = prompt | llm | StrOutputParser()
    
    # Get classification
    classification = classification_chain.invoke({"query": state["query"]}).strip().lower()
    
    # Ensure we get a valid classification
    if classification not in ['qa', 'summary']:
        # Default to qa if classification is unclear
        classification = 'qa'
        
    logger.info(f"Query classified as: {classification}")
    
    # Update state
    return {**state, "query_type": classification}

def retrieve_documents(state: RAGState) -> RAGState:
    """
    Retrieve relevant documents for the query
    """
    logger.info(f"Retrieving documents for query: {state['query']}")
    
    # Initialize ChromaIndexer
    indexer = ChromaIndexer()
    
    # Search for documents
    documents = indexer.search(state["query"])
    
    logger.info(f"Retrieved {len(documents)} documents")
    
    # Update state
    return {**state, "documents": documents}

def answer_question(state: RAGState) -> RAGState:
    """
    Answer a question using RAG
    """
    logger.info(f"Answering question: {state['query']}")
    
    # Format context
    documents = state["documents"]
    formatted_context = ""
    for i, doc in enumerate(documents):
        formatted_context += f"\n--- Document {i+1} ---\n"
        formatted_context += doc.page_content
        formatted_context += f"\n--- End Document {i+1} ---\n"
    
    # Create prompt for answering
    prompt = PromptTemplate(
        template="""You are a helpful assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't find the answer in the context, just say that you don't have that information.
        
        Context:
        {context}
        
        Question: {query}
        Answer:""",
        input_variables=["context", "query"],
    )
    
    # Create chain
    answer_chain = prompt | llm | StrOutputParser()
    
    # Get answer
    answer = answer_chain.invoke({"context": formatted_context, "query": state["query"]})
    
    # Update state
    return {**state, "final_answer": answer}

def summarize_documents(state: RAGState) -> RAGState:
    """
    Summarize each document
    """
    logger.info("Summarizing documents")
    
    # Get all documents from the system
    documents = []
    
    # Load previously processed documents from the output directory
    md_files = list(Path(OUTPUT_DIR).glob("*.md"))
    
    document_summaries = []
    
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
            
            # Create prompt for summarization
            prompt = PromptTemplate(
                template="""You are a document summarizer. Your task is to create a concise yet comprehensive summary of the document.
                Focus on the main points, key findings, and important details.
                Make the summary clear, well-structured, and informative.
                
                Document:
                {document}
                
                Summary:""",
                input_variables=["document"],
            )
            
            # Create chain
            summary_chain = prompt | llm | StrOutputParser()
            
            # Get summary
            summary = summary_chain.invoke({"document": content})
            
            # Add to summaries
            document_summaries.append({
                "summary": summary,
                "metadata": metadata
            })
            
            logger.info(f"Summarized document: {md_file.name}")
            
        except Exception as e:
            logger.error(f"Error summarizing document {md_file}: {str(e)}")
    
    # Update state
    return {**state, "document_summaries": document_summaries}

def create_final_summary(state: RAGState) -> RAGState:
    """
    Create a final summary from individual document summaries
    """
    logger.info("Creating final summary")
    
    # Format summaries
    summaries = state["document_summaries"]
    formatted_summaries = ""
    for i, summary in enumerate(summaries):
        formatted_summaries += f"\n--- Document {i+1}: {summary['metadata'].get('file_name', 'unknown')} ---\n"
        formatted_summaries += summary["summary"]
        formatted_summaries += f"\n--- End Document {i+1} ---\n"
    
    # Create prompt for final summary
    prompt = PromptTemplate(
        template="""You are a document synthesizer. Your task is to create a comprehensive final summary that reorganizes and integrates information from multiple document summaries.
        Identify common themes, highlight key points, and present a coherent overview that connects information across all documents.
        Make the final summary well-structured, informative, and easy to understand.
        
        Document summaries:
        {summaries}
        
        Final summary:""",
        input_variables=["summaries"],
    )
    
    # Create chain
    final_summary_chain = prompt | llm | StrOutputParser()
    
    # Get final summary
    final_summary = final_summary_chain.invoke({"summaries": formatted_summaries})
    
    # Update state
    return {**state, "final_answer": final_summary}

def route_by_query_type(state: RAGState) -> str:
    """
    Route to the appropriate node based on query type
    """
    query_type = state["query_type"]
    
    if query_type == "qa":
        return "qa"
    else:
        return "summary"

# Create the LangGraph workflow
workflow = StateGraph(RAGState)

# Add nodes
workflow.add_node("classify", classify_query)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("answer", answer_question)
workflow.add_node("summarize", summarize_documents)
workflow.add_node("final_summary", create_final_summary)

# Set entry point
workflow.set_entry_point("classify")

# Add edges
workflow.add_edge("classify", "retrieve")
workflow.add_conditional_edges(
    "retrieve",
    route_by_query_type,
    {
        "qa": "answer",
        "summary": "summarize"
    }
)
workflow.add_edge("summarize", "final_summary")
workflow.add_edge("answer", END)
workflow.add_edge("final_summary", END)

# Compile the graph
rag_app = workflow.compile()

class DocumentSystem:
    """Main system that integrates all components"""
    
    def __init__(self):
        """Initialize the document system"""
        self.processor = DocumentProcessor()
        self.indexer = ChromaIndexer()
        self.documents = {}  # Store processed documents
        
        # Create output directory if it doesn't exist
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        # Load previously processed documents
        self._load_processed_documents()
        
    def _load_processed_documents(self):
        """Load previously processed documents from the output directory and ChromaDB"""
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
        Process a user query using the LangGraph workflow
        
        Args:
            query: User query
            
        Returns:
            Response to the query
        """
        logger.info(f"Processing query: {query}")
        
        # Initialize state
        initial_state = {
            "query": query,
            "query_type": None,
            "documents": None,
            "document_summaries": None,
            "generation": None,
            "final_answer": None
        }
        
        # Run the workflow
        result = rag_app.invoke(initial_state)
        
        # Return the final answer
        return result["final_answer"]

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Document Processing and RAG System with LangGraph")
    
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
    # python ex30.py --process_docs --sources https://en.wikipedia.org/wiki/Artificial_intelligence https://en.wikipedia.org/wiki/Machine_learning
    # python ex30.py --query "What is artificial intelligence?"
    main()
