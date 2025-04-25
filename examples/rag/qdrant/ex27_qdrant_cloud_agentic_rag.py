from dotenv import load_dotenv
load_dotenv(override=True)

import os
import time
import urllib.request
from io import BytesIO
from typing import List, Dict, Any, Literal, TypedDict
from typing_extensions import TypedDict
from tqdm import tqdm
import asyncio
import logging

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TesseractOcrOptions
)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not QDRANT_API_KEY or not QDRANT_URL:
    raise ValueError("Please set QDRANT_API_KEY and QDRANT_URL environment variables")
if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY environment variable")

# Qdrant setup
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Text splitter for chunking documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)

# Functions to process different document types

def url_to_markdown(url: str) -> str:
    """Convert URL content to markdown using docling"""
    try:
        # Fetch the URL content
        text = urllib.request.urlopen(url).read()
        
        # Create input document
        in_doc = InputDocument(
            path_or_stream=BytesIO(text),
            format=InputFormat.HTML,
            backend=HTMLDocumentBackend,
            filename=f"{url.split('/')[-1]}.html",
        )
        
        # Convert to document
        backend = HTMLDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(text))
        dl_doc = backend.convert()
        
        # Export to markdown
        markdown_text = dl_doc.export_to_markdown()
        return markdown_text
    
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return f"# Error Processing URL\n\nFailed to process {url}. Error: {str(e)}"

def pdf_to_markdown(pdf_path: str) -> str:
    """Convert PDF content to markdown using docling"""
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
        print(f"Error processing PDF {pdf_path}: {e}")
        return f"# Error Processing PDF\n\nFailed to process {pdf_path}. Error: {str(e)}"

def process_document(source: str, source_type: Literal["url", "pdf"]) -> List[Document]:
    """Process a document and return chunks for vector store"""
    try:
        print(f"Processing {source_type}: {source}")
        
        # Convert to markdown based on source type
        if source_type == "url":
            markdown_text = url_to_markdown(source)
        elif source_type == "pdf":
            markdown_text = pdf_to_markdown(source)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        print(f"Converted {source_type} to markdown")

        # Create document
        doc = Document(
            page_content=markdown_text,
            metadata={"source": source, "type": source_type}
        )
        
        # Count number of tokens
        token_count = len(doc.page_content.split())
        print(f"Number of tokens: {token_count}")
        
        # Split into chunks
        chunks = text_splitter.split_documents([doc])
        print(f"Split into {len(chunks)} chunks")
        
        # Update metadata for each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
        
        return chunks
    except Exception as e:
        print(f"Error processing document {source}: {e}")
        return []

# Initialize vector store
def initialize_vector_store():
    """Initialize the Qdrant Vector Store with cloud configuration"""
    try:
        # Initialize Qdrant client with cloud configuration
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=True,  # Use gRPC for better performance
            timeout=30.0,  # Increase timeout for cloud operations
        )
        
        collection_name = "agentic_rag_collection"
        
        try:
            # Check if collection exists
            collection_info = client.get_collection(collection_name)
            print(f"Collection {collection_name} exists")
        except Exception as e:
            print(f"Collection {collection_name} does not exist, creating...")
            # Create collection with proper configuration
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding dimension
                    distance=Distance.COSINE
                ),
                optimizers_config={
                    "default_segment_number": 2,
                    "memmap_threshold": 20000
                }
            )
        
        # Initialize vector store with cloud configuration
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
        
        print("Vector store initialized successfully with cloud configuration")
        return vector_store
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        raise

# LangGraph components for agentic RAG

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langgraph.graph import END, StateGraph

# Data model for LLM output format
class GradeDocuments(TypedDict):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str  # "yes" or "no"

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: user question
        generation: LLM response generation
        documents: list of context documents
        document_relevance: whether documents are relevant - "yes" or "no"
    """
    question: str
    generation: str
    documents: List[Document]
    document_relevance: str

# Initialize Groq client and LLM
from groq import Groq
client = Groq()
MODEL = 'llama-3.3-70b-versatile'

# Function to grade document relevance
def grade_document(question: str, document_content: str) -> bool:
    """
    Determines whether a document is relevant to the question.
    Returns True if relevant, False otherwise.
    """
    print(f"Grading document relevance for question: {question}")
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert grader assessing relevance of a retrieved document to a user question. "
                      "Follow these instructions for grading: "
                      "- If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. "
                      "- Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not."
        },
        {
            "role": "user",
            "content": f"Retrieved document:\n{document_content}\n\nUser question:\n{question}"
        }
    ]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=False,
        max_tokens=10
    )
    
    result = response.choices[0].message.content.strip().lower()
    is_relevant = "yes" in result
    print(f"Document relevance: {'Relevant' if is_relevant else 'Not relevant'}")
    return is_relevant

def grade_documents_relevance(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    relevant_docs = []
    document_relevance = "no"
    
    if documents:
        for d in documents:
            is_relevant = grade_document(question, d.page_content)
            if is_relevant:
                print("---GRADE: DOCUMENT RELEVANT---")
                relevant_docs.append(d)
                document_relevance = "yes"
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
    else:
        print("---NO DOCUMENTS RETRIEVED---")
    
    return {
        "documents": relevant_docs, 
        "question": question, 
        "document_relevance": document_relevance
    }

def generate_answer(state: GraphState):
    """
    Generate answer from context document using LLM
    """
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]

    # Format documents into context string
    context = "\n\n".join(doc.page_content for doc in documents)
    
    messages = [
        {
            "role": "system",
            "content": "You are an assistant for question-answering tasks. "
                      "Use the following pieces of retrieved context to answer the question. "
                      "If no context is present or if you don't know the answer, just say that you don't know the answer. "
                      "Do not make up the answer unless it is there in the provided context. "
                      "Give a detailed answer and to the point answer with regard to the question."
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"
        }
    ]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=False
    )
    
    generation = response.choices[0].message.content.strip()
    return {
        "documents": documents, 
        "question": question, 
        "generation": generation
    }

def generate_no_context_answer(state: GraphState):
    """
    Generate answer when no relevant context is available
    """
    print("---GENERATE ANSWER WITHOUT CONTEXT---")
    question = state["question"]
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. "
                      "Please politely explain that you don't have enough information to provide a specific answer."
        },
        {
            "role": "user",
            "content": f"The user has asked: {question}"
        }
    ]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=False
    )
    
    generation = response.choices[0].message.content.strip()
    return {
        "documents": state["documents"],
        "question": question,
        "generation": generation
    }

def retrieve(state: GraphState, vector_store):
    """
    Retrieve documents from vector store
    """
    print("---RETRIEVAL FROM QDRANT CLOUD---")
    question = state["question"]

    try:
        # Retrieval with similarity search
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3  # Retrieve top 3 documents
            }
        )
        documents = retriever.invoke(question)
        print(f"Retrieved {len(documents)} documents from Qdrant Cloud")
        
        return {"documents": documents, "question": question}
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        # Return empty documents list in case of error
        return {"documents": [], "question": question}

def decide_to_generate(state: GraphState):
    """
    Determines whether to generate an answer with context or without context.
    """
    print("---ASSESS DOCUMENT RELEVANCE---")
    document_relevance = state["document_relevance"]

    if document_relevance == "yes":
        print("---DECISION: GENERATE RESPONSE WITH CONTEXT---")
        return "generate_with_context"
    else:
        print("---DECISION: GENERATE RESPONSE WITHOUT CONTEXT---")
        return "generate_without_context"

# Create and compile the graph
def create_agentic_rag_graph(vector_store):
    """Create the agentic RAG graph"""
    
    # Create the graph
    agentic_rag = StateGraph(GraphState)
    
    # Define the nodes with partial application for vector_store
    agentic_rag.add_node("retrieve", lambda state: retrieve(state, vector_store))
    agentic_rag.add_node("grade_documents", grade_documents_relevance)
    agentic_rag.add_node("generate_with_context", generate_answer)
    agentic_rag.add_node("generate_without_context", generate_no_context_answer)
    
    # Build graph
    agentic_rag.set_entry_point("retrieve")
    agentic_rag.add_edge("retrieve", "grade_documents")
    agentic_rag.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate_with_context": "generate_with_context", 
            "generate_without_context": "generate_without_context"
        },
    )
    agentic_rag.add_edge("generate_with_context", END)
    agentic_rag.add_edge("generate_without_context", END)
    
    # Compile
    return agentic_rag.compile()

# Main execution

def main():
    # Example sources
    sources = [
        {"source": "https://en.wikipedia.org/wiki/Artificial_intelligence", "type": "url"},
        {"source": "https://en.wikipedia.org/wiki/Machine_learning", "type": "url"},
        {"source": "https://www.ibm.com/fr-fr/think/topics/ai-agents", "type": "url"},
    ]
    
    # Initialize vector store with cloud configuration
    vector_store = initialize_vector_store()
    print("Vector store initialized with cloud configuration")
    
    # Process and load each document one by one using vector store
    from uuid import uuid4
    
    for source_info in sources:
        # Process document to get chunks
        chunks = process_document(source_info["source"], source_info["type"])
        
        if chunks:
            # Generate unique IDs for each chunk
            chunk_ids = [str(uuid4()) for _ in range(len(chunks))]
            
            # Add documents to vector store in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_ids = chunk_ids[i:i + batch_size]
                vector_store.add_documents(documents=batch, ids=batch_ids)
                print(f"Added batch {i//batch_size + 1} of {(len(chunks) + batch_size - 1)//batch_size}")
            
            print(f"Successfully loaded {len(chunks)} chunks from {source_info['source']} into Qdrant Cloud")
        else:
            print(f"No chunks to load from {source_info['source']}")
        
        time.sleep(1)  # Small delay between processing documents
    
    # Create the agentic RAG graph
    agentic_rag = create_agentic_rag_graph(vector_store)
    print("Agentic RAG graph created")
    
    # Example queries
    queries = [
        "What is machine learning?",
        "Explain artificial intelligence",
        "How do agentic architectures work?",
    ]
    
    # Run queries
    for query in queries:
        print("\n" + "="*50)
        print(f"QUERY: {query}")
        print("="*50)
        
        start = time.time()
        response = agentic_rag.invoke({"question": query})
        
        print("\nRESPONSE:")
        print(response["generation"])
        print(f"Time taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main() 