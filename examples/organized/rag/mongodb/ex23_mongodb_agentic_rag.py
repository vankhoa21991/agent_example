from dotenv import load_dotenv
load_dotenv()

import os
import time
import urllib.request
from io import BytesIO
from typing import List, Dict, Any, Literal, TypedDict
from typing_extensions import TypedDict
from tqdm import tqdm
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
MONGODB_URI = os.getenv("MONGO_AUTH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB setup
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq

# Initialize MongoDB python client
client = MongoClient(
    MONGODB_URI, appname="devrel.showcase.mongodb_langchain_agentic_rag"
)

DB_NAME = "agentic_rag_db"
COLLECTION_NAME = "documents"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client[DB_NAME][COLLECTION_NAME]

# Clear existing data
collection.delete_many({})

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

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

# Initialize vector store after loading documents
def initialize_vector_store():
    """Initialize the MongoDB Atlas Vector Search"""
    try:
        vector_store = MongoDBAtlasVectorSearch(
            # connection_string=MONGODB_URI,
            namespace=f"{DB_NAME}.{COLLECTION_NAME}",
            collection=collection,
            embedding=embeddings,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
            text_key="text",
            relevance_score_fn="cosine"  # Add relevance score function
        )
        
        # Create vector search index with appropriate dimensions for the embedding model
        # For OpenAI text-embedding-3-small, the dimension is 1536
        # vector_store.create_vector_search_index(dimensions=1536)
        print("Vector search index created successfully")
        
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

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# MODEL = 'llama-3.3-70b-versatile'

# # Initialize LLM for langchain
# llm = ChatGroq(
#     temperature=0,
#     model_name=MODEL,
#     api_key=os.getenv("GROQ_API_KEY")
# )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt template for grading
SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question.
                Follow these instructions for grading:
                  - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                  - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not.
             """
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Retrieved document:
                     {document}

                     User question:
                     {question}
                  """),
    ]
)

# Build grader chain
doc_grader = grade_prompt | structured_llm_grader

# Create RAG prompt for response generation
prompt = """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If no context is present or if you don't know the answer, just say that you don't know the answer.
            Do not make up the answer unless it is there in the provided context.
            Give a detailed answer and to the point answer with regard to the question.

            Question:
            {question}

            Context:
            {context}

            Answer:
         """
prompt_template = ChatPromptTemplate.from_template(prompt)

# Used for separating context docs with new lines
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create QA RAG chain
qa_rag_chain = (
    {
        "context": (itemgetter('context') | RunnableLambda(format_docs)),
        "question": itemgetter('question')
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# LangGraph node functions

def retrieve(state: GraphState, vector_store):
    """
    Retrieve documents from vector store
    """
    print("---RETRIEVAL FROM MONGODB---")
    question = state["question"]

    try:
        # Retrieval with similarity search and score threshold
        # This ensures we only get relevant documents
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,  # Retrieve top 3 documents
                "score_threshold": 0.0  # Only include documents with similarity score above threshold
            }
        )
        documents = retriever.invoke(question)
        print(f"Retrieved {len(documents)} documents from MongoDB Atlas")
        
        return {"documents": documents, "question": question}
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        # Return empty documents list in case of error
        return {"documents": [], "question": question}

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
            score = doc_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["binary_score"]
            if grade == "yes":
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

    # RAG generation
    generation = qa_rag_chain.invoke({"context": documents, "question": question})
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
    
    # Simple prompt for no-context answers
    no_context_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. The user has asked: {question}
        
        You don't have specific information about this in your knowledge base.
        Please politely explain that you don't have enough information to provide a specific answer.
        """
    )
    
    # Generate response
    generation = (no_context_prompt | llm | StrOutputParser()).invoke({"question": question})
    
    return {
        "documents": state["documents"],
        "question": question,
        "generation": generation
    }

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
        # {"source": "data/fibromyalgia.pdf", "type": "pdf"},
        {"source": "https://en.wikipedia.org/wiki/Machine_learning", "type": "url"},
        # {"source": "data/Weaviate Agentic Architectures-ebook.pdf", "type": "pdf"}
    ]
    
    # Initialize vector store first
    vector_store = initialize_vector_store()
    print("Vector store initialized")
    
    # Process and load each document one by one using vector store
    from uuid import uuid4
    
    for source_info in sources:
        # Process document to get chunks
        chunks = process_document(source_info["source"], source_info["type"])
        
        if chunks:
            # Generate unique IDs for each chunk
            chunk_ids = [str(uuid4()) for _ in range(len(chunks))]
            
            # Add documents to vector store
            vector_store.add_documents(documents=chunks, ids=chunk_ids)
            print(f"Successfully loaded {len(chunks)} chunks from {source_info['source']} into MongoDB Atlas")
        else:
            print(f"No chunks to load from {source_info['source']}")
        
        time.sleep(1)  # Small delay between processing documents
    
    # Create the agentic RAG graph
    agentic_rag = create_agentic_rag_graph(vector_store)
    print("Agentic RAG graph created")
    
    # Example queries
    queries = [
        # "What is machine learning?",
        "summarize the documents",
        # "What are the symptoms of fibromyalgia?",
        # "How do agentic architectures work?",
        # "What is the capital of France?"  # This should trigger the no-context path
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
