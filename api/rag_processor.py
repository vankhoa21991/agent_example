#!/usr/bin/env python3
"""
rag_processor.py - LangGraph RAG system for document processing and querying

This module provides a RAG system that:
1. Classifies queries as either Q/A or summary tasks
2. Retrieves relevant documents for Q/A tasks
3. Generates summaries for summary tasks
4. Uses Groq LLM and LangGraph for the workflow
"""

import os
import logging
from typing import Dict, List, Optional, Union, TypedDict, Annotated, Literal
from datetime import datetime
from pathlib import Path

# LangGraph imports
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Import local modules
from rag import create_embedding_model, create_rag_provider
from prompts import (
    QUERY_CLASSIFICATION_PROMPT,
    QUESTION_ANSWERING_PROMPT,
    DOCUMENT_SUMMARIZATION_PROMPT,
    FINAL_SUMMARY_PROMPT
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_processor.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY: {GROQ_API_KEY}")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")
RAG_PROVIDER_TYPE = os.getenv("RAG_PROVIDER_TYPE", "chromadb")
EMBEDDING_MODEL_TYPE = os.getenv("EMBEDDING_MODEL_TYPE", "groq")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Define the state for our LangGraph
class RAGState(TypedDict):
    query: str
    query_type: Optional[str]
    documents: Optional[List[Document]]
    document_summaries: Optional[List[Dict]]
    generation: Optional[str]
    final_answer: Optional[str]
    session_id: Optional[str]

class RAGProcessor:
    """Handles RAG processing with LangGraph and Groq"""
    
    def __init__(self, rag_provider_type: str = RAG_PROVIDER_TYPE, embedding_model_type: str = EMBEDDING_MODEL_TYPE):
        """Initialize the RAG processor"""
        # Initialize LLM
        self.llm = ChatGroq(
            temperature=0,
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY
        )
        
        # Initialize embedding model
        self.embedding_model = create_embedding_model(embedding_model_type, OPENAI_API_KEY)
        
        # Initialize RAG provider
        self.rag_provider = create_rag_provider(
            provider_type=rag_provider_type,
            embedding_model=self.embedding_model
        )
        self.rag_provider.initialize()
        
        # Create retriever
        self.retriever = self.rag_provider.get_retriever(top_k=TOP_K_RESULTS)
        
        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()
        
        logger.info(f"RAG processor initialized successfully with provider: {rag_provider_type}")
    
    def _create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow
        
        Returns:
            StateGraph: The compiled workflow
        """
        # Create the graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("classify", self._classify_query)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("answer", self._answer_question)
        workflow.add_node("summarize", self._summarize_documents)
        workflow.add_node("final_summary", self._create_final_summary)
        
        # Set entry point
        workflow.set_entry_point("classify")
        
        # Add edges
        workflow.add_edge("classify", "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self._route_by_query_type,
            {
                "qa": "answer",
                "summary": "summarize"
            }
        )
        workflow.add_edge("summarize", "final_summary")
        workflow.add_edge("answer", END)
        workflow.add_edge("final_summary", END)
        
        # Compile the graph
        return workflow.compile()
    
    def _classify_query(self, state: RAGState) -> RAGState:
        """
        Classify a query as either 'qa' or 'summary'
        
        Args:
            state: Current state
            
        Returns:
            Updated state with query_type
        """
        logger.info(f"Classifying query: {state['query']}")
        
        # Create chain
        classification_chain = QUERY_CLASSIFICATION_PROMPT | self.llm | StrOutputParser()
        
        # Get classification
        classification = classification_chain.invoke({"query": state["query"]}).strip().lower()
        
        # Ensure we get a valid classification
        if classification not in ['qa', 'summary']:
            # Default to qa if classification is unclear
            classification = 'qa'
            
        logger.info(f"Query classified as: {classification}")
        
        # Update state
        return {**state, "query_type": classification}
    
    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents for the query
        
        Args:
            state: Current state
            
        Returns:
            Updated state with documents
        """
        logger.info(f"Retrieving documents for query: {state['query']}")
        
        # Check if session_id is provided
        session_id = state.get("session_id")
        
        if session_id:
            logger.info(f"Using session_id: {session_id} to retrieve documents")
            # Import database handler here to avoid circular imports
            from database import MongoDBHandler
            db_handler = MongoDBHandler()
            
            # Get documents from MongoDB by session_id
            mongo_documents = db_handler.get_documents_by_session(session_id)
            
            # Convert MongoDB documents to LangChain documents
            documents = []
            for doc in mongo_documents:
                documents.append(
                    Document(
                        page_content=doc["content"],
                        metadata={
                            **doc["metadata"],
                            "id": str(doc["_id"]),
                            "session_id": doc.get("session_id")
                        }
                    )
                )
            
            logger.info(f"Retrieved {len(documents)} documents from session {session_id}")
        else:
            # If no session_id, use the regular retriever
            documents = self.retriever.invoke(state["query"])
            logger.info(f"Retrieved {len(documents)} documents from vector store")
        
        # Update state
        return {**state, "documents": documents}
    
    def _answer_question(self, state: RAGState) -> RAGState:
        """
        Answer a question using RAG
        
        Args:
            state: Current state
            
        Returns:
            Updated state with final_answer
        """
        logger.info(f"Answering question: {state['query']}")
        
        # Format context
        documents = state["documents"]
        formatted_context = ""
        for i, doc in enumerate(documents):
            formatted_context += f"\n--- Document {i+1} ---\n"
            formatted_context += doc.page_content
            formatted_context += f"\n--- End Document {i+1} ---\n"
        
        # Create chain
        answer_chain = QUESTION_ANSWERING_PROMPT | self.llm | StrOutputParser()
        
        # Get answer
        answer = answer_chain.invoke({"context": formatted_context, "query": state["query"]})
        
        # Update state
        return {**state, "final_answer": answer}
    
    def _summarize_documents(self, state: RAGState) -> RAGState:
        """
        Summarize each document
        
        Args:
            state: Current state
            
        Returns:
            Updated state with document_summaries
        """
        logger.info("Summarizing documents")
        
        # Get documents from the state
        documents = state["documents"]
        
        document_summaries = []
        
        # Summarize each document
        for i, doc in enumerate(documents):
            # Create chain
            summary_chain = DOCUMENT_SUMMARIZATION_PROMPT | self.llm | StrOutputParser()
            
            # Get summary
            summary = summary_chain.invoke({"document": doc.page_content})
            
            # Add to summaries
            document_summaries.append({
                "summary": summary,
                "metadata": doc.metadata
            })
            
            logger.info(f"Summarized document {i+1}")
        
        # Update state
        return {**state, "document_summaries": document_summaries}
    
    def _create_final_summary(self, state: RAGState) -> RAGState:
        """
        Create a final summary from individual document summaries
        
        Args:
            state: Current state
            
        Returns:
            Updated state with final_answer
        """
        logger.info("Creating final summary")
        
        # Format summaries
        summaries = state["document_summaries"]
        formatted_summaries = ""
        for i, summary in enumerate(summaries):
            formatted_summaries += f"\n--- Document {i+1} ---\n"
            formatted_summaries += summary["summary"]
            formatted_summaries += f"\n--- End Document {i+1} ---\n"
        
        # Create chain
        final_summary_chain = FINAL_SUMMARY_PROMPT | self.llm | StrOutputParser()
        
        # Get final summary
        final_summary = final_summary_chain.invoke({"summaries": formatted_summaries})
        
        # Update state
        return {**state, "final_answer": final_summary}
    
    def _route_by_query_type(self, state: RAGState) -> str:
        """
        Route to the appropriate node based on query type
        
        Args:
            state: Current state
            
        Returns:
            Next node name
        """
        query_type = state["query_type"]
        
        if query_type == "qa":
            return "qa"
        else:
            return "summary"
    
    def process_query(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Process a user query using the LangGraph workflow
        
        Args:
            query: User query
            session_id: Optional session ID to retrieve documents from
            
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
            "final_answer": None,
            "session_id": session_id
        }
        
        # Run the workflow
        result = self.workflow.invoke(initial_state)
        
        # Return the final answer
        return result["final_answer"]
    
    def index_document(self, content: str, metadata: Dict) -> List[str]:
        """
        Index a document
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            List of document IDs
        """
        return self.rag_provider.index_document(content, metadata)
