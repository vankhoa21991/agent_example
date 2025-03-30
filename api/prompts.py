#!/usr/bin/env python3
"""
prompts.py - Prompts for the RAG system

This module contains all the prompts used by the RAG system.
"""

from langchain_core.prompts import PromptTemplate

# Query classification prompt
QUERY_CLASSIFICATION_PROMPT = PromptTemplate(
    template="""You are a query classifier. Your task is to determine if a user query is asking for:
    1. A question-answering task (labeled as 'qa') - where the user is seeking specific information or answers
    2. A summarization task (labeled as 'summary') - where the user is asking for a summary or overview
    
    Respond with ONLY 'qa' or 'summary'.
    
    User query: {query}
    Classification:""",
    input_variables=["query"],
)

# Question answering prompt
QUESTION_ANSWERING_PROMPT = PromptTemplate(
    template="""You are a helpful assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't find the answer in the context, just say that you don't have that information.
    
    Context:
    {context}
    
    Question: {query}
    Answer:""",
    input_variables=["context", "query"],
)

# Document summarization prompt
DOCUMENT_SUMMARIZATION_PROMPT = PromptTemplate(
    template="""You are a document summarizer. Your task is to create a concise yet comprehensive summary of the document.
    Focus on the main points, key findings, and important details.
    Make the summary clear, well-structured, and informative.
    
    Document:
    {document}
    
    Summary:""",
    input_variables=["document"],
)

# Final summary prompt
FINAL_SUMMARY_PROMPT = PromptTemplate(
    template="""You are a document synthesizer. Your task is to create a comprehensive final summary that reorganizes and integrates information from multiple document summaries.
    Identify common themes, highlight key points, and present a coherent overview that connects information across all documents.
    Make the final summary well-structured, informative, and easy to understand.
    
    Document summaries:
    {summaries}
    
    Final summary:""",
    input_variables=["summaries"],
)
