#!/usr/bin/env python
"""
Generate Q&A pairs using Groq LLM from web articles about machine learning and AI
Using LangChain to load documents, ChromaDB for retrieval, and evidence-based answers
Includes fake answer generation and grading functionality
"""

import re
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import json
from typing import List, Dict
import time
import shutil

# Load environment variables
load_dotenv(override=True)

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"  # You can change to another Groq model like "llama3-70b-8192"
CHUNK_SIZE = 2000  # Size of chunks for ChromaDB indexing
CHUNK_OVERLAP = 100  # Overlap between chunks
QUESTIONS_PER_DOC = 5  # Number of questions to generate per document
OUTPUT_FILE = "ai_ml_qa_pairs.json"
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "ml_ai_documents"

# URLs about ML and AI
URLS = [
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Deep_learning",
    # "https://en.wikipedia.org/wiki/Natural_language_processing",
    # "https://en.wikipedia.org/wiki/Computer_vision"
]

# Initialize LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=0.2,
)

# Initialize text splitter for ChromaDB indexing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Initialize embedding model
embedding_model = OpenAIEmbeddings()

# Initialize a single vectorstore for all documents
def initialize_vectorstore():
    """Initialize a single vectorstore for all documents"""
    # Optionally clear existing data
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        shutil.rmtree(CHROMA_PERSIST_DIRECTORY)
        print(f"Removed existing ChromaDB directory: {CHROMA_PERSIST_DIRECTORY}")
        
    # Create an empty vectorstore
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    print(f"Initialized ChromaDB collection: {COLLECTION_NAME}")
    return vectorstore

def load_url_as_document(url: str):
    """Load a URL as a document using LangChain WebBaseLoader"""
    print(f"Loading document from {url}")
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Add source URL to metadata if not already present
        for doc in documents:
            if 'source' not in doc.metadata:
                doc.metadata['source'] = url
                
        return documents
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return []

def add_document_to_vectorstore(document, doc_id, url, topic, vectorstore):
    """Add a document to the vectorstore"""
    # Split document into chunks
    chunks = text_splitter.split_documents([document])
    print(f"Split document into {len(chunks)} chunks for indexing")
    
    # Add document ID and topic to metadata
    for chunk in chunks:
        chunk.metadata['doc_id'] = doc_id
        chunk.metadata['source'] = url
        chunk.metadata['topic'] = topic
    
    # Add chunks to vectorstore
    vectorstore.add_documents(chunks)
    print(f"Added {len(chunks)} chunks to vectorstore from {url}")
    
    return len(chunks)

def retrieve_evidence(query, vectorstore, doc_id=None, k=3):
    """Retrieve evidence using LangChain and ChromaDB"""
    # Set up filter if doc_id is provided
    filter_dict = {"doc_id": doc_id} if doc_id else None
    
    # Use the similarity search to find relevant chunks
    relevant_chunks = vectorstore.similarity_search(
        query, 
        k=k,
        filter=filter_dict
    )
    
    # Format the evidence
    evidence_list = []
    for chunk in relevant_chunks:
        evidence = {
            "text": chunk.page_content,
            "metadata": chunk.metadata
        }
        evidence_list.append(evidence)
    
    return evidence_list

def generate_questions(doc_content, num_questions=QUESTIONS_PER_DOC):
    """Generate questions from a document using Groq LLM"""
    # Truncate content if it's too long to avoid token limits
    max_length = 12000  # Adjust based on model's context window
    if len(doc_content) > max_length:
        doc_content = doc_content[:max_length] + "... [content truncated]"
    
    prompt = f"""You are an AI teaching assistant creating educational questions about machine learning and AI.

The following is content from an article about machine learning or AI:

{doc_content}

Based on the content provided, generate {num_questions} diverse questions that would be useful for students learning about AI and machine learning. Follow these rules:

1. Each question should be clear and directly answerable from the text.
2. The questions should cover different concepts mentioned in the text.
3. Focus on substantive questions that test understanding, not simple factual recall.
4. Make questions specific enough that they can be answered based on the text.

Format your response as a JSON array with the following structure:
[
  {{
    "question": "The question text here?"
  }},
  ...
]

Return ONLY the JSON array, nothing else."""

    try:
        response = llm.invoke(prompt)
        questions_text = response.content
        
        # Extract JSON from response if there's any additional text
        match = re.search(r'\[\s*{.*}\s*\]', questions_text, re.DOTALL)
        if match:
            questions_text = match.group(0)
        
        # Parse JSON response
        questions = json.loads(questions_text)
        return questions
    
    except Exception as e:
        print(f"Error generating questions: {e}")
        print(f"Response was: {response.content if 'response' in locals() else 'No response'}")
        return []

def generate_answer_from_evidence(question, evidence_list):
    """Generate an answer based on the retrieved evidence"""
    # Combine all evidence texts
    combined_evidence = "\n\n".join([evidence["text"] for evidence in evidence_list])
    
    prompt = f"""You are an AI assistant answering questions based only on the provided evidence.

Question: {question}

Evidence:
{combined_evidence}

Using ONLY the information from the evidence above, provide a comprehensive but concise answer (2-4 sentences).
If the evidence doesn't contain information to answer the question, state that clearly.
Do not use any external knowledge. Base your answer solely on the evidence provided."""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error: Could not generate an answer."

def generate_fake_answer(question, topic):
    """Generate a potentially incorrect answer to a question"""
    prompt = f"""You are an AI that deliberately generates answers that contain factual errors.

Question: {question}
Topic: {topic}

Generate a plausible-sounding but factually incorrect or partially incorrect answer (2-4 sentences).
Your answer should be misleading in at least one significant way, but should still sound authoritative.
The errors should be subtle enough that they might not be obvious to someone without domain knowledge.
"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error generating fake answer: {e}")
        return "Error: Could not generate a fake answer."

def grade_answer(question, fake_answer, evidence_list):
    """Grade a fake answer against the evidence"""
    # Combine all evidence texts
    combined_evidence = "\n\n".join([evidence["text"] for evidence in evidence_list])
    
    prompt = f"""You are an AI assistant grading the factual accuracy of an answer based only on the provided evidence.

Question: {question}

Student Answer: {fake_answer}

Evidence:
{combined_evidence}

Evaluate the student answer for factual accuracy based ONLY on the evidence provided.
Provide your assessment in the following format:

Score: [Give a score from 0-10, where 0 is completely incorrect and 10 is completely accurate]
Feedback: [Explain what parts of the answer are incorrect or missing, and why]
Corrected Answer: [Provide the correct answer based on the evidence]

Be specific about any factual errors, omissions, or misleading statements in the answer.
"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error grading answer: {e}")
        return "Error: Could not grade the answer."

def main():
    # Initialize a single vectorstore for all documents
    vectorstore = initialize_vectorstore()
    
    all_qa_pairs = []
    total_documents = 0
    total_chunks = 0
    
    for url in URLS:
        # Extract topic name from URL
        topic = url.split('/')[-1].replace('_', ' ')
        
        # Load document from URL
        documents = load_url_as_document(url)
        if not documents:
            continue
        
        # Process each document
        for doc_index, doc in enumerate(documents):
            print(f"Processing document {doc_index+1}/{len(documents)} from {url}")
            
            # Generate a unique ID for the document
            doc_id = f"doc_{topic}_{doc_index}"
            
            # Get document content and source URL
            doc_content = doc.page_content
            source_url = doc.metadata.get('source', url)
            
            # Add document to the shared vectorstore
            num_chunks = add_document_to_vectorstore(doc, doc_id, source_url, topic, vectorstore)
            total_chunks += num_chunks
            
            # Generate questions for the document
            print(f"Generating questions for document")
            questions = generate_questions(doc_content)
            
            # For each question, retrieve evidence and generate answer
            for i, question_obj in enumerate(questions):
                question = question_obj["question"]
                print(f"Processing question {i+1}: {question}")
                
                # Retrieve evidence for the question (filter by doc_id to stay within this document)
                evidence_list = retrieve_evidence(question, vectorstore, doc_id=doc_id)
                
                # Generate correct answer from evidence
                correct_answer = generate_answer_from_evidence(question, evidence_list)
                
                # Generate fake answer
                print(f"Generating fake answer for question {i+1}")
                fake_answer = generate_fake_answer(question, topic)
                
                # Grade the fake answer
                print(f"Grading fake answer for question {i+1}")
                grading_result = grade_answer(question, fake_answer, evidence_list)
                
                # Create QA pair with evidence and grading
                qa_pair = {
                    "question": question,
                    "correct_answer": correct_answer,
                    "fake_answer": fake_answer,
                    "grading_result": grading_result,
                    "evidence": [ev["text"] for ev in evidence_list],
                    "evidence_metadata": [ev["metadata"] for ev in evidence_list],
                    "source_url": source_url,
                    "doc_id": doc_id,
                    "topic": topic
                }
                
                all_qa_pairs.append(qa_pair)
            
            total_documents += 1
    
    # Persist the vectorstore to disk
    vectorstore.persist()
    print(f"Persisted vectorstore to {CHROMA_PERSIST_DIRECTORY}")
    
    # Save all Q&A pairs to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    print(f"\nGenerated {len(all_qa_pairs)} Q&A pairs from {total_documents} documents.")
    print(f"Indexed {total_chunks} chunks in ChromaDB.")
    print(f"Results saved to {OUTPUT_FILE}")
    
    # Print a few examples
    print("\nSample Q&A pairs with grading:")
    for i, qa in enumerate(all_qa_pairs[:2]):
        print(f"\nQ{i+1}: {qa['question']}")
        print(f"Correct A{i+1}: {qa['correct_answer']}")
        print(f"Fake A{i+1}: {qa['fake_answer']}")
        print(f"Grading: {qa['grading_result']}")
        print(f"Topic: {qa['topic']}")
        print(f"Source: {qa['source_url']}")

if __name__ == "__main__":
    main()
