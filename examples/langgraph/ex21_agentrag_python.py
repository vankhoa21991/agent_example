# Agentic RAG system without langgraph, using chatgroq and tools
from dotenv import load_dotenv
load_dotenv()
import os
import gzip
import json
from typing import List, Dict, Any, Optional
import time

# Import necessary libraries for embeddings and vector store
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma

# Import Groq for LLM
from groq import Groq
from langchain_groq import ChatGroq

# Initialize the Groq client and LLM
client = Groq()
MODEL = 'llama-3.3-70b-versatile'

# Initialize LLM for langchain
llm = ChatGroq(
    temperature=0,
    model_name=MODEL,
    api_key=os.getenv("GROQ_API_KEY")
)

# Use FastEmbed for embeddings (more efficient than OpenAI embeddings)
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Load and process Wikipedia data
wikipedia_filepath = 'data/simplewiki-2020-11-01.jsonl.gz'

print("Loading Wikipedia data...")
docs = []
with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        data = json.loads(line.strip())
        docs.append({
            'metadata': {
                'title': data.get('title'),
                'article_id': data.get('id')
            },
            'data': ' '.join(data.get('paragraphs')[0:3])  # restrict data to first 3 paragraphs
        })

# Subset data to only include documents about India for faster processing
docs = [doc for doc in docs for x in ['india']
        if x in doc['data'].lower().split()]

# Create Document objects
docs = [Document(page_content=doc['data'],
                 metadata=doc['metadata']) for doc in docs]

# Chunk documents
print("Chunking documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
chunked_docs = splitter.split_documents(docs)

print(f"Number of document chunks: {len(chunked_docs)}")

# Create vector DB
print("Creating vector database...")
chroma_db = Chroma.from_documents(
    documents=chunked_docs,
    collection_name='rag_wikipedia_db',
    embedding=embed_model,
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="./wikipedia_db"
)

# Create retriever with similarity threshold
similarity_threshold_retriever = chroma_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3}
)

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

# Function to rewrite query
def rewrite_query(question: str) -> str:
    """
    Rewrites the query to produce a better question for search.
    """
    print(f"Rewriting query: {question}")
    
    messages = [
        {
            "role": "system",
            "content": "Act as a question re-writer and perform the following task: "
                      "- Convert the following input question to a better version that is optimized for web search. "
                      "- When re-writing, look at the input question and try to reason about the underlying semantic intent / meaning."
        },
        {
            "role": "user",
            "content": f"Here is the initial question:\n{question}\n\nFormulate an improved question."
        }
    ]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=False
    )
    
    rewritten_query = response.choices[0].message.content.strip()
    print(f"Rewritten query: {rewritten_query}")
    return rewritten_query

# Function to perform web search (simulated)
def web_search(question: str) -> Document:
    """
    Simulates a web search for the question.
    In a real implementation, this would use an actual search API.
    """
    print(f"Performing web search for: {question}")
    
    # Simulate web search with a delay
    time.sleep(1)
    
    # For demonstration, we'll return a generic response about the topic
    if "india" in question.lower():
        content = "India is a country in South Asia. It is the seventh-largest country by area, the second-most populous country, and the most populous democracy in the world. New Delhi is the capital of India. The Indian subcontinent was home to the Indus Valley Civilisation. The capital of India is New Delhi."
    elif "champions league" in question.lower():
        content = "The UEFA Champions League is an annual club football competition organised by the Union of European Football Associations (UEFA) and contested by top-division European clubs, deciding the competition winners through a round robin group stage to qualify for a double-legged knockout format, and a single leg final. Real Madrid won the Champions League in 2024, defeating Borussia Dortmund in the final."
    else:
        content = f"Web search results for: {question}. Please note that this is a simulated response as this example doesn't include actual web search functionality."
    
    return Document(page_content=content)

# Function to generate answer from context
def generate_answer(question: str, documents: List[Document]) -> str:
    """
    Generate an answer from the context documents using LLM.
    """
    print(f"Generating answer for question: {question}")
    
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
    
    answer = response.choices[0].message.content.strip()
    return answer

# Main function to orchestrate the agentic RAG process
def agentic_rag(question: str) -> Dict[str, Any]:
    """
    Orchestrates the agentic RAG process without using langgraph.
    """
    print(f"\n--- AGENTIC RAG PROCESS FOR: '{question}' ---\n")
    
    # Step 1: Retrieve documents
    print("--- RETRIEVAL FROM VECTOR DB ---")
    documents = similarity_threshold_retriever.invoke(question)
    
    # Step 2: Grade documents
    print("--- CHECK DOCUMENT RELEVANCE TO QUESTION ---")
    filtered_docs = []
    web_search_needed = False
    
    if documents:
        for doc in documents:
            is_relevant = grade_document(question, doc.page_content)
            if is_relevant:
                print("--- GRADE: DOCUMENT RELEVANT ---")
                filtered_docs.append(doc)
            else:
                print("--- GRADE: DOCUMENT NOT RELEVANT ---")
                web_search_needed = True
    else:
        print("--- NO DOCUMENTS RETRIEVED ---")
        web_search_needed = True
    
    # Step 3: Decide whether to rewrite query and perform web search
    if web_search_needed:
        print("--- DECISION: SOME or ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, REWRITE QUERY ---")
        
        # Step 3a: Rewrite query
        better_question = rewrite_query(question)
        
        # Step 3b: Perform web search
        print("--- WEB SEARCH ---")
        web_result = web_search(better_question)
        filtered_docs.append(web_result)
    else:
        print("--- DECISION: GENERATE RESPONSE ---")
        # No need to rewrite query or perform web search
    
    # Step 4: Generate answer
    print("--- GENERATE ANSWER ---")
    answer = generate_answer(question, filtered_docs)
    
    # Return the final state
    return {
        "question": question,
        "documents": filtered_docs,
        "generation": answer
    }

# Example usage
if __name__ == "__main__":
    # Example 1: Question about India (should find in vector DB)
    query1 = "what is the capital of India?"
    response1 = agentic_rag(query1)
    print("\nFINAL ANSWER:")
    print(response1["generation"])
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Question not in vector DB (should trigger web search)
    query2 = "who won the champions league in 2024?"
    response2 = agentic_rag(query2)
    print("\nFINAL ANSWER:")
    print(response2["generation"])
