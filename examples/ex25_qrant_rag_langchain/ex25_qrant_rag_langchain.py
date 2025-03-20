import logging
import os
import asyncio
from uuid import uuid4
from datetime import datetime
import sys
from typing import List, Optional
import nest_asyncio

# Fix imports - remove relative paths
from db_utils import get_past_conversation_async, add_conversation_async
from langchain_utils import generate_chatbot_response, index_documents
from utils import extract_text_from_file

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_file(file_path: str, username: str):
    """Process a file and index its contents."""
    try:
        logger.info(f"Processing file: {file_path}")
        
        # Read file
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Extract file extension
        file_extension = file_path.split('.')[-1].lower()
        file_name = os.path.basename(file_path)
        
        # Extract text from file
        extracted_text = await extract_text_from_file(file_content, file_extension)
        logger.info(f"File content size: {len(file_content)} bytes")
        logger.info(f"Extracted text from file")
        
        # Index documents
        logger.info(f"Indexing documents in QdrantDB")
        await index_documents(username, extracted_text, file_name, file_extension)
        
        logger.info(f"File processed and indexed successfully: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return False

async def query_chatbot(query: str, username: str, session_id: Optional[str] = None, no_of_chunks: int = 3):
    """Generate a response to a query using RAG."""
    try:
        start_time = datetime.now()
        logger.info(f"Request started at {start_time}")
        logger.info(f"Received request from {username} for question: {query}")
        
        # Get past messages if session_id is provided
        past_messages = []
        if session_id is not None:
            logger.info(f"Fetching past messages")
            past_messages = await get_past_conversation_async(session_id)
            logger.info(f"Fetched past messages")
        else:
            session_id = str(uuid4())
            past_messages = []
        
        # Generate response
        logger.info(f"Generating chatbot response")
        response, response_time, input_tokens, output_tokens, total_tokens, final_context, refined_query, extracted_documents = await generate_chatbot_response(
            query, 
            past_messages,
            no_of_chunks,
            username
        )
        logger.info(f"Response generated for User question: {query}")
        
        # Add conversation to history
        logger.info(f"Adding conversation to chat history")
        await add_conversation_async(session_id, query, response)
        logger.info(f"Added conversation to chat history")
        
        # Create debug info
        sources = []
        if extracted_documents:
            sources = [{"file_name": doc.metadata.get("file_name", "unknown"), "context": doc.page_content} for doc in extracted_documents]
        
        end_time = datetime.now()
        logger.info(f"Request ended at {end_time}")
        
        # Print response and metrics
        print("\n" + "="*50)
        print(f"RESPONSE: {response}")
        print("="*50)
        print(f"Session ID: {session_id}")
        print(f"Response time: {response_time:.2f} seconds")
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Total tokens: {total_tokens}")
        print(f"Total processing time: {(end_time - start_time).total_seconds():.2f} seconds")
        print("="*50)
        
        # Print sources if available
        if sources:
            print("\nSOURCES:")
            for i, source in enumerate(sources, 1):
                print(f"{i}. {source['file_name']}")
                print(f"   {source['context'][:100]}...")
            print("="*50)
        
        return {
            "response": response,
            "session_id": session_id,
            "metrics": {
                "response_time": response_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "total_processing_time": (end_time - start_time).total_seconds()
            },
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        print(f"Error: {str(e)}")
        return None

async def main():
    """Example usage of the functions."""
    # Enable asyncio for Jupyter notebooks if needed
    try:
        nest_asyncio.apply()
    except:
        pass
    
    # Example: Process a file
    file_path = "/home/vankhoa@median.cad/code/github/agent_example/data/Weaviate Agentic Architectures-ebook.pdf"
    username = "example_user"
    await process_file(file_path, username)
    
    # Example: Query the chatbot
    query = "What is the main topic of the document?"
    result = await query_chatbot(query, username)
    print(result)
    
    # You can call these functions directly in your code
    # or modify this main function to suit your needs

if __name__ == "__main__":
    asyncio.run(main())
