import langsmith as ls
from langsmith.wrappers import wrap_openai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import time
import os
import logging
import asyncio
from dotenv import load_dotenv
from qdrant_client import QdrantClient, AsyncQdrantClient

# Fix imports - remove relative paths
from prompts import get_query_refiner_prompt, get_main_prompt
from qdrant_utils import DocumentIndexer

# Configure logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
qdrant_db_path = os.getenv("qdrant_db_path", "./qdrant_storage")  # Default path if not set


async def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

async def index_documents(username, extracted_text, filename, file_extension):
    try:
        indexer = DocumentIndexer(qdrant_db_path)
        start_time = time.time()
        logger.info("Indexing documents in QdrantDB...")

        await indexer.index_in_qdrantdb(
            extracted_text=extracted_text,
            file_name=filename,
            doc_type=file_extension,
            chunk_size=1500  
        )
        logger.info(f"Document indexing completed in {time.time() - start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise RuntimeError(f"Failed to process documents: {str(e)}")


async def retrieve_similar_documents(refined_query: str, num_of_chunks: int, username: str):
    try:
        indexer = DocumentIndexer(qdrant_db_path)
        start_time = time.time()
        logger.info("Searching for similar documents in QdrantDB...")

        if num_of_chunks is None:
            num_of_chunks = int(os.getenv('no_of_chunks', 3))
        if not isinstance(num_of_chunks, int) or num_of_chunks <= 0:
            raise ValueError(f"Invalid number of chunks: {num_of_chunks}")
            
        retriever = await indexer.get_retriever(top_k=num_of_chunks)
        if not retriever:
            raise ValueError("Failed to initialize document retriever")
            
        extracted_documents = await retriever.ainvoke(refined_query)
        if not extracted_documents:
            extracted_text_data = ""
        else:
            extracted_text_data = await format_docs(extracted_documents)
            
        logger.info(f"Document retrieval and formatting completed in {time.time() - start_time:.2f} seconds")
        return extracted_text_data, extracted_documents

    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise RuntimeError(f"Failed to process documents: {str(e)}")


async def invoke_chain(query, context, history, llm):
    """Handles the streamed response asynchronously."""
    logger.info("Initializing Chain...")
    final_chain = get_main_prompt() | llm | StrOutputParser()
    logger.info("Chain initialized.")
    input_data = {"user_query": query, "context": context, "messages": history.messages}

    with get_openai_callback() as cb:
        final_response = await final_chain.ainvoke(input_data)  # Asynchronous method

    return final_response, cb


def create_history(messages):
    history = InMemoryChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])

    return history


def initialize_llm(model=None, temperature=None, llm_provider=None):
    """Initialize the language model."""
    if temperature is None:
        temperature = float(os.getenv('temperature', 0.7))
    if llm_provider is None:
        llm_provider = os.getenv('llm_provider', "openai")
        model = os.getenv('model', "gpt-4o")

    if llm_provider == "openai":
        logger.info(f"Initializing OpenAI model with values {model} and {temperature}")
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=temperature, 
            model_name=model,
            streaming=True,
            stream_usage=True
        )
    return llm


async def refine_user_query(query, messages):
    """Refines the user query asynchronously."""
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    history = create_history(messages)
    prompt = get_query_refiner_prompt()
    refined_query_chain = prompt | llm | StrOutputParser()
    refined_query = await refined_query_chain.ainvoke({"query": query, "messages": history.messages})  # Async method
    return refined_query


@ls.traceable(run_type="chain", name="Chat Pipeline")
async def generate_chatbot_response(query, past_messages, no_of_chunks, username):
    """Main function to generate chatbot responses asynchronously."""
    logger.info("Refining user query")
    refined_query = await refine_user_query(query, past_messages)  # Async call
    logger.info(f"Generated refined query: {refined_query}")

    extracted_text_data, extracted_documents = await retrieve_similar_documents(refined_query, int(no_of_chunks), username)  # Async call
    logger.info("Extracted text data")

    llm = initialize_llm()  # Synchronous initialization
    history = create_history(past_messages)
    logger.info(f"Created history for session")

    logger.info("Fetching response")
    start_time = time.time()
    final_response, cb = await invoke_chain(query, extracted_text_data, history, llm)  # Async call
    response_time = time.time() - start_time

    logger.info("Got response from chain")

    return final_response, response_time, cb.prompt_tokens, cb.completion_tokens, cb.total_tokens, extracted_text_data, refined_query, extracted_documents
