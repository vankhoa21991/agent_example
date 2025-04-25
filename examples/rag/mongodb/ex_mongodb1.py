import os

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

MONGODB_URI = os.getenv('MONGO_AUTH')
openai_client = OpenAI()

from langchain_community.document_loaders import WebBaseLoader

web_loader = WebBaseLoader(
    [
        "https://peps.python.org/pep-0483/",
        "https://peps.python.org/pep-0008/",
        "https://peps.python.org/pep-0257/",
    ]
)

pages = web_loader.load()

from typing import Dict, List, Optional

from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

def fixed_token_split(
    docs: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """
    Fixed token chunking

    Args:
        docs (List[Document]): List of documents to chunk
        chunk_size (int): Chunk size (number of tokens)
        chunk_overlap (int): Token overlap between chunks

    Returns:
        List[Document]: List of chunked documents
    """
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def recursive_split(
    docs: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    language: Optional[Language] = None,
) -> List[Document]:
    """
    Recursive chunking

    Args:
        docs (List[Document]): List of documents to chunk
        chunk_size (int): Chunk size (number of tokens)
        chunk_overlap (int): Token overlap between chunks
        language (Optional[Language], optional): Language enum name. Defaults to None.

    Returns:
        List[Document]: List of chunked documents
    """
    separators = ["\n\n", "\n", " ", ""]

    if language is not None:
        try:
            separators = RecursiveCharacterTextSplitter.get_separators_for_language(
                language
            )
        except (NameError, ValueError):
            print(f"No separators found for language {language}. Using defaults.")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    return splitter.split_documents(docs)

def semantic_split(docs: List[Document]) -> List[Document]:
    """
    Semantic chunking

    Args:
        docs (List[Document]): List of documents to chunk

    Returns:
        List[Document]: List of chunked documents
    """
    splitter = SemanticChunker(
        OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
    )
    return splitter.split_documents(docs)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import RunConfig
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator

RUN_CONFIG = RunConfig(max_workers=4, max_wait=180)

# Generator with openai models
generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

# Set question type distribution
distributions = {simple: 0.5, multi_context: 0.4, reasoning: 0.1}

testset = generator.generate_with_langchain_docs(
    pages, 10, distributions, run_config=RUN_CONFIG
)

testset = testset.to_pandas()

len(testset)

from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

client = MongoClient(MONGODB_URI, appname="devrel.showcase.chunking_strategies")
DB_NAME = "evals"
COLLECTION_NAME = "chunking"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

def create_vector_store(docs: List[Document]) -> MongoDBAtlasVectorSearch:
    """
    Create MongoDB Atlas vector store

    Args:
        docs (List[Document]): List of documents to create the vector store

    Returns:
        MongoDBAtlasVectorSearch: MongoDB Atlas vector store
    """
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    return vector_store

import nest_asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from tqdm import tqdm

# Allow nested use of asyncio (used by RAGAS)
nest_asyncio.apply()

# Disable tqdm locks
tqdm.get_lock().locks = []

QUESTIONS = testset.question.to_list()
GROUND_TRUTH = testset.ground_truth.to_list()

def perform_eval(docs: List[Document]) -> Dict[str, float]:
    """
    Perform RAGAS evaluation

    Args:
        docs (List[Document]): List of documents to create the vector store

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    eval_data = {
        "question": QUESTIONS,
        "ground_truth": GROUND_TRUTH,
        "contexts": [],
    }

    print(f"Deleting existing documents in the collection {DB_NAME}.{COLLECTION_NAME}")
    MONGODB_COLLECTION.delete_many({})
    print("Deletion complete")
    vector_store = create_vector_store(docs)

    # Getting relevant documents for questions in the evaluation dataset
    print("Getting contexts for evaluation set")
    for question in tqdm(QUESTIONS):
        eval_data["contexts"].append(
            [doc.page_content for doc in vector_store.similarity_search(question, k=3)]
        )
    # RAGAS expects a Dataset object
    dataset = Dataset.from_dict(eval_data)

    print("Running evals")
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall],
        run_config=RUN_CONFIG,
        raise_exceptions=False,
    )
    return result

for chunk_size in [100, 200, 500, 1000]:
    chunk_overlap = int(0.15 * chunk_size)
    print(f"CHUNK SIZE: {chunk_size}")
    print("------ Fixed token without overlap ------")
    print(f"Result: {perform_eval(fixed_token_split(pages, chunk_size, 0))}")
    print("------ Fixed token with overlap ------")
    print(
        f"Result: {perform_eval(fixed_token_split(pages, chunk_size, chunk_overlap))}"
    )
    print("------ Recursive with overlap ------")
    print(f"Result: {perform_eval(recursive_split(pages, chunk_size, chunk_overlap))}")
    print("------ Recursive Python splitter with overlap ------")
    print(
        f"Result: {perform_eval(recursive_split(pages, chunk_size, chunk_overlap, Language.PYTHON))}"
    )
print("------ Semantic chunking ------")
print(f"Result: {perform_eval(semantic_split(pages))}")