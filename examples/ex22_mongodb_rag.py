from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from datasets import load_dataset
import os

MONGODB_URI = os.getenv("MONGO_AUTH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load MongoDB's embedded_movies dataset from Hugging Face
data = load_dataset("MongoDB/embedded_movies")

df = pd.DataFrame(data["train"])

df.head(1)

df = df[df["fullplot"].notna()]
df.rename(columns={"plot_embedding": "embedding"}, inplace=True)

from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Initialize MongoDB python client
client = MongoClient(
    MONGODB_URI, appname="devrel.showcase.mongodb_langchain_cache_memory"
)

DB_NAME = "langchain_chatbot"
COLLECTION_NAME = "data"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client[DB_NAME][COLLECTION_NAME]

collection.delete_many({})

# Data Ingestion
records = df.to_dict("records")
collection.insert_many(records)

print("Data ingestion into MongoDB completed")

from langchain_openai import OpenAIEmbeddings

# Using the text-embedding-ada-002 since that's what was used to create embeddings in the movies dataset
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002"
)

# Vector Store Creation
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGODB_URI,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    text_key="fullplot",
)

# Using the MongoDB vector store as a retriever in a RAG chain
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Generate context using the retriever, and pass the user question through
retrieve = {
    "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
    "question": RunnablePassthrough(),
}
template = """Answer the question based only on the following context: \
{context}

Question: {question}
"""
# Defining the chat prompt
prompt = ChatPromptTemplate.from_template(template)
# Defining the model to be used for chat completion
model = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
# Parse output as a string
parse_output = StrOutputParser()

# Naive RAG chain
naive_rag_chain = retrieve | prompt | model | parse_output

from langchain_core.globals import set_llm_cache
from langchain_mongodb.cache import MongoDBAtlasSemanticCache

# set_llm_cache(
#     MongoDBAtlasSemanticCache(
#         connection_string=MONGODB_URI,
#         embedding=embeddings,
#         collection_name="semantic_cache",
#         database_name=DB_NAME,
#         index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
#         wait_until_ready=True,  # Optional, waits until the cache is ready to be used
#     )
# )

import time

start = time.time()
response = naive_rag_chain.invoke("What is the best movie to watch when sad?")

print(response)
print("Time taken:", time.time() - start)

# pause 1s
 
# time.sleep(1)

# start = time.time()
# response = naive_rag_chain.invoke("What is the best movie to watch when sad?")

# print(response)
# print("Time taken:", time.time() - start)