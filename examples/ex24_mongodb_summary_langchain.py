# https://github.com/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/chat_with_pdf_mongodb_openai_langchain_POLM_AI_Stack.ipynb
import os

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()


# Set up your OpenAI API key

# Set up MongoDB connection
mongo_uri = os.getenv("MONGO_AUTH")
db_name = "anthropic_demo"
collection_name = "research"

client = MongoClient(mongo_uri, appname="devrel.showcase.chat_with_pdf")
db = client[db_name]
collection = db[collection_name]

# Set up document loading and splitting
loader = PyPDFLoader("/home/vankhoa@median.cad/code/github/agent_example/data/Weaviate Agentic Architectures-ebook.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Set up embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = MongoDBAtlasVectorSearch.from_documents(
    texts, embeddings, collection=collection, index_name="vector_index"
)

# Set up retriever and language model
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Set up RAG pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)


# Function to process user query
def process_query(query):
    result = qa_chain.invoke({"query": query})
    return result["result"], result["source_documents"]


# Example usage
query = "What is the document about?"
answer, sources = process_query(query)
print(f"Answer: {answer}")
print("Sources:")
for doc in sources:
    print(f"- {doc.metadata['source']}: {doc.page_content[:100]}...")

# Don't forget to close the MongoDB connection when done
client.close()