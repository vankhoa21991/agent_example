import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import bs4
from langchain.schema import Document

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Use WebBaseLoader instead of Wikipedia
try:
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    )
    documents = loader.load()
    print(f"Successfully loaded content from blog post")
except Exception as e:
    print(f"Error fetching web content: {e}")
    # Fallback content if web fetch fails
    documents = [Document(
        page_content="LLM Agents are systems that use LLMs to interact with tools, making them more capable of solving complex tasks. They use techniques like chain-of-thought reasoning and ReAct to intelligently invoke tools and respond to user queries.",
        metadata={"source": "fallback-content"}
    )]

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Initialize Chroma with OpenAI Embeddings
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# Save the DB
vectordb.persist()

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = OpenAI(temperature=0)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

query = "What are the key components of LLM agents?"
result = qa_chain(query)

print("Answer:\n", result['result'])
print("\nRetrieved Docs:\n", [doc.page_content for doc in result['source_documents']])


from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_similarity,
)
from datasets import Dataset

# Reference answer (relevant to agents)
reference_answer = "Key components of LLM agents include a base language model, a planning/reasoning mechanism, tool usage capabilities, memory systems, and feedback mechanisms."

# Build dataset
data = {
    "question": [query],
    "contexts": [[doc.page_content for doc in result['source_documents']]],
    "answer": [result['result']],
    "ground_truth": [reference_answer]  # only needed for answer_similarity
}

ragas_dataset = Dataset.from_dict(data)

# Evaluate all available metrics
eval_results = evaluate(
    ragas_dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_similarity,  # optional but nice to have if ground_truth is present
    ]
)

print("\n📊 Evaluation Metrics:")
print(eval_results)
