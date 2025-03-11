# https://medium.com/the-ai-forum/implementing-agentic-rag-using-langchain-b22af7f6a3b5

from uuid import uuid4
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Load the document pertaining to a particular topic
docs = ArxivLoader(query="Retrieval Augmented Generation", load_max_docs=5).load()

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=350, chunk_overlap=50
)

chunked_documents = text_splitter.split_documents(docs)

# Instantiate the Embedding Model
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.environ['OPENAI_API_KEY1'])
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Create Index - Load document chunks into the vectorstore
faiss_vectorstore = FAISS.from_documents(
    documents=chunked_documents,
    embedding=embeddings,
)

# Create a retriever
retriever = faiss_vectorstore.as_retriever()

from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = """\
Use the following context to answer the user's query. If you cannot answer the question, please respond with 'I don't know'.

Question:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

from langchain_groq import ChatGroq

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel

# Define the RAG chain
retriever_chain = RunnableParallel(
    {"context": itemgetter("question") | retriever, "question": RunnablePassthrough()}
)

format_docs = lambda docs: "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {"question": RunnablePassthrough()}
    | retriever_chain
    | {"context": lambda x: format_docs(x["context"]), "question": itemgetter("question")}
    | {"response": rag_prompt | llm | StrOutputParser(), "context": itemgetter("context")}
)

# Run the chain
result = rag_chain.invoke({"question": "What is Retrieval Augmented Generation?"})

# Print the response
print(result["response"])

print("\n" + "="*50 + "\n" + "AGENTIC RAG IMPLEMENTATION" + "\n" + "="*50 + "\n")

# ===== AGENTIC RAG IMPLEMENTATION =====

from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain.agents import tool

# Define tools
search_tool = DuckDuckGoSearchRun()
arxiv_tool = ArxivQueryRun()

# Create a simple tool executor function
def execute_tool(tool_name, tool_input):
    print(f"Executing tool: {tool_name} with input: {tool_input}")
    if tool_name == "duckduckgo_search":
        return search_tool.invoke(tool_input.get("query", ""))
    elif tool_name == "arxiv_query":
        return arxiv_tool.invoke(tool_input)
    else:
        return f"Tool {tool_name} not found"

from langchain_core.utils.function_calling import convert_to_openai_function

# Create a new LLM instance for the agent to avoid conflicts
agent_llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Convert tools to OpenAI tools format
tools = [search_tool, arxiv_tool]
agent_model = agent_llm.bind_tools(tools)

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

import json
from langchain_core.messages import FunctionMessage

def call_model(state):
    messages = state["messages"]
    response = agent_model.invoke(messages)
    return {"messages": [response]}

def call_tool(state):
    last_message = state["messages"][-1]
    
    # Check for tool_calls (newer format) or function_call (older format)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call.get("name", "")
        tool_input = tool_call.get("args", {})
    else:
        # Extract function call details from additional_kwargs
        tool_calls = last_message.additional_kwargs.get("tool_calls", [])
        if tool_calls:
            tool_call = tool_calls[0]
            tool_name = tool_call.get("function", {}).get("name", "")
            args_str = tool_call.get("function", {}).get("arguments", "{}")
            try:
                tool_input = json.loads(args_str)
            except json.JSONDecodeError:
                tool_input = {"query": args_str}
        else:
            function_call = last_message.additional_kwargs.get("function_call", {})
            tool_name = function_call.get("name", "")
            try:
                tool_input = json.loads(function_call.get("arguments", "{}"))
            except json.JSONDecodeError:
                tool_input = {"query": function_call.get("arguments", "")}
    
    # Execute the tool
    response = execute_tool(tool_name, tool_input)
    
    # Create a function message with the response
    function_message = FunctionMessage(content=str(response), name=tool_name)
    
    return {"messages": [function_message]}

from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")

def should_continue(state):
    last_message = state["messages"][-1]
    
    # Check for tool_calls (newer format) or function_call (older format)
    has_tool_call = False
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        has_tool_call = True
    elif "tool_calls" in last_message.additional_kwargs and last_message.additional_kwargs["tool_calls"]:
        has_tool_call = True
    elif "function_call" in last_message.additional_kwargs and last_message.additional_kwargs["function_call"]:
        has_tool_call = True
    
    if not has_tool_call:
        print("No more tool calls, ending conversation")
        return "end"
    
    print("Continuing with tool execution")
    return "continue"

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)

workflow.add_edge("action", "agent")

app = workflow.compile()

from langchain_core.messages import HumanMessage

# Create a system message to guide the agent
from langchain_core.messages import SystemMessage

system_message = SystemMessage(content="""You are an AI assistant that helps answer questions about AI and machine learning topics.
When you don't know the answer, use the available tools to search for information.
After using tools, always provide a comprehensive answer to the original question based on the information you found.
""")

inputs = {"messages": [
    system_message,
    HumanMessage(content="What is RAG in the context of Large Language Models? When did it break onto the scene?")
]}

try:
    response = app.invoke(inputs)
    
    # Extract and print the final response in a more readable format
    final_messages = response.get("messages", [])
    if final_messages:
        # Get the last AI message with content
        for msg in reversed(final_messages):
            if hasattr(msg, "content") and msg.content and not msg.content.isspace():
                print("\nAgentic RAG Response:")
                print(msg.content)
                break
        else:
            print("\nNo final response found in the agentic RAG output.")
    else:
        print("\nNo messages found in the agentic RAG output.")
        
except Exception as e:
    print(f"\nError in agentic RAG: {e}")
    print("Note: Basic RAG implementation completed successfully.")
