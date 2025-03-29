# https://blog.futuresmart.ai/langgraph-tutorial-for-beginners

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
load_dotenv()
ANTROPIC_API_KEY = os.getenv("ANTROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class State(TypedDict):
    # 'messages' will store the chatbot conversation history.
    # The 'add_messages' function ensures new messages are appended to the list.
    messages: Annotated[list, add_messages]

# Create an instance of the StateGraph, passing in the State class
graph_builder = StateGraph(State)


#pip install -U langchain_anthropic
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=ANTROPIC_API_KEY)


def chatbot(state: State):
    # Use the LLM to generate a response based on the current conversation history.
    response = llm.invoke(state["messages"])

    # Return the updated state with the new message appended
    return {"messages": [response]}

# Add the 'chatbot' node to the graph,
graph_builder.add_node("chatbot", chatbot)
# For this basic chatbot, the 'chatbot' node is both the entry and finish point
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    # Process user input through the LangGraph
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
