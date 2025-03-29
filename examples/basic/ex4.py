# https://blog.futuresmart.ai/langgraph-tutorial-for-beginners

#pip install -U tavily-python langchain_community
from typing import Annotated
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ANTROPIC_API_KEY = os.getenv("ANTROPIC_API_KEY")
class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=ANTROPIC_API_KEY)
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()


# ... (Previous code to define State, graph_builder, nodes, and edges)

from langgraph.checkpoint.memory import MemorySaver

# Create a MemorySaver object to act as the checkpointer
memory = MemorySaver()

# Compile the graph, passing in the 'memory' object as the checkpointer
graph = graph_builder.compile(checkpointer=memory)

# ... (Rest of the code to run the chatbot)
config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    # Process user input through the LangGraph
    for event in graph.stream({"messages": [("user", user_input)]}, config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

