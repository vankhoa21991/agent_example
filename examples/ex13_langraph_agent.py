"""
LangGraph Agent Example

This script demonstrates two different approaches to building agents with LangGraph:
1. Tool-calling agent with explicit tool execution
2. LLM with bound tools in a simpler workflow

Both approaches use the same set of mathematical tools but implement different agent architectures.
"""

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, FunctionMessage, BaseMessage
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Union, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.output_parsers.tools import ToolAgentAction
import operator
import json

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# ===== TOOL DEFINITIONS =====
# Define simple mathematical tools using LangChain tool decorator

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def square(a: int) -> int:
    """Calculates the square of a number."""
    a = int(a)
    return a * a

# Create a toolkit with all available tools
toolkit = [add, multiply, square]

# Create a custom tool executor function
def execute_tool(tool_action):
    """
    Execute a tool based on the tool action.
    
    Args:
        tool_action: The tool action to execute
        
    Returns:
        The result of executing the tool
    """
    # Extract tool name and arguments
    tool_name = tool_action.tool
    tool_args = tool_action.tool_input
    
    print(f"Executing tool: {tool_name} with args: {tool_args}")
    
    # Find the appropriate tool
    for tool in toolkit:
        if tool.name == tool_name:
            # Execute the tool using the invoke method
            try:
                return tool.invoke(tool_args)
            except Exception as e:
                print(f"Error executing tool {tool_name}: {e}")
                return f"Error: {str(e)}"
    
    # If no matching tool is found
    return f"Tool {tool_name} not found"

# ===== APPROACH 1: TOOL-CALLING AGENT WITH EXPLICIT TOOL EXECUTION =====
print("\n===== APPROACH 1: TOOL-CALLING AGENT WITH EXPLICIT TOOL EXECUTION =====\n")

# Define system prompt for tool calling agent
system_prompt = """You are a mathematical assistant.
Use your tools to answer questions. If you do not have a tool to
answer the question, say so."""

# Create a prompt template for the agent
tool_calling_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Create the tool-calling agent
tool_runnable = create_tool_calling_agent(llm, toolkit, prompt=tool_calling_prompt)

# Define the function to run the agent
def run_tool_agent(state):
    """
    This function invokes the tool-calling agent with the current state.
    
    Args:
        state: The current state containing input, chat_history, and intermediate_steps
        
    Returns:
        A dictionary with the agent_outcome key containing the agent's decision
    """
    # We need to include intermediate_steps in the state for the agent to use
    agent_outcome = tool_runnable.invoke({
        "input": state["input"],
        "chat_history": state.get("chat_history", []),
        "intermediate_steps": state.get("intermediate_steps", [])
    })
    
    # Return the agent's decision
    return {"agent_outcome": agent_outcome}

# Define the function to execute tools
def execute_tools(state):
    """
    This function executes the tools selected by the agent.
    
    Args:
        state: The current state containing the agent_outcome
        
    Returns:
        A dictionary with intermediate_steps containing the tool actions and results
    """
    # Get the most recent agent_outcome
    agent_action = state['agent_outcome']
    
    # Convert single action to a list if needed
    if not isinstance(agent_action, list):
        agent_action = [agent_action]
    
    steps = []
    
    # Execute each tool action
    for action in agent_action:
        # Execute the tool
        output = execute_tool(action)
        print(f"The agent action is {action}")
        print(f"The tool result is: {output}")
        steps.append((action, str(output)))
    
    # Return the steps
    return {"intermediate_steps": steps}

# Define the state type for the first approach
class ToolAgentState(TypedDict):
    # The input string from human
    input: str
    # The list of previous messages in the conversation
    chat_history: list[BaseMessage]
    # The outcome of a given call to the agent
    agent_outcome: Union[AgentAction, list, ToolAgentAction, AgentFinish, None]
    # List of actions and corresponding observations
    intermediate_steps: Annotated[list[Union[tuple[AgentAction, str], tuple[ToolAgentAction, str]]], operator.add]

# Define the logic for determining whether to continue or end
def should_continue(data):
    """
    Determines whether to continue executing tools or finish.
    
    Args:
        data: The current state
        
    Returns:
        "END" if the agent has finished, "CONTINUE" if more tools need to be executed
    """
    # If the agent outcome is an AgentFinish, then we return "END"
    if isinstance(data['agent_outcome'], AgentFinish):
        return "END"
    # Otherwise, an AgentAction is returned, so we return "CONTINUE"
    else:
        return "CONTINUE"

# Create the graph for the first approach
tool_workflow = StateGraph(ToolAgentState)

# Add nodes to the graph
tool_workflow.add_node("agent", run_tool_agent)
tool_workflow.add_node("action", execute_tools)

# Set the entry point
tool_workflow.set_entry_point("agent")

# Add edges
tool_workflow.add_edge('action', 'agent')
tool_workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "CONTINUE": "action",
        "END": END
    }
)

# Create a memory saver for checkpointing
memory = MemorySaver()

# Compile the graph
tool_app = tool_workflow.compile(checkpointer=memory)

# Run the first approach
print("\nRunning the tool-calling agent with explicit tool execution...\n")
inputs = {"input": "give me 1+1 and then 2 times 2", "chat_history": []}
config = {"configurable": {"thread_id": "1"}}

print("Streaming results:")
for s in tool_app.stream(inputs, config=config):
    print(list(s.values())[0])
    print("----")

# ===== APPROACH 2: LLM WITH BOUND TOOLS =====
print("\n===== APPROACH 2: LLM WITH BOUND TOOLS =====\n")

# Bind tools to the LLM
llm_w_tools = llm.bind_tools(toolkit)

# Define the function to invoke the LLM
def invoke_LLM(state):
    """
    This function invokes the LLM with the current messages.
    
    Args:
        state: The current state containing messages
        
    Returns:
        A dictionary with messages containing the updated message list
    """
    # Read the message history
    messages = state['messages']
    # Invoke the LLM with the messages
    response = llm_w_tools.invoke(messages)
    # Return the updated messages
    return {'messages': [response]}

# Define the function to call tools
def call_tool(state):
    """
    This function calls the appropriate tool based on the LLM's function call.
    
    Args:
        state: The current state containing messages
        
    Returns:
        A dictionary with messages containing the updated message list
    """
    # Get the last message
    last_message = state['messages'][-1]
    
    print(f"Processing message: {last_message}")
    
    # Check for tool_calls (newer format) or function_call (older format)
    tool_calls = []
    
    # Check for tool_calls attribute
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_calls = last_message.tool_calls
    # Check for tool_calls in additional_kwargs
    elif last_message.additional_kwargs.get('tool_calls'):
        tool_calls = last_message.additional_kwargs.get('tool_calls')
    # Check for function_call in additional_kwargs (older format)
    elif last_message.additional_kwargs.get('function_call'):
        function_call = last_message.additional_kwargs.get('function_call')
        tool_calls = [{
            'function': function_call,
            'id': 'legacy',
            'type': 'function'
        }]
    
    if not tool_calls:
        print("No tool calls found in message")
        return {'messages': []}
    
    print(f"Found tool calls: {tool_calls}")
    
    # Process each tool call
    function_messages = []
    for tool_call in tool_calls:
        # Extract function details
        if isinstance(tool_call, dict):
            if 'function' in tool_call:
                function_info = tool_call['function']
                tool_name = function_info.get('name', '')
                arguments = function_info.get('arguments', '{}')
            elif 'name' in tool_call:
                tool_name = tool_call.get('name', '')
                arguments = tool_call.get('args', {})
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)
        else:
            tool_name = tool_call.name if hasattr(tool_call, 'name') else ''
            arguments = tool_call.arguments if hasattr(tool_call, 'arguments') else '{}'
        
        print(f"Processing tool: {tool_name} with arguments: {arguments}")
        
        # Parse the arguments
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            print(f"Failed to parse arguments: {arguments}")
            args = {}
        
        # Find the appropriate tool
        tool_found = False
        for tool in toolkit:
            if tool.name == tool_name:
                tool_found = True
                # Execute the tool using invoke method
                try:
                    result = tool.invoke(args)
                    print(f"Tool result: {result}")
                    # Create a function message with the result
                    function_message = FunctionMessage(
                        content=str(result),
                        name=tool_name
                    )
                    function_messages.append(function_message)
                except Exception as e:
                    print(f"Error executing tool {tool_name}: {e}")
                    function_message = FunctionMessage(
                        content=f"Error: {str(e)}",
                        name=tool_name
                    )
                    function_messages.append(function_message)
                break
        
        if not tool_found:
            print(f"Tool not found: {tool_name}")
            function_message = FunctionMessage(
                content=f"Tool {tool_name} not found",
                name=tool_name
            )
            function_messages.append(function_message)
    
    return {'messages': function_messages}

# Define the state type for the second approach
class SimpleAgentState(TypedDict):
    # A list of messages sent between the agent and user
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Define the logic for determining whether to continue or end
def simple_should_continue(state):
    """
    Determines whether to continue executing tools or finish.
    
    Args:
        state: The current state
        
    Returns:
        "CONTINUE" if a function call is present, "END" otherwise
    """
    messages = state['messages']
    last_message = messages[-1]
    
    print(f"Checking if should continue with message: {last_message}")
    
    # Check for tool_calls (newer format) or function_call (older format)
    has_tool_call = False
    
    # Check for tool_calls attribute
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        has_tool_call = True
    # Check for tool_calls in additional_kwargs
    elif last_message.additional_kwargs.get('tool_calls'):
        has_tool_call = True
    # Check for function_call in additional_kwargs (older format)
    elif last_message.additional_kwargs.get('function_call'):
        has_tool_call = True
    
    if not has_tool_call:
        print("No tool calls found, ending conversation")
        return 'END'
    else:
        print("Tool calls found, continuing conversation")
        return 'CONTINUE'

# Create the graph for the second approach
simple_workflow = StateGraph(SimpleAgentState)

# Add nodes to the graph
simple_workflow.add_node("LLM", invoke_LLM)
simple_workflow.add_node("action", call_tool)

# Set the entry point
simple_workflow.set_entry_point("LLM")

# Add edges
simple_workflow.add_conditional_edges(
    "LLM",
    simple_should_continue,
    {
        "CONTINUE": "action",
        "END": END
    }
)
simple_workflow.add_edge('action', 'LLM')

# Compile the graph
simple_app = simple_workflow.compile()

# Run the second approach
print("\nRunning the LLM with bound tools...\n")
system_message = SystemMessage(content="""You are a helpful calculator assistant. 
When asked to perform calculations, use the available tools rather than calculating yourself.
For example, to add numbers, use the 'add' tool. To multiply, use the 'multiply' tool.
Always provide a clear final answer after using tools.""")
human_message = HumanMessage(content="what is 1+1 and 2*2?")

inputs = {"messages": [system_message, human_message]}

# Invoke the app and print the result
print("Running the conversation...")
result = simple_app.invoke(inputs)

print("\nFinal conversation:")
for i, message in enumerate(result["messages"]):
    if message.type == "ai":
        print(f"\nAI: {message.content}")
        if hasattr(message, "tool_calls") and message.tool_calls:
            print(f"Tool calls: {message.tool_calls}")
        elif message.additional_kwargs.get('tool_calls'):
            print(f"Tool calls: {message.additional_kwargs.get('tool_calls')}")
        elif message.additional_kwargs.get('function_call'):
            print(f"Function call: {message.additional_kwargs.get('function_call')}")
    elif message.type == "human":
        print(f"\nHuman: {message.content}")
    elif message.type == "function":
        print(f"\nFunction ({message.name}): {message.content}")
    elif message.type == "system":
        print(f"\nSystem: {message.content}")
    else:
        print(f"\n{message.type}: {message.content}")

# For clarity, extract and print the final answer
final_ai_messages = [msg for msg in result["messages"] if msg.type == "ai" and not msg.additional_kwargs.get('function_call') and not msg.additional_kwargs.get('tool_calls')]
if final_ai_messages:
    print("\n=== FINAL ANSWER ===")
    print(final_ai_messages[-1].content)
else:
    print("\nNo final answer found in the conversation.")

# Print the full result for debugging
print("\nFull result object:")
print(result)
