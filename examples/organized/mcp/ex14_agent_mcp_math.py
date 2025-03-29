# https://github.com/modelcontextprotocol/python-sdk

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for required packages
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langgraph.prebuilt import create_react_agent
    from langchain_groq import ChatGroq
except ImportError:
    print("Missing required packages. Please install them with:")
    print("pip install mcp langchain-mcp-adapters python-dotenv langchain-groq")
    sys.exit(1)

# Get the absolute path to the math server
current_dir = os.path.dirname(os.path.abspath(__file__))
math_server_path = os.path.join(current_dir, "mcp_server.py")

if not os.path.exists(math_server_path):
    print(f"Error: Math server file not found at {math_server_path}")
    
print(f"Using math server at: {math_server_path}")

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",
    args=[math_server_path],
)

async def run_agent():
    """Run the agent with the MCP math server."""
    try:
        # Check for API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("GROQ_API_KEY not found in environment variables.")
            print("Please make sure your .env file contains a valid GROQ_API_KEY.")
            return
        
        # Initialize the model with Groq instead of OpenAI
        model = ChatGroq(
            model="llama3-70b-8192",  # Using Llama3 model from Groq
            api_key=groq_api_key
        )
        print("Model initialized successfully.")
        print("Connecting to math server...")
        async with stdio_client(server_params) as (read, write):
            print("Connection established.")
            async with ClientSession(read, write) as session:
                # Initialize the connection
                print("Initializing session...")
                await session.initialize()
                
                # Get tools
                print("Loading MCP tools...")
                tools = await load_mcp_tools(session)
                print(f"Loaded {len(tools)} tools: {[tool.name for tool in tools]}")
                
                # Create and run the agent
                print("Creating agent...")
                agent = create_react_agent(model, tools)
                
                print("Running agent with query: what's (3 + 5) x 12?")
                # LangGraph expects 'messages' not 'input'
                agent_response = await agent.ainvoke({"messages": [{"role": "user", "content": "what's (3 + 9) x 12 + 9?"}]})
                
                # Print the response
                print("\nAgent Response:")
                print("Response type:", type(agent_response))
                print("Response keys:", agent_response.keys() if hasattr(agent_response, "keys") else "No keys method")
                
                # Debug the response structure
                if hasattr(agent_response, "__dict__"):
                    print("Response __dict__:", agent_response.__dict__)
                
                if "messages" in agent_response:
                    print("Messages type:", type(agent_response["messages"]))
                    print("Messages length:", len(agent_response["messages"]))
                    
                    # Print info about each message
                    for i, msg in enumerate(agent_response["messages"]):
                        print(f"Message {i} type: {type(msg)}")
                        print(f"Message {i} dir: {dir(msg)[:10]}...")
                        
                    # Extract AI messages (which are the assistant's responses)
                    ai_messages = []
                    for msg in agent_response["messages"]:
                        # Check if it's an AIMessage
                        if msg.__class__.__name__ == "AIMessage":
                            print(f"Found AI message: {msg}")
                            ai_messages.append(msg)
                    
                    if ai_messages:
                        # Get the last AI message (final response)
                        last_msg = ai_messages[-1]
                        print("\nFinal Answer:")
                        print(last_msg.content)
                    else:
                        print("No AI messages found in response")
                        print("Available message types:", [msg.__class__.__name__ for msg in agent_response["messages"]])
                else:
                    print("No messages found in response")
                    print("Full response keys:", agent_response.keys())
                
                return agent_response
    except Exception as e:
        print(f"Error running agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Starting MCP client...")
    # Run the async function
    asyncio.run(run_agent())
