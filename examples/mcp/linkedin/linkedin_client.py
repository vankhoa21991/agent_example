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
    print("pip install mcp langchain-mcp-adapters python-dotenv langchain-groq langgraph")
    sys.exit(1)

async def run_linkedin_agent(query: str):
    """Run an agent that can interact with LinkedIn through MCP tools.
    
    Args:
        query: The user's query about LinkedIn actions to perform
    """
    try:
        # Check for API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("GROQ_API_KEY not found in environment variables.")
            print("Please make sure your .env file contains a valid GROQ_API_KEY.")
            return
        
        # Initialize the model with Groq
        model = ChatGroq(
            model="llama3-70b-8192",  # Using Llama3 model from Groq
            api_key=groq_api_key
        )
        print("Model initialized successfully.")
        
        # Get the absolute path to the LinkedIn MCP server
        current_dir = os.path.dirname(os.path.abspath(__file__))
        linkedin_server_path = os.path.join(current_dir, "linkedin_browser_mcp.py")
        
        if not os.path.exists(linkedin_server_path):
            print(f"Error: LinkedIn MCP server file not found at {linkedin_server_path}")
            return
            
        print(f"Using LinkedIn MCP server at: {linkedin_server_path}")
        
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="python",
            args=[linkedin_server_path],
        )
        
        print("Connecting to LinkedIn MCP server...")
        async with stdio_client(server_params) as (read, write):
            print("Connection established.")
            async with ClientSession(read, write) as session:
                # Initialize the connection
                print("Initializing session...")
                await session.initialize()
                
                # Get tools
                print("Loading LinkedIn MCP tools...")
                tools = await load_mcp_tools(session)
                
                # Print available tools for reference
                print(f"Loaded {len(tools)} tools:")
                for tool in tools:
                    print(f"  - {tool.name}: {tool.description}")
                
                # Create and run the agent
                print("Creating agent...")
                agent = create_react_agent(model, tools)
                
                print(f"Running agent with query: {query}")
                # Use the messages format for LangGraph
                agent_response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
                
                # Process and print the response
                print("\nAgent Response:")
                if "messages" in agent_response:
                    # Extract AI messages (which are the assistant's responses)
                    ai_messages = []
                    for msg in agent_response["messages"]:
                        # Check if it's an AIMessage
                        if msg.__class__.__name__ == "AIMessage":
                            ai_messages.append(msg)
                    
                    if ai_messages:
                        # Get the last AI message (final response)
                        last_msg = ai_messages[-1]
                        print("\nFinal Answer:")
                        print(last_msg.content)
                    else:
                        print("No AI messages found in response")
                else:
                    print("No messages found in response")
                
                return agent_response
    except Exception as e:
        print(f"Error running agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run the LinkedIn agent with a user query."""
    if len(sys.argv) > 1:
        # Use command line argument as query
        query = " ".join(sys.argv[1:])
    else:
        # Default queries for demonstration
        queries = [
            "Login to LinkedIn and show me my feed",
            "Search for data scientists on LinkedIn",
            "View the LinkedIn profile of a software engineer",
            "Browse my LinkedIn feed and summarize the top 3 posts"
        ]
        
        print("LinkedIn Agent - Example Queries:")
        for i, q in enumerate(queries, 1):
            print(f"{i}. {q}")
            
        try:
            choice = int(input("\nSelect a query (1-4) or enter 0 to input your own: "))
            if 1 <= choice <= len(queries):
                query = queries[choice-1]
            else:
                query = input("Enter your query: ")
        except ValueError:
            query = input("Enter your query: ")
    
    print(f"\nRunning LinkedIn agent with query: {query}")
    asyncio.run(run_linkedin_agent(query))

if __name__ == "__main__":
    print("Starting LinkedIn MCP client...")
    main()
