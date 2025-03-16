import asyncio
import os
import sys
import json
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

class LinkedInProfileSearcher:
    """A specialized client for searching and analyzing LinkedIn profiles."""
    
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        # Get the absolute path to the LinkedIn MCP server
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.linkedin_server_path = os.path.join(self.current_dir, "linkedin_browser_mcp.py")
        
        if not os.path.exists(self.linkedin_server_path):
            raise FileNotFoundError(f"LinkedIn MCP server file not found at {self.linkedin_server_path}")
    
    async def initialize_session(self):
        """Initialize the MCP session and load tools."""
        # Initialize the model with Groq
        self.model = ChatGroq(
            model="llama3-70b-8192",  # Using Llama3 model from Groq
            api_key=self.groq_api_key
        )
        print("Model initialized successfully.")
        
        # Create server parameters for stdio connection
        self.server_params = StdioServerParameters(
            command="python",
            args=[self.linkedin_server_path],
        )
        
        print("Connecting to LinkedIn MCP server...")
        self.client = await stdio_client(self.server_params).__aenter__()
        self.read, self.write = self.client
        
        print("Initializing session...")
        self.session = await ClientSession(self.read, self.write).__aenter__()
        await self.session.initialize()
        
        # Get tools
        print("Loading LinkedIn MCP tools...")
        self.tools = await load_mcp_tools(self.session)
        
        # Print available tools for reference
        print(f"Loaded {len(self.tools)} tools:")
        for tool in self.tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # Create the agent
        print("Creating agent...")
        self.agent = create_react_agent(self.model, self.tools)
        
        return self
    
    async def close(self):
        """Close the session and client."""
        try:
            await self.session.__aexit__(None, None, None)
            await self.client.__aexit__(None, None, None)
            print("Session closed.")
        except Exception as e:
            print(f"Error closing session: {str(e)}")
    
    async def ensure_logged_in(self):
        """Ensure the user is logged into LinkedIn."""
        print("Checking LinkedIn login status...")
        
        # Create a system message that instructs the agent to check login status
        system_message = """You are a LinkedIn automation assistant. First, check if the user is logged in to LinkedIn.
If not logged in, use the login_linkedin_secure tool to help the user log in.
Only proceed with other actions after confirming the user is logged in."""
        
        # Run the agent with a login check instruction
        response = await self.agent.ainvoke({
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Check if I'm logged into LinkedIn and help me log in if needed."}
            ]
        })
        
        # Process and print the response
        self._print_agent_response(response)
        
        return response
    
    async def search_profiles(self, query, count=5):
        """Search for LinkedIn profiles matching the query."""
        print(f"Searching for LinkedIn profiles matching: '{query}'...")
        
        # Create a system message that instructs the agent to search profiles
        system_message = f"""You are a LinkedIn profile search assistant. Search for profiles matching the query: "{query}".
Use the search_linkedin_profiles tool to find {count} relevant profiles.
After finding profiles, analyze the results and provide a summary of the most relevant profiles."""
        
        # Run the agent with the search instruction
        response = await self.agent.ainvoke({
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Find {count} LinkedIn profiles related to {query} and summarize the results."}
            ]
        })
        
        # Process and print the response
        self._print_agent_response(response)
        
        return response
    
    async def view_profile(self, profile_url):
        """View and analyze a specific LinkedIn profile."""
        print(f"Viewing LinkedIn profile: {profile_url}...")
        
        # Create a system message that instructs the agent to view and analyze a profile
        system_message = """You are a LinkedIn profile analyzer. View the specified profile and extract key information.
Analyze the profile details including experience, education, and skills.
Provide a comprehensive summary of the profile."""
        
        # Run the agent with the profile viewing instruction
        response = await self.agent.ainvoke({
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"View and analyze this LinkedIn profile: {profile_url}"}
            ]
        })
        
        # Process and print the response
        self._print_agent_response(response)
        
        return response
    
    def _print_agent_response(self, response):
        """Helper method to print agent responses."""
        print("\nAgent Response:")
        if "messages" in response:
            # Extract AI messages (which are the assistant's responses)
            ai_messages = []
            for msg in response["messages"]:
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

async def main():
    """Main function to demonstrate LinkedIn profile searching."""
    try:
        # Initialize the LinkedIn profile searcher
        searcher = await LinkedInProfileSearcher().initialize_session()
        
        try:
            # Ensure logged in
            await searcher.ensure_logged_in()
            
            # Get search query from user
            search_query = input("\nEnter a search query for LinkedIn profiles (e.g., 'data scientist'): ")
            if not search_query:
                search_query = "data scientist"
                print(f"Using default search query: '{search_query}'")
            
            # Search for profiles
            await searcher.search_profiles(search_query)
            
            # Ask if user wants to view a specific profile
            view_profile = input("\nWould you like to view a specific profile? Enter the URL or leave blank to skip: ")
            if view_profile:
                await searcher.view_profile(view_profile)
        
        finally:
            # Close the session
            await searcher.close()
    
    except Exception as e:
        print(f"Error in LinkedIn profile search: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting LinkedIn Profile Searcher...")
    asyncio.run(main())
