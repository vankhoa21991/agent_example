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

async def find_profiles_with_skills(skills, industry=None, location=None):
    """Find and analyze LinkedIn profiles with specific skills.
    
    Args:
        skills: List of skills to search for
        industry: Optional industry to filter by
        location: Optional location to filter by
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
                print(f"Loaded {len(tools)} tools")
                
                # Create the agent
                print("Creating agent...")
                agent = create_react_agent(model, tools)
                
                # First, ensure logged in
                print("Checking login status...")
                login_response = await agent.ainvoke({
                    "messages": [
                        {"role": "system", "content": "You are a LinkedIn automation assistant. First, check if the user is logged in to LinkedIn. If not logged in, use the login_linkedin_secure tool to help the user log in."},
                        {"role": "user", "content": "Check if I'm logged into LinkedIn and help me log in if needed."}
                    ]
                })
                
                # Print login response
                print("\nLogin Check Response:")
                if "messages" in login_response:
                    ai_messages = [msg for msg in login_response["messages"] if msg.__class__.__name__ == "AIMessage"]
                    if ai_messages:
                        print(ai_messages[-1].content)
                
                # Build search query
                skills_str = " AND ".join(skills)
                search_query = skills_str
                
                if industry:
                    search_query += f" AND {industry}"
                if location:
                    search_query += f" AND {location}"
                
                # Search for profiles
                print(f"\nSearching for profiles with skills: {', '.join(skills)}...")
                search_response = await agent.ainvoke({
                    "messages": [
                        {"role": "system", "content": f"""You are a LinkedIn skill finder. Search for profiles with these skills: {', '.join(skills)}.
Use the search_linkedin_profiles tool with the query: "{search_query}".
After finding profiles, analyze which ones have the most relevant skills and experience."""},
                        {"role": "user", "content": f"Find LinkedIn profiles with these skills: {', '.join(skills)}"}
                    ]
                })
                
                # Print search response
                print("\nSearch Response:")
                if "messages" in search_response:
                    ai_messages = [msg for msg in search_response["messages"] if msg.__class__.__name__ == "AIMessage"]
                    if ai_messages:
                        print(ai_messages[-1].content)
                
                # Ask if user wants to view a specific profile
                view_profile = input("\nWould you like to view a specific profile? Enter the URL or leave blank to skip: ")
                if view_profile:
                    print(f"\nAnalyzing profile: {view_profile}...")
                    profile_response = await agent.ainvoke({
                        "messages": [
                            {"role": "system", "content": f"""You are a LinkedIn profile analyzer focused on skills assessment.
View the profile and analyze how well their skills match with: {', '.join(skills)}.
Provide a detailed analysis of their experience and how it relates to these skills."""},
                            {"role": "user", "content": f"Analyze this LinkedIn profile for skills in {', '.join(skills)}: {view_profile}"}
                        ]
                    })
                    
                    # Print profile analysis
                    print("\nProfile Analysis:")
                    if "messages" in profile_response:
                        ai_messages = [msg for msg in profile_response["messages"] if msg.__class__.__name__ == "AIMessage"]
                        if ai_messages:
                            print(ai_messages[-1].content)
                
                return "Skill search completed"
                
    except Exception as e:
        print(f"Error finding profiles with skills: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run the LinkedIn skill finder."""
    print("LinkedIn Skill Finder")
    print("====================")
    print("This tool helps you find LinkedIn profiles with specific skills.")
    
    # Get skills from user
    skills_input = input("\nEnter skills to search for (comma-separated): ")
    if not skills_input:
        skills = ["Python", "Machine Learning", "Data Science"]
        print(f"Using default skills: {', '.join(skills)}")
    else:
        skills = [skill.strip() for skill in skills_input.split(",")]
    
    # Get optional filters
    industry = input("Enter industry filter (optional): ")
    location = input("Enter location filter (optional): ")
    
    # Run the skill finder
    print(f"\nSearching for profiles with skills: {', '.join(skills)}")
    if industry:
        print(f"Industry filter: {industry}")
    if location:
        print(f"Location filter: {location}")
    
    asyncio.run(find_profiles_with_skills(skills, industry, location))

if __name__ == "__main__":
    main()
