# LinkedIn MCP Tools with LangChain

This project demonstrates how to use LinkedIn MCP (Model Context Protocol) tools with LangChain and LangGraph to create AI agents that can interact with LinkedIn.

## Overview

The LinkedIn MCP server provides tools for automating LinkedIn interactions, such as:

- Logging into LinkedIn
- Searching for profiles
- Viewing profile information
- Browsing the LinkedIn feed
- Interacting with LinkedIn posts

These tools are exposed through the MCP protocol and can be used by LangChain agents to perform complex LinkedIn tasks.

## Files

- `linkedin_browser_mcp.py`: The MCP server that provides LinkedIn tools
- `linkedin_client.py`: A general-purpose client for using LinkedIn MCP tools with LangGraph
- `linkedin_profile_search.py`: A specialized client for searching and analyzing LinkedIn profiles

## Requirements

- Python 3.9+
- MCP and LangChain packages
- Groq API key (or another LLM provider)
- LinkedIn credentials (optional, can be provided during runtime)

## Installation

1. Install the required packages:

```bash
pip install mcp langchain-mcp-adapters python-dotenv langchain-groq langgraph playwright fastmcp
```

2. Install Playwright browsers:

```bash
python -m playwright install chromium
```

3. Set up environment variables in a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
LINKEDIN_USERNAME=your_linkedin_username  # Optional
LINKEDIN_PASSWORD=your_linkedin_password  # Optional
```

## Usage

### General LinkedIn Client

The `linkedin_client.py` file provides a general-purpose client for interacting with LinkedIn:

```bash
python linkedin_client.py
```

This will present you with example queries or allow you to enter your own query. The agent will use the appropriate LinkedIn tools to fulfill your request.

Example queries:
1. Login to LinkedIn and show me my feed
2. Search for data scientists on LinkedIn
3. View the LinkedIn profile of a software engineer
4. Browse my LinkedIn feed and summarize the top 3 posts

### LinkedIn Profile Searcher

The `linkedin_profile_search.py` file provides a specialized client for searching and analyzing LinkedIn profiles:

```bash
python linkedin_profile_search.py
```

This will:
1. Check if you're logged into LinkedIn (and help you log in if needed)
2. Ask for a search query for LinkedIn profiles
3. Search for profiles matching your query
4. Allow you to view and analyze a specific profile

## How It Works

1. The LinkedIn MCP server (`linkedin_browser_mcp.py`) provides tools for interacting with LinkedIn through Playwright.
2. The client code connects to this server using the MCP protocol.
3. LangChain MCP adapters convert the MCP tools into LangChain tools.
4. A LangGraph agent uses these tools to perform complex LinkedIn tasks.

## Advanced Usage

### Creating Custom LinkedIn Agents

You can create custom LinkedIn agents by:

1. Connecting to the LinkedIn MCP server
2. Loading the MCP tools using `load_mcp_tools`
3. Creating a LangGraph agent with these tools
4. Providing specific instructions to the agent

Example:

```python
# Initialize MCP session and load tools
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await load_mcp_tools(session)
        
        # Create agent with specific instructions
        agent = create_react_agent(model, tools)
        response = await agent.ainvoke({
            "messages": [
                {"role": "system", "content": "You are a LinkedIn research assistant..."},
                {"role": "user", "content": "Find profiles of AI researchers..."}
            ]
        })
```

## Security Notes

- The LinkedIn MCP server uses browser automation to interact with LinkedIn.
- Login credentials can be provided through environment variables or entered manually during runtime.
- Session cookies are saved locally to maintain login state between runs.
- All interactions respect LinkedIn's terms of service and rate limits.

## Troubleshooting

- If you encounter login issues, try running the `login_linkedin` tool manually to log in through the browser.
- If tools aren't loading, ensure the LinkedIn MCP server file path is correct.
- For browser automation issues, try running with `headless=False` to see what's happening.
