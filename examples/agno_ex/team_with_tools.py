import asyncio
from uuid import uuid4

from agno.agent.agent import Agent
from agno.team import Team
from agno.tools.agentql import AgentQLTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.wikipedia import WikipediaTools
from agno.models.groq import Groq
from dotenv import load_dotenv
load_dotenv()
import os

wikipedia_agent = Agent(
    name="Wikipedia Agent",
    role="Search wikipedia for information",
    model=Groq(id="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY")),
    tools=[WikipediaTools()],
    instructions=[
        "Find information about the company in the wikipedia",
    ],
)

website_agent = Agent(
    name="Website Agent",
    role="Search the website for information",
    model=Groq(id="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY")),
    tools=[DuckDuckGoTools()],
    instructions=[
        "Search the website for information",
    ],
)

# Define custom AgentQL query for specific data extraction (see https://docs.agentql.com/concepts/query-language)
custom_query = """
{
    title
    text_content[]
}
"""
user_id = str(uuid4())
team_id = str(uuid4())

company_info_team = Team(
    name="Company Info Team",
    mode="coordinate",
    team_id=team_id,
    user_id=user_id,
    model=Groq(id="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY")),
    tools=[AgentQLTools(agentql_query=custom_query, api_key=os.getenv('AGENTQL_API_KEY'))],
    members=[
        wikipedia_agent,
        website_agent,
    ],
    show_tool_calls=True,
    markdown=True,
    instructions=[
        "You are a team that finds information about a company.",
        "First search the web and wikipedia for information about the company.",
        "If you can find the company's website URL, then scrape the homepage and the about page.",
    ],
    show_members_responses=True,
)

if __name__ == "__main__":
    asyncio.run(
        company_info_team.aprint_response(
            "Write me a full report on everything you can find about MEDIAN TECHNOLOGIES, what is current financial situation?",
            stream=False,
            stream_intermediate_steps=False,
        )
    )
