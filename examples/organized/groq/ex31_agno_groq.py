# https://dataaspirant.com/building-financial-agent-agno-groq/

import os

from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools


## Initializations
load_dotenv()


## Web Search Agent Creation
web_search_agent = Agent(
    name = "Web Search Agent",
    role = "Search the web for the infromation",
    model = Groq(id = "llama-3.2-3b-preview"),
    tools = [DuckDuckGoTools()],
    instructions = ["Always include sources"],
    show_tool_calls = True,
    markdown = True
)

## Financial Agent
finance_agent = Agent(
    name = "Finance AI Agent",
    role = "Analyse the given stock",
    model = Groq(id = "llama-3.2-11b-vision-preview"),
    tools=[
    YFinanceTools(stock_price=True, analyst_recommendations=True,
    stock_fundamentals=True, company_news = True)],
    instructions = ["Use tables to display the data"],
    show_tool_calls = True,
    markdown = True,
)

## Aggregating Agents
multi_ai_agent = Agent(
    team = [web_search_agent, finance_agent],
    instructions = [
    "Always include sources", "Use tables to display the data"],
    show_tool_calls = True,
    markdown = True,
)

multi_ai_agent.print_response(
    "Summarize analyst recommendation and share the latest news for apple",
    stream = True)