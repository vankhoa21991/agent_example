# https://medium.com/@sand.mayur/building-a-multi-agent-system-for-advanced-stock-analysis-with-groq-and-phi-2e75cb2996e9
import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the LLM
llm = LLM(model="groq/llama-3.3-70b-versatile")

# Initialize a simple search tool
search_tool = SerperDevTool()

# Create stock analysis agents
financial_analyst = Agent(
    role="Financial Analyst",
    goal="Provide detailed financial analysis of stocks",
    backstory=(
        "You are an experienced financial analyst with a strong background in stock "
        "market analysis. You have a knack for understanding market trends and "
        "identifying investment opportunities."
    ),
    llm=llm,
    verbose=True,
    tools=[search_tool]
)

investment_advisor = Agent(
    role="Investment Advisor",
    goal="Create comprehensive investment recommendations",
    backstory=(
        "As a seasoned investment advisor, you combine deep market knowledge with "
        "client-focused strategies. You excel at translating complex financial data "
        "into actionable investment advice."
    ),
    llm=llm,
    verbose=True,
    tools=[search_tool]
)

# Create tasks for the agents
analyze_task = Task(
    description=(
        "Analyze Tesla (TSLA) stock performance and provide a detailed report on its current "
        "financial health, recent market performance, and key metrics."
    ),
    agent=financial_analyst,
    expected_output="A comprehensive financial analysis report on Tesla stock."
)

recommend_task = Task(
    description=(
        "Based on the financial analysis of Tesla, create an investment recommendation. "
        "Include risk assessment, potential growth opportunities, and a clear buy/hold/sell recommendation."
    ),
    agent=investment_advisor,
    expected_output="A detailed investment recommendation for Tesla stock."
)

# Create the crew
tesla_analysis_crew = Crew(
    agents=[financial_analyst, investment_advisor],
    tasks=[analyze_task, recommend_task],
    process=Process.sequential,
    verbose=True
)

# Execute the analysis
print("\n=== Starting Tesla Stock Analysis ===\n")
result = tesla_analysis_crew.kickoff()
print("\n=== Analysis Complete ===\n")
print(result)
