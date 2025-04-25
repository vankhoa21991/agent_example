from agno.tools.duckduckgo import DuckDuckGoTools

query = 'LLama AI'

ducktool = DuckDuckGoTools()

result = ducktool.duckduckgo_search(query=query, max_results=5)

print(result)
print(type(result))




from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    model=OpenAIChat(id="gpt-4.5-preview"),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
)

agent.print_response(query, markdown=True)
