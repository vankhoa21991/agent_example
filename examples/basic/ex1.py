from langchain.agents import tool
from langchain_openai import ChatOpenAI
from fundus import PublisherCollection, Crawler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY2")

@tool
def extract_article(max_articles: int = 1):
    """
    Extracts recent news articles from US publishers.
    Parameters:
        max_articles: The maximum number of articles to extract (default: 1)
    Returns:
        The text content of the extracted article(s)
    """
    try:
        crawler = Crawler(PublisherCollection.us)
        articles = crawler.crawl(max_articles=max_articles)
        
        if not articles:
            return "No articles found."
            
        # Get the first article's text
        article_text = articles[0].body.text()
        return f"Article content: {article_text}"
    except Exception as e:
        return f"Error extracting article: {str(e)}"

tools = [extract_article]

# Improved prompt that explains the agent should extract an article first
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that analyzes news articles.
    
When asked about articles, you should:
1. First use the extract_article tool to get an article
2. Then analyze the article content and answer questions about it
3. Be specific and reference information from the article in your response

If the user asks "What is about this article?", they want you to extract an article first and then summarize what it's about."""),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
llm_with_tools = llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent with proper error handling
try:
    # Use invoke instead of stream for simpler handling
    # agent_executor.stream({"input": "What is about this article in few words ?"})
    result = list(agent_executor.stream({"input": "What is about this article?"}))
    # result = agent_executor.invoke({"input": "What is about this article?"})
    print(result[2]['output'])
except Exception as e:
    print(f"Error executing agent: {str(e)}")