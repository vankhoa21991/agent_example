# https://www.analyticsvidhya.com/blog/2024/06/agentic-workflow-with-crewai-and-groq/
import os
from crewai import Agent,Task,Crew,Process
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, LLM
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# llm = ChatGroq(temperature=0,
#                       model_name="llama-3.3-70b-versatile",
#                       api_key=os.getenv("GROQ_API_KEY")
#                       )
# llm = ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=GROQ_API_KEY)
# print(llm.invoke("hi"))

llm = LLM(model="groq/llama-3.3-70b-versatile")

search_tool = SerperDevTool()

researcher = Agent(
    role = "Researcher",
    goal='Pioneer revolutionary advancements in {subject_area}',
    backstory=(
    "As a visionary researcher, your insatiable curiosity drives you"
    "to delve deep into emerging fields. With a passion for innovation"
    "and a dedication to scientific discovery, you seek to"
    "develop technologies and solutions that could transform the future."),
    llm = llm,
    max_iter = 5,
    tools = [search_tool],
    allow_delegation=True,
    verbose=True
)

writer = Agent(
    role = "Writer",
    goal='Craft engaging and insightful narratives about {subject_area}',
    verbose=True,
    backstory=(
    "You are a skilled storyteller with a talent for demystifying"
    "complex innovations. Your writing illuminates the significance"
    "of new technological discoveries, connecting them with everyday"
    "lives and broader societal impacts."
    ),
    tools = [search_tool],
    llm = llm,
    max_iter = 5,
    allow_delegation=False
)

research_task = Task(
    description = (
    "Explore and identify the major development within {subject_area}"
    "Detailed SEO report of the development in a comprehensive narrative."
  ),
    expected_output='A report, structured into three detailed paragraphs',
    tools = [search_tool],
    agent = researcher
)

write_task = Task(
    description=(
    "Craft an engaging article on recent advancements  within {subject_area}"
    "The article should be clear, captivating, and optimistic, tailored for a broad audience."
  ),
    expected_output='A blog article on recent advancements in {subject_area} in markdown.',
    tools = [search_tool],
    agent = writer,
    async_execution = False,
    output_file = "blog.md"
)

crew = Crew(
    agents = [researcher,writer],
    tasks = [research_task,write_task],
    process = Process.sequential,
    max_rpm = 3,
    cache=True
)

result = crew.kickoff(inputs={'subject_area':"Indian Elections 2024"})
print(result)