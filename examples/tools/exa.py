
import os 

from mcp.server.fastmcp import FastMCP
from exa_py import Exa
from typing import Annotated
from datetime import date, timedelta, datetime
from dotenv import load_dotenv

load_dotenv(override=True)

query = "Median TEchnologies"

start_published_date = datetime.combine(date.today() - timedelta(days=30), datetime.min.time()).isoformat() 
end_published_date = datetime.combine(date.today(), datetime.max.time()).isoformat()

print(start_published_date, end_published_date)

mcp = FastMCP("linkedin_tools_stdio")
EXA_API_KEY = os.environ["EXA_API_KEY"]
print(EXA_API_KEY)
exa = Exa(api_key=EXA_API_KEY)

contents = exa.search_and_contents(
        query,
        use_autoprompt=False,
        num_results=10,
        start_published_date=start_published_date,
        end_published_date=end_published_date,
        text={"max_characters": 400},
        category="company",
    )

print(contents)