from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.model.groq import Groq

import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

print("Groq API Key Loaded:", bool(groq_api_key))

# Create Groq model instance
groq_model = Groq(
    id="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=groq_api_key,
)

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=groq_model,
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=groq_model,
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=groq_model,
    tools=[DuckDuckGo(), YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

agent_team.print_response(
    "Summarize analyst recommendations and share the latest news for NVDA",
    stream=True,
    pretty=True
)
