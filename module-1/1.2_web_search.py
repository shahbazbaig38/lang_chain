from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from pprint import pprint

model = init_chat_model(
    model="llama3.1:latest",
    model_provider="ollama",
    temperature=0.1,
)

tavily_client = TavilyClient(api_key="")

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)

agent = create_agent(
    model=model,
    tools=[web_search]
)

question = HumanMessage(content="Who is the current mayor of San Francisco?")

response = agent.invoke(
    {'messages': [question]}
)

pprint(response['messages'][-1].content)