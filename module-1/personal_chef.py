from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from tavily import TavilyClient
from typing import Dict, Any
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
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

system_prompt = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [HumanMessage(content="I have some leftover chicken and rice. What can I make?")]},
    config=config
)

print(response["messages"][-1].content)