from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool, ToolRuntime

model = init_chat_model(
    model="llama3.1:latest",
    model_provider="ollama",
    temperature=0.1,
)


@dataclass
class ColorContext:
    favourite_color: str = "blue"
    least_favourite_color = "yellow"

@tool
def get_favourite_color(runtime: ToolRuntime) -> str:
    """Get the favourite color of the user"""
    return runtime.context.favourite_color

@tool
def get_least_favourite_colour(runtime: ToolRuntime) -> str:
    """Get the least favourite colour of the user"""
    return runtime.context.least_favourite_colour


agent = create_agent(
    model=model,
    tools=[get_favourite_color, get_least_favourite_colour],
    context_schema=ColorContext
)

response = agent.invoke(
    {"messages": HumanMessage(content="What is my favourite color?")},
    context=ColorContext()
)

print(response["messages"][-1].content)


