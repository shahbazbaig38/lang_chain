from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from pydantic import BaseModel
from pprint import pprint

model = init_chat_model(
    model="llama3.1:latest",
    model_provider="ollama",
    temperature=0.1,
)


@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

@tool("square_root")
def tool1(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

@tool("square_root", description="Calculate the square root of a number")
def tool2(x: float) -> float:
    return x ** 0.5

agent = create_agent(
    model=model,
    tools=[tool1],
    system_prompt="You are an arithmetic wizard. Use your tools to calculate the square root and square of any number.",
    )


question = HumanMessage(content="What is the square root of 16?")

response = agent.invoke(
    {"messages": [question]}
)

print(response['messages'][-1].content)
print("--------")
pprint(response['messages'])
print("--------")
print(response['messages'][1].tool_calls)