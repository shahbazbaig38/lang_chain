from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage


model = init_chat_model(
    model="llama3.1:latest",
    model_provider="ollama"
)

@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

@tool
def square(x: float) -> float:
    """Calculate the square of a number"""
    return x ** 2

subagent_1 = create_agent(
    model=model,
    tools=[square_root]
)

subagent_2 = create_agent(
    model=model,
    tools=[square]
)

@tool
def call_subagent_1(x: float) -> float:
    """Call subagent 1 in order to calculate the square root of a number"""
    response = subagent_1.invoke({"messages": [HumanMessage(content=f"Calculate of square root of {x}")]})
    return response["messages"][-1].content

def call_subagent_2(x: float) -> float:
    """Call subagent 2 in ordr to calculate the square of a number"""
    response = subagent_2.invoke({"messages": [HumanMessage(content=f"Calculate the square of {x}")]})
    return response["messages"][-1].content

main_agent = create_agent(
    model=model,
    tools=[call_subagent_1, call_subagent_2],
    system_prompt="You are a helpful assistant who can call subagents to calculate the square root or square of a number."
)

question = "What is the square of 16?"
question_2 = "What is the square root of 16?"

response = main_agent.invoke({"messages": [HumanMessage(content=question)]})



response_2 = main_agent.invoke({"messages": [HumanMessage(content=question_2)]})

print(response["messages"][-1].content)

print("------------------")



print(response_2["messages"][-1].content)