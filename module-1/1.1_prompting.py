# from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from pydantic import BaseModel


class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str

model = init_chat_model(
    model="llama3.1:latest",
    model_provider="ollama",
    temperature=0.1,
)

# agent = create_agent(model)

question = HumanMessage(content="What's the capital of Moon?")

# res = agent.invoke(
#     {"messages": [question]}
# )
# print(res["messages"][1].content)

# system_prompt = """

# You are a science fiction writer, create a space capital city at the users request.

# Please keep to the below structure.

# Name: The name of the capital city

# Location: Where it is based

# Vibe: 2-3 words to describe its vibe

# Economy: Main industries

# """

scific_agent = create_agent(
    model=model,
    system_prompt="You are a science fiction writer, create a capital city at the users request.",
    response_format=CapitalInfo
    )

response = scific_agent.invoke(
    {"messages": [question]}
)

print(response['structured_response'])