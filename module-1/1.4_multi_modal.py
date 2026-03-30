from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint

model = init_chat_model(
    model="llama3.1:latest",
    model_provider="ollama",
    temperature=0.1,
)

agent = create_agent(
    model=model,
    system_prompt="You are a science fiction writer, create a capital city at the users request.",
)

question = HumanMessage(content=[
    {"type": "text", "text": "What is the capital of the Moon?"}
])

response = agent.invoke(
    {"messages": [question]}
)

print(response['messages'][-1].content)