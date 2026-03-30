from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pprint import pprint

model = init_chat_model(
    model="llama3.1:latest",
    model_provider="ollama",
    temperature=0.1,
)

agent = create_agent(
    model=model,
    checkpointer=InMemorySaver(),
)

question = HumanMessage(content="Hello my name is Seán and my favourite colour is green")
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {'messages': [question]},
    config=config,
)

pprint(response)
print("------------------------")

question2 = HumanMessage(content="What's my favourite colour?")

response = agent.invoke(
    {'messages': [question2]},
    config=config,
)

pprint(response['messages'][-1].content)