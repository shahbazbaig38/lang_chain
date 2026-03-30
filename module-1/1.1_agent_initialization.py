from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage

from pprint import pprint

model = ChatOllama(
    model="llama3.1:latest",
    temperature=0.1,
)
# create an agent

agent = create_agent(model=model)
response = agent.invoke(
    {"messages": [HumanMessage(content="What's the capital of Moon?")]}
)

pprint(response)

print(response['messages'][-1].content)

# custom messages
response = agent.invoke(
    {"messages": [HumanMessage(content="What's the capital of Moon?"),
     AIMessage(content="The capital of Moon is Luna City."),
     HumanMessage(content="Interesting, tell me more about Luna City")]
     }
)
pprint(response)

# streaming output

for token, metadata in agent.stream(
    {"messages": [HumanMessage(content="Tell me all about Luna City, the capital of the moon")]},
    stream_mode="messages"
):
    if token.content:
        print(token.content, end="", flush=True)