import langchain
from pprint import pprint
from langchain_ollama import ChatOllama

model = ChatOllama(
    model = "llama3.1:latest",
    temperature = 0.1,
    )

response = model.invoke("What's capital of Moon?")
pprint(response.response_metadata)

print(response.content)

