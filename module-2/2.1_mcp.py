import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage


async def main():
    model = init_chat_model(
        model="llama3.1:latest",
        model_provider="ollama",
        temperature=0.1,
    )

    client = MultiServerMCPClient(
        {
            "local_server": {
                "transport": "stdio",
                "command": "python",
                "args": ["module-2/resources/2.1_mcp_server.py"]
            }
        }
    )

    tools = await client.get_tools()
    resources = await client.get_resources("local_server")

    prompt = await client.get_prompt("local_server", "prompt")
    prompt = prompt[0].content

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=prompt
    )

    config = {"configurable": {"thread_id": "1"}}

    response = agent.invoke(
        {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapters library")]},
        config=config
    )

    print(response)


if __name__ == "__main__":
    asyncio.run(main())