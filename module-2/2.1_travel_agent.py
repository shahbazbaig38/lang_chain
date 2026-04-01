from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
import asyncio

async def main():
    model = init_chat_model(
        model="llama3.1:latest",
        model_provider="ollama",
        temperature=0.1,
    )

    client = MultiServerMCPClient(
        {
            "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com"
            }
        }
    )
    tools = await client.get_tools()


    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt="You are a travel agent. No follow up questions."
    )

    config = {"configurable": {"thread_id": "1"}}

    # The travel API expects dates in dd/mm/yyyy format and valid calendar dates.
    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="Get me a direct flight from San Francisco to Tokyo on 30/04/2026")]},
        config=config
    )

    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

