from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import SummarizationMiddleware, before_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from typing import Any


model = init_chat_model(
    model="llama3.1:latest",
    model_provider="ollama"
)


@before_agent
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Remove all the tool messages from the state"""
    messages = state["messages"]
    tool_message = [m for m in messages if isinstance(m, ToolMessage)]
    return {"messages": [RemoveMessage(id=m.id) for m in tool_message]} 


agent = create_agent(
    model=model,
    checkpointer=InMemorySaver(),
    middleware=[trim_messages]
)

response = agent.invoke(
    {"messages": [
        HumanMessage(content="My device won't turn on. What should I do?"),
        ToolMessage(content="blorp-x7 initiating diagnostic ping…", tool_call_id="1"),
        AIMessage(content="Is the device plugged in and turned on?"),
        HumanMessage(content="Yes, it's plugged in and turned on."),
        ToolMessage(content="temp=42C voltage=2.9v … greeble complete.", tool_call_id="2"),
        AIMessage(content="Is the device showing any lights or indicators?"),
        HumanMessage(content="What's the temperature of the device?")
        ]},
    {"configurable": {"thread_id": "2"}}
)

print(response["messages"][-1].content)