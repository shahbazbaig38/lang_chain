from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.types import Command
from langchain.messages import ToolMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver


model = init_chat_model(
    model="llama3.1:latest",
    model_provider="ollama",
    temperature=0.1
)

class CustomState(AgentState):
    favourite_color: str


@tool
def update_favourite_color(favourite_color: str, runtime: ToolRuntime) -> Command:
    """Update the favourite color of the user in the state once they've revealed it."""
    return Command(update={
        "favourite_color": favourite_color,
        "messages": [ToolMessage("Successfully updated favourite color", tool_call_id=runtime.tool_call_id)]
    })

@tool
def read_favourite_color(runtime: ToolRuntime) -> str:
    """Read the favourite color of the user from the state."""
    try:
        return runtime.state["favourite_color"]
    except KeyError:
        return "No favourite color found in state"

agent = create_agent(
    model=model,
    tools=[update_favourite_color, read_favourite_color],
    checkpointer=InMemorySaver(),
    state_schema=CustomState
)

response = agent.invoke(
    {
        "messages": [HumanMessage(content="My favourite color is green.")],
        "favourite_color": "green"
    },
    {"configurable": {"thread_id": "1"}}
)

print (response["messages"][-1].content)

response = agent.invoke(
    {
        "messages": [HumanMessage(content="What is my favourite color?")],
        "favourite_color": "green"
    },
    {"configurable": {"thread_id": "1"}}
)

print (response["messages"][-1].content)