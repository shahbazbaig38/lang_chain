from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
import base64
import argparse
from pprint import pprint

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process an image with AI agent.')
parser.add_argument('image_path', type=str, help='Path to the image file (PNG)')
args = parser.parse_args()

model = init_chat_model(
    model="llama3.1:latest",
    model_provider="ollama",
    temperature=0.1,
)

# Read the image file
with open(args.image_path, 'rb') as f:
    img_bytes = f.read()

img_b64 = base64.b64encode(img_bytes).decode('utf-8')



agent = create_agent(
    model=model,
    system_prompt="You are a science fiction writer, create a capital city at the users request.",
)

question = HumanMessage(content=[
    {"type": "text", "text": "Tell me about this capital"},
    {"type": "image", "base64": img_b64, "mime_type": "image/png"}
])

response = agent.invoke(
    {"messages": [question]}
)

print(response['messages'][-1].content)