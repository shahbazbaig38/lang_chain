# LangChain Fundamentals

A comprehensive tutorial project covering the core concepts and practical implementations of LangChain, a framework for building applications with large language models.

## 📋 Overview

This project provides hands-on examples and implementations of LangChain fundamentals, organized into progressive modules that build from basic concepts to advanced patterns. Each module contains executable Python scripts demonstrating key LangChain features.

## 🚀 Features

- **Agent Creation & Management**: Learn how to create and configure LangChain agents
- **Model Integration**: Work with various language models including Ollama-hosted models
- **Tool Integration**: Implement custom tools and MCP (Model Context Protocol) servers
- **Message Management**: Handle conversation history with summarization middleware
- **State Management**: Use LangGraph for checkpointing and state persistence
- **Web Search Integration**: Incorporate real-time web search capabilities

## 📁 Project Structure

```
lang-chain/
├── main.py                    # Main entry point
├── pyproject.toml            # Project configuration and dependencies
├── README.md                 # This file
├── module-1/                 # Foundational Concepts
│   ├── 1.1_agent_initialization.py
│   ├── 1.1_foundational_models.py
│   ├── 1.1_prompting.py
│   ├── 1.2_tools.py
│   ├── 1.2_web_search.py
│   └── 1.3_memory.py
├── module-2/                 # Advanced Patterns
│   ├── 2.1_mcp.py
│   ├── 2.1_travel_agent.py
│   ├── 2.2_runtime_context.py
│   └── 2.2_state.py
├── module-3/                 # Production Features
│   └── 3.2_managing_messages.py
└── resources/
    └── 2.1_mcp_server.py    # MCP server implementation
```

## 📚 Modules Overview

### Module 1: Foundations
- **1.1 Agent Initialization**: Creating and configuring basic LangChain agents
- **1.1 Foundational Models**: Working with ChatOllama and model responses
- **1.1 Prompting**: Effective prompt engineering techniques
- **1.2 Tools**: Implementing and integrating custom tools
- **1.2 Web Search**: Adding web search capabilities with Tavily
- **1.3 Memory**: Managing conversation memory and context

### Module 2: Advanced Patterns
- **2.1 MCP**: Model Context Protocol implementation and server integration
- **2.1 Travel Agent**: Building specialized agents for specific domains
- **2.2 Runtime Context**: Managing runtime state and context
- **2.2 State**: Advanced state management with LangGraph

### Module 3: Production Features
- **3.2 Managing Messages**: Message summarization and middleware for long conversations

## 🛠️ Prerequisites

- Python 3.12 or higher
- [Ollama](https://ollama.ai/) installed and running
- Llama 3.1 model: `ollama pull llama3.1:latest`

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lang-chain
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Ensure Ollama is running with the required model:
```bash
ollama serve
ollama pull llama3.1:latest
```

## 🚀 Usage

Run individual modules to explore specific concepts:

```bash
# Run foundational examples
uv run module-1/1.1_agent_initialization.py
uv run module-1/1.1_foundational_models.py

# Run advanced patterns
uv run module-2/2.1_mcp.py
uv run module-2/2.1_travel_agent.py

# Run production features
uv run module-3/3.2_managing_messages.py
```

## 🔧 Key Dependencies

- **langchain**: Core framework for LLM applications
- **langchain-ollama**: Ollama model integration
- **langchain-mcp-adapters**: Model Context Protocol support
- **tavily-python**: Web search capabilities
- **langgraph**: State management and checkpointing

## 📖 Learning Path

1. **Start with Module 1** to understand basic LangChain concepts
2. **Progress to Module 2** for advanced agent patterns and MCP
3. **Explore Module 3** for production-ready message management

Each script includes comments and examples to help you understand the implementation details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve this tutorial.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama Models](https://ollama.ai/library)
- [Model Context Protocol](https://modelcontextprotocol.io/)