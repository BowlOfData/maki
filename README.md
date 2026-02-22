# maki

A Python service for interacting with Ollama models. This library provides a simple interface to communicate with Ollama's API for generating text and retrieving model information.

## Features

- Send prompts to Ollama models
- Retrieve model version information
- Support for image-based prompts
- Simple and intuitive API
- Multi-agent system support for complex task coordination
- **Flexible LLM configuration per agent** - Different agents can use different models, temperatures, and endpoints

## Installation

```bash
pip install requests
```

## Usage

### Basic Setup

```python
from maki.maki import Maki

# Initialize the Maki object
maki = Maki(url="localhost", port="11434", model="llama3", temperature=0.7)
```

### Generate Text

```python
# Simple text generation
prompt = "Explain quantum computing in simple terms"
response = maki.request(prompt)
print(response)
```

### With Images

```python
# Generate text with image input
response = maki.request_with_images("Describe this image", "path/to/image.jpg")
print(response)
```

### Get Model Version

```python
# Get the version of the connected model
version = maki.version()
print(version)
```

## Multi-Agent System

The maki framework now supports multi-agent systems for complex task coordination:

### Creating Agents

```python
from maki.agents import AgentManager

# Create an agent manager
agent_manager = AgentManager(maki)

# Add agents with specific roles and instructions
researcher = agent_manager.add_agent(
    name="Researcher",
    role="research analyst",
    instructions="You are an expert researcher who can find and analyze information on various topics."
)

writer = agent_manager.add_agent(
    name="Writer",
    role="content writer",
    instructions="You are a skilled writer who can create clear, well-structured content based on research."
)
```

### Assigning Tasks

```python
# Assign tasks to specific agents
result = agent_manager.assign_task("Researcher", "Research the benefits of renewable energy")
print(result)
```

### Collaborative Tasks

```python
# Have multiple agents work together
result = agent_manager.collaborative_task(
    task="Write an article about AI ethics",
    agents=["Researcher", "Writer", "Editor"]
)
```

### Workflows

```python
# Execute a complete workflow
workflow = [
    {
        "name": "research",
        "agent": "Researcher",
        "task": "Research the latest developments in quantum computing",
        "context": {"focus": "applications"}
    },
    {
        "name": "write",
        "agent": "Writer",
        "task": "Write a summary of the quantum computing research findings",
        "context": {"tone": "technical"}
    }
]

results = agent_manager.run_workflow(workflow)
```

## Advanced: Different LLMs for Each Agent

The framework now supports different LLM configurations for each agent, enabling specialized agents for different tasks:

### Creating Specialized Agents

```python
from maki.maki import Maki

# Create different Maki instances for different models
research_maki = Maki(url="localhost", port="11434", model="llama3", temperature=0.7)
analysis_maki = Maki(url="localhost", port="11434", model="mixtral", temperature=0.3)

# Create agent manager with default Maki instance
agent_manager = AgentManager(research_maki)

# Add agent using default Maki instance (backward compatibility)
researcher = agent_manager.add_agent(
    name="Researcher",
    role="research analyst",
    instructions="You are an expert researcher who can find and analyze information on various topics."
)

# Add agent using specialized Maki instance (new functionality)
analyst = agent_manager.add_agent(
    name="Analyst",
    role="data analyst",
    instructions="You are a skilled data analyst who can interpret and explain complex data.",
    maki_instance=analysis_maki
)

# Both agents can now use different LLMs
print(f"Researcher uses: {researcher.maki.model}")  # llama3
print(f"Analyst uses: {analyst.maki.model}")        # mixtral
```

### Benefits of Different LLMs

- **Specialized Tasks**: Use more capable models for complex reasoning, lighter models for simple tasks
- **Optimized Performance**: Match model capabilities to task requirements
- **Backward Compatibility**: Existing code works without changes
- **Flexible Configuration**: Different temperature settings, endpoints, and models per agent

## API Reference

### Maki Class

- `__init__(url: str, port: str, model: str, temperature=0)`: Initialize the Maki object
- `request(prompt: str) -> str`: Send a prompt to the LLM and return the response
- `version() -> str`: Get the version of the connected LLM
- `request_with_images(prompt: str, img: str) -> str`: Send a prompt with image input

### Agent Class

- `__init__(name: str, maki_instance: Maki, role: str = "", instructions: str = "")`: Initialize an agent
- `execute_task(task: str, context: Optional[Dict] = None) -> str`: Execute a task
- `remember(key: str, value: Any)`: Store information in memory
- `recall(key: str) -> Any`: Retrieve information from memory
- `clear_memory()`: Clear the agent's memory

### AgentManager Class

- `__init__(maki_instance: Maki)`: Initialize the agent manager
- `add_agent(name: str, role: str = "", instructions: str = "", maki_instance: Maki = None) -> Agent`: Add a new agent
  - `maki_instance`: Optional Maki instance to use for this agent. If not provided, uses the manager's default Maki instance.
- `get_agent(name: str) -> Optional[Agent]`: Get an agent by name
- `remove_agent(name: str)`: Remove an agent
- `list_agents() -> List[str]`: List all agent names
- `assign_task(agent_name: str, task: str, context: Optional[Dict] = None) -> str`: Assign task to agent
- `coordinate_agents(tasks: List[Dict], coordination_prompt: str = "") -> Dict[str, str]`: Coordinate multiple agents
- `collaborative_task(task: str, agents: List[str], context: Optional[Dict] = None) -> str`: Have agents collaborate
- `run_workflow(workflow: List[Dict]) -> Dict[str, Any]`: Execute a complete workflow

## Requirements

- Python 3.6+
- requests library

## License

MIT