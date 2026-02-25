![Maki Logo](img/logo.png#center)

# Maki

Maki is a Python framework for multi-agent LLM interactions using Ollama.

## Features

- Simple interface for interacting with Ollama LLMs
- Multi-agent system for coordinated task execution
- Support for image inputs
- Easy-to-use API for both single and multi-agent workflows
- Enhanced reasoning capabilities:
  - Step-by-step thinking
  - Self-correction mechanisms
  - Task decomposition

## Installation

### Option 1: Install with pip (recommended)
```bash
pip install .
```

### Option 2: Copy folder to your project
You can also use the Maki framework by copying the `maki` folder directly into your project:

1. Copy the entire `maki` folder from this repository to your project directory
2. Make sure your project has the required dependencies:
   ```bash
   pip install requests
   ```
3. Import and use as normal:
   ```python
   from maki import Maki
   from maki.agents import AgentManager, Agent
   ```

This approach allows you to use Maki without installing it as a package, which is useful for:
- Embedding Maki directly in your project
- Development and testing
- Projects with restricted package installation

## Quick Start

```python
from maki import Maki
from maki.agents import AgentManager, Agent

# Initialize Maki
maki = Maki(url="localhost", port="11434", model="llama3", temperature=0.7)

# Create an agent manager
agent_manager = AgentManager(maki)

# Add some agents
researcher = agent_manager.add_agent(
    name="Researcher",
    role="research analyst",
    instructions="You are an expert researcher who can find and analyze information on various topics."
)

# Execute a task
result = agent_manager.assign_task("Researcher", "Research the benefits of renewable energy")
print(result)
```

## Enhanced Reasoning Capabilities

The enhanced Agent class now supports:

1. **Step-by-step thinking**: `agent.think_step_by_step(problem, steps=3)`
2. **Self-correction**: `agent.self_correct(initial_response, feedback)`
3. **Task decomposition**: `agent.decompose_task(task, max_subtasks=5)`

These capabilities allow agents to handle more complex reasoning tasks and improve their performance through iterative refinement.

## Project Structure

```
maki/
├── __init__.py
├── maki.py
├── connector.py
├── utils.py
├── urls.py
├── llm_objects/
│   ├── __init__.py
│   └── ollama_payload.py
├── agents/
│   ├── __init__.py
│   └── agents.py
└── examples/
    ├── __init__.py
    └── agent_example.py
```

## Examples

Run the example:
```bash
python -m maki.examples.agent_example
```

## License

MIT