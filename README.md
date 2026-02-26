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

## Python Compatibility

This framework is compatible with Python 3.7 and above. The code has been updated to remove all dependencies on Python 3.11+ features like `StrEnum`, ensuring compatibility with older Python versions.

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
├── __main__.py
├── maki.py
├── connector.py
├── utils.py
├── urls.py
├── logging_config.py
├── agents/
│   ├── __init__.py
│   └── agents.py
├── plugins/
│   └── __init__.py
└── test/
    ├── __init__.py
    ├── test_maki_functionality.py
    ├── test_agent_functionality.py
    ├── test_error_handling.py
    └── test_different_llms.py
```

## Examples

Run the example:
```bash
python -m maki
```

## Logging Configuration

Starting from version 0.1.0, Maki no longer automatically configures logging when imported. This prevents unwanted side effects and conflicts with host application logging.

To configure logging in your application, you need to explicitly call the `configure_logging()` function from `maki.logging_config`:

```python
from maki.logging_config import setup_logging
import logging

# Setup logging with default settings (StreamHandler only)
setup_logging()

# Or setup with custom settings
setup_logging(log_level=logging.DEBUG, log_file_path="my_app.log")
```

If you want to use only console output (recommended for most cases):
```python
from maki.logging_config import setup_logging
setup_logging()
```

If you want to log to both console and file:
```python
from maki.logging_config import setup_logging
setup_logging(log_file_path="app.log")
```

## License

MIT