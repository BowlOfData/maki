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
- Workflow management system:
  - Task dependencies and conditions
  - Retry logic with configurable delays
  - Parallelizable task execution
  - Comprehensive workflow state tracking
  - Execution strategies (sequential, parallel, dependency-based)

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

## Multi-Agent Coordination

The AgentManager now supports advanced coordination patterns:

1. **Coordinate multiple agents**: `agent_manager.coordinate_agents(tasks, coordination_prompt)`
2. **Collaborative task execution**: `agent_manager.collaborative_task(task, agents, context)`
3. **Workflow execution**: `agent_manager.run_workflow(workflow)`

These methods enable sophisticated multi-agent workflows that can coordinate complex tasks across multiple agents.

## Plugin Support

Agents now support plugins for extending functionality. Plugins can be loaded and used within agent tasks:

```python
# Load a plugin
file_reader = agent.load_plugin("file_reader")

# Use the plugin
result = file_reader.read_file("example.txt")

# Get a loaded plugin
plugin = agent.get_plugin("file_reader")

# Unload a plugin
agent.unload_plugin("file_reader")
```

Plugins are automatically available to agents and can be used to extend agent capabilities with additional functionality like file reading, data processing, or other specialized tools.

Available built-in plugins include `file_reader`, `directory_reader`, `file_writer`, `web_to_md`, and `ftp_client`.

## Workflow Management

The enhanced workflow system allows for complex multi-agent coordination:

```python
from maki.agents import WorkflowTask, TaskStatus, WorkflowState

# Create workflow tasks with dependencies
tasks = [
    WorkflowTask(
        name="research_task",
        agent="Researcher",
        task="Research the latest developments in AI",
        dependencies=[],
        max_retries=2
    ),
    WorkflowTask(
        name="write_task",
        agent="Writer",
        task="Write a summary of the research findings",
        dependencies=["research_task"],
        max_retries=1
    )
]

# Execute workflow with different strategies
result = agent_manager.execute_enhanced_workflow(
    workflow_id="research_workflow",
    tasks=tasks,
    execution_strategy="dependency"
)
```

The workflow system supports:
- Task dependencies and conditions
- Retry logic with configurable delays
- Parallelizable task execution
- Comprehensive workflow state tracking
- Execution strategies (sequential, parallel, dependency-based)

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
├── exceptions.py
├── agents/
│   ├── __init__.py
│   ├── agents.py
│   └── workflow.py
├── plugins/
│   ├── __init__.py
│   ├── file_reader/
│   │   ├── __init__.py
│   │   ├── file_reader.py
│   │   ├── README.md
│   │   ├── USAGE.md
│   │   ├── example_usage.py
│   │   └── test_file_reader.py
│   └── file_writer/
│       ├── __init__.py
│       ├── file_writer.py
│       ├── README.md
│       ├── example_usage.py
│       └── test_file_writer.py
└── test/
    ├── __init__.py
    ├── test_maki_functionality.py
    ├── test_agent_functionality.py
    ├── test_error_handling.py
    ├── test_different_llms.py
    ├── test_logging.py
    ├── test_history_cleanup.py
    └── test_enhanced_workflow.py
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
from maki.logging_config import configure_logging
import logging

# Setup logging with default settings (StreamHandler only)
configure_logging()

# Or setup with custom settings
configure_logging(log_level=logging.DEBUG, log_file_path="my_app.log")
```

If you want to use only console output (recommended for most cases):
```python
from maki.logging_config import configure_logging
configure_logging()
```

If you want to log to both console and file:
```python
from maki.logging_config import configure_logging
configure_logging(log_file_path="app.log")
```

### Logging Levels

The framework uses standard Python logging levels:
- `DEBUG`: Detailed information, typically only of interest when diagnosing problems
- `INFO`: Confirmation that things are working as expected
- `WARNING`: An indication that something unexpected happened
- `ERROR`: Due to a more serious problem, the software has not been able to perform some function

### Log Format

All logs follow this format:
```
2026-03-03 14:30:45,123 - module_name - LEVEL - Message
```

### Example Usage with Logging

```python
import logging
from maki import Maki
from maki.logging_config import configure_logging

# Configure logging
configure_logging(log_level=logging.INFO)

# Create a logger for your application
logger = logging.getLogger(__name__)

# Initialize Maki
maki = Maki(url="localhost", port="11434", model="llama3", temperature=0.7)
logger.info("Maki initialized successfully")

# Use Maki
result = maki.request("Hello, world!")
logger.debug(f"Request result: {result}")
```

## License

MIT
