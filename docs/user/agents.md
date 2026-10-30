# Agents

An `Agent` wraps a backend with a role, instructions, memory and plugins.

```python
from maki import MakiLLama
from maki.agents import Agent

llm = MakiLLama(model="llama3.2")
agent = Agent(
    name="Reviewer",
    maki_instance=llm,
    role="code reviewer",
    instructions="Focus on bugs, regressions, and missing validation.",
    stateful=True,
)

print(agent.execute_task("Review this design: a plugin system with file access."))
```

With `stateful=True` the agent keeps token-budgeted conversation history between tasks.

## Memory and reasoning helpers

```python
agent.remember("repo", "maki")
print(agent.recall("repo"))

steps = agent.think_step_by_step("How should we structure plugin validation?")
subtasks = agent.decompose_task("Prepare this repository for a public release")
```

## Streaming

```python
for chunk in agent.stream_task("Draft a short changelog entry."):
    print(chunk, end="", flush=True)
```

## Long-running tasks

By default `execute_task` sends one blocking request. For tasks that may exceed the configured timeout (default 120 s), set `use_streaming=True`; the timeout then applies per chunk rather than to the whole response.

```python
agent = Agent(name="Ranker", maki_instance=llm, role="content ranker", use_streaming=True)
result = agent.execute_task("Rank these 50 articles by relevance: ...")
```

## Tools

Load a [plugin](plugins.md) and let the model call it:

```python
agent.load_plugin("file_reader")
result = agent.execute_task("Read the first lines of README.md and summarize them.", use_plugins=True)
```

## Verified tasks

A workflow task can carry a checker agent that must accept the result before the task counts as complete. See [Workflows](workflows.md#verified-tasks-makerchecker).

## Loops

`agent.run_loop(goal, requirements, judge=reviewer)` keeps an agent working until every requirement is confirmed, with stall detection, token and time budgets, resume and optional human approval. See [Agent loops](loops.md).
