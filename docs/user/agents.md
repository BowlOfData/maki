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

With `stateful=True` the agent keeps token-budgeted conversation history between tasks. To send that history as chat messages, or to cap plugin results and stored history, pass a `MemoryPolicy`; see [Agent memory](agent-memory.md).

`AgentManager.add_agent()` accepts the same options as `Agent` (`stateful`, `allow_dangerous_tools`, `memory`, `tool_policy`, `callbacks`), so a manager-created agent needs no setup afterwards:

```python
manager.add_agent("Researcher", role="researcher", stateful=True,
                  memory={"token_budget": 2000, "summarize_older": True},
                  tool_policy={"retries": 2, "parallel": True})
```

## Async

Every blocking entry point has an awaitable twin, so an event loop (FastAPI, a bot, a notebook) keeps
running while a model answers:

```python
answer = await agent.aexecute_task("Summarise this", use_plugins=True)

async for chunk in agent.astream_task("Tell me a story"):
    print(chunk, end="")

results = await manager.agather_tasks([
    {"agent": "Researcher", "task": "Find three sources"},
    {"agent": "Writer", "task": "Draft the intro"},
])                                   # different agents run at the same time, results keep this order
```

Available: `Agent.aexecute_task`, `aexecute_task_with_retry`, `astream_task`, `arun_loop`;
`AgentManager.aassign_task`, `agather_tasks` (with `return_exceptions=`), `acollaborative_task`,
`arun_workflow`; `AgentProxy.aexecute_task`, `astream_task`.

These run the existing synchronous code on worker threads, so behaviour, memory, tool policies and
callbacks are identical. Calls to one agent are still serialised by that agent (a stateful agent's
history never interleaves) and the waiting happens on the event loop, so a long queue for one agent
never uses up the worker threads other agents need; different agents overlap. Cancelling an `await` stops the wait, not a
call already sent to the model, and breaking out of `astream_task` stops the stream after the chunk in
flight. Worker threads come from the event loop's default executor, and context variables (trace ids, logging context) are carried over. Callbacks and plugin methods run on those threads, so a callback that touches asyncio objects must hand off with `loop.call_soon_threadsafe`.

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
