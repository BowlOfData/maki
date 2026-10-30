# Agent loops

A loop gives an agent a goal and a list of requirements, then lets it iterate until every requirement is met. After each attempt the requirements are checked, the failures are fed back to the agent, and it tries again. Use it for work that needs several passes: fixing code until the tests pass, rewriting text until it meets a brief, gathering data until nothing is missing.

```python
from maki.agents import Agent, Requirement

result = writer.run_loop(
    "Write a product description for the Aurora water bottle.",
    ["Between 40 and 80 words", "Mentions that it is dishwasher safe"],
    judge=reviewer,           # a separate, stateless agent
    max_iterations=5,
)
print(result.stop_reason, result.iterations)
print(result.artifact)
```

`writer` must be created with `stateful=True` (the default `context="shared"` keeps one conversation) or you must pass `context="fresh"`. `reviewer` must be a different, stateless agent. Back it with a JSON-capable model with thinking off; on Ollama create it with `MakiLLama(json_format=True, think=False)`.

## Requirements and who checks them

| Kind | How to write it | Checked by |
|---|---|---|
| Text | a plain string, or `Requirement("id", "text")` | one batched call to the judge agent per iteration |
| Command | `Requirement("tests", "pytest passes", command=["pytest", "-q"])` | the command's exit code |
| Callable | `Requirement("short", "under 100 chars", check=lambda text: len(text) < 100)` | your function |

!!! warning "Commands are not sandboxed"
    A command runs as you, with no shell, a minimal environment, a timeout and a process-group kill. It can run whatever the agent just wrote. Commands and callables come only from code you write; requirements an agent drafts (`draft_requirements`) are text only and need `approve_requirements()` before they can run.

## When a loop stops

`result.stop_reason` is one of:

| Reason | Meaning |
|---|---|
| `succeeded` | every requirement passed |
| `max_iterations` | the iteration cap was reached |
| `timeout` | `max_seconds` of wall-clock time passed |
| `token_budget` | `max_tokens` (maker plus judge) was crossed |
| `stalled` | no improvement for `stall_limit` iterations (default 3) |
| `blocked` | the judge says the goal cannot be met as written |
| `stopped` | your `on_iteration` hook returned `Decision.STOP`, or an approval was rejected |
| `error` | the agent raised an error |
| `suspended` | waiting for a human approval |

Caps are checked between iterations, so a model call in flight finishes first. The token budget can overshoot by one iteration. Tokens come from the backends' reported usage (`agent.usage`); a backend that reports none counts as zero.

## Watching and steering

`on_iteration` is called with each `LoopEvent` and can stop the loop or give the agent a note to address next:

```python
from maki.agents import Decision

def hook(event):
    if event is None:                  # before the first iteration
        return None
    print(event.summary(), event.tokens, "tokens")
    return Decision.proceed("keep it under 60 words") if event.passing == 0 else None

writer.run_loop(goal, reqs, judge=reviewer, on_iteration=hook)
```

Every iteration also writes a **scratchpad** (`~/.maki/loops/<id>/scratchpad.md`) with the current judge feedback and the agent's own notes. You can edit the feedback and notes while the loop runs; the engine section is restored if you touch it.

## Resuming

State is checkpointed to `~/.maki/loops/<id>/state.json` after every iteration. `resume_loop(agent, judge, spec)` continues after a crash, a cap, an error or a suspension, with the same goal and requirements. If a requirement's text or command changed, the resume is refused. Only one process can run a loop at a time (`LoopBusyError`). Over HTTP a resume keeps the loop's original token budget (within the server cap) unless you send a new `max_tokens`; a loop that already spent its budget stops again at once, so raise it when you resume.

## Human approval before each iteration

Pass an `ApprovalStore` and the loop asks before every iteration, then returns `suspended` until someone answers:

```python
from maki.agents import LocalApprovalStore

store = LocalApprovalStore()
result = writer.run_loop(goal, reqs, judge=reviewer, approvals=store, loop_id="demo")   # -> "suspended"
store.approve("demo", "loop-gate")                  # or store.reject("demo", "loop-gate", note="try X")
writer.resume_loop(goal, reqs, judge=reviewer, approvals=store, loop_id="demo")
```

Approve lets the next iteration run. Reject with a note continues with that note; reject without a note stops the loop.

## Loops over HTTP

Serve an agent with a `judge:` section and loops become available:

```yaml
name: writer
backend: ollama
model: gemma4:26b
think: false
stateful: true
judge:
  backend: ollama
  model: gemma4:26b
loop:
  max_iterations: 20     # caps for remote loops
  max_seconds: 3600
  max_tokens: 200000
  retention_days: 14     # finished loops older than this are deleted
```

| Endpoint | Purpose |
|---|---|
| `POST /loop` | start a loop; returns 202 and an id. Fields: `goal`, `requirements`, `max_iterations`, `max_seconds`, `max_tokens`, `stall_limit`, `context`, `use_plugins`, `require_approval` |
| `GET /loop`, `GET /loop/{id}` | list loops, or one loop's status (includes `tokens` and `pending_approval`) |
| `GET /loop/{id}/events` | server-sent events; replay with `Last-Event-ID` or `?after=N` |
| `DELETE /loop/{id}` | cancel after the current iteration |
| `POST /loop/{id}/approval` | answer a suspended loop: `{"approve": true, "note": ""}` |
| `POST /loop/{id}/resume` | continue a stopped, interrupted or approved loop |

Remote loops accept **text requirements only**; a command or any other field is rejected with 400. The server runs one loop at a time (409 when busy), answers 503 when no judge is configured, and the served agent never judges its own work.

```python
from maki.distributed.proxy import AgentProxy

proxy = AgentProxy("http://127.0.0.1:8123", timeout=120)
result = proxy.run_loop(goal, ["Between 40 and 80 words"], max_iterations=5, max_tokens=50_000)

gated = proxy.run_loop(goal, reqs, require_approval=True)      # ends "suspended"
proxy.approve_loop(gated.loop_id, True, resume=True)           # approve, then continue
```

!!! note
    Anyone holding the server's API key can approve; there is no separate approver role. Cancel and the token budget take effect between iterations.

## In workflows

```python
from maki.agents import LoopTask, WorkflowTask

WorkflowTask("draft", "writer", "Write the announcement",
             loop=LoopTask(judge="reviewer", requirements=["Under 80 words"], max_tokens=40_000))
```

The task completes only when its loop succeeds; any other outcome fails it and skips the tasks that depend on it. Loop tasks run on local agents, not remote proxies, and cannot be combined with `verify=`. A resumed workflow resumes the loop if its goal and requirements are unchanged.

See `examples/loop_fix_tests.py`, `loop_server.yaml`, `loop_remote.py` and `loop_workflow.py`.
