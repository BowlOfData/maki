# Run traces

A trace is a file that records what an agent did, step by step: each task, each model round inside it, and each tool call a round asked for. It answers "why did the model fail at round 4" without a debugger, and it is the base for replaying and resuming runs later.

```python
from maki import Agent

agent = Agent("helper", backend, trace="runs.jsonl")
agent.execute_task("Summarise today's news", use_plugins=True)
```

```bash
maki trace list runs.jsonl          # one line per run: id, spans, errors, tokens, seconds
maki trace show runs.jsonl          # the latest run as a tree
maki trace show runs.jsonl --trace 3f2a    # an id or a prefix of one
maki trace show runs.jsonl --json   # the raw records of that run
```

```text
trace 9c1e4b7a02d35f68
task   'Summarise today's news'  4.21s
├─ model  3 messages 812 in / 40 out tokens -> web_search__search  1.84s
├─ tool   web_search__search {"query": "news today"}  0.42s  -> 'Top stories: ...'
└─ model  5 messages 1204 in / 310 out tokens  1.95s
```

## What is recorded

One JSON object per line, appended as events happen (so a crashed run still shows its last open span, marked `running`):

| Field | Meaning |
|---|---|
| `trace`, `span`, `parent` | ids: every span of one run shares `trace`; `parent` is the span it ran inside |
| `kind`, `ph` | `task`, `model` or `tool`; `start`, `end`, `error` or `retry` |
| `agent`, `ts`, `seconds` | agent name, UTC time, elapsed seconds (on `end` and `error`) |
| task | `task`, `result` or `error` |
| model | `messages`, `prompt_tokens`, `completion_tokens`, `cached_tokens`, `tool_calls` |
| tool | `tool`, `args`, `result` or `error`, retries as `retry` lines |

A plain task (no tools) has one model span; a streamed task (`stream_task`) has one model span for the whole stream, without token counts since they are not known while streaming.

## Other ways in

```python
manager.add_agent("writer", "writer", "Be brief.", trace="runs.jsonl")   # AgentManager
report = run_eval(suite, models, trace="evals.jsonl")                       # every eval run
```

`maki eval --trace evals.jsonl` does the same from the command line, and a failed case prints `[trace <id>]`, so `maki trace show evals.jsonl --trace <id>` shows what the model did in that run. `RunResult.trace_id` (and the JSON report) carry the id; setup turns and the scored turn of a case share one trace.

## Keeping it safe

Traces hold prompts, answers and tool arguments, so:

- the file is created readable by you only (`0600`);
- every text field is clipped to 500 characters by default, and arguments stored under keys that look like credentials (`key`, `token`, `secret`, `password`, ...) are replaced by `***`;
- `TraceCallback(path, include_content=False)` records timing, token counts and names only, never text;
- the file rolls over to `<path>.1` at 50 MB (`max_bytes`, 0 turns it off; one backup is kept) (`maki trace show runs.jsonl.1 runs.jsonl` reads both);
- a trace that cannot be written logs one warning and never breaks the run.

```python
from maki import Agent, TraceCallback

agent = Agent("helper", backend, callbacks=[TraceCallback("runs.jsonl", max_chars=200, include_content=False)])
```

## Workflows and nested agents

`coordinate_agents`, `collaborative_task` and `run_workflow` (parallel tasks included) put all their agents' tasks in one trace, so each agent's file shows the run as a whole. When an agent's tool runs another agent (or a whole workflow) inside a traced task, the inner spans hang under the tool call's span, also when the two agents write to different files; join the files with `maki trace show a.jsonl b.jsonl`.

If a log rotator renames, deletes or truncates the file while an agent runs, the next event goes to a fresh file at the same path.

## Across `maki serve`

The trace id travels with the request: a client sends it in `X-Maki-Trace-Id` (and its open span in `X-Maki-Parent-Span`), and the server uses it for its own spans and echoes it back. An `AgentProxy` call made inside a traced run does this automatically, so the two files join into one tree:

```bash
maki trace show client.jsonl server.jsonl
```

The server records a trace when its agent has one; in the YAML config:

```yaml
trace: /var/log/maki/runs.jsonl
```

The server accepts only short plain ids (letters, digits, `.`, `_`, `-`, up to 64 characters) from the headers and invents its own for anything else, since the id ends up in logs and files.

## Reading traces in code

```python
from maki.trace import read_trace, build_spans, list_traces

records = read_trace("runs.jsonl", "server.jsonl")     # time-ordered; torn lines are skipped
for trace_id, roots in build_spans(records).items():
    print(trace_id, [span.status for span in roots])
```

`trace_context("my-trace-id", parent_span)` makes the agent runs inside a `with` block part of an existing trace.
