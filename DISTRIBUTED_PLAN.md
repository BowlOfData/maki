# Maki Distributed Agents — Feasibility & Implementation Plan

## Verdict: Feasible, but non-trivial

The core framework is well-structured and can be extended for distribution without a full rewrite. The main surgery points are **state serialization**, **agent location**, and **inter-agent communication**. The existing `Connector`, `LLMBackend`, and `WorkflowTask` abstractions give good seams to cut along.

Estimated effort: **3–5 weeks** for a minimal working system; 2–3 more to harden it.

---

## Current Architecture: What Works and What Doesn't

### Works in our favor
- `Agent` state is already logically separable: `memory`, `task_history`, `_conversation_history` are all plain dicts/deques.
- `WorkflowTask` already models dependencies and data passing between tasks — this is the DAG we need.
- `LLMBackend` is abstract; each backend is already an HTTP client to a remote endpoint.
- `objects.py` data classes (`Message`, `LLMResponse`, `GenerationConfig`) are nearly JSON-serializable already.
- `Connector` already handles HTTP with retry and error classification.

### Hard gaps
| Gap | Current state | What's missing |
|-----|--------------|----------------|
| Agent state | In-process deques/dicts | Serialization + external store |
| Agent location | `AgentManager.agents` local dict | Network registry + RPC |
| Inter-agent calls | Direct Python method calls | HTTP/gRPC protocol |
| Workflow coordination | Local topological sort | Distributed scheduler |
| Plugin execution | Filesystem imports | Remote plugin invocation |
| `WorkflowTask.conditions` | Python callables (can't serialize) | Expression language or DSL |

---

## Target Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Orchestrator Node                  │
│                                                     │
│  AgentManager ──► DistributedRegistry               │
│       │                  │                          │
│  WorkflowRunner    ◄── AgentProxy (per remote agent)│
│       │                                             │
│  WorkflowState ──► StateStore (Redis / Postgres)    │
└──────────────────────────┬──────────────────────────┘
                           │ HTTP/gRPC
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼───────┐
   │  Agent Node │  │  Agent Node │  │  Agent Node │
   │             │  │             │  │             │
   │  AgentServer│  │  AgentServer│  │  AgentServer│
   │  (FastAPI)  │  │  (FastAPI)  │  │  (FastAPI)  │
   │             │  │             │  │             │
   │  Agent(s)   │  │  Agent(s)   │  │  Agent(s)   │
   │  LLMBackend │  │  LLMBackend │  │  LLMBackend │
   │  Plugins    │  │  Plugins    │  │  Plugins    │
   └─────────────┘  └─────────────┘  └─────────────┘
```

Each **Agent Node** is a standalone FastAPI process exposing an HTTP API. The **Orchestrator** uses `AgentProxy` objects that look like local `Agent` instances but dispatch over the network. `AgentManager` and `WorkflowRunner` work unchanged — they only ever call the proxy interface.

---

## Phased Implementation Plan

### Phase 1 — Serialization Foundation
**Goal**: Make agent state fully serializable/deserializable. No networking yet.

#### 1.1 Add `agent_id` and `to_dict` / `from_dict` to `Agent`

`maki/agents/agent.py`
- Add `agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))`.
- Add `to_dict() -> dict`: serializes `name`, `role`, `instructions`, `memory`, and the last N entries of `task_history` and `_conversation_history`.
- Add `from_dict(cls, data, maki_backend) -> Agent`: reconstructs from dict, re-attaches a backend.

#### 1.2 Make data classes JSON-clean

`maki/objects.py`
- `Message.to_dict()` already exists. Add `Message.from_dict()`.
- `LLMResponse`: add `to_dict()` / `from_dict()`.
- `GenerationConfig`: add `to_dict()` / `from_dict()` (map from existing `to_openai_kwargs()` pattern).

#### 1.3 Serialize `WorkflowTask`

`maki/agents/workflow.py`
- Add `WorkflowTask.to_dict()` / `from_dict()`.
- `conditions` (Python callables) are **not serializable** — document that conditions are local-only. Remote workflows must use the `data` field and evaluate conditions at the orchestrator.

**Deliverable**: `pytest maki/test/test_serialization.py` passes. Agents can be pickled to JSON and reconstructed.

---

### Phase 2 — Agent Node (Server Side)
**Goal**: Run agents as standalone HTTP services.

#### 2.1 `AgentServer` — `maki/distributed/server.py`

A thin FastAPI wrapper around an `Agent`. Exposes:

```
POST /execute          body: {task, context?, generation_config?}
                       returns: {result, agent_id, elapsed}

GET  /stream           SSE stream for execute_task results

POST /memory/set       body: {key, value}
GET  /memory/{key}     returns: {value}
DELETE /memory/{key}

GET  /history          returns: [{role, content, timestamp}]
DELETE /history        clears conversation history

GET  /health           returns: {agent_id, name, role, status: "ok"}
GET  /info             returns: {agent_id, name, role, plugins, backend}
```

Design notes:
- One `Agent` instance per server process (or multiple agents on different URL prefixes).
- Auth via a pre-shared API key passed in `Authorization: Bearer <key>` header — checked on every request.
- Background task logs to a local file; no external dependency in Phase 2.

#### 2.2 Launch CLI

```bash
maki serve --agent-config agents/researcher.yaml --host 0.0.0.0 --port 8100
```

Agent config YAML:
```yaml
name: researcher
role: Research specialist
instructions: |
  You are a research agent...
backend: ollama          # or openai, anthropic
model: llama3.2
plugins:
  - web_to_md
  - file_reader
```

**Deliverable**: `maki serve` starts a process; `curl /health` returns 200.

---

### Phase 3 — Agent Proxy (Client Side)
**Goal**: `AgentProxy` makes remote agents look like local `Agent` objects to `AgentManager`.

#### 3.1 `AgentProxy` — `maki/distributed/proxy.py`

Implements the same interface as `Agent`:
- `execute_task(task, context)` → HTTP POST `/execute`, returns `str`.
- `execute_task_with_retry(...)` → retry loop around `execute_task`.
- `stream_task(task)` → SSE stream from `/stream`, yields `str` chunks.
- `remember(key, value)`, `recall(key)`, `clear_memory()` → proxy to `/memory/*`.

Properties forwarded from `/info`:
- `name`, `role`, `agent_id`, `plugins`.

Exceptions:
- HTTP 4xx → `MakiAPIError` (same as existing classification).
- Timeout → `MakiTimeoutError`.
- Connection refused → `MakiNetworkError`.

#### 3.2 `AgentRegistry` — `maki/distributed/registry.py`

Extends `AgentManager`'s agent dict with remote entries:

```python
manager = AgentManager(maki=local_backend)

# Register local agent
manager.add_agent(researcher_agent)

# Register remote agent
manager.register_remote("writer", endpoint="http://writer-node:8101", api_key="...")
```

`register_remote` creates an `AgentProxy` and inserts it under the same `agents` dict. `AgentManager.assign_task`, `coordinate_agents`, `collaborative_task`, and `run_workflow` all call `.execute_task()` — they work unchanged because `AgentProxy` speaks the same interface.

**Deliverable**: `manager.assign_task("researcher", "find papers on X")` dispatches to the remote node and returns the result.

---

### Phase 4 — Distributed Workflow State
**Goal**: `WorkflowState` survives node restarts; tasks can be resumed.

#### 4.1 `StateStore` interface — `maki/distributed/state_store.py`

```python
class StateStore(ABC):
    def save_workflow(self, state: WorkflowState) -> None: ...
    def load_workflow(self, workflow_id: str) -> WorkflowState: ...
    def update_task(self, workflow_id: str, task_name: str, update: dict) -> None: ...
    def list_workflows(self) -> list[str]: ...
```

Two implementations:
- `LocalStateStore` — writes JSON files to `~/.maki/workflows/`. No extra deps. Good for dev.
- `RedisStateStore` — stores workflow JSON in Redis with TTL. Requires `redis-py`.

#### 4.2 Wire `WorkflowRunner` to `StateStore`

`maki/agents/agent_manager.py` → `run_workflow` accepts optional `state_store` kwarg. If provided:
- Persist `WorkflowState` after each task completes.
- On startup, check if `workflow_id` already exists — resume from last checkpoint.
- Log errors to `state.error_log` and persist immediately.

**Deliverable**: Kill the orchestrator mid-workflow, restart it, and it resumes from the last completed task.

---

### Phase 5 — Hardening & Observability
**Goal**: Production-ready reliability.

#### 5.1 Health checks & circuit breaker

`AgentProxy` tracks consecutive failure count. After N failures, mark the remote agent as `DEGRADED` and raise `MakiNetworkError` immediately without waiting for timeout. Reset after a successful call.

#### 5.2 Distributed tracing

Add `trace_id` (UUID) to every task request. Propagate via HTTP header `X-Maki-Trace-Id`. Log at entry/exit of each agent node. `WorkflowState` records `trace_id` per task.

#### 5.3 Streaming support in proxy

`AgentProxy.stream_task()` consumes SSE from the remote `/stream` endpoint and re-yields locally. Caller code that uses `agent.stream_task()` works without changes.

#### 5.4 mTLS (optional, flagged)

Document how to configure `uvicorn` with TLS certs. `AgentProxy` passes `ssl_verify` and `cert` to `httpx` session. Leave disabled by default; enable with `--tls-cert` / `--tls-key` flags on `maki serve`.

---

## Key Technical Decisions

### Transport: HTTP + JSON (not gRPC)
**Reason**: Simpler to deploy and debug; `Connector` already provides the HTTP pattern; no Protobuf schema required. gRPC can be added later as an optional transport if latency becomes a concern.

### Streaming: Server-Sent Events
**Reason**: Native in HTTP; no WebSocket state management; works through proxies. `FastAPI` has first-class SSE support.

### State store: pluggable, default to local files
**Reason**: No forced infrastructure dependency for single-machine dev. Redis optional for multi-node.

### Auth: pre-shared API key
**Reason**: Simple to implement, sufficient for trusted internal networks. Not designed for public internet exposure — document this clearly.

### `conditions` in `WorkflowTask`: local-only
**Reason**: Python callables cannot be serialized. Remote workflows evaluate conditions at the orchestrator before dispatching tasks. This is a deliberate constraint, not a bug.

---

## Dependencies to Add

| Package | Phase | Why |
|---------|-------|-----|
| `fastapi` | 2 | Agent HTTP server |
| `uvicorn` | 2 | ASGI server |
| `httpx` | 3 | Async-capable HTTP client for proxy |
| `pyyaml` | 2 | Agent config files |
| `redis` | 4 (optional) | `RedisStateStore` |

All additions go into `pyproject.toml` as optional extras:
```toml
[project.optional-dependencies]
distributed = ["fastapi", "uvicorn", "httpx", "pyyaml"]
distributed-redis = ["maki[distributed]", "redis"]
```

Core `maki` install stays dependency-free except for existing packages.

---

## File Structure

```
maki/
├── distributed/
│   ├── __init__.py
│   ├── server.py        # AgentServer (FastAPI app)
│   ├── proxy.py         # AgentProxy
│   ├── registry.py      # AgentRegistry mixin for AgentManager
│   └── state_store.py   # StateStore interface + LocalStateStore + RedisStateStore
├── cli/
│   └── serve.py         # `maki serve` entrypoint
└── agents/
    └── agent.py         # + to_dict / from_dict
```

New test files:
```
maki/test/
├── test_serialization.py      # Phase 1
├── test_agent_server.py       # Phase 2 (TestClient)
├── test_agent_proxy.py        # Phase 3 (mock server)
└── test_distributed_workflow.py  # Phase 4
```

---

## What This Is NOT

- **Not a Kubernetes/cloud deployment solution** — that's infra, not framework. `maki serve` gives you a process; how you deploy and scale it is out of scope.
- **Not a message queue system** — no Kafka, RabbitMQ, or Celery. Tasks are dispatched synchronously (with retry). Async queuing can be layered on top later.
- **Not a multi-tenant SaaS** — no user isolation, no billing, no tenant routing. Single-operator use.

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Network latency degrades workflow throughput | Medium | Parallelizable tasks already batched; proxy adds little overhead |
| Agent state diverges across replicas | Low (stateless per request) | No shared mutable state; history is per-agent |
| `conditions` incompatibility | Medium | Clear docs + validation error if callable passed to remote workflow |
| Proxy hides errors from `WorkflowRunner` | Medium | Re-raise as typed Maki exceptions; tested in Phase 3 |
| API key leaks in logs | Low | Scrub `Authorization` header from logs in `AgentServer` |
