# Serving agents

Expose any agent over HTTP. Requires the `distributed` extra:

```bash
pip install "maki-framework[distributed]"
```

## Serve

```yaml
# agent.yaml
name: MyAgent
model: llama3.2
role: assistant
# backend: ollama | llamacpp | vllm | openai | anthropic | openrouter (default: ollama)
# base_url: http://127.0.0.1:8080/v1   # llamacpp / vllm only
plugins:
  - web_search
  - file_reader
```

```bash
maki serve --config agent.yaml --host 127.0.0.1 --port 8100
```

Options: `--api-key KEY` or the `MAKI_API_KEY` environment variable (bearer-token auth), `--tls-cert FILE` and `--tls-key FILE` (HTTPS), `--max-body-mb`, `--max-concurrency` and `--graceful-timeout` (see [Running in production](#running-in-production)). The server exposes `/execute`, `/stream`, `/health` and memory endpoints, plus `/loop` endpoints when the config has a `judge:` section (see [Agent loops](loops.md)).

## Running in production

`maki serve` is a small, single-agent server. Before exposing it beyond your own machine:

| Do | Why |
|---|---|
| Set the key with `MAKI_API_KEY` (not `--api-key`) | A key on the command line shows up in the process list and shell history; maki warns when you use the flag. |
| Terminate TLS in front of it, or pass `--tls-cert/--tls-key` | The bearer token and every prompt and answer cross the network in clear text otherwise. maki prints a warning when it listens off-loopback without a key or TLS. |
| Bind to `127.0.0.1` behind a reverse proxy when you can | The default host is loopback. Put rate limiting, per-client quotas and IP allow-lists in the proxy: the server has one shared bearer key and no rate limiting of its own. |
| Keep `--max-body-mb` modest (default 4) | Every request body is capped (413 beyond it); `/loop` has its own 256 KB cap. |
| Set `--max-concurrency` (for example 2 to 4 times the tasks you can queue) | The agent runs one task at a time and the rest wait in worker threads. Beyond the limit the server answers 503 instead of queueing without end. `/health` is not affected by the queue (it needs no worker thread). |
| Leave `--graceful-timeout` at a value your longest task fits in (default 30 s) | On SIGTERM in-flight requests get that long before the server exits, so a deploy never waits for a stuck model call. |
| Do not put `allow_dangerous_tools: true` or write-capable plugins on a network-facing agent | Anyone holding the key can then make the agent write files, move them or trade. Plugins are fail-closed for exactly this reason. |
| Give the agent model timeouts and a `tool_policy` (`max_rounds`, `retries`) | They bound how long one request can run and keep a failing tool from looping. |

With a key set, `/docs`, `/redoc` and `/openapi.json` are switched off (they are served without authentication); without a key they stay on for development. Errors returned to callers are generic; details are logged on the server only, and URLs in log lines and error messages have credential-like query parameters (`?api_key=`, `?token=`, ...) masked.

To check a deployment under load, `python -m maki.test.load_check --url http://host:8100 --api-key KEY --duration 60 --concurrency 8` drives a running server (execute, health and info only) and reports latency percentiles and any unexpected response. Without `--url` it starts the real server stack with a fake backend in a separate process and runs a mixed workload (valid calls plus malformed JSON, oversized bodies, wrong credentials, unknown paths) while sampling the server's memory, threads and open file descriptors; it fails on errors, hangs, or sustained drift.

What it does not do: one agent per process (run several processes for several agents), no persistence of a conversation across restarts, no metrics endpoint, and logs are plain text. Hosted backends other than OpenRouter were verified against test doubles, not live; try your own model and provider under your own load before relying on them.

## Loops over HTTP

Add a `judge:` section and the server also runs loops for remote callers (`POST /loop`, replayable event streams, cancel, resume and approvals):

```yaml
judge:
  backend: ollama
  model: gemma4:26b
loop:
  max_iterations: 20
  max_tokens: 200000
```

Remote loops take text requirements only and run one at a time. The server refuses to start with a `judge:` section on a non-loopback host unless `--api-key` is set. See [Agent loops](loops.md#loops-over-http) and `AgentProxy.run_loop`.

## Connect from another process

```python
from maki.distributed.proxy import AgentProxy

agent = AgentProxy(name="MyAgent", base_url="http://127.0.0.1:8100")
print(agent.execute_task("Summarize the latest AI news."))
```

`AgentProxy` has the same surface as `Agent` and includes a per-proxy circuit breaker.

## Mix local and remote agents

`DistributedAgentManager` lets one workflow use both:

```python
from maki.distributed.registry import DistributedAgentManager
```

Workflow checkpoints can be stored as local JSON files (`LocalStateStore`) or in Redis (`RedisStateStore`, extra `distributed-redis`).

!!! warning
    Bind to `127.0.0.1` unless you also set `--api-key` and TLS. An agent with plugins can read files and reach the network on behalf of whoever can call it.

## Tool policy

An optional `fallback:` list (`provider:model` strings or backend mappings) adds backends tried in order when the main one fails (see [Backends](backends.md#falling-back-to-another-backend)).

An optional `tool_policy:` section in the agent YAML sets how tool calls run: `max_rounds`, `retries` (transient errors, with backoff), `parallel` and `max_workers`. Unknown fields are rejected at startup. See [Plugins](plugins.md#tool-schemas-retries-and-callbacks).
