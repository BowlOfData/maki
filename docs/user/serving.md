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

Options: `--api-key KEY` (bearer-token auth), `--tls-cert FILE` and `--tls-key FILE` (HTTPS). The server exposes `/execute`, `/stream`, `/health` and memory endpoints, plus `/loop` endpoints when the config has a `judge:` section (see [Agent loops](loops.md)).

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
