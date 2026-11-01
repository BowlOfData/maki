# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `MemoryPolicy` and `Agent(memory=...)`: opt-in limits on what an agent keeps and sends, all off by default. `history_mode="messages"` sends a stateful agent's prior turns as structured chat history (backends declare `supports_history`; the others keep the text form), `token_budget` sets the conversation budget, `tool_result_chars` clips plugin results on both the native and `TOOL:` paths, and `history_entry_chars` bounds what `task_history` and `reasoning_history` store. `maki serve` YAML accepts the same fields under `memory:`; `to_dict`/`from_dict` round-trip the policy
- `LLMResponse.cached_tokens` and `Usage.cached_tokens`: prompt tokens served from the provider's cache, reported by the Anthropic and OpenAI backends (0 elsewhere)
- `maki/test/memory_bench.py`: an offline benchmark of prompt tokens, recall and history size (`python -m maki.test.memory_bench`)

## [0.4.1] - 2026-11-01

### Added
- `file_mover` plugin: `move(src, dst, overwrite=False, dry_run=False, create_dirs=False)`, `undo_last()` and `list_moves()` move and rename files and directories identically on Linux, macOS and Windows (pure Python, no shell). Atomic rename on one volume; across volumes it copies, verifies size and SHA-256, moves the copy into place and only then removes the source. Case-only renames work on case-insensitive disks; symlinks move as links. `dst` is always the full new path; conflicts are refused and `overwrite=True` replaces a file with a file only (and cannot be undone). Scope is the operator's: `base_dir` by default, plus `allowed_roots`, or `allow_any_path=True`; the model cannot widen it. Every move is written to a monthly journal in `~/.maki/moves` (`retention_months`, default 12) and `undo_last()` reverses only the latest move, refusing if the item changed since. Windows file-name rules apply on a Windows host, or everywhere with `strict_names=True`. `move` and `undo_last` need `Agent(allow_dangerous_tools=True)`. `list_moves` and `undo_last` only see journal entries inside the plugin's own scope, so agents with different scopes can share a journal safely
- `Agent.load_plugin(name, plugin_path=None, **options)` forwards options to the plugin's `register_plugin(maki, **options)`
- CI runs the `file_mover` tests on macOS and Windows (Windows is best effort)

### Documentation
- `docs/user/plugins.md` lists `file_mover` and `postgres_store` and describes `load_plugin` options; plugin count corrected to 18 in the README and index
- `docs/user/loops.md` and `examples/loop_fix_tests.py`: `on_iteration` is not called for the iteration that ends the loop; the example prints that last event from `result.events`

## [0.4.0] - 2026-10-30

### Added
- Agent loops: `agent.run_loop(goal, requirements, judge)` iterates an agent until every requirement is confirmed. Requirements are checked by one batched call to a separate, stateless judge agent (JSON mode, thinking off, one retry, fail closed), by a user-written argv command (exit code; **not sandboxed**) or by a callable. A loop stops on success, `max_iterations`, `max_seconds`, `max_tokens` (`token_budget`), stall detection (default 3 non-improving iterations; judge-error iterations are skipped), a judge `block`, an agent error or a human `Decision.STOP` from the `on_iteration` hook. Context is `"shared"` (stateful agent) or `"fresh"`; long artifacts are clipped for the judge and the next prompt (`max_artifact_chars`). A human-editable scratchpad and a JSON checkpoint are written every iteration; `resume_loop()` continues after a crash, a cap or an error, an exclusive run lock prevents two runs of one loop, and `run_loop` refuses to overwrite an unfinished loop unless `overwrite=True`. With an `ApprovalStore` the loop suspends before each iteration until approved (reject with a note continues with that note). Agent-drafted requirements are text only and need `approve_requirements()` first. Exported: `Requirement`, `LoopSpec`, `LoopTask`, `LoopEvent`, `LoopResult`, `StopReason`, `Decision`, `run_loop`, `resume_loop`, `approve_requirements`, `draft_requirements`, `LoopBusyError`
- Loops over HTTP: a `judge:` section (and optional `loop:` caps) in the `maki serve` YAML enables `POST /loop` (text requirements only; 202 + server-chosen id), `GET /loop[/{id}]`, replayable `GET /loop/{id}/events` (SSE), `DELETE /loop/{id}` (cancel between iterations), `POST /loop/{id}/resume` and `POST /loop/{id}/approval` (for loops started with `require_approval`). One loop runs at a time; finished loops are pruned after `loop.retention_days` (default 14). `AgentProxy.run_loop()`, `resume_loop()`, `approve_loop()`, `loop_status()` and `cancel_loop()` are the client. `maki serve` refuses to start loops on a non-loopback host without `--api-key`; loop request bodies are limited to 256 KiB. `maki serve` configs also accept `think:` for Ollama
- `WorkflowTask(loop=LoopTask(judge=..., requirements=...))`: the task completes only when its loop succeeds, otherwise it fails and its dependents are skipped. Local agents only; exclusive with `verify=`
- Token usage: `Agent.usage` (prompt, completion, total tokens and calls) counts every backend response, and `LoopSpec.max_tokens` budgets a loop. A counted backend can still be copied and pickled
- New guide `docs/user/loops.md`; loops added to the README, index, agents, workflows and serving docs; examples `loop_fix_tests.py`, `loop_server.yaml`, `loop_remote.py`, `loop_workflow.py`

- Verified workflows (maker/checker): `WorkflowTask(verify=VerifySpec(verify=..., max_revisions=2))`. The result must be accepted by an independent verifier (a callable returning `VerifyResult(Verdict.ACCEPT|REVISE|BLOCK, evidence, fixes)` or a bool, or `agent_verifier(agent_name, criteria)` which asks a *different* agent for a JSON verdict and fails closed) before the task is `COMPLETED`. `REVISE` re-runs the task with `__verification_feedback`/`__previous_result` in its context up to `max_revisions`; `BLOCK`, an exhausted budget or any verifier error fails the task and skips its dependents. Verdict, evidence and history are stored in `WorkflowState.tasks[name]["verification"]` and survive checkpoint/resume. A verifier agent equal to the producing agent is rejected with `ValueError`
- Structured outputs for Ollama: `MakiLLama.chat()`/`async_chat()` accept `response_format` (`"json"` or a JSON-schema dict, sent as Ollama's `format`) and a per-call `think` override; the constructor accepts `keep_alive`. Structured replies never fall back to the thinking trace. New `maki.structured.generate_structured(backend, PydanticModel, prompt)` validates with pydantic and retries with the validation errors fed back, raising `StructuredOutputError` (a `MakiValidationError`) when the budget is exhausted (extra: `structured`)
- Human-approval gate for workflows: `WorkflowTask(approval=ApprovalSpec(key_fn=...))` parks the task as `TaskStatus.AWAITING_APPROVAL` instead of skipping it; downstream tasks stay `PENDING`, `run_workflow(..., approval_store=...)` returns with `last_workflow_state.status == "awaiting_approval"`, and re-running with the same `workflow_id` after `store.approve(...)` resumes. Approval is bound to a content key (a changed key invalidates it), supports expiry and rejection, and fails closed. `LocalApprovalStore` (JSON files) and `InMemoryApprovalStore` provided
- `TaskOutput(text, data)`: an agent may return it from `execute_task_with_retry`; `data` is stored on `WorkflowTask.data` and reaches dependents in their context (previously nothing in Maki populated `WorkflowTask.data`)
- `GateResult(ok, reason)`: conditions may return it; the reason is recorded in `WorkflowState` (`tasks[name]["reason"]`) and `WorkflowTask.gate_reason`
- `MakiLlamaCpp` backend for llama.cpp's `llama serve` (extra: `llamacpp`). Subclasses `MakiOpenAI`, so chat, streaming, async, vision and native tool calling work unchanged; adds `top_k`/`repeat_penalty` passthrough, `reasoning_content` capture, a streaming `chat_collect()` (per-chunk timeout for long local generations), and `health()`, `embed()`/`embed_batch()` (usable as `RagMemory(embedder=llm.embed)`) and `rerank()`. API key optional (`LLAMACPP_API_KEY`); base URL defaults to `http://127.0.0.1:8080/v1` (`MAKI_LLAMACPP_BASE_URL`)
- `MakiVllm` backend for vLLM servers (extra: `vllm`). Subclasses `MakiOpenAI` (chat, streaming, async, vision, native tools unchanged); adds `top_k`/`repetition_penalty` passthrough, structured outputs (`chat_json()` with a JSON schema or pydantic model, `chat_choice()`, `chat_regex()`), concurrent `chat_batch()`/`async_chat_batch()`, `reasoning` capture, `chat_collect()`, and `health()`, `models()`, `embed()`/`embed_batch()` and `rerank()`. LoRA adapters are selected via `model=`. API key optional (`VLLM_API_KEY`); base URL defaults to `http://127.0.0.1:8000/v1` (`MAKI_VLLM_BASE_URL`); model defaults to `Qwen/Qwen2.5-1.5B-Instruct` (`MAKI_VLLM_MODEL`); `maki serve` configs accept `backend: vllm`
- `LLMResponse.reasoning`: optional thinking text for backends that return it separately
- `maki serve` YAML configs accept `backend: llamacpp` (with optional `base_url`) and `backend: openrouter`, which was previously missing

### Documentation
- New guide `docs/user/memory.md` with worked examples for the `obsidian_memory` and `rag_memory` plugins (agent memory across runs, document Q&A, reranking, indexing a vault for semantic search) using vLLM for chat and embeddings; vLLM added to the README, index and quickstart backend lists

### Changed
- `maki serve` builds its agent through `load_server_config()` (the served agent, the optional loop judge and the loop caps); `load_agent_from_config()` is unchanged
- `create_app()` accepts `judge=` and `loop_caps=`; its routes are now module-level handlers on one router, and `_build_backend` is table-driven (no change in behaviour)
- `MakiLlamaCpp` and `MakiVllm` share a new abstract base, `OpenAICompatServer` (`maki/makiOpenAICompat.py`), holding the constructor, `chat_collect()`, reasoning capture, `health()`, `embed()`/`embed_batch()` and `rerank()`; each backend now only declares its defaults and `_server_extra_body()`. Public behaviour is unchanged
- `MakiOpenAI` gained `_request_kwargs()` and `_backend_type` hooks; `MakiOpenRouter` uses them instead of duplicating `_parse_response()`

### Security
- `web_to_md` now fetches with the full SSRF-protected connector. It previously ran with `ssrf_protect=False`, so only IP *literals* in the first URL were checked: a hostname whose DNS pointed at a private or metadata address (e.g. `169.254.169.254`) was fetched, and redirects (`allow_redirects=True`) were never re-validated, so a public page could bounce the fetch to an internal host. Every hostname, including each redirect hop, is now resolved and validated at connect time

### Fixed
- Pinned (SSRF-protected) HTTPS connections sent `Host: <ip>` instead of the real hostname: the connect-time IP swap on `conn._dns_host` was never undone, and urllib3 connects HTTPS connections before writing the request. CDNs such as Fastly answered `421 Misdirected Request` (and Cloudflare/GitHub Pages `403`/`404`) because the Host header no longer matched the SNI name. This was the real cause of the "DNS pinning breaks CDN-hosted sites" workaround (`ssrf_protect=False`) used in several plugins. Verified against pypi.org, docs.python.org, raw.githubusercontent.com, reddit.com, theverge.com, bbc.com, blog.cloudflare.com and GitHub Pages, all of which now return 200 with pinning on

## [0.3.2] - 2026-10-23

### Changed
- New project logo (`img/logo.png`), used by the PyPI page and the documentation site

## [0.3.1] - 2026-10-23

### Fixed
- PyPI project page: the logo did not render because the README used a relative image path. The README is now a short landing page with absolute image and link URLs

### Added
- User documentation site (MkDocs Material) at https://bowlofdata.github.io/maki/docs/, and `[project.urls]` metadata

## [0.3.0] - 2026-09-08

### Added
- RAG support and initial refactoring toward a provider-agnostic architecture
- Equity market data support (`alpaca_data.get_equity_bars`/`get_equity_latest_quote`) and forex market data support (`get_forex_bars`/`get_forex_latest_quote`, via Yahoo Finance) — all four now exposed to LLM tool-calling via `ALLOWED_METHODS` (previously implemented but unreachable through the tool layer, and forex/equities were undocumented)
- Optional-dependency extras: `gui`, `ftp`, `web`, `trends`, `alpaca`, `distributed-redis`, `all`
- `maki/py.typed` marker so downstream type checkers pick up maki's type hints
- `maki.__version__`, resolved from installed package metadata
- `LICENSE` file (MIT), matching what `pyproject.toml` and `README.md` already declared
- `.github/workflows/publish.yml` — PyPI publish workflow via Trusted Publishing (OIDC); requires one-time trusted-publisher setup on pypi.org before it can actually publish

### Changed
- **Packaging**: PyPI distribution renamed from `maki` to `maki-framework` — the name `maki` was already taken on PyPI by an unrelated abandoned package. The import path is unchanged (`import maki` still works); only the `pip install` target and extras syntax change (e.g. `pip install "maki-framework[alpaca]"`)
- `alpaca_trading`'s live-trading gate env var renamed `TRANDING_ALLOW_LIVE` → `MAKI_ALPACA_ALLOW_LIVE` (old name kept as a deprecated, warning-emitting alias)
- **Security**: `Connector` reworked into the single hardened HTTP layer for the whole framework — it owns URL validation, DNS resolution with resolved-IP validation and connect-time pinning (a hostname whose records include any private/reserved address is rejected, and the socket connects to the exact validated IP, closing the DNS-rebinding window; TLS still verifies against the original hostname), timeouts from `maki.config` (the hardcoded 180 s is gone), and the mapping of transport/status failures onto the Maki exception tree (transport timeout & HTTP 408/504 → `MakiTimeoutError`, connection failures & 5xx → `MakiNetworkError`, other 4xx → `MakiAPIError`). A new `AsyncConnector` (httpx) covers the async paths with pre-flight validation
- All outbound HTTP now routes through the hardened layer: `MakiLLama`, `AgentProxy`, and the `web_to_md`, `web_search`, `provider_updates`, `media_search`, and `rag_memory` plugins no longer make raw `requests`/`httpx` calls. Operator-configured endpoints (Ollama base URL, registered remote agents, the RAG embedder) use `allow_private=True` so LAN/loopback deployments keep working; content-derived URLs (web plugins) get the full SSRF treatment
- Connection pinning falls back through the validated address list, so dual-stack hosts whose service listens on one address family only (e.g. `localhost` → `::1` first, server on IPv4) still connect — every attempted address has passed validation
- `MakiLLama` HTTP error classes changed with the unified mapping: 5xx responses now raise the retryable `MakiNetworkError` (previously `MakiAPIError`), and `stream()` failures are classified instead of escaping as raw `requests` exceptions
- `AgentProxy` no longer requires `httpx` (the distributed client now rides on the core `requests`-based layer)
- `Utils.PRIVATE_IP_RANGES`: removed `128.0.0.0/16` (allocated public space, wrongly blocked for decades' worth of hosts) and corrected the TEST-NET labels (`198.18.0.0/15` is the RFC 2544 benchmarking range, not TEST-NET)
- `web_to_md` retries on exception *types* (`MakiNetworkError`) instead of matching exception-name substrings
- **Breaking**: `AgentServer` `/stream` is now `POST` with the same body as `/execute`, so task text stays out of access logs and URL length limits no longer apply; `AgentProxy` updated to match, and its `context` argument — previously dropped by the GET query string — is now forwarded
- **Security**: `AgentServer` API-key comparison is constant-time (`secrets.compare_digest`)
- **Security**: `/execute` no longer echoes internal error details to remote callers — validation errors (`MakiValidationError`/`ValueError`) return 400 with the validation message, backend timeouts return 504 (which `AgentProxy` maps back to `MakiTimeoutError`), network failures 502, and everything else a generic 400/500 body with the full exception logged server-side only; the `/stream` SSE error event is sanitized the same way
- `GET /health` no longer requires the API key, so load-balancer/k8s probes work against authenticated servers
- YAML agent configs: plugin names are validated against `maki.plugins.PLUGIN_REGISTRY` (previously any string reached `importlib`), and served agents keep dangerous tool methods disabled unless the config sets the new `allow_dangerous_tools: true` field
- **Breaking / security**: plugin tool calls are now **fail-closed** — a plugin exposes only the methods named in its class-level `ALLOWED_METHODS`; a plugin without one exposes nothing to `TOOL:` directives (a warning is logged at load time). Destructive methods listed in a plugin's `DANGEROUS_METHODS` (file writes, FTP uploads/downloads/deletes, trade submission/cancellation) additionally require `Agent(allow_dangerous_tools=True)`
- `file_reader`, `file_writer`, `directory_reader`, `ftp_client`, and `web_to_md` now declare `ALLOWED_METHODS`; `file_writer`, `ftp_client`, and `alpaca_trading` mark their destructive methods in `DANGEROUS_METHODS`
- **Packaging**: `pyproject.toml` is the single manifest — `setup.py` and `requirements.txt` removed; version bumped to 0.2.0
- **Breaking**: `PySide6` and `paramiko` are no longer core dependencies — install `maki[gui]` / `maki[ftp]`; core install is just `requests`, `httpx`, `python-dotenv`
- `requires-python` raised to `>=3.10` (the code already used 3.9+/3.10+ syntax); CI matrix now 3.10–3.13
- Plan documents moved from the repo root to `docs/`

### Fixed
- `alpaca_data`'s forex methods depended on `yfinance`, an undeclared dependency (not listed in the `alpaca` extra) — `pip install "maki-framework[alpaca]"` now installs it, and the import is wrapped with the same friendly-error convention as every other optional dependency
- `alpaca_news.get_rss_news` fetched feeds via raw `feedparser.parse(url)`, bypassing the hardened `Connector` entirely (no timeout, no SSRF validation, no Maki error classification) — now routed through `Connector` like every other RSS fetch in the codebase
- `web_search`'s default Reddit User-Agent identified requests as an unrelated downstream project (`maki_newsletter`/`maki_bot`) — now a generic maki identity, still overridable via `MAKI_REDDIT_USER_AGENT`
- `MakiLLama.chat()` computed `elapsed` before checking the response status, and a 200 response with a non-JSON body leaked a raw `json.JSONDecodeError`; both now go through the hardened layer (`MakiAPIError` for invalid JSON)
- `rag_memory._ollama_embed` called `Connector.post()`, a method that did not exist (the old `Connector` only had `simple`/`version`) — embedding via Ollama would have crashed with `AttributeError`; the reworked instance API makes the call real
- `LocalStateStore` checkpoint writes are now atomic (temp file + `os.replace`) — a crash mid-save no longer leaves a corrupt checkpoint, which is the scenario checkpointing exists for; applies to both `save_workflow` and `update_task`
- Workflow IDs that sanitize to the same filename/Redis key (e.g. `a/b` and `a_b`) no longer overwrite each other's state — sanitized IDs get a short hash suffix (IDs that need no sanitization are unchanged)
- `MakiLLama.pull()` crashed with `TypeError` on the first progress chunk (`log.info(..., end="\r")`); progress is now logged at ~10% intervals
- `AgentProxy` streaming raised `httpx.ResponseNotRead` on non-2xx responses instead of the mapped Maki error, and the circuit breaker never recorded the failure
- `Utils.convert64` rejected any path under a symlinked directory (e.g. `/tmp` on macOS); now resolves symlinks and optionally enforces containment via a new `allowed_dirs` parameter. It also returns a base64 `str` (as documented) instead of `bytes`
- `MakiLLama.__call__` silently dropped the `system` and `images` kwargs
- `HFBackend.stream()` ignored the configured `GenerationConfig` and streamed with defaults
- `Utils.cleanup_response` kept no reference to the scheduled `aclose()` task, so the cleanup could be garbage-collected before running
- `PluginHandler.load_plugin` fell through to calling a module object (always `TypeError`); now raises a clear `MakiValidationError` naming the plugin
- `ChatSession` streaming lost both turns of history when the consumer abandoned the stream mid-way
- Ten plugins (`trend_search`, `web_search`, `provider_updates`, `media_search`, `rag_memory`, `obsidian_memory`, and the four `alpaca_*` plugins) declared `ALLOWED_METHODS` only at module level, where tool-call validation (which reads the plugin instance) never saw it — their whitelists were silently ineffective. The lists are now mirrored onto the plugin classes

### Removed
- Dead `media_search` plugin (a Pexels stock-photo integration, already deleted from `PLUGIN_REGISTRY` in a prior commit) — removed its leftover bytecode artifacts and its stale mention in `CLAUDE.md`
- Stale `maki_newsletter` packaging exclude in `pyproject.toml`, and business/planning docs (`docs/IDEAS_startup_investor_report.md`, `docs/news_letter_plan.md`) untracked from the repo — content preserved locally, just no longer shipped
- Stale root-level files: `local_llm.py`, `local_llm_v2.py`, `orchestrator.py`, `review.md`, `examples/demo_implementation.py` (all referenced the deleted `Maki` class or were pre-package copies)
- `Utils.compose_url` (110 lines of URL sanitisation serving no remaining caller), `Utils._validate_port`, and `maki/urls.py` (`Actions` enum, `GENERIC_LLAMA_URL`) — dead since the `Maki` class was removed; the hardened `Connector` is the one place URLs are validated now
- `Connector.simple()` / `Connector.version()` static methods (unused; replaced by the instance `get`/`post`/`delete` API)

---

## [0.1.0] — 2025

### Added
- **Agent system** — `Agent` class with task execution, memory (remember/recall/clear), conversation history, plugin support, and step-by-step reasoning via `PluginHandler` and `ReasoningEngine` mixins
- **AgentManager** — orchestrates multiple agents with `assign_task`, `coordinate_agents`, `collaborative_task`, and `run_workflow`
- **Workflow engine** — `WorkflowTask` / `WorkflowState` with topological sort, retry logic, and optional parallel execution
- **Ollama backends** — `Maki` (single-turn generate API) and `MakiLLama` (full chat API with streaming, async, multi-turn sessions); convenience factory functions `gemma3()`, `qwen()`, `llama()`, `mistral()`
- **HuggingFace backend** — `HFBackend` for direct Transformers integration with quantization and device selection (no Ollama required)
- **Plugin system** — plugins triggered by `TOOL:` directives in LLM output; built-in plugins: `file_reader`, `file_writer`, `directory_reader`, `web_to_md`, `ftp_client`
- **Alpaca trading support** — integration for brokerage operations
- **OCR plugin** — image-to-text via external OCR tooling
- **Infrastructure layer** — `Connector` HTTP client with SSRF protection, error classification, timeout handling, and rate limiting
- **Shared data classes** — `Message`, `GenerationConfig`, `LLMResponse` in `objects.py`
- **Custom exception hierarchy** — `MakiNetworkError`, `MakiTimeoutError`, `MakiAPIError`, `MakiValidationError`
- **Lazy loading** — `__init__.py` defers module imports until first access
- **Logging** — `configure_logging()` in `logging_config.py`; not auto-configured on import

### Security
- SSRF protection on all outbound URLs (loopback allowed for local Ollama)
- FTP/SFTP remote path traversal fixes
- Plugin path injection hardening
- Robust LLM output parsing to prevent injection via model responses
- Connection pooling and rate limiting added to HTTP layer
- Bounded conversation history using `deque(maxlen=1000)`

### Fixed
- Backend return type consistency across all LLM backends
- Mixin contracts enforced between `Agent`, `PluginHandler`, and `ReasoningEngine`
- Temperature validation and payload construction
- Workflow dependency resolution and parallel execution race conditions
- Undefined logger reference in `utils.py`
- ChatSession system prompt mutation bug
- FTP absolute path injection
