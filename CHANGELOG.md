# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- RAG support and initial refactoring toward a provider-agnostic architecture
- Equity market data support
- Forex market data support

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
