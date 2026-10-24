"""
MakiVllm — vLLM server backend for the Maki framework.

`vllm serve` (https://docs.vllm.ai) is a high-throughput GPU inference server
with continuous batching and an OpenAI-compatible REST API. This backend
reuses the openai Python SDK, pointed at the vLLM server, and adds the
vLLM-specific pieces: extra sampling parameters, structured outputs
(JSON schema / regex / choice), concurrent batch generation, reasoning
capture, and the /health, /v1/models, /v1/embeddings and /v1/rerank
endpoints (called through the hardened Connector).

Requirements:
    pip install "maki-framework[vllm]"
    vllm serve Qwen/Qwen2.5-1.5B-Instruct

The ``model`` passed to MakiVllm must match the name the server exposes
(the model path, or ``--served-model-name``). A LoRA adapter registered
with ``--lora-modules name=path`` is selected the same way: ``model="name"``.
An API key is only needed when the server was started with ``--api-key``;
pass ``api_key=`` or set VLLM_API_KEY.
"""

from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from .config import (
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_VLLM_BASE_URL,
    DEFAULT_VLLM_MODEL,
    VLLM_API_KEY_ENV,
)
from .connector import Connector
from .makiOpenAICompat import OpenAICompatServer
from .objects import BackendType, GenerationConfig, LLMResponse

# Structured-output constraint for the chat call in flight. A ContextVar keeps
# concurrent threads / tasks sharing one backend from seeing each other's.
_structured: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "maki_vllm_structured", default=None
)


class MakiVllm(OpenAICompatServer):
    """
    vLLM server backend (OpenAI-compatible chat completions API).

    Subclasses OpenAICompatServer (itself a MakiOpenAI): chat(), stream(),
    async_chat(), session(), vision (``images=``), chat_collect(), health(),
    embed() and rerank() are inherited unchanged. Native tool calling needs
    the server started with ``--enable-auto-tool-choice
    --tool-call-parser <parser>``.

    Usage
    -----
        llm = MakiVllm(model="Qwen/Qwen2.5-1.5B-Instruct")
        print(llm.chat("What is the capital of France?").content)

        # Output constrained to a JSON schema
        reply = llm.chat_json("Extract: Ada, 36", schema={...})

        # Many prompts at once; the server batches them
        replies = llm.chat_batch(["Summarise A", "Summarise B"])

        # Embeddings for RagMemory (server must be serving an embedding model)
        rag = RagMemory(dsn="memory://", embedder=llm.embed)
    """
    _backend_type: BackendType = BackendType.VLLM
    _api_key_env = VLLM_API_KEY_ENV
    _install_extra = "vllm"
    # Newer vLLM releases return `reasoning`, older ones `reasoning_content`.
    _reasoning_fields = ("reasoning", "reasoning_content")

    def __init__(
        self,
        model: str = DEFAULT_VLLM_MODEL,
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        rate_limit: Optional[int] = None,
        base_url: str = DEFAULT_VLLM_BASE_URL,
    ) -> None:
        super().__init__(
            model=model, base_url=base_url, api_key=api_key, config=config,
            system_prompt=system_prompt, timeout=timeout, rate_limit=rate_limit,
        )

    def _server_extra_body(self, cfg: GenerationConfig) -> dict:
        """Add the vLLM sampling and structured-output parameters."""
        # vLLM names the penalty `repetition_penalty` (llama.cpp: `repeat_penalty`).
        extra: dict = {"top_k": cfg.top_k, "repetition_penalty": cfg.repeat_penalty}
        structured = _structured.get()
        if structured:
            extra["structured_outputs"] = structured
        return extra

    # ------------------------------------------------------------------
    # Structured outputs
    # ------------------------------------------------------------------

    def _chat_structured(self, spec: dict, prompt: str, **chat_kwargs: Any) -> LLMResponse:
        token = _structured.set(spec)
        try:
            return self.chat(prompt, **chat_kwargs)
        finally:
            _structured.reset(token)

    def chat_json(self, prompt: str, schema: Any, **chat_kwargs: Any) -> LLMResponse:
        """Like ``chat()`` but the reply is guaranteed to match a JSON schema.

        *schema* is a JSON-schema dict or a pydantic model class. The reply's
        ``content`` is a JSON string; parse it with ``json.loads``. Accepts
        the same keyword arguments as ``chat()`` (history, config, system, images).
        """
        if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
            schema = schema.model_json_schema()
        if not isinstance(schema, dict):
            raise TypeError("schema must be a JSON-schema dict or a pydantic model class")
        return self._chat_structured({"json": schema}, prompt, **chat_kwargs)

    def chat_choice(self, prompt: str, choices: List[str], **chat_kwargs: Any) -> LLMResponse:
        """Like ``chat()`` but the reply is exactly one of *choices*."""
        if not choices or not all(isinstance(c, str) for c in choices):
            raise ValueError("choices must be a non-empty list of strings")
        return self._chat_structured({"choice": list(choices)}, prompt, **chat_kwargs)

    def chat_regex(self, prompt: str, pattern: str, **chat_kwargs: Any) -> LLMResponse:
        """Like ``chat()`` but the reply matches the regular expression *pattern*."""
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("pattern must be a non-empty string")
        return self._chat_structured({"regex": pattern}, prompt, **chat_kwargs)

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    async def async_chat_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
        max_concurrency: int = 32,
    ) -> List[LLMResponse]:
        """Run *prompts* concurrently; results keep the order of *prompts*.

        vLLM batches in-flight requests on the GPU, so concurrency is what
        buys throughput. *max_concurrency* caps requests in flight. The first
        failure is raised.
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        semaphore = asyncio.Semaphore(max_concurrency)

        async def one(prompt: str) -> LLMResponse:
            async with semaphore:
                return await self.async_chat(prompt, config=config, system=system)

        return list(await asyncio.gather(*(one(p) for p in prompts)))

    def chat_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
        max_concurrency: int = 32,
    ) -> List[LLMResponse]:
        """Run *prompts* concurrently from synchronous code.

        Uses a thread pool over the sync client, so it works from plain
        scripts and from inside a running event loop, and can be called
        repeatedly (an ``asyncio.run`` per call would orphan the async
        client's connections). Results keep the order of *prompts*; the
        first failure is raised.
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=min(max_concurrency, len(prompts))) as pool:
            return list(pool.map(
                lambda p: self.chat(p, config=config, system=system), prompts))

    def models(self) -> List[str]:
        """Ids of the models and LoRA adapters the server exposes (``/v1/models``)."""
        r = self._http.get(f"{self.base_url}/models", headers=self._headers())
        return [m["id"] for m in Connector.json_or_raise(r).get("data", [])]


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def vllm(model: str = DEFAULT_VLLM_MODEL, system: Optional[str] = None, **kwargs) -> MakiVllm:
    """Generic vLLM wrapper — defaults to a server on :8000."""
    return MakiVllm(model=model, system_prompt=system, **kwargs)
