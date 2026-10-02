"""
MakiLlamaCpp — llama.cpp server backend for the Maki framework.

`llama serve` (https://llama.app/docs/serve) runs GGUF models locally and
exposes an OpenAI-compatible REST API. This backend reuses the openai Python
SDK, pointed at the llama.cpp server, and adds the llama.cpp-specific
pieces: extra sampling parameters, reasoning-content capture, streaming
collection for long local generations, and the /health, /v1/embeddings and
/v1/rerank endpoints (called through the hardened Connector).

Requirements:
    pip install "maki-framework[llamacpp]"
    llama serve -hf ggml-org/gemma-4-e4b-it-GGUF:Q4_0

An API key is only needed when the server was started with ``--api-key``;
pass ``api_key=`` or set LLAMACPP_API_KEY.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

from . import makiOpenAI
from .config import (
    DEFAULT_LLAMACPP_BASE_URL,
    DEFAULT_LLAMACPP_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
    LLAMACPP_API_KEY_ENV,
)
from .connector import Connector
from .exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError
from .objects import BackendType, GenerationConfig, LLMResponse, Message, RateLimiter

log = logging.getLogger(__name__)

# llama.cpp accepts any bearer token when started without --api-key, but the
# openai SDK refuses to build a client without one.
_NO_KEY = "no-key"


class MakiLlamaCpp(makiOpenAI.MakiOpenAI):
    """
    llama.cpp server backend (OpenAI-compatible chat completions API).

    Subclasses MakiOpenAI: chat(), stream(), async_chat(), session(),
    and the native tool-calling methods are inherited unchanged.

    Usage
    -----
        llm = MakiLlamaCpp()                       # http://127.0.0.1:8080/v1
        print(llm.chat("What is the capital of France?").content)

        # Router mode (`llama serve` without a model): name the model.
        llm = MakiLlamaCpp(model="ggml-org/gemma-3-4b-it-qat-GGUF:Q4_0")

        # Embeddings for RagMemory (server started with --embedding)
        rag = RagMemory(dsn="memory://", embedder=llm.embed)

    In single-model mode the server ignores ``model``, so the default
    placeholder works; in router mode it selects which model to load.
    """
    _backend_type: BackendType = BackendType.LLAMACPP

    def __init__(
        self,
        model: str = DEFAULT_LLAMACPP_MODEL,
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        rate_limit: Optional[int] = None,
        base_url: str = DEFAULT_LLAMACPP_BASE_URL,
    ) -> None:
        if makiOpenAI._openai_sdk is None:
            raise ImportError(
                'openai package is required: pip install "maki-framework[llamacpp]"'
            )

        self.model = model
        self.config = config or GenerationConfig()
        self.temperature = self.config.temperature
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get(LLAMACPP_API_KEY_ENV)
        self._rate_limiter = RateLimiter(rate_limit) if rate_limit is not None else None

        # Operator-configured endpoint (usually loopback or LAN), so
        # allow_private=True, matching MakiLLama's Ollama connection.
        self._http = Connector(timeout=timeout, allow_private=True)
        self._http.validate_url(self.base_url)

        client_kwargs: dict = {
            "api_key": self._api_key or _NO_KEY,
            "base_url": self.base_url,
            "timeout": timeout,
        }
        self._client = makiOpenAI._openai_sdk.OpenAI(**client_kwargs)
        self._async_client = makiOpenAI._openai_sdk.AsyncOpenAI(**client_kwargs)
        log.info("MakiLlamaCpp · model=%s · base_url=%s", model, self.base_url)

    # ------------------------------------------------------------------
    # MakiOpenAI hooks
    # ------------------------------------------------------------------

    @property
    def _model_family(self) -> str:
        """Local GGUF models all take the standard chat parameters."""
        return "chat"

    def _request_kwargs(self, cfg: GenerationConfig) -> dict:
        """Add the llama.cpp sampling parameters the OpenAI schema lacks."""
        kwargs = super()._request_kwargs(cfg)
        kwargs["extra_body"] = {"top_k": cfg.top_k, "repeat_penalty": cfg.repeat_penalty}
        return kwargs

    def _parse_response(self, response: object, elapsed: float) -> LLMResponse:
        result = super()._parse_response(response, elapsed)
        reasoning = getattr(response.choices[0].message, "reasoning_content", None)  # type: ignore[attr-defined]
        if isinstance(reasoning, str) and reasoning:
            result.reasoning = reasoning
        return result

    # ------------------------------------------------------------------
    # Streaming collection
    # ------------------------------------------------------------------

    def chat_collect(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
        images: Optional[list[str]] = None,
    ) -> LLMResponse:
        """Like ``chat()`` but streams internally.

        The SDK's read timeout then applies per chunk rather than to the
        whole response, so long local generations don't time out.
        """
        log.debug("chat_collect: %s", prompt[:100])
        if self._rate_limiter:
            self._rate_limiter.acquire()
        cfg = config or self.config
        messages = self._build_messages(prompt, history, system=system, images=images)
        content: list[str] = []
        reasoning: list[str] = []
        usage = None
        model = self.model
        t0 = time.perf_counter()
        try:
            with self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **self._request_kwargs(cfg),
            ) as stream:
                for chunk in stream:
                    if getattr(chunk, "usage", None) is not None:
                        usage = chunk.usage
                    if isinstance(getattr(chunk, "model", None), str) and chunk.model:
                        model = chunk.model
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if isinstance(delta.content, str) and delta.content:
                        content.append(delta.content)
                    thinking = getattr(delta, "reasoning_content", None)
                    if isinstance(thinking, str) and thinking:
                        reasoning.append(thinking)
        except makiOpenAI._openai_sdk.APITimeoutError as e:
            raise MakiTimeoutError(f"chat_collect() timed out: {e}") from e
        except makiOpenAI._openai_sdk.APIConnectionError as e:
            raise MakiNetworkError(f"chat_collect() connection failed: {e}") from e
        except makiOpenAI._openai_sdk.APIStatusError as e:
            raise MakiAPIError(f"chat_collect() HTTP error {e.status_code}: {e}") from e
        elapsed = time.perf_counter() - t0
        result = LLMResponse(
            content="".join(content),
            model=model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            elapsed_seconds=elapsed,
            done=True,
            backend=self._backend_type,
            reasoning="".join(reasoning) or None,
        )
        log.info("chat_collect: %.2fs, %d tokens", elapsed, result.total_tokens)
        return result

    # ------------------------------------------------------------------
    # llama.cpp endpoints beyond chat
    # ------------------------------------------------------------------

    @property
    def _server_root(self) -> str:
        """Server origin for native endpoints (base_url minus the /v1 suffix)."""
        return self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def health(self) -> bool:
        """Return True when the server is up and its model is loaded.

        ``/health`` answers 503 while a model is still loading; that and
        an unreachable server both return False.
        """
        try:
            r = self._http.get(f"{self._server_root}/health", timeout=5, raise_on_status=False)
        except (MakiNetworkError, MakiTimeoutError) as e:
            log.debug("health: %s unreachable: %s", self._server_root, e)
            return False
        return r.status_code == 200

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed *texts* via ``/v1/embeddings`` (server needs ``--embedding``)."""
        r = self._http.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": list(texts)},
            headers=self._headers(),
        )
        data = Connector.json_or_raise(r).get("data", [])
        return [item["embedding"] for item in sorted(data, key=lambda d: d.get("index", 0))]

    def embed(self, text: str) -> List[float]:
        """Embed a single *text*; usable as ``RagMemory(embedder=llm.embed)``."""
        vectors = self.embed_batch([text])
        if not vectors:
            raise MakiAPIError("Embeddings response contained no vectors")
        return vectors[0]

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[dict]:
        """Rank *documents* against *query* via ``/v1/rerank``.

        Requires a reranker model (``--rerank``). Returns dicts with
        ``index``, ``relevance_score`` and ``document``, best match first.
        """
        payload: dict = {"model": self.model, "query": query, "documents": list(documents)}
        if top_n is not None:
            payload["top_n"] = top_n
        r = self._http.post(f"{self.base_url}/rerank", json=payload, headers=self._headers())
        results = Connector.json_or_raise(r).get("results", [])
        ranked = [
            {
                "index": item["index"],
                "relevance_score": item["relevance_score"],
                "document": documents[item["index"]],
            }
            for item in results
        ]
        ranked.sort(key=lambda d: d["relevance_score"], reverse=True)
        return ranked

    def close(self) -> None:
        """Close the HTTP session used for the native endpoints."""
        self._http.close()

    def __repr__(self) -> str:
        return f"MakiLlamaCpp(model={self.model!r}, base_url={self.base_url!r})"


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def llamacpp(model: str = DEFAULT_LLAMACPP_MODEL, system: Optional[str] = None, **kwargs) -> MakiLlamaCpp:
    """Generic llama.cpp wrapper — defaults to a single-model server on :8080."""
    return MakiLlamaCpp(model=model, system_prompt=system, **kwargs)
