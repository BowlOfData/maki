"""
OpenAICompatServer — shared base for self-hosted OpenAI-compatible servers.

llama.cpp (``llama serve``) and vLLM (``vllm serve``) both expose the OpenAI
chat completions API plus ``/health``, ``/v1/embeddings`` and ``/v1/rerank``.
This base reuses the openai Python SDK pointed at such a server and holds
everything the two backends have in common: construction, the streaming
``chat_collect()``, reasoning capture, and the native endpoints (called
through the hardened Connector).

A concrete backend sets ``_backend_type``, ``_api_key_env`` and
``_install_extra``, optionally ``_reasoning_fields``, and implements
``_server_extra_body()`` with the server-specific request parameters.
"""

from __future__ import annotations

import logging
import os
import time
from abc import abstractmethod
from typing import List, Optional

from . import makiOpenAI
from .config import DEFAULT_REQUEST_TIMEOUT
from .connector import Connector
from .exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError
from .objects import GenerationConfig, LLMResponse, Message, RateLimiter

log = logging.getLogger(__name__)

# These servers accept any bearer token when started without --api-key, but
# the openai SDK refuses to build a client without one.
_NO_KEY = "no-key"


class OpenAICompatServer(makiOpenAI.MakiOpenAI):
    """Base class for self-hosted OpenAI-compatible inference servers.

    Subclasses MakiOpenAI: chat(), stream(), async_chat(), session() and the
    native tool-calling methods are inherited unchanged.
    """

    _api_key_env: str                   # env var holding the optional API key
    _install_extra: str                 # pip extra named in the ImportError
    # Message/delta attributes that may carry thinking text, in priority order.
    _reasoning_fields: tuple = ("reasoning_content",)

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        rate_limit: Optional[int] = None,
    ) -> None:
        if makiOpenAI._openai_sdk is None:
            raise ImportError(
                f'openai package is required: pip install "maki-framework[{self._install_extra}]"'
            )

        self.model = model
        self.config = config or GenerationConfig()
        self.temperature = self.config.temperature
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get(self._api_key_env)
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
        log.info("%s · model=%s · base_url=%s", type(self).__name__, model, self.base_url)

    # ------------------------------------------------------------------
    # MakiOpenAI hooks
    # ------------------------------------------------------------------

    @property
    def _model_family(self) -> str:
        """Self-hosted models all take the standard chat parameters."""
        return "chat"

    @abstractmethod
    def _server_extra_body(self, cfg: GenerationConfig) -> dict:
        """Server-specific request parameters the OpenAI schema lacks."""

    def _request_kwargs(self, cfg: GenerationConfig) -> dict:
        kwargs = super()._request_kwargs(cfg)
        kwargs["extra_body"] = self._server_extra_body(cfg)
        return kwargs

    def _reasoning_of(self, obj: object) -> Optional[str]:
        """Thinking text carried by a response message or stream delta, if any."""
        for attr in self._reasoning_fields:
            value = getattr(obj, attr, None)
            if isinstance(value, str) and value:
                return value
        return None

    def _parse_response(self, response: object, elapsed: float) -> LLMResponse:
        result = super()._parse_response(response, elapsed)
        reasoning = self._reasoning_of(response.choices[0].message)  # type: ignore[attr-defined]
        if reasoning:
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
                    thinking = self._reasoning_of(delta)
                    if thinking:
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
    # Native endpoints beyond chat
    # ------------------------------------------------------------------

    @property
    def _server_root(self) -> str:
        """Server origin for native endpoints (base_url minus the /v1 suffix)."""
        return self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def health(self) -> bool:
        """Return True when the server is up and its model is ready.

        Any non-200 answer (llama.cpp returns 503 while a model loads) and
        an unreachable server both return False.
        """
        try:
            r = self._http.get(f"{self._server_root}/health", timeout=5, raise_on_status=False)
        except (MakiNetworkError, MakiTimeoutError) as e:
            log.debug("health: %s unreachable: %s", self._server_root, e)
            return False
        return r.status_code == 200

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed *texts* via ``/v1/embeddings`` (needs an embedding model)."""
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

        Requires a reranker model. Returns dicts with ``index``,
        ``relevance_score`` and ``document``, best match first.
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
        return f"{type(self).__name__}(model={self.model!r}, base_url={self.base_url!r})"
