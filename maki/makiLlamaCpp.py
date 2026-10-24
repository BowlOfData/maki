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

from typing import Optional

from .config import (
    DEFAULT_LLAMACPP_BASE_URL,
    DEFAULT_LLAMACPP_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
    LLAMACPP_API_KEY_ENV,
)
from .makiOpenAICompat import OpenAICompatServer
from .objects import BackendType, GenerationConfig


class MakiLlamaCpp(OpenAICompatServer):
    """
    llama.cpp server backend (OpenAI-compatible chat completions API).

    Subclasses OpenAICompatServer (itself a MakiOpenAI): chat(), stream(),
    async_chat(), session(), the native tool-calling methods, chat_collect(),
    health(), embed() and rerank() are inherited unchanged.

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
    _api_key_env = LLAMACPP_API_KEY_ENV
    _install_extra = "llamacpp"
    _reasoning_fields = ("reasoning_content",)

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
        super().__init__(
            model=model, base_url=base_url, api_key=api_key, config=config,
            system_prompt=system_prompt, timeout=timeout, rate_limit=rate_limit,
        )

    def _server_extra_body(self, cfg: GenerationConfig) -> dict:
        """Add the llama.cpp sampling parameters the OpenAI schema lacks."""
        return {"top_k": cfg.top_k, "repeat_penalty": cfg.repeat_penalty}


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def llamacpp(model: str = DEFAULT_LLAMACPP_MODEL, system: Optional[str] = None, **kwargs) -> MakiLlamaCpp:
    """Generic llama.cpp wrapper — defaults to a single-model server on :8080."""
    return MakiLlamaCpp(model=model, system_prompt=system, **kwargs)
