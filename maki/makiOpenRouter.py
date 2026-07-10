"""
MakiOpenRouter — OpenRouter backend for the Maki framework.

OpenRouter (https://openrouter.ai) exposes an OpenAI-compatible chat
completions API in front of dozens of vendors/models behind one API key.
This backend reuses the openai Python SDK, pointed at OpenRouter's base
URL, so model selection is just a vendor-namespaced model string (e.g.
"openai/gpt-4o-mini", "anthropic/claude-sonnet-4-6",
"meta-llama/llama-3.3-70b-instruct").

Requirements:
    pip install openai
    export OPENROUTER_API_KEY=sk-or-...
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from . import makiOpenAI
from .config import (
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_APP_NAME_ENV,
    OPENROUTER_SITE_URL_ENV,
)
from .objects import BackendType, GenerationConfig, LLMResponse, RateLimiter

log = logging.getLogger(__name__)


class MakiOpenRouter(makiOpenAI.MakiOpenAI):
    """
    OpenRouter backend (OpenAI-compatible chat completions API).

    Subclasses MakiOpenAI: chat(), stream(), async_chat(), session(),
    and the native tool-calling methods (to_tool_schemas(),
    chat_with_tools(), append_tool_results()) are all inherited
    unchanged. Only client construction (base_url, optional attribution
    headers), the reasoning-model family check, and response BackendType
    stamping are overridden.

    Usage
    -----
        llm = MakiOpenRouter(model="anthropic/claude-sonnet-4-6")
        response = llm.chat("What is the capital of France?")
        print(response.content)

        for chunk in llm.stream("Tell me a joke"):
            print(chunk, end="", flush=True)

    Note: not every OpenRouter-routed model supports native tool calling.
    If the underlying model doesn't, OpenRouter returns an API error,
    which surfaces here as MakiAPIError — no special-casing is needed.
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENROUTER_MODEL,
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        rate_limit: Optional[int] = None,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
    ) -> None:
        if makiOpenAI._openai_sdk is None:
            raise ImportError("openai package is required: pip install openai")

        resolved_key = api_key or os.environ.get(OPENROUTER_API_KEY_ENV)
        if not resolved_key:
            raise ValueError(
                f"OpenRouter API key not found. Pass api_key= or set {OPENROUTER_API_KEY_ENV}."
            )

        self.model = model
        self.config = config or GenerationConfig()
        self.temperature = self.config.temperature
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.base_url = base_url
        self._rate_limiter = RateLimiter(rate_limit) if rate_limit is not None else None

        # OpenRouter's recommended (optional) attribution headers — used for
        # its public model rankings. Omitted entirely unless configured, so
        # default usage needs no extra setup.
        effective_site_url = site_url or os.environ.get(OPENROUTER_SITE_URL_ENV)
        effective_app_name = app_name or os.environ.get(OPENROUTER_APP_NAME_ENV)
        default_headers: dict = {}
        if effective_site_url:
            default_headers["HTTP-Referer"] = effective_site_url
        if effective_app_name:
            default_headers["X-Title"] = effective_app_name

        client_kwargs: dict = {"api_key": resolved_key, "base_url": base_url, "timeout": timeout}
        if default_headers:
            client_kwargs["default_headers"] = default_headers

        self._client = makiOpenAI._openai_sdk.OpenAI(**client_kwargs)
        self._async_client = makiOpenAI._openai_sdk.AsyncOpenAI(**client_kwargs)
        log.info("MakiOpenRouter · model=%s · base_url=%s", model, base_url)

    @property
    def _model_family(self) -> str:
        """Return 'reasoning' for o1/o3/o4 models, 'chat' for everything else.

        OpenRouter model ids are vendor-namespaced (e.g. "openai/o1-preview"),
        so the bare model name is checked after stripping the "vendor/" prefix.
        """
        bare = self.model.rsplit("/", 1)[-1]
        if bare.startswith(("o1", "o3", "o4")):
            return "reasoning"
        return "chat"

    def _parse_response(self, response: object, elapsed: float) -> LLMResponse:
        usage = response.usage  # type: ignore[attr-defined]
        return LLMResponse(
            content=response.choices[0].message.content or "",  # type: ignore[attr-defined]
            model=response.model,  # type: ignore[attr-defined]
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            elapsed_seconds=elapsed,
            done=True,
            backend=BackendType.OPENROUTER,
        )

    def __repr__(self) -> str:
        return f"MakiOpenRouter(model={self.model!r}, base_url={self.base_url!r})"


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def openrouter(model: str = DEFAULT_OPENROUTER_MODEL, system: Optional[str] = None, **kwargs) -> MakiOpenRouter:
    """Generic OpenRouter wrapper — pass any OpenRouter-routed model id."""
    return MakiOpenRouter(model=model, system_prompt=system, **kwargs)


def openrouter_gpt4o(system: Optional[str] = None, **kwargs) -> MakiOpenRouter:
    """Pre-configured wrapper for GPT-4o via OpenRouter."""
    return MakiOpenRouter(model="openai/gpt-4o", system_prompt=system, **kwargs)


def openrouter_claude_sonnet(system: Optional[str] = None, **kwargs) -> MakiOpenRouter:
    """Pre-configured wrapper for Claude Sonnet via OpenRouter."""
    return MakiOpenRouter(model="anthropic/claude-sonnet-4-6", system_prompt=system, **kwargs)


def openrouter_llama(system: Optional[str] = None, **kwargs) -> MakiOpenRouter:
    """Pre-configured wrapper for Llama 3.3 70B via OpenRouter."""
    return MakiOpenRouter(model="meta-llama/llama-3.3-70b-instruct", system_prompt=system, **kwargs)
