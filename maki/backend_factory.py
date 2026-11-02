"""
Build a backend from a ``"provider:model"`` string.

    from maki import init_backend

    backend = init_backend("openai:gpt-4o")
    local = init_backend("ollama:gemma4:26b", temperature=0.2)   # only the first colon splits
    default_model = init_backend("anthropic")                     # provider alone: its default model

Backend modules are imported only for the provider asked for, so a missing optional extra
(``maki[openai]``, ...) only matters when that provider is used.
"""

import difflib
import importlib
from typing import Optional, Tuple

from .backend import LLMBackend
from .objects import GenerationConfig

# provider name -> (module in maki, class)
BACKENDS = {
    "ollama": ("makiLLama", "MakiLLama"),
    "openai": ("makiOpenAI", "MakiOpenAI"),
    "anthropic": ("makiAnthropic", "MakiAnthropic"),
    "openrouter": ("makiOpenRouter", "MakiOpenRouter"),
    "llamacpp": ("makiLlamaCpp", "MakiLlamaCpp"),
    "vllm": ("makiVllm", "MakiVllm"),
}


def split_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Split ``"provider:model"`` into ``(provider, model)``; ``model`` is None for a bare provider.

    Raises:
        ValueError: If *spec* is empty or its provider is not a known backend (so a bare
            Ollama tag such as ``"gemma4:26b"`` is reported, not misread).
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Backend spec must be a non-empty string like 'openai:gpt-4o'")
    provider, _, model = spec.strip().partition(":")
    provider = provider.lower()
    if provider not in BACKENDS:
        close = difflib.get_close_matches(provider, BACKENDS, n=1)
        # A bare Ollama tag ("llama3:8b") can look like a typo of a provider, so offer both readings.
        hint = (f"Did you mean '{close[0]}'? " if close else "") + \
            f"For an Ollama model write 'ollama:{spec.strip()}'."
        raise ValueError(
            f"Unknown backend '{provider}' in '{spec}'. {hint} "
            f"Use 'provider:model' with a provider from: {', '.join(BACKENDS)}"
        )
    return provider, model.strip() or None


def init_backend(spec: str, *, temperature: Optional[float] = None, **kwargs) -> LLMBackend:
    """Create the backend named by *spec* (``"provider:model"`` or just ``"provider"``).

    Args:
        spec: Provider and optional model, e.g. ``"openai:gpt-4o"``. Everything after the
            first colon is the model id, so ``"ollama:gemma4:26b"`` works.
        temperature: Shorthand for ``config=GenerationConfig(temperature=...)``; not allowed
            together with an explicit ``config``.
        **kwargs: Passed to the backend constructor (``base_url``, ``api_key``, ``config``, ...).

    Raises:
        ValueError: For a malformed spec, an unknown provider, or ``temperature`` with ``config``.
    """
    provider, model = split_spec(spec)
    if "model" in kwargs and model is not None:
        raise ValueError("Pass the model either in the spec or as model=, not both")
    if temperature is not None:
        if "config" in kwargs:
            raise ValueError("Pass either temperature= or config=, not both")
        kwargs["config"] = GenerationConfig(temperature=float(temperature))
    if model is not None:
        kwargs["model"] = model
    module_name, class_name = BACKENDS[provider]
    backend_class = getattr(importlib.import_module(f".{module_name}", __package__), class_name)
    return backend_class(**kwargs)
