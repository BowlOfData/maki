"""
Public package interface for Maki.
"""

from importlib import import_module



__all__ = ["LLMBackend", "Maki", "Connector", "Utils", "Agent", "AgentManager", "MakiLLama",
           "HFBackend", "LLMResponse", "GenerationConfig", "Message", "RateLimiter"]

_LAZY_EXPORTS = {
    "LLMBackend": (".backend", "LLMBackend"),
    "Maki": (".maki", "Maki"),
    "Connector": (".connector", "Connector"),
    "Utils": (".utils", "Utils"),
    "Agent": (".agents", "Agent"),
    "AgentManager": (".agents", "AgentManager"),
    "MakiLLama": (".makiLLama", "MakiLLama"),
    "HFBackend": (".makiHG", "HFBackend"),
    "LLMResponse": (".objects", "LLMResponse"),
    "GenerationConfig": (".objects", "GenerationConfig"),
    "Message": (".objects", "Message"),
    "RateLimiter": (".objects", "RateLimiter"),
}


def __getattr__(name):
    """Load public exports only when they are requested."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__():
    """Expose lazy exports to introspection tools."""
    return sorted(set(globals()) | set(__all__))
