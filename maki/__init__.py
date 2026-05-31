"""
Public package interface for Maki.
"""

from importlib import import_module

from . import config

__all__ = ["LLMBackend", "Connector", "Utils", "Agent", "AgentManager", "MakiLLama",
           "HFBackend", "MakiOpenAI", "MakiAnthropic", "LLMResponse", "GenerationConfig",
           "Message", "RateLimiter", "BackendType", "config"]

_LAZY_EXPORTS = {
    "LLMBackend":     (".backend",       "LLMBackend"),
    "Connector":      (".connector",     "Connector"),
    "Utils":          (".utils",         "Utils"),
    "Agent":          (".agents",        "Agent"),
    "AgentManager":   (".agents",        "AgentManager"),
    "MakiLLama":      (".makiLLama",     "MakiLLama"),
    "HFBackend":      (".makiHG",        "HFBackend"),
    "MakiOpenAI":     (".makiOpenAI",    "MakiOpenAI"),
    "MakiAnthropic":  (".makiAnthropic", "MakiAnthropic"),
    "LLMResponse":    (".objects",       "LLMResponse"),
    "GenerationConfig": (".objects",     "GenerationConfig"),
    "Message":        (".objects",       "Message"),
    "RateLimiter":    (".objects",       "RateLimiter"),
    "BackendType":    (".objects",       "BackendType"),
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
