"""
Protocols for Maki agent mixins.

These Protocols define the attributes that a host class must provide when
using PluginHandler or ReasoningEngine as mixins. If a class uses either
mixin without satisfying its protocol, an error is raised at initialisation
time rather than silently failing at the first method call.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class PluginHostProtocol(Protocol):
    """
    Contract required by :class:`PluginHandler`.

    The host class must expose these attributes *before* calling
    ``_init_plugins()``:

    * ``name``  – a non-empty string identifier for the agent.
    * ``maki``  – a Maki LLM backend instance (used when loading plugins).
    * ``plugins`` – populated by ``_init_plugins()`` itself; listed here
      for documentation completeness.
    """

    name: str
    maki: Any
    plugins: Dict


@runtime_checkable
class ReasoningHostProtocol(Protocol):
    """
    Contract required by :class:`ReasoningEngine`.

    The host class must expose these attributes *before* calling
    ``_init_reasoning()``:

    * ``maki``              – a Maki LLM backend instance.
    * ``reasoning_history`` – a :class:`collections.deque` that stores
      reasoning steps.
    """

    maki: Any
    reasoning_history: deque
