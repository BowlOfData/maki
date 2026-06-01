"""Distributed agent infrastructure for Maki."""

from .proxy import AgentProxy
from .registry import DistributedAgentManager

__all__ = ["AgentProxy", "DistributedAgentManager"]
