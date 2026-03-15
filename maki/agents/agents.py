"""
Backward-compatibility re-exports.

The agent system has been split into:
- agent.py         — Core Agent class
- agent_manager.py — AgentManager class
- plugin_handler.py — Plugin loading and tool-call execution
- reasoning.py      — Step-by-step reasoning, self-correction, task decomposition

All public names are still importable from this module.
"""

from .agent import Agent
from .agent_manager import AgentManager

__all__ = ['Agent', 'AgentManager']
