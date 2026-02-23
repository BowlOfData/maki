"""
Agent module for Maki Framework

This module provides the infrastructure for creating and managing multi-agent systems
using the Maki framework. Agents can work together to solve complex tasks through
coordination, delegation, and collaboration.
"""

from .agents import Agent, AgentManager

__all__ = ['Agent', 'AgentManager']