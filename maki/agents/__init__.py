"""
Agent module for Maki Framework

This module provides the infrastructure for creating and managing multi-agent systems
using the Maki framework. Agents can work together to solve complex tasks through
coordination, delegation, and collaboration.
"""

from .agent import Agent
from .agent_manager import AgentManager
from .workflow import WorkflowTask, WorkflowState, TaskStatus

__all__ = ['Agent', 'AgentManager', 'WorkflowTask', 'TaskStatus', 'WorkflowState']
