"""
Agent module for Maki Framework

This module provides the infrastructure for creating and managing multi-agent systems
using the Maki framework. Agents can work together to solve complex tasks through
coordination, delegation, and collaboration.
"""

from .agent import Agent
from .agent_manager import AgentManager
from .workflow import WorkflowTask, WorkflowState, TaskStatus, GateResult, TaskOutput
from .approvals import (
    ApprovalSpec, ApprovalStore, ApprovalRecord, ApprovalDecision,
    LocalApprovalStore, InMemoryApprovalStore,
)
from .verification import (
    Verdict, VerifyResult, VerifySpec, AgentVerifier, agent_verifier,
)
from .loop import (
    Decision, LoopEvent, LoopResult, LoopSpec, Requirement, StopReason, approve_requirements,
    draft_requirements, run_loop,
)
from .protocols import PluginHostProtocol, ReasoningHostProtocol

__all__ = [
    'Agent', 'AgentManager', 'WorkflowTask', 'TaskStatus', 'WorkflowState',
    'PluginHostProtocol', 'ReasoningHostProtocol',
    'GateResult', 'TaskOutput', 'ApprovalSpec', 'ApprovalStore', 'ApprovalRecord', 'ApprovalDecision',
    'LocalApprovalStore', 'InMemoryApprovalStore',
    'Verdict', 'VerifyResult', 'VerifySpec', 'AgentVerifier', 'agent_verifier',
    'Decision', 'LoopEvent', 'LoopResult', 'LoopSpec', 'Requirement', 'StopReason',
    'approve_requirements', 'draft_requirements', 'run_loop',
]
