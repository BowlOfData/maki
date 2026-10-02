"""
Workflow management for Maki Framework

This module provides classes and utilities for creating and managing workflows
with multiple tasks that can be executed in various strategies.
"""

from typing import Dict, List, Any, Optional, Callable
import time
import logging
from dataclasses import dataclass
from enum import Enum

from .approvals import ApprovalSpec

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Enumeration for task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"
    AWAITING_APPROVAL = "awaiting_approval"


@dataclass(frozen=True)
class GateResult:
    """Outcome of a gate (condition) with a human-readable reason.

    Conditions may return a plain bool or a GateResult; ``bool(result)`` is ``ok``,
    so existing truthiness checks keep working.
    """
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass(frozen=True)
class TaskOutput:
    """Return value for agents that produce typed data alongside text.

    An agent's ``execute_task_with_retry`` may return ``TaskOutput`` instead of ``str``:
    ``text`` becomes the task ``result`` and ``data`` becomes ``WorkflowTask.data``,
    which dependents receive in their context under the dependency name.
    """
    text: str
    data: Optional[Dict[str, Any]] = None


class WorkflowTask:
    """Represents a task within a workflow with enhanced capabilities"""

    def __init__(self, name: str, agent: str, task: str, dependencies: Optional[List[str]] = None,
                 conditions: Optional[List[Callable]] = None, max_retries: int = 3,
                 retry_delay: float = 1.0, parallelizable: bool = False,
                 approval: Optional[ApprovalSpec] = None):
        """
        Initialize a workflow task

        Args:
            name: Unique identifier for the task
            agent: Agent name that will execute this task
            task: The task description
            dependencies: List of task names that must complete before this task
            conditions: List of functions that must return True for task to execute
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            parallelizable: Whether this task can be executed in parallel
            approval: If set, the task waits for human approval (see maki.agents.approvals)
                before running. Unlike a failed condition, this does not skip the task:
                it parks as AWAITING_APPROVAL until approved or rejected.
        """
        self.name = name
        self.agent = agent
        self.task = task
        self.dependencies = dependencies or []
        self.conditions = conditions or []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.parallelizable = parallelizable
        self.approval = approval
        self.gate_reason: Optional[str] = None
        self.status = TaskStatus.PENDING
        self.result = None
        # Structured output from this task.  Populated by the agent (or post-processor)
        # after execution.  Dependents receive it via the ``context`` dict under their
        # dependency name, so downstream agents can consume typed data directly rather
        # than parsing free-text strings.
        self.data: Optional[Dict[str, Any]] = None
        self.timestamp = None
        self.attempts = 0
        self.execution_time = 0.0
        self.resources_used = {}

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict.

        Note: conditions (Python callables) are intentionally excluded.
        Remote workflows must evaluate conditions at the orchestrator.
        """
        return {
            'name': self.name,
            'agent': self.agent,
            'task': self.task,
            'dependencies': list(self.dependencies),
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'parallelizable': self.parallelizable,
            'requires_approval': self.approval is not None,
            'gate_reason': self.gate_reason,
            'status': self.status.value,
            'result': self.result,
            'data': self.data,
            'timestamp': self.timestamp,
            'attempts': self.attempts,
            'execution_time': self.execution_time,
            'resources_used': dict(self.resources_used),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'WorkflowTask':
        """Reconstruct a WorkflowTask from a dict produced by to_dict().

        conditions are not restored — attach them after construction if needed.
        """
        wt = cls(
            name=data['name'],
            agent=data['agent'],
            task=data['task'],
            dependencies=data.get('dependencies', []),
            max_retries=data.get('max_retries', 3),
            retry_delay=data.get('retry_delay', 1.0),
            parallelizable=data.get('parallelizable', False),
        )
        wt.status = TaskStatus(data.get('status', 'pending'))
        wt.result = data.get('result')
        wt.data = data.get('data')
        wt.timestamp = data.get('timestamp')
        wt.attempts = data.get('attempts', 0)
        wt.execution_time = data.get('execution_time', 0.0)
        wt.resources_used = data.get('resources_used', {})
        return wt

    def evaluate_conditions(self, context: Optional[Dict] = None) -> GateResult:
        """Evaluate all conditions; return the first failure (with reason) or ok."""
        for condition in self.conditions:
            cname = getattr(condition, '__name__', repr(condition))
            try:
                outcome = condition(context)
            except Exception as e:
                logger.error("Task '%s': condition '%s' raised an exception: %s",
                             self.name, cname, e, exc_info=True)
                return GateResult(False, f"condition '{cname}' raised {type(e).__name__}: {e}")
            if not outcome:
                detail = getattr(outcome, 'reason', '') or "returned False"
                logger.debug("Task '%s' skipped: condition '%s' %s", self.name, cname, detail)
                return GateResult(False, f"condition '{cname}' {detail}")
        return GateResult(True)

    def should_execute(self, context: Optional[Dict] = None) -> bool:
        """Check if task should execute based on conditions"""
        result = self.evaluate_conditions(context)
        self.gate_reason = None if result.ok else result.reason
        return result.ok


class WorkflowState:
    """Tracks the state of a workflow execution"""

    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.tasks = {}
        self.status = "running"
        self.start_time = time.time()
        self.end_time = None
        self.metrics = {}
        self.error_log = []

    def update_task_status(self, task_name: str, status: TaskStatus, result: Any = None,
                          execution_time: float = 0.0, resources_used: Dict = None,
                          data: Optional[Dict[str, Any]] = None,
                          reason: Optional[str] = None):
        """Update task status and metrics, including optional structured data payload."""
        self.tasks[task_name] = {
            'status': status,
            'result': result,
            'data': data,
            'reason': reason,
            'timestamp': time.time(),
            'execution_time': execution_time,
            'resources_used': resources_used or {}
        }

    def get_workflow_progress(self) -> Dict:
        """Get overall workflow progress"""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks.values() if t['status'] == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t['status'] == TaskStatus.FAILED)
        return {
            'total_tasks': total,
            'completed_tasks': completed,
            'failed_tasks': failed,
            'progress_percentage': (completed/total*100) if total > 0 else 0,
            'status': self.status
        }

    def add_error(self, task_name: str, error: str):
        """Add an error to the workflow error log"""
        self.error_log.append({
            'task': task_name,
            'error': error,
            'timestamp': time.time()
        })

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        serialized_tasks = {}
        for name, task_data in self.tasks.items():
            entry = dict(task_data)
            status = entry.get('status')
            if isinstance(status, TaskStatus):
                entry['status'] = status.value
            serialized_tasks[name] = entry
        return {
            'workflow_id': self.workflow_id,
            'tasks': serialized_tasks,
            'status': self.status,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'metrics': dict(self.metrics),
            'error_log': list(self.error_log),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'WorkflowState':
        """Reconstruct a WorkflowState from a dict produced by to_dict()."""
        state = cls(workflow_id=data['workflow_id'])
        state.status = data.get('status', 'running')
        state.start_time = data.get('start_time', time.time())
        state.end_time = data.get('end_time')
        state.metrics = data.get('metrics', {})
        state.error_log = list(data.get('error_log', []))
        for name, task_data in data.get('tasks', {}).items():
            entry = dict(task_data)
            status_val = entry.get('status', 'pending')
            entry['status'] = TaskStatus(status_val) if isinstance(status_val, str) else status_val
            state.tasks[name] = entry
        return state