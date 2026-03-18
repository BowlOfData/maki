"""
Agent Manager for the Maki Framework.

Manages a collection of agents and facilitates their coordination,
collaboration, and workflow execution.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..maki import Maki
from .agent import Agent
from .workflow import WorkflowTask, WorkflowState, TaskStatus

logger = logging.getLogger(__name__)


class AgentManager:
    """Manages a collection of agents and facilitates their coordination."""

    def __init__(self, maki_instance: Maki):
        """
        Initialize the agent manager.

        Args:
            maki_instance: Default Maki instance used for synthesis and agents that
                           don't have their own instance
        """
        self.maki = maki_instance
        self.agents: Dict[str, Agent] = {}
        self.task_queue: List[Dict] = []

        logger.info("AgentManager initialized")

    def add_agent(self, name: str, role: str = "", instructions: str = "",
                  maki_instance: Maki = None) -> Agent:
        """
        Add a new agent to the manager.

        Args:
            name: Unique identifier for the agent
            role: The role of the agent
            instructions: Specific instructions for this agent
            maki_instance: Optional per-agent Maki instance (different model/temperature).
                           Falls back to the manager's default instance.

        Returns:
            The created Agent instance
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Agent name must be a non-empty string")

        if not isinstance(role, str):
            raise ValueError("Role must be a string")

        if not isinstance(instructions, str):
            raise ValueError("Instructions must be a string")

        maki_to_use = maki_instance if maki_instance is not None else self.maki

        if not isinstance(maki_to_use, Maki):
            raise TypeError("maki_instance must be a Maki instance")

        agent = Agent(name, maki_to_use, role, instructions)
        self.agents[name] = agent
        logger.info(f"Added agent '{name}' with role '{role}'")
        return agent

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name, or None if not found."""
        return self.agents.get(name)

    def remove_agent(self, name: str):
        """Remove an agent from the manager."""
        if name in self.agents:
            del self.agents[name]

    def list_agents(self) -> List[str]:
        """Return a list of all registered agent names."""
        return list(self.agents.keys())

    def assign_task(self, agent_name: str, task: str, context: Optional[Dict] = None) -> str:
        """
        Assign a task to a specific agent.

        Args:
            agent_name: The name of the agent to assign the task to
            task: The task to assign
            context: Additional context for the task

        Returns:
            The result of the task execution
        """
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("Agent name must be a non-empty string")

        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be a non-empty string")

        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")

        try:
            return agent.execute_task(task, context)
        except Exception as e:
            raise Exception(f"Failed to assign task '{task}' to agent '{agent_name}': {str(e)}")

    def coordinate_agents(self, tasks: List[Dict], coordination_prompt: str = "") -> Dict[str, str]:
        """
        Coordinate multiple agents to complete a set of tasks sequentially.

        Args:
            tasks: List of task dicts with 'agent', 'task', and optional 'context' keys
            coordination_prompt: If provided, a synthesis LLM call merges all results

        Returns:
            A dict mapping unique task keys to results, plus 'final_synthesis' if requested
        """
        results = {}
        task_metadata = {}

        for i, task_dict in enumerate(tasks):
            agent_name = task_dict.get('agent')
            task = task_dict.get('task')
            context = task_dict.get('context')

            if not agent_name or not task:
                continue

            result = self.assign_task(agent_name, task, context)

            key = f"task_{i}_{agent_name}"
            results[key] = result
            task_metadata[key] = {
                'agent': agent_name,
                'original_task': task,
                'context': context
            }

        if coordination_prompt:
            synthesis_prompt = f"""
            {coordination_prompt}

            Here are the individual results from the agents:
            {json.dumps(results, indent=2)}

            Task metadata:
            {json.dumps(task_metadata, indent=2)}

            Please synthesize these results into a comprehensive response that
            properly attributes each result to its original task and agent.
            """
            try:
                results['final_synthesis'] = self.maki.request(synthesis_prompt)
            except Exception as e:
                logger.error(f"Failed to create synthesis: {str(e)}")

        return results

    def collaborative_task(self, task: str, agents: List[str],
                           context: Optional[Dict] = None, strict: bool = False) -> str:
        """
        Have multiple agents collaborate on a single task.

        Args:
            task: The main task for collaboration
            agents: List of agent names to participate
            context: Additional context for the task
            strict: If True, raise RuntimeError when any agent fails instead of
                    proceeding with partial results

        Returns:
            A synthesised response from all successful agents

        Raises:
            RuntimeError: If all agents fail, or if strict=True and any agent fails
        """
        agent_results = {}
        agent_errors = {}

        for agent_name in agents:
            agent = self.get_agent(agent_name)
            if agent is None:
                msg = f"Agent '{agent_name}' not found"
                logger.error(msg)
                agent_errors[agent_name] = msg
                continue
            try:
                agent_results[agent_name] = agent.execute_task(task, context)
            except Exception as e:
                logger.error(f"Failed to get result from agent {agent_name}: {str(e)}")
                agent_errors[agent_name] = str(e)

        if agent_errors:
            logger.warning(
                f"collaborative_task: {len(agent_errors)} agent(s) failed: "
                + ", ".join(f"{n}: {e}" for n, e in agent_errors.items())
            )
            if strict:
                raise RuntimeError(
                    f"collaborative_task (strict): {len(agent_errors)} agent(s) failed: "
                    + "; ".join(f"{n}: {e}" for n, e in agent_errors.items())
                )

        if not agent_results:
            raise RuntimeError(
                f"All agents failed for collaborative task '{task}': "
                + "; ".join(f"{n}: {e}" for n, e in agent_errors.items())
            )

        synthesis_prompt = f"""
        You are synthesizing results from multiple agents working on the same task.

        Original task: {task}

        Individual agent responses:
        {json.dumps(agent_results, indent=2)}

        Please provide a comprehensive, coordinated response that synthesizes
        the insights from all agents into a final answer.
        """

        try:
            return self.maki.request(synthesis_prompt)
        except Exception as e:
            logger.error(f"Failed to create final synthesis: {str(e)}")
            return json.dumps(agent_results, indent=2)

    def run_workflow(self, workflow: List) -> Dict[str, Any]:
        """
        Execute a workflow.

        Accepts either a list of plain dicts or a list of WorkflowTask objects.

        - **Dict workflow**: same behaviour as before; steps with
          ``parallelizable: True`` that appear consecutively are batched into a
          thread pool.
        - **WorkflowTask workflow**: dependencies are enforced via topological sort,
          WorkflowTask fields (status, result, attempts, execution_time) are updated
          during execution, conditions are evaluated, and retries are driven by each
          task's own max_retries / retry_delay settings.

        Args:
            workflow: List of step dicts or WorkflowTask objects

        Returns:
            Dict mapping step/task names to result dicts with keys
            'agent', 'task', 'result'
        """
        if not workflow:
            return {}
        if isinstance(workflow[0], WorkflowTask):
            return self._run_workflow_tasks(workflow)
        return self._run_workflow_dicts(workflow)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_workflow_dicts(self, workflow: List[Dict]) -> Dict[str, Any]:
        """Execute a list-of-dicts workflow (original behaviour)."""
        results: Dict[str, Any] = {}

        batches: List[List[Dict]] = []
        for step in workflow:
            if step.get('parallelizable') and batches and batches[-1][0].get('parallelizable'):
                batches[-1].append(step)
            else:
                batches.append([step])

        for batch in batches:
            if batch[0].get('parallelizable') and len(batch) > 1:
                def _run_step(step: Dict, idx: int):
                    step_name = step.get('name', f'step_{idx}')
                    agent_name = step.get('agent')
                    task = step.get('task')
                    context = step.get('context')
                    if not agent_name or not task:
                        return step_name, None
                    result = self.assign_task(agent_name, task, context)
                    return step_name, {'agent': agent_name, 'task': task, 'result': result}

                base_idx = len(results)
                with ThreadPoolExecutor() as pool:
                    futures = {
                        pool.submit(_run_step, step, base_idx + i): step
                        for i, step in enumerate(batch)
                    }
                    for future in as_completed(futures):
                        step_name, value = future.result()
                        if value is not None:
                            results[step_name] = value
            else:
                step = batch[0]
                step_name = step.get('name', f'step_{len(results)}')
                agent_name = step.get('agent')
                task = step.get('task')
                context = step.get('context')
                if agent_name and task:
                    result = self.assign_task(agent_name, task, context)
                    results[step_name] = {
                        'agent': agent_name,
                        'task': task,
                        'result': result
                    }

        return results

    def _run_workflow_tasks(self, tasks: List[WorkflowTask]) -> Dict[str, Any]:
        """
        Execute a list of WorkflowTask objects.

        - Dependencies are enforced via topological sort.
        - WorkflowTask status / result / attempts / execution_time fields are updated.
        - Conditions are evaluated at runtime; tasks that fail their conditions are skipped.
        - Retries use each task's own max_retries and retry_delay.
        - Consecutive parallelizable tasks are batched into a thread pool.
        """
        state = WorkflowState(f"workflow_{int(time.time())}")
        results: Dict[str, Any] = {}

        ordered = self._topological_sort(tasks)

        # Batch consecutive parallelizable tasks
        batches: List[List[WorkflowTask]] = []
        for wt in ordered:
            if wt.parallelizable and batches and batches[-1][0].parallelizable:
                batches[-1].append(wt)
            else:
                batches.append([wt])

        for batch in batches:
            if batch[0].parallelizable and len(batch) > 1:
                def _run_wf_task(wt: WorkflowTask):
                    return self._execute_workflow_task(wt, results, state)

                with ThreadPoolExecutor() as pool:
                    futures = {pool.submit(_run_wf_task, wt): wt for wt in batch}
                    for future in as_completed(futures):
                        name, value = future.result()
                        if value is not None:
                            results[name] = value
            else:
                wt = batch[0]
                name, value = self._execute_workflow_task(wt, results, state)
                if value is not None:
                    results[name] = value

        state.status = "completed"
        state.end_time = time.time()
        return results

    def _execute_workflow_task(self, wt: WorkflowTask, results: Dict,
                               state: WorkflowState):
        """Run a single WorkflowTask, updating its fields and the WorkflowState."""
        # Evaluate conditions; skip if any returns False
        if not wt.should_execute(results):
            logger.info(f"WorkflowTask '{wt.name}' skipped (conditions not met)")
            wt.status = TaskStatus.PENDING
            return wt.name, None

        agent = self.get_agent(wt.agent)
        if not agent:
            err = f"Agent '{wt.agent}' not found"
            wt.status = TaskStatus.FAILED
            state.update_task_status(wt.name, TaskStatus.FAILED, err)
            state.add_error(wt.name, err)
            logger.error(f"WorkflowTask '{wt.name}' failed: {err}")
            return wt.name, None

        wt.status = TaskStatus.IN_PROGRESS
        wt.attempts += 1
        start = time.time()

        try:
            result = agent.execute_task_with_retry(
                wt.task,
                max_retries=wt.max_retries,
                retry_delay=wt.retry_delay
            )
            wt.result = result
            wt.status = TaskStatus.COMPLETED
            wt.execution_time = time.time() - start
            state.update_task_status(
                wt.name, TaskStatus.COMPLETED, result, wt.execution_time
            )
            return wt.name, {'agent': wt.agent, 'task': wt.task, 'result': result}

        except Exception as e:
            wt.status = TaskStatus.FAILED
            wt.execution_time = time.time() - start
            state.update_task_status(
                wt.name, TaskStatus.FAILED, str(e), wt.execution_time
            )
            state.add_error(wt.name, str(e))
            logger.error(f"WorkflowTask '{wt.name}' failed: {str(e)}")
            return wt.name, None

    def _topological_sort(self, tasks: List[WorkflowTask]) -> List[WorkflowTask]:
        """
        Return tasks sorted so every dependency comes before the task that needs it.

        Raises:
            ValueError: If a circular dependency is detected
        """
        task_map = {t.name: t for t in tasks}
        visited: set = set()
        visiting: set = set()  # cycle detection
        order: List[WorkflowTask] = []

        def visit(name: str):
            if name in visiting:
                raise ValueError(f"Circular dependency detected involving task '{name}'")
            if name in visited:
                return
            visiting.add(name)
            task = task_map.get(name)
            if task:
                for dep in task.dependencies:
                    visit(dep)
            visiting.discard(name)
            visited.add(name)
            if task:
                order.append(task)

        for task in tasks:
            visit(task.name)

        return order
