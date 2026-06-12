"""
Agent Manager for the Maki Framework.

Manages a collection of agents and facilitates their coordination,
collaboration, and workflow execution.
"""

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..backend import LLMBackend
from .agent import Agent
from .workflow import WorkflowTask, WorkflowState, TaskStatus

logger = logging.getLogger(__name__)


_WORKFLOW_MAX_WORKERS = 4


class AgentManager:
    """Manages a collection of agents and facilitates their coordination."""

    def __init__(self, maki_instance: LLMBackend):
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
                  maki_instance: LLMBackend = None, use_streaming: bool = False) -> Agent:
        """
        Add a new agent to the manager.

        Args:
            name: Unique identifier for the agent
            role: The role of the agent
            instructions: Specific instructions for this agent
            maki_instance: Optional per-agent Maki instance (different model/temperature).
                           Falls back to the manager's default instance.
            use_streaming: If True, execute_task uses streaming internally so the timeout
                           applies per-chunk rather than to the full response. Useful for
                           agents with large prompts or long expected outputs.

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

        if not (isinstance(maki_to_use, LLMBackend) or hasattr(maki_to_use, 'request')):
            raise TypeError("maki_instance must implement the LLMBackend interface")

        agent = Agent(name, maki_to_use, role, instructions, use_streaming=use_streaming)
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
            The result of the task execution as a string.

        Raises:
            ValueError: If agent_name or task are invalid, or the agent is not registered.
            RuntimeError: If the agent raises an exception during task execution.
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
            raise RuntimeError(f"Failed to assign task '{task}' to agent '{agent_name}': {str(e)}") from e

    def coordinate_agents(self, tasks: List[Dict], coordination_prompt: str = "") -> Dict[str, Any]:
        """
        Coordinate multiple agents to complete a set of tasks sequentially.

        Args:
            tasks: List of task dicts with 'agent', 'task', and optional 'context' keys
            coordination_prompt: If provided, a synthesis LLM call merges all results.
                When synthesis fails, 'final_synthesis' is set to None so that individual
                task results are still accessible (graceful degradation).

        Returns:
            A dict mapping unique task keys (``"task_{i}_{agent}"`` format) to agent
            result strings.  When *coordination_prompt* is supplied, also contains a
            ``'final_synthesis'`` key whose value is the synthesised string, or ``None``
            if synthesis failed.

        Raises:
            RuntimeError: Propagated from :meth:`assign_task` when an individual agent
                task raises an exception.  Synthesis failures do NOT raise; they set
                ``'final_synthesis'`` to ``None`` so callers can still access the
                individual results.
        """
        results: Dict[str, Any] = {}
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
                results['final_synthesis'] = self.maki.chat(synthesis_prompt).content if hasattr(self.maki, 'chat') else self.maki.request(synthesis_prompt).content
            except Exception as e:
                logger.error(f"Failed to create synthesis: {str(e)}", exc_info=True)
                results['final_synthesis'] = None

        return results

    def collaborative_task(self, task: str, agents: List[str],
                           context: Optional[Dict] = None, strict: bool = False) -> str:
        """
        Have multiple agents collaborate on a single task.

        Unlike :meth:`coordinate_agents`, the synthesised response *is* the
        output of this method.  If synthesis fails after agents have produced
        results, a ``RuntimeError`` is raised rather than falling back to raw
        data, because there is no meaningful partial result to return.

        Args:
            task: The main task for collaboration
            agents: List of agent names to participate
            context: Additional context for the task
            strict: If True, raise RuntimeError when any agent fails instead of
                    proceeding with partial results

        Returns:
            A synthesised string response produced by the LLM from all successful
            agents' outputs.

        Raises:
            RuntimeError: If all agents fail; if strict=True and any agent fails;
                or if the final synthesis LLM call fails.
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
            return self.maki.chat(synthesis_prompt).content if hasattr(self.maki, 'chat') else self.maki.request(synthesis_prompt).content
        except Exception as e:
            logger.error(f"Failed to create final synthesis: {str(e)}", exc_info=True)
            raise RuntimeError(
                f"collaborative_task: synthesis failed after all agents succeeded: {str(e)}"
            ) from e

    def run_workflow(
        self,
        workflow: List,
        workflow_id: Optional[str] = None,
        state_store: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a workflow.

        Accepts either a list of plain dicts or a list of WorkflowTask objects.

        - **Dict workflow**: same behaviour as before; steps with
          ``parallelizable: True`` that appear consecutively are batched into a
          thread pool.  state_store is ignored for dict workflows.
        - **WorkflowTask workflow**: dependencies are enforced via topological sort,
          WorkflowTask fields (status, result, attempts, execution_time) are updated
          during execution, conditions are evaluated, and retries are driven by each
          task's own max_retries / retry_delay settings.

        Args:
            workflow:     List of step dicts or WorkflowTask objects.
            workflow_id:  Stable identifier used for checkpointing and resume.
                          Required when state_store is provided; ignored otherwise.
            state_store:  Optional StateStore instance.  When supplied, state is
                          persisted after every task so the workflow can be resumed
                          after a restart.  On resume, already-COMPLETED tasks are
                          skipped; FAILED / PENDING tasks are re-executed.

        Returns:
            Dict mapping step/task names to result dicts with keys
            'agent', 'task', 'result'
        """
        if not workflow:
            return {}
        if isinstance(workflow[0], WorkflowTask):
            return self._run_workflow_tasks(
                workflow, workflow_id=workflow_id, state_store=state_store
            )
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
                with ThreadPoolExecutor(max_workers=_WORKFLOW_MAX_WORKERS) as pool:
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

    def _run_workflow_tasks(
        self,
        tasks: List[WorkflowTask],
        workflow_id: Optional[str] = None,
        state_store: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a list of WorkflowTask objects.

        - Dependencies are enforced via topological sort.
        - WorkflowTask status / result / attempts / execution_time fields are updated.
        - Conditions are evaluated at runtime; tasks that fail their conditions are skipped.
        - Retries use each task's own max_retries and retry_delay.
        - Consecutive parallelizable tasks are batched into a thread pool.
        - When state_store + workflow_id are supplied, state is persisted after every
          task and COMPLETED tasks from a prior run are skipped on resume.
        """
        wf_id = workflow_id or f"workflow_{int(time.time())}"

        # ------------------------------------------------------------------
        # Checkpoint resume: load prior state and pre-populate results
        # ------------------------------------------------------------------
        checkpoint = (
            state_store.load_workflow(wf_id)
            if (state_store is not None and workflow_id)
            else None
        )
        state = checkpoint if checkpoint is not None else WorkflowState(wf_id)
        results: Dict[str, Any] = {}

        if checkpoint is not None:
            task_map = {t.name: t for t in tasks}
            for task_name, task_data in checkpoint.tasks.items():
                if task_data.get("status") == TaskStatus.COMPLETED:
                    wt = task_map.get(task_name)
                    if wt is not None:
                        wt.status = TaskStatus.COMPLETED
                        wt.result = task_data.get("result")
                        results[task_name] = {
                            "agent": wt.agent,
                            "task": wt.task,
                            "result": task_data.get("result"),
                            "data": task_data.get("data"),
                        }
            if results:
                logger.info(
                    "Resuming workflow '%s': %d task(s) already completed, skipping.",
                    wf_id, len(results),
                )

        if state_store is not None:
            state_store.save_workflow(state)

        # ------------------------------------------------------------------
        # Topological sort + batch construction (unchanged from before)
        # ------------------------------------------------------------------
        ordered = self._topological_sort(tasks)

        # Batch consecutive parallelizable tasks, but never place a task in the
        # same batch as one of its dependencies. That would allow a dependent
        # task to start before its prerequisite has produced a result.
        batches: List[List[WorkflowTask]] = []
        for wt in ordered:
            current_batch = batches[-1] if batches else None
            current_batch_names = {task.name for task in current_batch} if current_batch else set()
            can_share_batch = (
                wt.parallelizable
                and current_batch is not None
                and current_batch[0].parallelizable
                and not any(dep in current_batch_names for dep in wt.dependencies)
            )
            if can_share_batch:
                batches[-1].append(wt)
            else:
                batches.append([wt])

        # Lock used when persisting state from the main thread after parallel tasks.
        _save_lock = threading.Lock()

        def _persist():
            if state_store is not None:
                with _save_lock:
                    state_store.save_workflow(state)

        # ------------------------------------------------------------------
        # Execute batches
        # ------------------------------------------------------------------
        for batch in batches:
            if batch[0].parallelizable and len(batch) > 1:
                # Skip tasks already completed in a prior run.
                pending = [wt for wt in batch if wt.name not in results]
                if not pending:
                    continue

                # Snapshot results before submitting so all tasks in this batch
                # evaluate conditions against the same consistent state, not a
                # partially-updated dict from concurrently finishing siblings.
                results_snapshot = dict(results)

                def _run_wf_task(wt: WorkflowTask, _snap=results_snapshot):
                    return self._execute_workflow_task(wt, _snap, state)

                with ThreadPoolExecutor(max_workers=_WORKFLOW_MAX_WORKERS) as pool:
                    futures = {pool.submit(_run_wf_task, wt): wt for wt in pending}
                    for future in as_completed(futures):
                        name, value = future.result()
                        if value is not None:
                            results[name] = value
                        _persist()
            else:
                wt = batch[0]
                if wt.name in results:
                    logger.debug("Skipping task '%s' (resumed from checkpoint).", wt.name)
                    continue
                name, value = self._execute_workflow_task(wt, results, state)
                if value is not None:
                    results[name] = value
                _persist()

        any_skipped = any(
            t.get("status") == TaskStatus.SKIPPED
            for t in state.tasks.values()
        )
        state.status = "completed_with_skips" if any_skipped else "completed"
        state.end_time = time.time()
        if state_store is not None:
            state_store.save_workflow(state)

        return results

    def _execute_workflow_task(self, wt: WorkflowTask, results: Dict,
                               state: WorkflowState):
        """Run a single WorkflowTask, updating its fields and the WorkflowState.

        The agent receives a ``context`` dict built from the structured ``data``
        payloads of all completed dependencies (keyed by dependency name).  This
        lets downstream agents consume typed data directly rather than parsing
        free-text strings.  The text ``result`` of each dependency is also
        included under ``<dep_name>__result`` for backward compatibility.
        """
        # Propagate skip: a SKIPPED dependency skips this task transitively.
        skipped_dep = next(
            (dep for dep in wt.dependencies if results.get(dep, {}).get("skipped")),
            None,
        )
        if skipped_dep is not None or not wt.should_execute(results):
            reason = (
                f"dependency '{skipped_dep}' was skipped"
                if skipped_dep is not None
                else "conditions not met"
            )
            logger.info(f"WorkflowTask '{wt.name}' skipped ({reason})")
            wt.status = TaskStatus.SKIPPED
            state.update_task_status(wt.name, TaskStatus.SKIPPED, f"Skipped: {reason}")
            return wt.name, {
                "agent": wt.agent, "task": wt.task, "result": None, "data": None, "skipped": True
            }

        agent = self.get_agent(wt.agent)
        if not agent:
            err = f"Agent '{wt.agent}' not found"
            wt.status = TaskStatus.FAILED
            state.update_task_status(wt.name, TaskStatus.FAILED, err)
            state.add_error(wt.name, err)
            logger.error(f"WorkflowTask '{wt.name}' failed: {err}")
            return wt.name, None

        # Build context from completed (non-skipped) dependency outputs.
        context: dict = {}
        for dep in wt.dependencies:
            dep_result = results.get(dep)
            if dep_result and not dep_result.get("skipped"):
                if dep_result.get("data") is not None:
                    context[dep] = dep_result["data"]
                if dep_result.get("result") is not None:
                    context[f"{dep}__result"] = dep_result["result"]

        wt.status = TaskStatus.IN_PROGRESS
        wt.attempts += 1
        start = time.time()

        try:
            result = agent.execute_task_with_retry(
                wt.task,
                context=context or None,
                max_retries=wt.max_retries,
                retry_delay=wt.retry_delay
            )
            wt.result = result
            wt.status = TaskStatus.COMPLETED
            wt.execution_time = time.time() - start
            state.update_task_status(
                wt.name, TaskStatus.COMPLETED, result, wt.execution_time,
                data=wt.data,
            )
            return wt.name, {
                'agent': wt.agent,
                'task': wt.task,
                'result': result,
                'data': wt.data,
            }

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
            ValueError: If a circular dependency is detected or a dependency is missing
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
            if task is None:
                raise ValueError(f"Workflow dependency '{name}' is not defined")
            for dep in task.dependencies:
                visit(dep)
            visiting.discard(name)
            visited.add(name)
            order.append(task)

        for task in tasks:
            visit(task.name)

        return order
