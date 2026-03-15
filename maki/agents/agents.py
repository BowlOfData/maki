"""
Multi-Agent System for Maki Framework

This module provides the infrastructure for creating and managing multi-agent systems
using the Maki framework. Agents can work together to solve complex tasks through
coordination, delegation, and collaboration.
"""

from collections import deque
from typing import Dict, List, Any, Optional, Callable
from ..maki import Maki
from ..exceptions import MakiNetworkError, MakiTimeoutError, MakiAPIError
from .workflow import WorkflowTask, WorkflowState, TaskStatus
import json
import re
import time
import logging
import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = logging.getLogger(__name__)

# Errors that are worth retrying (transient network/timeout issues)
_RETRYABLE_ERRORS = (MakiNetworkError, MakiTimeoutError)


class Agent:
    """An individual agent that can perform tasks using the Maki framework"""

    def __init__(self, name: str, maki_instance: Maki, role: str = "", instructions: str = "",
                 stateful: bool = False):
        """
        Initialize an agent

        Args:
            name: Unique identifier for the agent
            maki_instance: Maki instance to use for LLM interactions
            role: The role of the agent (e.g., "researcher", "writer", "analyst")
            instructions: Specific instructions for this agent
            stateful: If True, prior task results are included in subsequent prompts

        Raises:
            ValueError: If name is not a valid string
            TypeError: If maki_instance is not a Maki instance
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Agent name must be a non-empty string")

        if not isinstance(role, str):
            raise ValueError("Role must be a string")

        if not isinstance(instructions, str):
            raise ValueError("Instructions must be a string")

        if not isinstance(maki_instance, Maki):
            raise TypeError("maki_instance must be a Maki instance")

        self.name = name.strip()
        self.maki = maki_instance
        self.role = role
        self.instructions = instructions
        self.stateful = stateful
        self.memory = {}

        # Maximum number of entries to keep in history; deque enforces this automatically
        self._max_history_entries = 1000
        self.reasoning_history: deque = deque(maxlen=self._max_history_entries)
        self.task_history: deque = deque(maxlen=self._max_history_entries)

        # Stateful multi-turn conversation memory (separate from task_history)
        self._conversation_history: List[Dict] = []

        # Plugin support
        self.plugins = {}

        logger.info(f"Agent '{self.name}' initialized with role '{self.role}'")

    def __repr__(self):
        return f"Agent(name='{self.name}', role='{self.role}')"

    def execute_task(self, task: str, context: Optional[Dict] = None, use_plugins: bool = False) -> str:
        """
        Execute a task using this agent.

        Args:
            task: The task to perform
            context: Additional context for the task
            use_plugins: When True and plugins are loaded, their descriptions are included
                in the prompt and the LLM may issue TOOL: calls that are executed before
                the final response is synthesised.

        Returns:
            The result of the task execution

        Raises:
            ValueError: If task is not a valid string
            MakiNetworkError: For network-related errors
            MakiTimeoutError: For timeout errors
            MakiAPIError: For API response errors
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be a non-empty string")

        # Build conversation history section (stateful mode)
        history_section = ""
        if self.stateful and self._conversation_history:
            lines = []
            for turn in self._conversation_history[-10:]:
                lines.append(f"Task: {turn['task']}")
                lines.append(f"Response: {turn['result'][:300]}")
            history_section = "\n\nPrior conversation:\n" + "\n".join(lines)

        # Build plugin section when use_plugins is requested
        plugin_section = ""
        if use_plugins and self.plugins:
            descriptions = []
            for pname, plugin in self.plugins.items():
                methods = [
                    m for m in dir(plugin)
                    if not m.startswith('_') and callable(getattr(plugin, m))
                ]
                descriptions.append(f"- {pname}: {', '.join(methods)}")
            plugin_section = (
                "\n\nAvailable plugins:\n" + "\n".join(descriptions) +
                '\n\nTo call a plugin output a line in this exact format before your answer:\n'
                'TOOL: {"plugin": "<name>", "method": "<method>", "args": {<key>: <value>}}'
            )

        prompt = f"""
        You are {self.name}, a {self.role}.
        {self.instructions}{history_section}

        Task: {task}

        Context: {json.dumps(context) if context else 'None'}{plugin_section}

        Please provide a detailed response to the task.
        """

        try:
            logger.debug(f"Executing task '{task}' for agent '{self.name}'")
            start_time = time.time()
            result = self.maki.request(prompt)

            # Execute any TOOL: directives the LLM emitted
            if use_plugins and self.plugins:
                result = self._handle_plugin_calls(result, task, context)

            execution_time = time.time() - start_time
            logger.debug(f"Task '{task}' completed in {execution_time:.2f}s for agent '{self.name}'")
        except Exception as e:
            logger.error(f"Failed to execute task '{task}' for agent '{self.name}': {str(e)}", exc_info=True)
            # Re-raise Maki exceptions and programming errors (ValueError, TypeError) as-is so
            # callers (e.g. execute_task_with_retry) can distinguish retryable vs non-retryable.
            if isinstance(e, (MakiNetworkError, MakiTimeoutError, MakiAPIError, ValueError, TypeError)):
                raise
            raise MakiNetworkError(f"Failed to execute task '{task}' for agent '{self.name}': {str(e)}")

        # Record the task execution in history
        self.task_history.append({
            'task': task,
            'context': context,
            'result': result,
            'timestamp': time.time()
        })

        # Maintain stateful conversation history
        if self.stateful:
            self._conversation_history.append({'task': task, 'result': result})

        return result

    def _handle_plugin_calls(self, llm_response: str, task: str, context: Optional[Dict]) -> str:
        """
        Parse TOOL: directives from the LLM response, execute them, and synthesise
        a final answer that incorporates the tool results.

        The expected format emitted by the LLM is:
            TOOL: {"plugin": "<name>", "method": "<method>", "args": {...}}
        """
        tool_pattern = re.compile(r'^TOOL:\s*(\{.*\})', re.MULTILINE)
        matches = tool_pattern.findall(llm_response)
        if not matches:
            return llm_response

        tool_results = []
        for match in matches:
            try:
                call = json.loads(match)
                plugin_name = call.get("plugin")
                method_name = call.get("method")
                args = call.get("args", {})
                plugin = self.plugins.get(plugin_name)
                if plugin and hasattr(plugin, method_name):
                    method = getattr(plugin, method_name)
                    output = method(**args)
                    tool_results.append({
                        "plugin": plugin_name,
                        "method": method_name,
                        "result": str(output)
                    })
                else:
                    tool_results.append({
                        "plugin": plugin_name,
                        "method": method_name,
                        "error": f"Plugin '{plugin_name}' or method '{method_name}' not available"
                    })
            except Exception as e:
                logger.warning(f"Plugin call failed: {str(e)}")
                tool_results.append({"error": str(e)})

        # Strip TOOL: lines from the partial response, then ask for a final answer
        clean_response = tool_pattern.sub('', llm_response).strip()
        follow_up = f"""
        Task: {task}
        Tool results: {json.dumps(tool_results, indent=2)}
        Previous partial response: {clean_response}
        Please provide your final answer incorporating the tool results.
        """
        return self.maki.request(follow_up)

    def reset_conversation(self):
        """Clear the stateful conversation history"""
        self._conversation_history.clear()

    def execute_task_with_retry(self, task: str, context: Optional[Dict] = None,
                               max_retries: int = 3, retry_delay: float = 1.0) -> str:
        """
        Execute a task with retry logic.

        Only transient network/timeout errors are retried.  Non-retryable errors
        (MakiAPIError, ValueError, etc.) are re-raised immediately without waiting.

        Args:
            task: The task to perform
            context: Additional context for the task
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            The result of the task execution

        Raises:
            Exception: If task fails after all retries (transient) or immediately (non-transient)
        """
        for attempt in range(max_retries):
            try:
                return self.execute_task(task, context)
            except _RETRYABLE_ERRORS as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    f"Task '{task}' failed on attempt {attempt + 1}/{max_retries}: "
                    f"{str(e)}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            except Exception:
                # Non-retryable (API error, bad input, etc.) — fail fast
                raise

        # Unreachable, but satisfies type checkers
        raise RuntimeError(f"Task '{task}' failed after {max_retries} attempts")

    def stream_task(self, task: str, context: Optional[Dict] = None):
        """
        Stream a task response token by token.

        Requires a backend that supports streaming (e.g., MakiLLama). Raises
        NotImplementedError if the configured backend has no stream() method.

        Args:
            task: The task to perform
            context: Additional context for the task

        Returns:
            A generator that yields response chunks
        """
        if not hasattr(self.maki, 'stream'):
            raise NotImplementedError(
                f"Backend '{type(self.maki).__name__}' does not support streaming. "
                "Use MakiLLama instead."
            )
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be a non-empty string")

        prompt = f"""
        You are {self.name}, a {self.role}.
        {self.instructions}

        Task: {task}

        Context: {json.dumps(context) if context else 'None'}

        Please provide a detailed response to the task.
        """
        return self.maki.stream(prompt)

    def remember(self, key: str, value: Any):
        """Store information in the agent's memory"""
        self.memory[key] = value

    def recall(self, key: str) -> Any:
        """Retrieve information from the agent's memory"""
        return self.memory.get(key, None)

    def clear_memory(self):
        """Clear the agent's memory"""
        self.memory.clear()

    def set_max_history_entries(self, max_entries: int):
        """Set the maximum number of entries to keep in history.

        Existing history is preserved up to the new limit (most recent entries kept).

        Args:
            max_entries: Maximum number of entries to keep in history
        """
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_history_entries = max_entries
        # Recreate deques with new maxlen, preserving most recent entries
        self.reasoning_history = deque(self.reasoning_history, maxlen=max_entries)
        self.task_history = deque(self.task_history, maxlen=max_entries)

    def load_plugin(self, plugin_name: str, plugin_path: str = None):
        """
        Load a plugin for this agent

        Args:
            plugin_name: Name of the plugin to load
            plugin_path: Optional path to the plugin (if not in standard location)

        Returns:
            The loaded plugin instance

        Raises:
            ImportError: If plugin cannot be loaded
            Exception: If plugin initialization fails
        """
        try:
            if plugin_path:
                # Load plugin from a custom path without mutating sys.path.
                import importlib.util
                import os
                plugin_file = os.path.join(plugin_path, plugin_name, "__init__.py")
                if not os.path.exists(plugin_file):
                    plugin_file = os.path.join(plugin_path, f"{plugin_name}.py")
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot locate plugin '{plugin_name}' at '{plugin_path}'")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                module = importlib.import_module(f"maki.plugins.{plugin_name}")

            if hasattr(module, 'register_plugin'):
                plugin_instance = module.register_plugin(self.maki)
            elif hasattr(module, plugin_name):
                plugin_class = getattr(module, plugin_name)
                plugin_instance = plugin_class(self.maki)
            else:
                plugin_instance = module(self.maki)

            self.plugins[plugin_name] = plugin_instance
            logger.info(f"Plugin '{plugin_name}' loaded successfully for agent '{self.name}'")
            return plugin_instance

        except Exception as e:
            logger.error(f"Failed to load plugin '{plugin_name}' for agent '{self.name}': {str(e)}")
            raise

    def get_plugin(self, plugin_name: str):
        """Get a loaded plugin instance, or None if not loaded"""
        return self.plugins.get(plugin_name)

    def unload_plugin(self, plugin_name: str):
        """Unload a plugin from this agent"""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]

    def _cleanup_history(self):
        """No-op: deque(maxlen=...) enforces the size limit automatically on every append."""
        pass

    def think_step_by_step(self, problem: str, steps: int = 3) -> str:
        """Execute reasoning through multiple steps"""
        prompt = f"""
        Break down the following problem into {steps} clear reasoning steps:
        Problem: {problem}

        Provide a structured approach with:
        1. Initial analysis
        2. Key considerations
        3. Solution approach
        """
        result = self.maki.request(prompt)

        self.reasoning_history.append({
            'problem': problem,
            'steps': steps,
            'result': result,
            'timestamp': time.time()
        })

        return result

    def self_correct(self, initial_response: str, feedback: str, max_iterations: int = 1) -> str:
        """
        Iteratively improve a response based on feedback.

        Args:
            initial_response: The response to improve
            feedback: Guidance on how to improve it
            max_iterations: Number of improvement rounds (default 1)

        Returns:
            The improved response after all iterations
        """
        current = initial_response
        for i in range(max_iterations):
            prompt = f"""
            Improve the following response based on feedback:

            Current response: {current}
            Feedback: {feedback}

            Please revise your response to be more accurate and complete.
            """
            current = self.maki.request(prompt)

            self.reasoning_history.append({
                'iteration': i + 1,
                'original_response': initial_response,
                'feedback': feedback,
                'corrected_response': current,
                'timestamp': time.time()
            })

        return current

    def decompose_task(self, task: str, max_subtasks: int = 5) -> List[Dict]:
        """Decompose a complex task into subtasks using LLM.

        Args:
            task: The complex task to decompose
            max_subtasks: Maximum number of subtasks to create

        Returns:
            A list of dicts with keys: 'description', 'resources', 'expected_outcome'

        Raises:
            ValueError: If task is not a valid string or the LLM returns invalid JSON
            MakiNetworkError / MakiTimeoutError / MakiAPIError: For LLM errors
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be a non-empty string")

        prompt = f"""
        Decompose the following task into {max_subtasks} or fewer subtasks.
        Return your response as a JSON array of objects.

        Task: {task}

        For each subtask, provide an object with these exact keys:
        - "description": A clear description of the subtask
        - "resources": Required resources (tools, data, skills needed)
        - "expected_outcome": What successful completion looks like

        Return ONLY the JSON array, no additional text.
        Example format:
        [
            {{
                "description": "Research existing solutions",
                "resources": "Internet access, documentation",
                "expected_outcome": "List of 3-5 comparable solutions with pros/cons"
            }}
        ]
        """

        try:
            result = self.maki.request(prompt)
        except Exception as e:
            if isinstance(e, (MakiNetworkError, MakiTimeoutError, MakiAPIError)):
                raise
            raise MakiNetworkError(f"Failed to get task decomposition from LLM: {str(e)}")

        self.reasoning_history.append({
            'original_task': task,
            'decomposition': result,
            'timestamp': time.time()
        })

        # Strip markdown code fences if present
        json_str = result.strip()
        if json_str.startswith('```json'):
            json_str = json_str[7:]
        elif json_str.startswith('```'):
            json_str = json_str[3:]
        if json_str.endswith('```'):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        try:
            subtasks = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM did not return valid JSON for task decomposition: {str(e)}. "
                f"Raw response (first 300 chars): {result[:300]}"
            )

        if not isinstance(subtasks, list):
            raise ValueError("LLM response is not a JSON array")

        validated = []
        for i, subtask in enumerate(subtasks[:max_subtasks]):
            if not isinstance(subtask, dict):
                subtask = {"description": str(subtask)}
            validated.append({
                "description": subtask.get("description", f"Subtask {i + 1}"),
                "resources": subtask.get("resources", "Not specified"),
                "expected_outcome": subtask.get("expected_outcome", "Not specified")
            })

        return validated


class AgentManager:
    """Manages a collection of agents and facilitates their coordination"""

    def __init__(self, maki_instance: Maki):
        """
        Initialize the agent manager

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
        Add a new agent to the manager

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
        """Get an agent by name, or None if not found"""
        return self.agents.get(name)

    def remove_agent(self, name: str):
        """Remove an agent from the manager"""
        if name in self.agents:
            del self.agents[name]

    def list_agents(self) -> List[str]:
        """Return a list of all registered agent names"""
        return list(self.agents.keys())

    def assign_task(self, agent_name: str, task: str, context: Optional[Dict] = None) -> str:
        """
        Assign a task to a specific agent

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
            agent_prompt = f"""
            You are working on the following task:

            Task: {task}

            Your role: {agent_name}

            Context: {json.dumps(context) if context else 'None'}

            Please provide your specific response to this task.
            """
            try:
                agent_results[agent_name] = self.maki.request(agent_prompt)
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
