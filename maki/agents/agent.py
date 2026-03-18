"""
Core Agent class for the Maki Framework.

An Agent wraps a Maki LLM backend and provides task execution, memory,
stateful conversations, and extension points via the PluginHandler and
ReasoningEngine mixins.
"""

from collections import deque
from typing import Dict, List, Any, Optional
import json
import logging
import time

from ..backend import LLMBackend
from ..exceptions import MakiNetworkError, MakiTimeoutError, MakiAPIError
from .plugin_handler import PluginHandler
from .reasoning import ReasoningEngine

logger = logging.getLogger(__name__)

# Errors that are worth retrying (transient network/timeout issues)
_RETRYABLE_ERRORS = (MakiNetworkError, MakiTimeoutError)


class Agent(PluginHandler, ReasoningEngine):
    """An individual agent that can perform tasks using the Maki framework."""

    def __init__(self, name: str, maki_instance: LLMBackend, role: str = "", instructions: str = "",
                 stateful: bool = False):
        """
        Initialize an agent.

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

        if not isinstance(maki_instance, LLMBackend):
            raise TypeError("maki_instance must be an LLMBackend instance")

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

        # Validate and initialise mixin contracts (must come after all attrs are set).
        self._init_reasoning()
        self._init_plugins()

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
            plugin_section = self.build_plugin_prompt_section()

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
            result = self.maki.request(prompt).content

            # Execute any TOOL: directives the LLM emitted
            if use_plugins and self.plugins:
                result = self.handle_plugin_calls(result, task, context)

            execution_time = time.time() - start_time
            logger.debug(f"Task '{task}' completed in {execution_time:.2f}s for agent '{self.name}'")
        except Exception as e:
            logger.error(f"Failed to execute task '{task}' for agent '{self.name}': {str(e)}", exc_info=True)
            # Re-raise Maki exceptions and programming errors (ValueError, TypeError) as-is so
            # callers (e.g. execute_task_with_retry) can distinguish retryable vs non-retryable.
            if isinstance(e, (MakiNetworkError, MakiTimeoutError, MakiAPIError, ValueError, TypeError)):
                raise
            raise MakiNetworkError(f"Failed to execute task '{task}' for agent '{self.name}': {str(e)}") from e

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

    def reset_conversation(self):
        """Clear the stateful conversation history."""
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
                # Non-retryable (API error, bad input, etc.) -- fail fast
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
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be a non-empty string")

        prompt = f"""
        You are {self.name}, a {self.role}.
        {self.instructions}

        Task: {task}

        Context: {json.dumps(context) if context else 'None'}

        Please provide a detailed response to the task.
        """
        try:
            return self.maki.stream(prompt)
        except NotImplementedError as e:
            raise NotImplementedError(
                f"Backend '{type(self.maki).__name__}' does not support streaming. "
                "Use MakiLLama or another streaming-capable backend instead."
            ) from e

    def remember(self, key: str, value: Any):
        """Store information in the agent's memory."""
        self.memory[key] = value

    def recall(self, key: str) -> Any:
        """Retrieve information from the agent's memory."""
        return self.memory.get(key, None)

    def clear_memory(self):
        """Clear the agent's memory."""
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

    def _cleanup_history(self):
        """No-op: deque(maxlen=...) enforces the size limit automatically on every append."""
        pass
