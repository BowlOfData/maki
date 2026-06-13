"""
Core Agent class for the Maki Framework.

An Agent wraps a Maki LLM backend and provides task execution, memory,
stateful conversations, and extension points via the PluginHandler and
ReasoningEngine mixins.
"""

from collections import deque
from typing import Dict, Any, Optional
import json
import logging
import threading
import time
import uuid

from ..backend import LLMBackend
from ..exceptions import MakiError, MakiNetworkError, MakiTimeoutError, MakiAPIError
from ..objects import ConversationMemory, Message
from .plugin_handler import PluginHandler
from .reasoning import ReasoningEngine

logger = logging.getLogger(__name__)

# Errors that are worth retrying (transient network/timeout issues)
_RETRYABLE_ERRORS = (MakiNetworkError, MakiTimeoutError)


class Agent(PluginHandler, ReasoningEngine):
    """An individual agent that can perform tasks using the Maki framework."""

    def __init__(self, name: str, maki_instance: LLMBackend, role: str = "", instructions: str = "",
                 stateful: bool = False, use_streaming: bool = False,
                 allow_dangerous_tools: bool = False):
        """
        Initialize an agent.

        Args:
            name: Unique identifier for the agent
            maki_instance: Maki instance to use for LLM interactions
            role: The role of the agent (e.g., "researcher", "writer", "analyst")
            instructions: Specific instructions for this agent
            stateful: If True, prior task results are included in subsequent prompts
            use_streaming: If True, execute_task uses streaming internally (chat_collect)
                so the timeout applies per-chunk rather than to the full response.
                Useful for tasks with very long outputs that exceed the backend timeout.
            allow_dangerous_tools: If True, plugin methods marked in a plugin's
                DANGEROUS_METHODS (file writes, uploads, deletes, …) may be
                invoked via TOOL: directives. Off by default — a prompt-injected
                instruction in scraped content should not be able to write
                files or delete remote directories.

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

        if not isinstance(maki_instance, LLMBackend) and not hasattr(maki_instance, 'request'):
            raise TypeError("maki_instance must implement the LLMBackend interface")

        self.agent_id = str(uuid.uuid4())
        self.name = name.strip()
        self.maki = maki_instance
        self.role = role
        self.instructions = instructions
        self.stateful = stateful
        self.use_streaming = use_streaming
        self.allow_dangerous_tools = allow_dangerous_tools
        self.memory = {}

        # Maximum number of entries to keep in history; deque enforces this automatically
        self._max_history_entries = 1000
        self.reasoning_history: deque = deque(maxlen=self._max_history_entries)
        self.task_history: deque = deque(maxlen=self._max_history_entries)

        # Stateful multi-turn conversation memory (separate from task_history).
        # Token-budgeted and eviction-based; replaces the old fixed-window + 300-char
        # truncation approach.
        self._conversation_memory = ConversationMemory(
            max_entries=self._max_history_entries,
        )

        # One in-flight task at a time; acquired by execute_task / stream_task so
        # concurrent callers (workflow thread pools, FastAPI handlers) are serialized.
        self._execution_lock = threading.Lock()

        # Validate and initialise mixin contracts (must come after all attrs are set).
        self._init_reasoning()
        self._init_plugins()

        logger.info(f"Agent '{self.name}' initialized with role '{self.role}'")

    def __repr__(self):
        return f"Agent(name='{self.name}', role='{self.role}')"

    def _build_history_section(self) -> str:
        """Return the prior-conversation block for stateful prompts, or empty string."""
        if not self.stateful:
            return ""
        return self._conversation_memory.format_as_text()

    def _build_system_message(self) -> str:
        """Return the system message (role + instructions) for this agent."""
        return f"You are {self.name}, a {self.role}. {self.instructions}".strip()

    def _build_user_message(self, task: str, context: Optional[Dict], use_plugins: bool) -> str:
        """Return the user message (task + optional context/plugins/history)."""
        plugin_section = self.build_plugin_prompt_section() if use_plugins and self.plugins else ""
        history_section = self._build_history_section()
        parts = [task]
        if context:
            parts.append(f"Context: {json.dumps(context)}")
        if plugin_section:
            parts.append(plugin_section)
        if history_section:
            parts.append(history_section)
        return "\n\n".join(p for p in parts if p)

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

        with self._execution_lock:
            try:
                logger.debug(f"Executing task '{task}' for agent '{self.name}'")
                start_time = time.time()
                system_msg = self._build_system_message()

                if use_plugins and self.plugins and getattr(self.maki, "supports_native_tools", False) is True:
                    # Native tool-calling path: backend drives structured tool calls.
                    result = self.execute_with_native_tools(task, context, system_msg)
                else:
                    user_msg = self._build_user_message(task, context, use_plugins)
                    if self.use_streaming:
                        result = self.maki.chat_collect(user_msg, system=system_msg).content
                    else:
                        result = self.maki.chat(user_msg, system=system_msg).content
                    if result is None:
                        raise MakiAPIError(f"Backend returned None content for task '{task}'")

                    # Legacy TOOL: directive path for non-native backends.
                    if use_plugins and self.plugins:
                        result = self.handle_plugin_calls(result, task, context)

                execution_time = time.time() - start_time
                logger.debug(f"Task '{task}' completed in {execution_time:.2f}s for agent '{self.name}'")
            except (MakiError, ValueError, TypeError) as e:
                logger.error(f"Failed to execute task '{task}' for agent '{self.name}': {str(e)}", exc_info=True)
                raise
            except Exception as e:
                logger.error(f"Unexpected error executing task '{task}' for agent '{self.name}': {str(e)}", exc_info=True)
                raise MakiError(f"Failed to execute task '{task}' for agent '{self.name}': {str(e)}") from e

            # Record the task execution in history
            self.task_history.append({
                'task': task,
                'context': context,
                'result': result,
                'timestamp': time.time()
            })

            # Maintain stateful conversation history
            if self.stateful:
                self._conversation_memory.append(Message("user", task))
                self._conversation_memory.append(Message("assistant", result))

        return result

    def reset_conversation(self):
        """Clear the stateful conversation history."""
        self._conversation_memory.clear()

    def execute_task_with_retry(self, task: str, context: Optional[Dict] = None,
                               max_retries: int = 3, retry_delay: float = 1.0,
                               use_plugins: bool = False) -> str:
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
                return self.execute_task(task, context, use_plugins=use_plugins)
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

    def stream_task(self, task: str, context: Optional[Dict] = None, use_plugins: bool = False):
        """
        Stream a task response token by token.

        Requires a backend that supports streaming (e.g., MakiLLama). Raises
        NotImplementedError if the configured backend has no stream() method.

        History (task_history and stateful conversation history) is recorded
        via a finally block in the returned generator, so it is always updated
        regardless of whether the caller consumes the stream fully or abandons
        it early.

        Args:
            task: The task to perform
            context: Additional context for the task
            use_plugins: When True and plugins are loaded, plugin descriptions
                are included in the prompt. Note: TOOL: directives emitted
                during streaming are not executed mid-stream; use execute_task()
                if you need plugin call/response cycles.

        Returns:
            A generator that yields response chunks
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be a non-empty string")

        system_msg = self._build_system_message()
        user_msg = self._build_user_message(task, context, use_plugins)
        stream_kwargs = {"system": system_msg}

        def _tracked():
            chunks = []
            with self._execution_lock:
                try:
                    raw_stream = self.maki.stream(user_msg, **stream_kwargs)
                except NotImplementedError as e:
                    raise NotImplementedError(
                        f"Backend '{type(self.maki).__name__}' does not support streaming. "
                        "Use MakiLLama or another streaming-capable backend instead."
                    ) from e
                try:
                    for chunk in raw_stream:
                        chunks.append(chunk)
                        yield chunk
                finally:
                    # Always record whatever was produced, even on early abandonment.
                    if chunks:
                        full_result = "".join(chunks)
                        self.task_history.append({
                            'task': task,
                            'context': context,
                            'result': full_result,
                            'timestamp': time.time(),
                        })
                        if self.stateful:
                            self._conversation_memory.append(Message("user", task))
                            self._conversation_memory.append(Message("assistant", full_result))

        return _tracked()

    def remember(self, key: str, value: Any):
        """Store information in the agent's memory."""
        self.memory[key] = value

    def recall(self, key: str) -> Any:
        """Retrieve information from the agent's memory."""
        return self.memory.get(key, None)

    def clear_memory(self):
        """Clear the agent's memory."""
        self.memory.clear()

    def to_dict(self) -> dict:
        """Serialize the agent's state to a JSON-compatible dict.

        The LLM backend is not included — pass it explicitly to from_dict().
        """
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'role': self.role,
            'instructions': self.instructions,
            'stateful': self.stateful,
            'use_streaming': self.use_streaming,
            'memory': dict(self.memory),
            'task_history': list(self.task_history),
            'conversation_memory': self._conversation_memory.to_list(),
            'conversation_token_budget': self._conversation_memory.token_budget,
            'max_history_entries': self._max_history_entries,
        }

    @classmethod
    def from_dict(cls, data: dict, maki_instance: LLMBackend) -> 'Agent':
        """Reconstruct an Agent from a dict produced by to_dict().

        Args:
            data: Dict previously returned by to_dict().
            maki_instance: LLM backend to attach (not serialized).
        """
        agent = cls(
            name=data['name'],
            maki_instance=maki_instance,
            role=data.get('role', ''),
            instructions=data.get('instructions', ''),
            stateful=data.get('stateful', False),
            use_streaming=data.get('use_streaming', False),
        )
        agent.agent_id = data['agent_id']
        agent.memory = dict(data.get('memory', {}))
        max_entries = data.get('max_history_entries', 1000)
        agent._max_history_entries = max_entries
        agent.task_history = deque(data.get('task_history', []), maxlen=max_entries)
        token_budget = data.get('conversation_token_budget', ConversationMemory.DEFAULT_TOKEN_BUDGET)
        if 'conversation_memory' in data:
            agent._conversation_memory = ConversationMemory.from_list(
                data['conversation_memory'],
                token_budget=token_budget,
                max_entries=max_entries,
            )
        elif 'conversation_history' in data:
            # Migration from old {'task': ..., 'result': ...} format
            agent._conversation_memory = ConversationMemory(
                token_budget=token_budget, max_entries=max_entries,
            )
            for turn in data['conversation_history']:
                agent._conversation_memory._messages.append(Message('user', turn.get('task', '')))
                agent._conversation_memory._messages.append(Message('assistant', turn.get('result', '')))
        else:
            agent._conversation_memory = ConversationMemory(
                token_budget=token_budget, max_entries=max_entries,
            )
        return agent

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
        # max_entries setter trims immediately if the new cap is smaller
        self._conversation_memory.max_entries = max(2, max_entries)

