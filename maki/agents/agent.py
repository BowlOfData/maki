"""
Core Agent class for the Maki Framework.

An Agent wraps a Maki LLM backend and provides task execution, memory,
stateful conversations, and extension points via the PluginHandler and
ReasoningEngine mixins.
"""

from collections import deque
from typing import Dict, Any, List, Optional, Union
import json
import logging
import threading
import time
import uuid

from ..aio import aiterate, async_gate, run_in_thread
from ..backend import LLMBackend
from ..callbacks import AgentCallback, fire
from ..exceptions import MakiError, MakiNetworkError, MakiTimeoutError, MakiAPIError
from ..objects import ConversationMemory, MemoryPolicy, Message, ToolCall, clip_text
from ..skills import coerce_registry
from ..usage import Usage, attach_usage
from .plugin_handler import PluginHandler
from .reasoning import ReasoningEngine
from .tool_policy import ToolPolicy

logger = logging.getLogger(__name__)

# Default cap on the running summary when MemoryPolicy(summarize_older=True) sets no summary_chars.
_DEFAULT_SUMMARY_CHARS = 1200
_CONTEXT_OVERHEAD_TOKENS = 2048  # system message, tool schemas and the task itself, on top of the history budget
# Each evicted message is clipped to this many characters in the summarisation prompt, and the
# turns together to _SUMMARY_TOTAL_CHARS, so a large eviction cannot overflow the model's window.
_SUMMARY_INPUT_CHARS = 1500
_SUMMARY_TOTAL_CHARS = 12000

# Errors that are worth retrying (transient network/timeout issues)
_RETRYABLE_ERRORS = (MakiNetworkError, MakiTimeoutError)


class Agent(PluginHandler, ReasoningEngine):
    """An individual agent that can perform tasks using the Maki framework."""

    MAX_MEMORY_ENTRIES = 10_000     # keys in ``remember()`` memory
    MAX_MEMORY_KEY_CHARS = 512

    def __init__(self, name: str, maki_instance: LLMBackend, role: str = "", instructions: str = "",
                 stateful: bool = False, use_streaming: bool = False,
                 allow_dangerous_tools: bool = False,
                 memory: Optional[Union[MemoryPolicy, dict]] = None,
                 tool_policy: Optional[Union[ToolPolicy, dict]] = None,
                 callbacks: Optional[List[AgentCallback]] = None,
                 skills: Any = None):
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
            memory: Optional ``MemoryPolicy`` (or a dict of its fields) limiting what the
                agent keeps and sends. ``None`` keeps the default behaviour.
            tool_policy: Optional ``ToolPolicy`` (or a dict of its fields): tool round limit,
                per-tool retries with backoff, parallel tool calls. ``None`` keeps the defaults.
            callbacks: Optional ``AgentCallback`` observers for task, model and tool events.
            skills: Agent Skills the model can load on demand: ``True`` (user skills), a folder or list of
                folders, a :class:`~maki.skills.SkillRegistry`, or an options dict (see :mod:`maki.skills`).
                Adds the ``skills`` plugin (``skills.load``, ``skills.read``) and, for tasks run with
                ``use_plugins=True``, a catalog of skill names and descriptions in the system message.

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
        attach_usage(maki_instance)  # count tokens from the first call on
        self.role = role
        self.instructions = instructions
        self.stateful = stateful
        self.use_streaming = use_streaming
        self.allow_dangerous_tools = allow_dangerous_tools
        self.memory: Dict[str, Any] = {}
        self.memory_policy: Optional[MemoryPolicy] = (
            MemoryPolicy.from_dict(memory) if isinstance(memory, dict) else memory)
        if self.memory_policy is not None and not isinstance(self.memory_policy, MemoryPolicy):
            raise TypeError("memory must be a MemoryPolicy, a dict of its fields, or None")
        self.tool_policy: Optional[ToolPolicy] = (
            ToolPolicy.from_dict(tool_policy) if isinstance(tool_policy, dict) else tool_policy)
        if self.tool_policy is not None and not isinstance(self.tool_policy, ToolPolicy):
            raise TypeError("tool_policy must be a ToolPolicy, a dict of its fields, or None")
        self.callbacks: List[AgentCallback] = list(callbacks or [])
        # Optional override for the summary tier: summarizer(previous_summary, [Message, ...]) -> str.
        self.summarizer = None
        if self.memory_policy is not None and self.memory_policy.summarize_older:
            summary_tokens = (self.memory_policy.summary_chars or _DEFAULT_SUMMARY_CHARS) // 4
            if summary_tokens * 2 > self._history_token_budget():
                logger.warning(
                    f"Agent '{self.name}': the summary (up to ~{summary_tokens} tokens) takes more than half "
                    f"of token_budget={self._history_token_budget()}; lower summary_chars or raise token_budget")
        self._warned_history_fallback = False
        self._pinned: Dict[str, Any] = {}   # facts that go in every system message; see remember()

        # Maximum number of entries to keep in history; deque enforces this automatically
        self._max_history_entries = 1000
        self.reasoning_history: deque = deque(maxlen=self._max_history_entries)
        self.task_history: deque = deque(maxlen=self._max_history_entries)

        # Stateful multi-turn conversation memory (separate from task_history).
        # Token-budgeted and eviction-based; replaces the old fixed-window + 300-char
        # truncation approach.
        self._conversation_memory = ConversationMemory(
            token_budget=self._history_token_budget(),
            max_entries=self._max_history_entries,
        )
        self._apply_tiers()
        hint = getattr(maki_instance, "hint_context", None)
        if callable(hint):  # a backend that sizes its context window ('auto') learns what this agent needs
            hint(self._history_token_budget() + _CONTEXT_OVERHEAD_TOKENS)

        # One in-flight task at a time; acquired by execute_task / stream_task so
        # concurrent callers (workflow thread pools, FastAPI handlers) are serialized.
        self._execution_lock = threading.Lock()
        self._last_tool_trace: List[Message] = []  # tool loop messages of the latest native task

        # Validate and initialise mixin contracts (must come after all attrs are set).
        self._init_reasoning()
        self._init_plugins()
        self.skills = coerce_registry(skills)
        if self.skills:
            self.plugins["skills"] = self.skills.plugin()

        logger.info(f"Agent '{self.name}' initialized with role '{self.role}'")

    def __repr__(self):
        return f"Agent(name='{self.name}', role='{self.role}')"

    def _history_token_budget(self) -> int:
        policy = self.memory_policy
        if policy is not None and policy.token_budget is not None:
            return policy.token_budget
        return ConversationMemory.DEFAULT_TOKEN_BUDGET

    def _apply_tiers(self) -> None:
        policy = self.memory_policy
        if policy is not None:
            self._conversation_memory.configure_tiers(policy.recent_turns, policy.older_turn_chars)
            self._conversation_memory.collect_evicted = policy.summarize_older

    def _fold_summary(self) -> None:
        """Fold turns evicted since the last call into the running summary (summary tier).

        Never raises: when the summariser fails the turns stay dropped, which the history shows as
        "[N earlier turn(s) dropped]", exactly as without the tier.
        """
        memory, policy = self._conversation_memory, self.memory_policy
        pairs = memory.take_pending()
        if not pairs or policy is None or not policy.summarize_older:
            return
        limit = policy.summary_chars or _DEFAULT_SUMMARY_CHARS
        messages = [m for pair in pairs for m in pair]
        try:
            if self.summarizer is not None:
                text = self.summarizer(memory.summary, messages)
            else:
                text = self._summarize(memory.summary, messages, limit)
            text = str(text or "").strip()
        except Exception as exc:
            logger.warning(f"Agent '{self.name}': summarising {len(pairs)} evicted turn(s) failed: {exc}")
            return
        if not text:
            logger.warning(f"Agent '{self.name}': the summariser returned nothing; turns stay dropped")
            return
        memory.set_summary(clip_text(text, limit), len(pairs))

    @staticmethod
    def _tools_note(message: Message, limit: int) -> str:
        """A one-line note of the tools behind a remembered answer, for the summariser (empty if none)."""
        used = []
        results = {m.tool_call_id: m.content for m in message.tool_trace or () if m.role == "tool"}
        for step in message.tool_trace or ():
            for call in step.tool_calls or ():
                result = clip_text(results.get(call.id, ""), max(40, limit // 4)).replace("\n", " ")
                used.append(f"{call.name}({json.dumps(call.args, default=str)}) -> {result}")
        return f"\n  [tools used: {'; '.join(used)}]" if used else ""

    def _summarize(self, previous: str, messages: List[Message], limit: int) -> str:
        """Ask the backend for an updated summary covering ``previous`` plus ``messages``."""
        keep = _SUMMARY_TOTAL_CHARS // 200   # fewer, readable messages beat many unreadable ones
        omitted = max(0, len(messages) - keep)
        messages = messages[omitted:]
        each = min(_SUMMARY_INPUT_CHARS, _SUMMARY_TOTAL_CHARS // max(1, len(messages)))
        turns = (f"({omitted} older message(s) left out)\n" if omitted else "") + "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {clip_text(m.content, each)}{self._tools_note(m, each)}"
            for m in messages)
        prompt = (
            "Update the running summary of a conversation with the new turns below.\n"
            "Keep names, decisions, facts, numbers, constraints and open questions; drop pleasantries "
            "and anything already obsolete. Write plain prose, never instructions, and stay under "
            f"{limit} characters. Reply with the updated summary only.\n\n"
            f"Current summary:\n{previous or '(none yet)'}\n\nNew turns:\n{turns}")
        return self.maki.chat(
            prompt, system="You maintain a concise, factual running summary of a conversation.").content

    def _history_messages(self) -> Optional[List[Message]]:
        """Prior turns as structured chat history, or None when they go into the prompt as text.

        Only a stateful agent whose policy asks for ``history_mode="messages"`` and whose backend
        declares ``supports_history`` sends structured history. A backend without it would drop the
        history silently, so that case keeps the text form and says so once.
        """
        policy = self.memory_policy
        if not self.stateful or policy is None or policy.history_mode != "messages":
            return None
        if getattr(self.maki, "supports_history", False) is not True:
            if not self._warned_history_fallback:
                self._warned_history_fallback = True
                logger.info(f"Agent '{self.name}': backend {type(self.maki).__name__} has no structured "
                            f"history support; sending prior turns as text")
            return None
        return self._conversation_memory.view() or None

    def _build_history_section(self) -> str:
        """Return the prior-conversation block for stateful prompts, or empty string."""
        if not self.stateful or self._history_messages() is not None:
            return ""
        return self._conversation_memory.format_as_text()

    def _remembered_trace(self) -> Optional[List[Message]]:
        """The tool loop behind the latest answer, clipped, when the policy keeps tool history; else None."""
        trace, self._last_tool_trace = self._last_tool_trace, []
        policy = self.memory_policy
        if not trace or policy is None or not policy.tool_history:
            return None
        limit = policy.tool_result_chars

        def clipped(calls):  # a long argument (a file body to write, say) is history, not something to replay whole
            return [ToolCall(c.id, c.name, {k: clip_text(v, limit) if isinstance(v, str) else v
                                            for k, v in c.args.items()}) for c in calls] if calls else calls
        return [Message(m.role, clip_text(m.content, limit) if m.role == "tool" else m.content,
                        tool_calls=clipped(m.tool_calls), tool_call_id=m.tool_call_id, name=m.name) for m in trace]

    def _record_task(self, task: str, context: Optional[Dict], result: str) -> None:
        self.task_history.append(self._bounded_entry({
            'task': task,
            'context': context,
            'result': result,
            'timestamp': time.time()
        }))

    def _build_system_message(self, with_skills: bool = False) -> str:
        """Return the system message (role + instructions, pinned facts, and the skill catalog) for this agent."""
        base = f"You are {self.name}, a {self.role}. {self.instructions}".strip()
        parts = [base]
        pinned = self._pinned_block()
        if pinned:
            parts.append(pinned)
        if with_skills and self.skills:
            parts.append(self.skills.catalog())
        return "\n\n".join(parts)

    @staticmethod
    def _pinned_lines(pinned: Dict[str, Any]) -> str:
        return "\n".join(f"- {key}: {value}" for key, value in pinned.items())

    def _pinned_block(self) -> str:
        pinned = dict(self._pinned)  # a snapshot: unpin() may run on another thread
        if not pinned:
            return ""
        return "Pinned facts (always true for this conversation):\n" + self._pinned_lines(pinned)

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

    def _generate(self, task: str, context: Optional[Dict], use_plugins: bool) -> str:
        """The model round(s) for one task: native tool loop, or chat plus ``TOOL:`` directives."""
        if self.stateful:
            self._fold_summary()
        self._last_tool_trace = []  # a trace is only ever the one of this task
        system_msg = self._build_system_message(with_skills=use_plugins)

        if use_plugins and self.plugins and getattr(self.maki, "supports_native_tools", False) is True:
            # Native tool-calling path: backend drives structured tool calls.
            return self.execute_with_native_tools(task, context, system_msg)
        user_msg = self._build_user_message(task, context, use_plugins)
        chat_kwargs: Dict[str, Any] = {"system": system_msg}
        history = self._history_messages()
        if history is not None:
            chat_kwargs["history"] = history
        if self.use_streaming:
            result = self.maki.chat_collect(user_msg, **chat_kwargs).content
        else:
            result = self.maki.chat(user_msg, **chat_kwargs).content
        if result is None:
            raise MakiAPIError(f"Backend returned None content for task '{task}'")

        # Legacy TOOL: directive path for non-native backends.
        if use_plugins and self.plugins:
            result = self.handle_plugin_calls(result, task, context)
        return result

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
            start_time = time.time()
            fire(self.callbacks, "on_task_start", self, task)
            try:
                logger.debug(f"Executing task '{task}' for agent '{self.name}'")
                result = self._generate(task, context, use_plugins)

                execution_time = time.time() - start_time
                logger.debug(f"Task '{task}' completed in {execution_time:.2f}s for agent '{self.name}'")
            except (MakiError, ValueError, TypeError) as e:
                logger.error(f"Failed to execute task '{task}' for agent '{self.name}': {str(e)}", exc_info=True)
                fire(self.callbacks, "on_task_error", self, task, e, time.time() - start_time)
                raise
            except Exception as e:
                logger.error(f"Unexpected error executing task '{task}' for agent '{self.name}': {str(e)}", exc_info=True)
                fire(self.callbacks, "on_task_error", self, task, e, time.time() - start_time)
                raise MakiError(f"Failed to execute task '{task}' for agent '{self.name}': {str(e)}") from e

            fire(self.callbacks, "on_task_end", self, task, result, time.time() - start_time)

            # Record the task execution in history
            self._record_task(task, context, result)

            # Maintain stateful conversation history
            if self.stateful:
                self._conversation_memory.append(Message("user", task))
                self._conversation_memory.append(Message("assistant", result, tool_trace=self._remembered_trace()))

        return result

    @property
    def usage(self) -> Usage:
        """Token totals of this agent's backend since counting started (shared if the backend is)."""
        return attach_usage(self.maki)

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

    async def aexecute_task(self, task: str, context: Optional[Dict] = None, use_plugins: bool = False) -> str:
        """Awaitable :meth:`execute_task`; the model call runs on a worker thread (see :mod:`maki.aio`)."""
        async with async_gate(self):
            return await run_in_thread(self.execute_task, task, context, use_plugins)

    async def aexecute_task_with_retry(self, task: str, context: Optional[Dict] = None, max_retries: int = 3,
                                       retry_delay: float = 1.0, use_plugins: bool = False) -> str:
        """Awaitable :meth:`execute_task_with_retry`; the waits between attempts happen on the worker thread."""
        async with async_gate(self):
            return await run_in_thread(self.execute_task_with_retry, task, context, max_retries, retry_delay,
                                       use_plugins)

    async def arun_loop(self, goal: str, requirements, judge, **options):
        """Awaitable :meth:`run_loop` (same arguments); the whole loop runs on one worker thread."""
        async with async_gate(self):
            return await run_in_thread(self.run_loop, goal, requirements, judge, **options)

    async def astream_task(self, task: str, context: Optional[Dict] = None, use_plugins: bool = False):
        """Async generator over :meth:`stream_task`: ``async for chunk in agent.astream_task(...)``.

        Breaking out of the loop stops the stream after the chunk in flight and releases the agent.
        """
        async with async_gate(self):
            chunks = aiterate(lambda: self.stream_task(task, context, use_plugins))
            try:
                async for chunk in chunks:
                    yield chunk
            finally:
                await chunks.aclose()  # stops the worker and closes stream_task before the gate opens

    def run_loop(self, goal: str, requirements, judge, on_iteration=None, confirm_requirements=None,
                 approvals=None, overwrite=False, **options):
        """
        Work on ``goal`` in a loop until a judge confirms every requirement is met.

        Args:
            goal: What to achieve.
            requirements: ``Requirement`` objects, or plain strings (judged by ``judge``).
            judge: A separate, stateless Agent that judges the requirements without a
                command or check. Back it with a JSON-mode, non-thinking model.
            on_iteration: Hook called before the first iteration (``None``) and after each
                one (a ``LoopEvent``); return a ``Decision`` to stop or add a reviewer note.
            confirm_requirements: Called once with every requirement (commands in full);
                return False to cancel.
            approvals: An ``ApprovalStore`` for durable approval before every iteration; a
                pending approval returns a suspended result, continued by ``resume_loop``.
            overwrite: Start over even if ``loop_id``/``loop_dir`` holds an unfinished checkpoint.
            **options: ``LoopSpec`` fields: ``max_iterations``, ``max_seconds``,
                ``stall_limit``, ``context`` ("shared" needs ``stateful=True``),
                ``max_tokens``, ``max_artifact_chars``, ``use_plugins``, ``artifact_fn``, ``cwd``, ``loop_dir``, ``loop_id``.

        Returns:
            A ``LoopResult``; see :mod:`maki.agents.loop`.
        """
        from .loop import LoopSpec, run_loop

        spec = LoopSpec(goal, self._loop_requirements(requirements), **options)
        return run_loop(self, judge, spec, on_iteration=on_iteration,
                        confirm_requirements=confirm_requirements, approvals=approvals, overwrite=overwrite)

    def resume_loop(self, goal: str, requirements, judge, on_iteration=None, approvals=None, **options):
        """Continue a loop from its checkpoint; ``options`` must include ``loop_id`` (or ``loop_dir``).

        The goal and requirement ids must match the original run. See :func:`maki.agents.loop.resume_loop`.
        """
        from .loop import LoopSpec, resume_loop

        spec = LoopSpec(goal, self._loop_requirements(requirements), **options)
        return resume_loop(self, judge, spec, on_iteration=on_iteration, approvals=approvals)

    @staticmethod
    def _loop_requirements(requirements):
        from .loop import Requirement

        return [r if isinstance(r, Requirement) else Requirement(f"r{n}", r)
                for n, r in enumerate(requirements, 1)]

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

        if self.stateful:
            with self._execution_lock:
                self._fold_summary()
        system_msg = self._build_system_message()
        user_msg = self._build_user_message(task, context, use_plugins)
        stream_kwargs: Dict[str, Any] = {"system": system_msg}
        history = self._history_messages()
        if history is not None:
            stream_kwargs["history"] = history

        def _tracked():
            chunks = []
            start_time = time.time()
            failure: Optional[BaseException] = None
            with self._execution_lock:
                fire(self.callbacks, "on_task_start", self, task)
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
                except Exception as exc:
                    failure = exc
                    raise
                finally:
                    # Always record whatever was produced, even on early abandonment.
                    elapsed = time.time() - start_time
                    if failure is not None:
                        fire(self.callbacks, "on_task_error", self, task, failure, elapsed)
                    if chunks:
                        full_result = "".join(chunks)
                        if failure is None:
                            fire(self.callbacks, "on_task_end", self, task, full_result, elapsed)
                        self._record_task(task, context, full_result)
                        if self.stateful:
                            self._conversation_memory.append(Message("user", task))
                            self._conversation_memory.append(Message("assistant", full_result))

        return _tracked()

    def remember(self, key: str, value: Any, pin: bool = False):
        """Store information in the agent's memory.

        With ``pin=True`` the fact is also put in the system message of every call and is never
        evicted, so a goal or constraint survives a long conversation. Pinning is the operator's
        call (the model has no tool for it) and needs ``MemoryPolicy(pinned_chars=...)``, which
        caps the total size of the pinned facts; going over it raises ``ValueError``. A pinned key
        can only be changed with ``pin=True`` or after ``unpin()``: a plain ``remember`` (which the
        HTTP ``/memory/set`` endpoint uses) must not rewrite what the operator pinned.

        Raises:
            ValueError: For a key that is not a short non-empty string, a new key once ``MAX_MEMORY_ENTRIES``
                are stored (a remote caller must not be able to grow the agent's memory without end), or a
                pinning conflict.
        """
        if not isinstance(key, str) or not key or len(key) > self.MAX_MEMORY_KEY_CHARS:
            raise ValueError(f"memory keys must be non-empty strings of at most {self.MAX_MEMORY_KEY_CHARS} characters")
        if key not in self.memory and len(self.memory) >= self.MAX_MEMORY_ENTRIES:
            raise ValueError(f"agent memory is full ({self.MAX_MEMORY_ENTRIES} entries); forget something first")
        if pin:
            policy = self.memory_policy
            if policy is None or policy.pinned_chars is None:
                raise ValueError("pin=True needs Agent(memory=MemoryPolicy(pinned_chars=...))")
            candidate = {**self._pinned, key: value}
            size = len(self._pinned_lines(candidate))
            if size > policy.pinned_chars:
                raise ValueError(f"pinned facts would take {size} characters; pinned_chars is {policy.pinned_chars}")
            self._pinned = candidate
        elif key in self._pinned:
            raise ValueError(f"'{key}' is pinned; use remember(..., pin=True) to change it, or unpin() first")
        self.memory[key] = value

    def unpin(self, key: str) -> None:
        """Stop sending a pinned fact (it stays in ``memory``)."""
        self._pinned.pop(key, None)

    def recall(self, key: str) -> Any:
        """Retrieve information from the agent's memory."""
        return self.memory.get(key, None)

    def clear_memory(self):
        """Clear the agent's memory. Pinned facts stay until ``unpin()``: they are the operator's,
        and ``DELETE /memory`` must not be able to remove them."""
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
            'memory_policy': self.memory_policy.to_dict() if self.memory_policy else None,
            'tool_policy': self.tool_policy.to_dict() if self.tool_policy else None,
            'pinned': dict(self._pinned),
            'conversation_dropped': self._conversation_memory.dropped_pairs,
            'conversation_summary': self._conversation_memory.summary,
            'conversation_summarized': self._conversation_memory.summarized_pairs,
            'conversation_pending': self._conversation_memory.pending_to_list(),
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
            memory=data.get('memory_policy'),
            tool_policy=data.get('tool_policy'),
        )
        agent.agent_id = data['agent_id']
        agent.memory = dict(data.get('memory', {}))
        agent._pinned = dict(data.get('pinned', {}))
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
        agent._conversation_memory.dropped_pairs = data.get('conversation_dropped', 0)
        agent._conversation_memory.summary = data.get('conversation_summary', '')
        agent._conversation_memory.summarized_pairs = data.get('conversation_summarized', 0)
        agent._conversation_memory.restore_pending(data.get('conversation_pending', []))
        agent._apply_tiers()
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
