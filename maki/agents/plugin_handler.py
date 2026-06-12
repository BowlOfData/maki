"""
Plugin handling for Maki agents.

Provides the PluginHandler mixin that gives agents the ability to load,
manage, and invoke plugins via LLM-emitted TOOL: directives (legacy fallback)
or via the native tool-calling APIs of capable backends.
"""

import functools
import importlib
import importlib.util
import inspect
import json
import logging
import os
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ..exceptions import MakiValidationError
from ..objects import ToolCall

if TYPE_CHECKING:
    from .protocols import PluginHostProtocol

logger = logging.getLogger(__name__)

# Maximum number of keyword arguments a tool call may pass.
_MAX_ARGS_COUNT = 20
# Maximum length of any single string argument value.
_MAX_ARG_STRING_LENGTH = 10_000
# JSON-safe primitive types accepted as argument values.
_SAFE_ARG_TYPES = (str, int, float, bool, list, dict, type(None))

# Attributes the host class must provide before _init_plugins() is called.
_REQUIRED_ATTRS = ("name", "maki")

# Maximum tool-call rounds before forcing a final text answer.
_MAX_TOOL_ROUNDS = 5


# ---------------------------------------------------------------------------
# Module-level helpers for TOOL: extraction (regex-free; uses raw_decode)
# ---------------------------------------------------------------------------

def _extract_tool_calls(text: str) -> List[Tuple[bool, str, object]]:
    """Extract all TOOL: directives from *text*.

    Uses ``json.JSONDecoder.raw_decode`` so multi-line and pretty-printed JSON
    is handled correctly, unlike a bare regex.

    Returns a list of ``(success, raw_snippet, value)`` triples where:
    * ``success=True``: ``value`` is the parsed dict.
    * ``success=False``: ``value`` is an error message string (fed back to
      the model so it can self-correct).
    """
    results: List[Tuple[bool, str, object]] = []
    marker = "TOOL:"
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        idx = text.find(marker, pos)
        if idx == -1:
            break
        json_start = idx + len(marker)
        # Skip horizontal whitespace between "TOOL:" and the opening brace.
        while json_start < len(text) and text[json_start] in " \t":
            json_start += 1
        try:
            # raw_decode returns (obj, end) where end is an absolute index in text.
            obj, end = decoder.raw_decode(text, json_start)
            raw = text[idx:end]
            results.append((True, raw, obj))
            pos = end
        except json.JSONDecodeError as exc:
            line_end = text.find("\n", json_start)
            if line_end == -1:
                line_end = len(text)
            raw = text[idx: min(line_end, idx + 200)]
            results.append((False, raw, f"JSON parse error: {exc}"))
            pos = idx + len(marker)
    return results


def _strip_tool_calls(text: str, extractions: List[Tuple[bool, str, object]]) -> str:
    """Remove all TOOL: snippets from *text* (uses positions from *extractions*)."""
    if not extractions:
        return text
    result = text
    # Remove from back to front to preserve offsets.
    for _, raw, _ in reversed(extractions):
        idx = result.rfind(raw)
        if idx != -1:
            result = result[:idx] + result[idx + len(raw):]
    return "\n".join(line for line in result.split("\n") if line.strip()).strip()


class PluginHandler:
    """
    Mixin that adds plugin loading and tool-call execution to an agent.

    **Two execution paths:**

    1. **Native** (``backend.supports_native_tools = True``): plugins are
       translated to the backend's JSON Schema tool format; the backend drives
       structured tool calls; results are fed back in a bounded loop (max
       ``_MAX_TOOL_ROUNDS``).

    2. **Legacy TOOL: regex** (fallback): the LLM emits ``TOOL: {...}``
       directives in plain text; they are parsed with
       ``json.JSONDecoder.raw_decode`` (handles multi-line JSON), executed,
       and their results are synthesised in a follow-up prompt — also
       looping up to ``_MAX_TOOL_ROUNDS`` so the model can chain calls.
       Parse failures are fed back so the model can self-correct.

    Tool calls are **fail-closed**: a plugin exposes only the methods named in
    its class-level ``ALLOWED_METHODS`` whitelist; a plugin without one exposes
    nothing. Methods additionally listed in ``DANGEROUS_METHODS`` (writes,
    uploads, deletes, …) are blocked unless the host sets
    ``allow_dangerous_tools = True``.

    **Contract** – the host class must set the following instance attributes
    before ``__init__`` returns:

    * ``name`` (str)  – a non-empty identifier used in log messages.
    * ``maki``        – a Maki LLM backend instance, passed to plugins on load.

    Enforcement is automatic: :meth:`__init_subclass__` wraps every subclass
    ``__init__`` so that a :exc:`TypeError` is raised immediately after
    construction if any required attribute is missing — even if
    ``super().__init__()`` was never called.  The ``plugins`` dict is also
    auto-initialized if the subclass did not set it.

    See :class:`~maki.agents.protocols.PluginHostProtocol` for the full
    contract definition.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Wrap the subclass ``__init__`` to enforce the plugin contract."""
        super().__init_subclass__(**kwargs)
        orig = cls.__dict__.get('__init__')
        # Only wrap a __init__ defined directly on this class, and avoid
        # double-wrapping if PluginHandler already processed it.
        if orig is None or getattr(orig, '_plugin_contract_checked', False):
            return

        @functools.wraps(orig)
        def _checked(self, *args, **kw):
            orig(self, *args, **kw)
            missing = [a for a in _REQUIRED_ATTRS if not hasattr(self, a)]
            if missing:
                raise TypeError(
                    f"'{type(self).__name__}.__init__' completed without setting "
                    f"required PluginHandler attribute(s): {missing}. "
                    f"Ensure super().__init__() is called or set these "
                    f"attributes before __init__ returns."
                )
            # Auto-initialize the plugins dict if the subclass did not.
            if not hasattr(self, 'plugins'):
                self.plugins = {}

        _checked._plugin_contract_checked = True
        cls.__init__ = _checked

    def _call_llm(self, prompt: str) -> str:
        """Send a plain prompt to the backend and return the response text."""
        return self.maki.chat(prompt).content

    def _init_plugins(self) -> None:
        """
        Initialize plugin storage.

        Must be called from the host class ``__init__`` *after* ``name`` and
        ``maki`` have been set.

        Raises:
            TypeError: If the host class has not set the required attributes.
        """
        missing = [a for a in _REQUIRED_ATTRS if not hasattr(self, a)]
        if missing:
            raise TypeError(
                f"'{type(self).__name__}' uses PluginHandler but is missing "
                f"required attribute(s): {missing}. "
                f"Set these before calling _init_plugins()."
            )
        self.plugins: Dict = {}

    def load_plugin(self, plugin_name: str, plugin_path: str = None):
        """
        Load a plugin for this agent.

        Args:
            plugin_name: Name of the plugin to load
            plugin_path: Optional path to the plugin (if not in standard location)

        Returns:
            The loaded plugin instance

        Raises:
            ImportError: If plugin cannot be loaded
            MakiValidationError: If the module exposes neither a
                register_plugin() function nor a class named after the plugin
            Exception: If plugin initialization fails
        """
        try:
            if plugin_path:
                plugin_path = os.path.realpath(plugin_path)
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
                raise MakiValidationError(
                    f"Plugin '{plugin_name}' has no register_plugin() function "
                    f"or '{plugin_name}' class"
                )

            if getattr(plugin_instance, "ALLOWED_METHODS", None) is None:
                logger.warning(
                    f"Plugin '{plugin_name}' declares no ALLOWED_METHODS; none of "
                    f"its methods will be callable via TOOL: directives. Declare a "
                    f"class-level ALLOWED_METHODS whitelist to expose methods."
                )

            self.plugins[plugin_name] = plugin_instance
            logger.info(f"Plugin '{plugin_name}' loaded successfully for agent '{self.name}'")
            return plugin_instance

        except Exception as e:
            logger.error(f"Failed to load plugin '{plugin_name}' for agent '{self.name}': {str(e)}")
            raise

    def get_plugin(self, plugin_name: str):
        """Get a loaded plugin instance, or None if not loaded."""
        return self.plugins.get(plugin_name)

    def unload_plugin(self, plugin_name: str):
        """Unload a plugin from this agent."""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]

    def _validate_plugin_call(
        self, plugin, plugin_name: str, method_name: str, args: dict
    ) -> Optional[str]:
        """Return an error string if the call is not allowed, else None."""
        # Block private and dunder methods.
        if method_name.startswith("_"):
            return f"Method '{method_name}' is not callable via tool directives"

        # Enforce the whitelist declared on the plugin class. Fail closed:
        # a plugin that declares no ALLOWED_METHODS exposes nothing, and an
        # ALLOWED_METHODS of an unrecognized collection type (e.g. an empty
        # dict {}) blocks all calls rather than silently bypassing the
        # whitelist. The attribute must live on the class/instance — a
        # module-level constant is invisible here.
        allowed = getattr(plugin, "ALLOWED_METHODS", None)
        if allowed is None:
            return (
                f"Plugin '{plugin_name}' declares no ALLOWED_METHODS; "
                f"no methods are exposed to tool calls"
            )
        if not isinstance(allowed, (list, set, tuple, frozenset)):
            return (
                f"Plugin '{plugin_name}' has an invalid ALLOWED_METHODS type "
                f"'{type(allowed).__name__}'; all calls blocked"
            )
        if method_name not in allowed:
            return (
                f"Method '{method_name}' is not in the allowed methods "
                f"for plugin '{plugin_name}'"
            )

        # Destructive methods (writes, uploads, deletes) may additionally be
        # listed in DANGEROUS_METHODS; they require the host agent to opt in
        # via allow_dangerous_tools=True.
        dangerous = getattr(plugin, "DANGEROUS_METHODS", None)
        if dangerous is not None and not isinstance(
            dangerous, (list, set, tuple, frozenset)
        ):
            return (
                f"Plugin '{plugin_name}' has an invalid DANGEROUS_METHODS type "
                f"'{type(dangerous).__name__}'; all calls blocked"
            )
        if (
            dangerous
            and method_name in dangerous
            and not getattr(self, "allow_dangerous_tools", False)
        ):
            return (
                f"Method '{method_name}' on plugin '{plugin_name}' is marked "
                f"dangerous; create the agent with allow_dangerous_tools=True "
                f"to permit it"
            )

        # args must be a plain dict.
        if not isinstance(args, dict):
            return "Tool args must be a JSON object"

        if len(args) > _MAX_ARGS_COUNT:
            return f"Too many arguments ({len(args)} > {_MAX_ARGS_COUNT})"

        for key, val in args.items():
            if not isinstance(key, str):
                return "Argument keys must be strings"
            if not isinstance(val, _SAFE_ARG_TYPES):
                return (
                    f"Argument '{key}' has unsupported type "
                    f"'{type(val).__name__}'"
                )
            if isinstance(val, str) and len(val) > _MAX_ARG_STRING_LENGTH:
                return (
                    f"Argument '{key}' exceeds the maximum allowed length "
                    f"of {_MAX_ARG_STRING_LENGTH} characters"
                )

        return None

    def _allowed_methods(self, plugin) -> list:
        """Return the list of method names exposed by *plugin* to the LLM."""
        explicit = getattr(plugin, "ALLOWED_METHODS", None)
        # Fail closed: no whitelist or an invalid whitelist type → expose nothing.
        if explicit is None or not isinstance(explicit, (list, set, tuple, frozenset)):
            return []
        methods = [m for m in explicit if callable(getattr(plugin, m, None))]
        dangerous = getattr(plugin, "DANGEROUS_METHODS", None)
        if dangerous is not None:
            if not isinstance(dangerous, (list, set, tuple, frozenset)):
                return []  # invalid marker type blocks all calls; advertise nothing
            if not getattr(self, "allow_dangerous_tools", False):
                # Don't advertise methods the agent would refuse to execute.
                methods = [m for m in methods if m not in dangerous]
        return methods

    def build_plugin_prompt_section(self) -> str:
        """Build the plugin description section for inclusion in a prompt."""
        if not self.plugins:
            return ""
        descriptions = []
        for pname, plugin in self.plugins.items():
            methods = self._allowed_methods(plugin)
            descriptions.append(f"- {pname}: {', '.join(methods)}")
        return (
            "\n\nAvailable plugins:\n" + "\n".join(descriptions) +
            '\n\nTo call a plugin output a line in this exact format before your answer:\n'
            'TOOL: {"plugin": "<name>", "method": "<method>", "args": {<key>: <value>}}'
        )

    # ------------------------------------------------------------------
    # Native tool-calling path
    # ------------------------------------------------------------------

    def _build_tool_specs(self) -> List[dict]:
        """Build backend-agnostic tool specs from loaded plugins.

        Each spec has keys ``name`` (``"plugin__method"``), ``description``
        (first docstring line or a fallback), and ``parameters`` (JSON Schema
        built via ``inspect.signature`` — parameter types are assumed string
        since Python annotations are not enforced at this layer).
        """
        specs: List[dict] = []
        for plugin_name, plugin in self.plugins.items():
            for method_name in self._allowed_methods(plugin):
                method = getattr(plugin, method_name, None)
                if method is None:
                    continue
                raw_doc = inspect.getdoc(method) or ""
                description = (
                    raw_doc.splitlines()[0]
                    if raw_doc
                    else f"{method_name} from plugin {plugin_name}"
                )
                properties: dict = {}
                required: List[str] = []
                try:
                    sig = inspect.signature(method)
                    for param_name, param in sig.parameters.items():
                        if param_name == "self":
                            continue
                        properties[param_name] = {"type": "string"}
                        if param.default is inspect.Parameter.empty:
                            required.append(param_name)
                except (ValueError, TypeError):
                    pass
                specs.append({
                    "name": f"{plugin_name}__{method_name}",
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                })
        return specs

    def _execute_tool_call(self, tc: ToolCall) -> str:
        """Execute one native tool call and return a string result."""
        try:
            plugin_name, method_name = tc.name.split("__", 1)
        except ValueError:
            return f"Error: tool name '{tc.name}' must be in 'plugin__method' format"

        plugin = self.plugins.get(plugin_name)
        if plugin is None:
            return f"Error: plugin '{plugin_name}' not loaded"

        error = self._validate_plugin_call(plugin, plugin_name, method_name, tc.args)
        if error:
            logger.warning(
                "Blocked native tool call %s.%s: %s", plugin_name, method_name, error
            )
            return f"Error: {error}"

        try:
            result = getattr(plugin, method_name)(**tc.args)
            return str(result)
        except Exception as exc:
            logger.warning(
                "Native tool call %s.%s raised: %s", plugin_name, method_name, exc
            )
            return f"Error: {exc}"

    def execute_with_native_tools(
        self, task: str, context: Optional[Dict], system: str
    ) -> str:
        """Drive the native tool-calling loop (up to ``_MAX_TOOL_ROUNDS`` rounds).

        Builds the initial user message without a TOOL: prompt section,
        then alternates between ``chat_with_tools`` calls and plugin
        executions.  After the maximum rounds a final call with an empty
        tools list forces a plain-text answer.
        """
        tool_specs = self._build_tool_specs()
        tools = self.maki.to_tool_schemas(tool_specs)

        # Build the initial user message without the TOOL: prompt section.
        parts = [task]
        if context:
            parts.append(f"Context: {json.dumps(context)}")
        # Include stateful history if the host agent exposes it.
        history_section = getattr(self, "_build_history_section", lambda: "")()
        if history_section:
            parts.append(history_section)
        user_content = "\n\n".join(parts)
        messages = [{"role": "user", "content": user_content}]

        for _ in range(_MAX_TOOL_ROUNDS):
            response, tool_calls, messages = self.maki.chat_with_tools(
                messages, tools, system=system
            )
            if tool_calls is None:
                return response.content
            results = [(tc, self._execute_tool_call(tc)) for tc in tool_calls]
            messages = self.maki.append_tool_results(messages, results)

        # Max rounds reached — force a text answer by omitting tools.
        response, _, _ = self.maki.chat_with_tools(messages, [], system=system)
        return response.content if response else ""

    # ------------------------------------------------------------------
    # Legacy TOOL: directive path (fallback for non-native backends)
    # ------------------------------------------------------------------

    def handle_plugin_calls(self, llm_response: str, task: str,
                            context: Optional[Dict]) -> str:
        """
        Parse TOOL: directives from the LLM response, execute them, and
        synthesise a final answer incorporating the results.

        Loops up to ``_MAX_TOOL_ROUNDS`` so the model can chain multiple
        tool calls.  Multi-line and pretty-printed JSON is supported via
        ``json.JSONDecoder.raw_decode``.  Parse failures are included in the
        synthesis prompt so the model can self-correct its output format.

        The expected format emitted by the LLM is::

            TOOL: {"plugin": "<name>", "method": "<method>", "args": {...}}
        """
        for _ in range(_MAX_TOOL_ROUNDS):
            extractions = _extract_tool_calls(llm_response)
            if not extractions:
                return llm_response

            tool_results = []
            for success, raw_match, value in extractions:
                if not success:
                    # Parse failure — feed the error back so the model can fix it.
                    tool_results.append({"raw": raw_match, "error": value})
                    continue

                call = value  # already a dict
                plugin_name = call.get("plugin")
                method_name = call.get("method")
                args = call.get("args", {})

                if not isinstance(plugin_name, str) or not plugin_name:
                    tool_results.append({"error": "Tool call missing 'plugin' name"})
                    continue
                if not isinstance(method_name, str) or not method_name:
                    tool_results.append({
                        "plugin": plugin_name,
                        "error": "Tool call missing 'method' name",
                    })
                    continue

                plugin = self.plugins.get(plugin_name)
                if not plugin:
                    tool_results.append({
                        "plugin": plugin_name,
                        "method": method_name,
                        "error": f"Plugin '{plugin_name}' not loaded",
                    })
                    continue

                validation_error = self._validate_plugin_call(
                    plugin, plugin_name, method_name, args
                )
                if validation_error:
                    logger.warning(
                        f"Blocked plugin call {plugin_name}.{method_name}: "
                        f"{validation_error}"
                    )
                    tool_results.append({
                        "plugin": plugin_name,
                        "method": method_name,
                        "error": validation_error,
                    })
                elif hasattr(plugin, method_name):
                    try:
                        output = getattr(plugin, method_name)(**args)
                        tool_results.append({
                            "plugin": plugin_name,
                            "method": method_name,
                            "result": str(output),
                        })
                    except Exception as exc:
                        logger.warning(
                            f"Plugin call {plugin_name}.{method_name} failed: {exc}",
                            exc_info=True,
                        )
                        tool_results.append({
                            "plugin": plugin_name,
                            "method": method_name,
                            "error": str(exc),
                        })
                else:
                    tool_results.append({
                        "plugin": plugin_name,
                        "method": method_name,
                        "error": (
                            f"Method '{method_name}' not found on plugin '{plugin_name}'"
                        ),
                    })

            clean_response = _strip_tool_calls(llm_response, extractions)
            role = getattr(self, "role", "")
            instructions = getattr(self, "instructions", "")
            agent_context = (
                f"You are a {role}.\n{instructions}\n\n"
                if role
                else (f"{instructions}\n\n" if instructions else "")
            )
            follow_up = (
                f"{agent_context}"
                f"Task: {task}\n\n"
                f"Tool results:\n{json.dumps(tool_results, indent=2)}\n\n"
                f"Previous partial response: {clean_response}\n\n"
                f"Please continue. If you need more tool calls, emit TOOL: directives. "
                f"Otherwise, provide your final answer."
            )
            llm_response = self._call_llm(follow_up)

        return llm_response
