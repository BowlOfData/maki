"""
Plugin handling for Maki agents.

Provides the PluginHandler mixin that gives agents the ability to load,
manage, and invoke plugins via LLM-emitted TOOL: directives.
"""

import functools
import importlib
import importlib.util
import json
import logging
import os
import re
from typing import Dict, Optional, TYPE_CHECKING

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


class PluginHandler:
    """
    Mixin that adds plugin loading and tool-call execution to an agent.

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
            Exception: If plugin initialization fails
        """
        try:
            if plugin_path:
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

        # Respect an explicit whitelist declared on the plugin class.
        # Only treat it as a whitelist if it is an actual collection type,
        # so duck-typed mocks or other non-collection attributes are ignored.
        allowed = getattr(plugin, "ALLOWED_METHODS", None)
        if isinstance(allowed, (list, set, tuple, frozenset)) and method_name not in allowed:
            return (
                f"Method '{method_name}' is not in the allowed methods "
                f"for plugin '{plugin_name}'"
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
        if isinstance(explicit, (list, set, tuple, frozenset)):
            return [m for m in explicit if callable(getattr(plugin, m, None))]
        return [
            m for m in dir(plugin)
            if not m.startswith("_") and callable(getattr(plugin, m))
        ]

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

    def handle_plugin_calls(self, llm_response: str, task: str,
                            context: Optional[Dict]) -> str:
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
                if not plugin:
                    tool_results.append({
                        "plugin": plugin_name,
                        "method": method_name,
                        "error": f"Plugin '{plugin_name}' not loaded"
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
                        "error": validation_error
                    })
                elif hasattr(plugin, method_name):
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
                        "error": f"Method '{method_name}' not found on plugin '{plugin_name}'"
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
        return self.maki.request(follow_up).content
