"""
Plugin handling for Maki agents.

Provides the PluginHandler mixin that gives agents the ability to load,
manage, and invoke plugins via LLM-emitted TOOL: directives.
"""

import importlib
import importlib.util
import json
import logging
import os
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PluginHandler:
    """Mixin that adds plugin loading and tool-call execution to an agent."""

    def _init_plugins(self):
        """Initialize plugin storage. Call from Agent.__init__."""
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

    def build_plugin_prompt_section(self) -> str:
        """Build the plugin description section for inclusion in a prompt."""
        if not self.plugins:
            return ""
        descriptions = []
        for pname, plugin in self.plugins.items():
            methods = [
                m for m in dir(plugin)
                if not m.startswith('_') and callable(getattr(plugin, m))
            ]
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
