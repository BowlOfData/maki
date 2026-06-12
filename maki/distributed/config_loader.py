"""
Load a Maki Agent from a YAML configuration file.

YAML schema
-----------
name: researcher            # required
role: Research specialist   # optional
instructions: |             # optional
  You are a research agent...
backend: ollama             # ollama | openai | anthropic  (default: ollama)
model: llama3.2             # optional; falls back to each backend's default
temperature: 0.7            # optional, default 0.7
stateful: false             # optional, default false
use_streaming: false        # optional, default false
allow_dangerous_tools: false  # optional, default false; opt-in to let TOOL:
                              # directives call methods a plugin marks in
                              # DANGEROUS_METHODS (file writes, FTP, trades)
plugins:                    # optional list of built-in plugin names; must be
  - web_to_md               # registered in maki.plugins.PLUGIN_REGISTRY
  - file_reader

Install optional dependencies before use:
    pip install "maki[distributed]"
"""
try:
    import yaml
except ImportError as _e:
    raise ImportError(
        "Config loading requires PyYAML. "
        'Install it with: pip install "maki[distributed]"'
    ) from _e

import os
from typing import Optional

from ..agents.agent import Agent
from ..backend import LLMBackend
from ..objects import GenerationConfig
from ..plugins import PLUGIN_REGISTRY


def _build_backend(cfg: dict) -> LLMBackend:
    backend_name = cfg.get("backend", "ollama").lower()
    model: Optional[str] = cfg.get("model")
    temperature = float(cfg.get("temperature", 0.7))
    gen_config = GenerationConfig(temperature=temperature)

    if backend_name == "ollama":
        from ..makiLLama import MakiLLama
        kwargs: dict = {"config": gen_config}
        if model:
            kwargs["model"] = model
        return MakiLLama(**kwargs)

    if backend_name == "openai":
        from ..makiOpenAI import MakiOpenAI
        kwargs = {"config": gen_config}
        if model:
            kwargs["model"] = model
        return MakiOpenAI(**kwargs)

    if backend_name == "anthropic":
        from ..makiAnthropic import MakiAnthropic
        kwargs = {"config": gen_config}
        if model:
            kwargs["model"] = model
        return MakiAnthropic(**kwargs)

    raise ValueError(
        f"Unknown backend: '{backend_name}'. Supported values: ollama, openai, anthropic"
    )


def load_agent_from_config(path: str) -> Agent:
    """
    Parse *path* (YAML) and return a fully configured Agent instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required fields are missing or backend is unknown.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Agent config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if not cfg or not cfg.get("name"):
        raise ValueError("Agent config must include a 'name' field")

    # Validate plugin names up front (before the backend constructor, which
    # may touch the network): load_plugin() feeds the name into
    # importlib.import_module, so only registry names are acceptable (§2.6).
    plugin_names = cfg.get("plugins") or []
    unknown = sorted(set(plugin_names) - set(PLUGIN_REGISTRY))
    if unknown:
        raise ValueError(
            f"Unknown plugin(s) in config: {', '.join(unknown)}. "
            f"Valid plugins: {', '.join(sorted(PLUGIN_REGISTRY))}"
        )

    backend = _build_backend(cfg)
    agent = Agent(
        name=cfg["name"],
        maki_instance=backend,
        role=cfg.get("role", ""),
        instructions=cfg.get("instructions", ""),
        stateful=bool(cfg.get("stateful", False)),
        use_streaming=bool(cfg.get("use_streaming", False)),
        allow_dangerous_tools=bool(cfg.get("allow_dangerous_tools", False)),
    )

    for plugin_name in plugin_names:
        agent.load_plugin(plugin_name)

    return agent
