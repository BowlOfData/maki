"""
Load a Maki Agent from a YAML configuration file.

YAML schema
-----------
name: researcher            # required
role: Research specialist   # optional
instructions: |             # optional
  You are a research agent...
backend: ollama             # ollama | openai | anthropic | openrouter | llamacpp | vllm
                            # (default: ollama)
model: llama3.2             # optional; falls back to each backend's default
base_url: http://127.0.0.1:8080/v1  # optional; llamacpp / vllm only
temperature: 0.7            # optional, default 0.7
think: false                # optional, ollama only: false turns model thinking off (a thinking
                            # model can leak its reasoning into the reply, which pollutes loop artifacts)
stateful: false             # optional, default false
use_streaming: false        # optional, default false
allow_dangerous_tools: false  # optional, default false; opt-in to let TOOL:
                              # directives call methods a plugin marks in
                              # DANGEROUS_METHODS (file writes, FTP, trades)
plugins:                    # optional list of built-in plugin names; must be
  - web_to_md               # registered in maki.plugins.PLUGIN_REGISTRY
  - file_reader
judge:                      # optional; enables POST /loop. A separate, stateless
  backend: ollama           # reviewer agent that judges loop requirements. Keys:
  model: gemma4:26b         # backend, model, base_url, temperature, instructions.
                            # (ollama: JSON mode on, thinking off.)  Without it /loop
                            # answers 503: the served agent never judges its own work.
loop:                       # optional caps for remote loops
  max_iterations: 20        # default 20
  max_seconds: 3600         # default 3600
  max_tokens: 200000        # optional cap on a loop's token budget (maker + judge); default none
  loop_dir_base: ~/.maki/loops
  retention_days: 14        # finished loops older than this are deleted (null = keep all)

Install optional dependencies before use:
    pip install "maki-framework[distributed]"
"""
try:
    import yaml
except ImportError as _e:
    raise ImportError(
        "Config loading requires PyYAML. "
        'Install it with: pip install "maki-framework[distributed]"'
    ) from _e

import os
from dataclasses import dataclass
from typing import Optional

from ..agents.agent import Agent
from ..backend import LLMBackend
from ..objects import GenerationConfig
from ..plugins import PLUGIN_REGISTRY
from .loop_jobs import LoopCaps


def _build_backend(cfg: dict, judge: bool = False) -> LLMBackend:
    backend_name = cfg.get("backend", "ollama").lower()
    model: Optional[str] = cfg.get("model")
    temperature = float(cfg.get("temperature", 0.7))
    gen_config = GenerationConfig(temperature=temperature)

    if backend_name == "ollama":
        from ..makiLLama import MakiLLama
        kwargs: dict = {"config": gen_config}
        if model:
            kwargs["model"] = model
        if "think" in cfg:  # ollama only: False stops reasoning text leaking into replies
            kwargs["think"] = bool(cfg["think"])
        if judge:  # a thinking judge often returns truncated or reasoning-polluted verdicts
            kwargs.update(json_format=True, think=False)
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

    if backend_name == "openrouter":
        from ..makiOpenRouter import MakiOpenRouter
        kwargs = {"config": gen_config}
        if model:
            kwargs["model"] = model
        return MakiOpenRouter(**kwargs)

    if backend_name == "llamacpp":
        from ..makiLlamaCpp import MakiLlamaCpp
        kwargs = {"config": gen_config}
        if model:
            kwargs["model"] = model
        if cfg.get("base_url"):
            kwargs["base_url"] = cfg["base_url"]
        return MakiLlamaCpp(**kwargs)

    if backend_name == "vllm":
        from ..makiVllm import MakiVllm
        kwargs = {"config": gen_config}
        if model:
            kwargs["model"] = model
        if cfg.get("base_url"):
            kwargs["base_url"] = cfg["base_url"]
        return MakiVllm(**kwargs)

    raise ValueError(
        f"Unknown backend: '{backend_name}'. "
        "Supported values: ollama, openai, anthropic, openrouter, llamacpp, vllm"
    )


def _read_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Agent config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not cfg or not cfg.get("name"):
        raise ValueError("Agent config must include a 'name' field")
    return cfg


def _agent_from_cfg(cfg: dict) -> Agent:
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


def load_agent_from_config(path: str) -> Agent:
    """
    Parse *path* (YAML) and return a fully configured Agent instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required fields are missing or backend is unknown.
    """
    return _agent_from_cfg(_read_config(path))


_JUDGE_KEYS = {"backend", "model", "base_url", "temperature", "instructions"}  # think/json are forced
_LOOP_KEYS = {"max_iterations", "max_seconds", "max_tokens", "loop_dir_base", "retention_days"}


def _judge_from_cfg(agent_name: str, section: dict) -> Agent:
    if not isinstance(section, dict):
        raise ValueError("'judge' must be a mapping")
    unknown = sorted(set(section) - _JUDGE_KEYS)
    if unknown:
        raise ValueError(f"Unknown key(s) in 'judge': {', '.join(unknown)}. "
                         f"Valid keys: {', '.join(sorted(_JUDGE_KEYS))} (a judge has no plugins or tools)")
    return Agent(name=f"{agent_name}-judge", maki_instance=_build_backend(section, judge=True), role="reviewer",
                 instructions=section.get("instructions", ""), stateful=False)


def _caps_from_cfg(section: dict) -> LoopCaps:
    if not isinstance(section, dict):
        raise ValueError("'loop' must be a mapping")
    unknown = sorted(set(section) - _LOOP_KEYS)
    if unknown:
        raise ValueError(f"Unknown key(s) in 'loop': {', '.join(unknown)}. Valid keys: {', '.join(sorted(_LOOP_KEYS))}")
    kwargs: dict = {}
    if "max_iterations" in section:
        kwargs["max_iterations"] = section["max_iterations"]
    if "max_seconds" in section:
        kwargs["max_seconds"] = None if section["max_seconds"] is None else float(section["max_seconds"])
    if "max_tokens" in section:
        kwargs["max_tokens"] = section["max_tokens"]
    if "retention_days" in section:
        kwargs["retention_days"] = None if section["retention_days"] is None else float(section["retention_days"])
    if section.get("loop_dir_base"):
        kwargs["loop_dir_base"] = os.path.abspath(os.path.expanduser(str(section["loop_dir_base"])))
    return LoopCaps(**kwargs)


@dataclass
class ServerConfig:
    """Everything ``maki serve`` builds from the YAML: the served agent, the optional loop
    judge (``None`` = ``/loop`` disabled) and the caps for remote loops."""
    agent: Agent
    judge: Optional[Agent]
    loop_caps: LoopCaps


def load_server_config(path: str) -> ServerConfig:
    """Like :func:`load_agent_from_config`, plus the optional ``judge:`` and ``loop:`` sections.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: For missing fields, an unknown backend, or bad ``judge``/``loop`` sections.
    """
    cfg = _read_config(path)
    agent = _agent_from_cfg(cfg)
    judge = _judge_from_cfg(cfg["name"], cfg["judge"]) if cfg.get("judge") is not None else None
    caps = _caps_from_cfg(cfg["loop"]) if cfg.get("loop") is not None else LoopCaps()
    return ServerConfig(agent=agent, judge=judge, loop_caps=caps)
