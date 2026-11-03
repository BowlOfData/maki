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
model: llama3.2             # optional; falls back to each backend's default. Without a `backend:`
                            # key, "openai:gpt-4o" selects both the provider and the model
base_url: http://127.0.0.1:8080/v1  # optional; llamacpp / vllm only
temperature: 0.7            # optional, default 0.7
think: false                # optional, ollama only: false turns model thinking off (a thinking
                            # model can leak its reasoning into the reply, which pollutes loop artifacts)
stateful: false             # optional, default false
use_streaming: false        # optional, default false
memory:                     # optional; limits on what the agent keeps and sends
  history_mode: messages    # text (default) | messages: prior turns as structured chat history
  token_budget: 4096        # stateful conversation budget, in estimated tokens
  tool_result_chars: 8000   # clip each plugin result to this many characters
  history_entry_chars: 4000 # clip what task_history / reasoning_history store
  pinned_chars: 400         # room for facts pinned with agent.remember(..., pin=True)
  recent_turns: 3           # with older_turn_chars: last N turns verbatim, older ones shortened
  older_turn_chars: 200
  summarize_older: true     # fold evicted turns into a running summary (one extra model call)
  summary_chars: 800        # cap on that summary
tool_policy:                # optional; how tool calls run
  max_rounds: 8             # model/tool round trips before a final answer is forced
  retries: 2                # extra attempts for transient tool errors (network, timeout)
  parallel: false           # run one turn's tool calls concurrently (plugins must be thread-safe)
fallback:                   # optional; backends tried in order when the main one fails
  - openai:gpt-4o          # (network error, timeout, 408/429), as "provider:model" specs
  - ollama:gemma4:26b
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

import importlib
import os
from dataclasses import dataclass
from typing import Optional

from ..agents.agent import Agent
from ..agents.tool_policy import ToolPolicy
from ..backend import LLMBackend
from ..backend_factory import BACKENDS
from ..objects import GenerationConfig, MemoryPolicy
from ..plugins import PLUGIN_REGISTRY
from .loop_jobs import LoopCaps


_BACKENDS = BACKENDS
_BASE_URL_BACKENDS = ("llamacpp", "vllm")


def _ollama_kwargs(cfg: dict, judge: bool) -> dict:
    kwargs: dict = {}
    if "think" in cfg:  # False stops reasoning text leaking into replies
        kwargs["think"] = bool(cfg["think"])
    if judge:  # a thinking judge often returns truncated or reasoning-polluted verdicts
        kwargs.update(json_format=True, think=False)
    return kwargs


def _build_fallback(primary: LLMBackend, specs) -> LLMBackend:
    """Wrap *primary* so the ``fallback:`` specs are tried when it fails; no section means no wrapper."""
    if specs is None:
        return primary
    if not isinstance(specs, list) or not specs or not all(isinstance(s, str) and s.strip() for s in specs):
        raise ValueError("'fallback' must be a non-empty list of 'provider:model' strings")
    from ..fallback import FallbackBackend
    return FallbackBackend([primary, *specs])


def _build_backend(cfg: dict, judge: bool = False) -> LLMBackend:
    backend_name = cfg.get("backend", "ollama").lower()
    model = cfg.get("model")
    provider, _, rest = str(model or "").partition(":")
    if "backend" not in cfg and provider.lower() in _BACKENDS and rest:
        backend_name, model = provider.lower(), rest  # model: "openai:gpt-4o" names both
    if backend_name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend: '{backend_name}'. "
            "Supported values: ollama, openai, anthropic, openrouter, llamacpp, vllm"
        )
    module_name, class_name = _BACKENDS[backend_name]
    backend_class = getattr(importlib.import_module(f"..{module_name}", __package__), class_name)

    kwargs: dict = {"config": GenerationConfig(temperature=float(cfg.get("temperature", 0.7)))}
    if model:
        kwargs["model"] = model
    if backend_name in _BASE_URL_BACKENDS and cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    if backend_name == "ollama":
        kwargs.update(_ollama_kwargs(cfg, judge))
    return backend_class(**kwargs)


def _read_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Agent config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not cfg or not cfg.get("name"):
        raise ValueError("Agent config must include a 'name' field")
    return cfg


def _memory_from_cfg(section) -> Optional[MemoryPolicy]:
    if section is None:
        return None
    try:
        return MemoryPolicy.from_dict(section)
    except ValueError as exc:
        raise ValueError(f"Invalid 'memory' section: {exc}") from exc


def _tool_policy_from_cfg(section) -> Optional[ToolPolicy]:
    if section is None:
        return None
    try:
        return ToolPolicy.from_dict(section)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid 'tool_policy' section: {exc}") from exc


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

    backend = _build_fallback(_build_backend(cfg), cfg.get("fallback"))
    agent = Agent(
        name=cfg["name"],
        maki_instance=backend,
        role=cfg.get("role", ""),
        instructions=cfg.get("instructions", ""),
        stateful=bool(cfg.get("stateful", False)),
        use_streaming=bool(cfg.get("use_streaming", False)),
        allow_dangerous_tools=bool(cfg.get("allow_dangerous_tools", False)),
        memory=_memory_from_cfg(cfg.get("memory")),
        tool_policy=_tool_policy_from_cfg(cfg.get("tool_policy")),
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
