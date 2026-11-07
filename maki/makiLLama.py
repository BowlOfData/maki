"""
A production-grade Python wrapper for local LLMs served via Ollama.
Supports: Gemma 3, Qwen, Llama, Mistral, Phi, DeepSeek and any other
model available through `ollama pull <model>`.

Quick-start:
    # 1. Install Ollama  →  https://ollama.com
    # 2. Pull a model    →  ollama pull gemma3
    # 3. Use it          →  from maki import MakiLLama
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import replace
from typing import Generator, Optional, Union


from .backend import LLMBackend
from .config import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_TEMPERATURE,
)
from .connector import AsyncConnector, Connector
from .utils import Utils
from .objects import LLMResponse, Message, GenerationConfig, RateLimiter, BackendType, ToolCall
from .session import ChatSession
from .context import (DEFAULT_NUM_CTX_CAP, ContextAware, ContextInfo, choose_num_ctx, estimate_tokens,
                      messages_chars)
from .exceptions import MakiNetworkError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

_PROBE_TIMEOUT = 5  # seconds: a probe must not hold up a request the way a generation may

# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------


class MakiLLama(ContextAware, LLMBackend):
    """
    A flexible wrapper around the Ollama HTTP API.

    Supports native tool calling via Ollama's ``tools=`` parameter.
    Set ``agent.use_plugins=True`` and the agent layer will prefer the
    native path over the legacy ``TOOL:`` directive fallback.

    Usage
    -----
        llm = LocalLLM(model="gemma3")
        response = llm.chat("What is the capital of France?")
        response.print()

        # Streaming
        for chunk in llm.stream("Tell me a joke"):
            print(chunk, end="", flush=True)

        # Multi-turn conversation
        session = llm.session(system="You are a senior Python engineer.")
        session.say("Explain list comprehensions.")
        session.say("Now show me a real-world example.")
    """

    OLLAMA_BASE_URL = DEFAULT_OLLAMA_BASE_URL
    supports_native_tools: bool = True
    supports_history: bool = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        rate_limit: Optional[int] = None,
        think: Optional[bool] = None,
        json_format: bool = False,
        keep_alive: Optional[str] = None,
        num_ctx: Union[int, str, None] = None,
        num_ctx_cap: int = DEFAULT_NUM_CTX_CAP,
        on_truncation: str = "warn",
    ) -> None:
        self.model = model
        self.temperature = config.temperature if config else DEFAULT_TEMPERATURE
        self._rate_limiter = RateLimiter(rate_limit) if rate_limit is not None else None
        self.base_url = base_url.rstrip("/")
        self.config = config or GenerationConfig()
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.think = think
        self.json_format = json_format
        self.keep_alive = keep_alive
        self._init_context(num_ctx, num_ctx_cap, on_truncation)
        # Operator-configured endpoint: private/LAN addresses are legitimate
        # (loopback Ollama is the common case), so allow_private=True.
        self._http = Connector(timeout=timeout, allow_private=True)
        self._async_http = AsyncConnector(timeout=timeout, allow_private=True)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP session and release connections."""
        self._http.close()

    def __del__(self) -> None:
        try:
            http = getattr(self, "_http", None)
            if http is not None:
                http.close()
        except Exception:
            pass

    def verify(self) -> None:
        """Ping the Ollama daemon and log model availability.

        Call this explicitly after construction if you need to confirm the
        server is reachable before the first inference call.  The constructor
        no longer blocks on this so that ``maki serve`` can start without
        Ollama being up.

        Raises:
            MakiNetworkError: if the Ollama daemon is unreachable or times out.
        """
        try:
            r = self._http.get(f"{self.base_url}/api/tags", timeout=5)
            available = [m["name"] for m in r.json().get("models", [])]
            log.debug("Available models: %s", available)
            if not any(self.model in m for m in available):
                log.warning(
                    "Model '%s' not found locally. Run: ollama pull %s",
                    self.model, self.model,
                )
            else:
                log.info("Connected to Ollama · model=%s", self.model)
        except MakiNetworkError as e:
            log.error("Cannot reach Ollama at %s: %s", self.base_url, e)
            raise

    # Backward-compat alias kept so any callers of the private method still work.
    _verify_connection = verify

    def list_models(self) -> list[str]:
        """Return names of all locally pulled models."""
        r = self._http.get(f"{self.base_url}/api/tags", timeout=10)
        return [m["name"] for m in r.json().get("models", [])]

    def pull(self, model: Optional[str] = None) -> None:
        """Pull a model from the Ollama registry (blocking, shows progress)."""
        target = model or self.model
        log.info("Pulling model '%s' …", target)
        response = None
        try:
            response = self._http.post(
                f"{self.base_url}/api/pull",
                json={"name": target},
                stream=True,
                timeout=600,
            )
            last_pct = -10
            for line in Connector.iter_lines(response):
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if data.get("total"):
                        pct = int(data.get("completed", 0) / data["total"] * 100)
                        if pct < last_pct:  # a new layer started downloading
                            last_pct = -10
                        if pct >= last_pct + 10:
                            log.info("  %s [%d%%]", status, pct)
                            last_pct = pct
                    else:
                        log.info("  %s", status)
                        last_pct = -10
        except Exception as e:
            log.error("Failed to pull model '%s': %s", target, str(e))
            raise
        finally:
            Utils.cleanup_response(response)
        log.info("Model '%s' ready.", target)

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Context window (see maki.context)
    # ------------------------------------------------------------------

    def _init_context(self, num_ctx: Union[int, str, None], cap: int, on_truncation: str) -> None:
        if num_ctx is not None and num_ctx != "auto" and (
                isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or num_ctx < 1):
            raise ValueError("num_ctx must be a positive integer, 'auto' or None")
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError("num_ctx_cap must be a positive integer")
        self._init_size_check(on_truncation)
        self.num_ctx_cap = cap
        self._auto_num_ctx = num_ctx == "auto"
        self._resolved_num_ctx: Optional[int] = None  # chosen once, then kept (a change reloads the model)
        self._context_hint = 0
        self._max_context: Optional[int] = None
        if isinstance(num_ctx, int):
            self.config = replace(self.config, num_ctx=num_ctx)

    def hint_context(self, tokens: int) -> None:
        """Tell an ``num_ctx="auto"`` backend how many prompt tokens its user expects to need.

        :class:`~maki.agents.agent.Agent` calls this with its history budget. It costs nothing: the window is
        chosen at the first request. Without a hint the first request's own size decides.
        """
        self._context_hint = max(self._context_hint, int(tokens))

    def context_info(self) -> ContextInfo:
        """The model's maximum context (``/api/show``) and the window it is loaded with (``/api/ps``).

        Either value is None when Ollama does not report it or cannot be reached.
        """
        max_length = loaded = None
        try:
            shown = Connector.json_or_raise(self._http.post(
                f"{self.base_url}/api/show", json={"model": self.model, "name": self.model}, timeout=_PROBE_TIMEOUT))
            sizes = [v for k, v in (shown.get("model_info") or {}).items()
                     if k.endswith(".context_length") and isinstance(v, int)]
            max_length = max(sizes) if sizes else None
        except Exception as exc:  # a probe must never break a request
            log.debug("context probe (/api/show) failed: %s", exc)
        try:
            running = Connector.json_or_raise(self._http.get(f"{self.base_url}/api/ps", timeout=_PROBE_TIMEOUT))
            for entry in running.get("models", []):
                if entry.get("name") in (self.model, f"{self.model}:latest") or entry.get("model") == self.model:
                    loaded = entry.get("context_length") if isinstance(entry.get("context_length"), int) else None
        except Exception as exc:
            log.debug("context probe (/api/ps) failed: %s", exc)
        if max_length:
            self._max_context = max_length
        return ContextInfo(max_length, loaded, "ollama")

    def _learn_window(self) -> Optional[int]:
        """The window the model is loaded with (``/api/ps``), remembered once found."""
        known = getattr(self, "_loaded_window", None)
        if known:
            return known
        try:
            running = Connector.json_or_raise(self._http.get(f"{self.base_url}/api/ps", timeout=_PROBE_TIMEOUT))
        except Exception as exc:  # the check just stays silent
            log.debug("context probe (/api/ps) failed: %s", exc)
            return None
        for entry in running.get("models", []):
            if entry.get("name") in (self.model, f"{self.model}:latest") or entry.get("model") == self.model:
                length = entry.get("context_length")
                if isinstance(length, int) and length > 0:
                    self._loaded_window = length
                    return length
        return None

    def _prepare_context(self, messages: list, options: dict) -> None:
        """Fix ``num_ctx`` in *options* for ``"auto"`` backends and note the request for truncation checks."""
        if getattr(self, "_auto_num_ctx", False) and "num_ctx" not in options:
            if getattr(self, "_resolved_num_ctx", None) is None:
                info = self.context_info()
                need = max(getattr(self, "_context_hint", 0), estimate_tokens(messages_chars(messages)))
                need += options.get("num_predict") or 0
                self._resolved_num_ctx = choose_num_ctx(need, model_max=info.max_length, loaded=info.loaded_length,
                                                        cap=getattr(self, "num_ctx_cap", DEFAULT_NUM_CTX_CAP))
                log.info("num_ctx fixed at %d for %s (need about %d tokens)", self._resolved_num_ctx, self.model, need)
            options["num_ctx"] = self._resolved_num_ctx
        self._note_request(messages, options.get("num_ctx") or getattr(self, "_max_context", None))

    def _build_messages(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        system: Optional[str] = None,
        images: Optional[list[str]] = None,
    ) -> list[dict]:
        msgs: list[dict] = []
        effective_system = system if system is not None else self.system_prompt
        if effective_system:
            msgs.append(Message("system", effective_system).to_dict())
        if history:
            # Plain chat sends the text of each turn; a remembered tool trace is only replayed by chat_turn().
            msgs.extend(Message(m.role, m.content, m.images).to_dict() for m in history)
        msgs.append(Message("user", prompt, images=images).to_dict())
        return msgs

    def _build_payload(
        self,
        prompt: str,
        history: Optional[list[Message]],
        config: Optional[GenerationConfig],
        *,
        stream: bool,
        system: Optional[str] = None,
        images: Optional[list[str]] = None,
        response_format: Optional[Union[str, dict]] = None,
        think: Optional[bool] = None,
    ) -> dict:
        cfg = config or self.config
        payload: dict = {
            "model": self.model,
            "messages": self._build_messages(prompt, history, system=system, images=images),
            "stream": stream,
            "options": cfg.to_ollama_options(),
        }
        self._prepare_context(payload["messages"], payload["options"])
        effective_think = think if think is not None else self.think
        if effective_think is not None:
            payload["think"] = effective_think
        if response_format is not None:
            # "json" or a JSON-schema dict: Ollama constrains decoding to it.
            payload["format"] = response_format
        elif self.json_format:
            payload["format"] = "json"
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload

    def _parse_response(self, data: dict, elapsed: float, *, thinking_fallback: bool = True) -> LLMResponse:
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        self._report_prompt_size(prompt_tokens)
        content = data["message"]["content"]
        # Thinking models (gemma4, qwen3, …) may leave content empty when the model
        # exhausted its budget on the reasoning trace. Fall back to the thinking field
        # so callers always receive something extractable.
        if thinking_fallback and not content.strip():
            thinking = data["message"].get("thinking", "")
            if thinking:
                log.debug("_parse_response: content empty, falling back to thinking field")
                content = thinking
        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            elapsed_seconds=elapsed,
            done=data.get("done", True),
            backend=BackendType.OLLAMA,
        )

    def chat(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
        images: Optional[list[str]] = None,
        response_format: Optional[Union[str, dict]] = None,
        think: Optional[bool] = None,
    ) -> LLMResponse:
        """
        Single-turn (or multi-turn with explicit history) generation.
        Returns a fully resolved LLMResponse. Pass base64 strings in images for vision models.

        ``response_format`` is ``"json"`` or a JSON-schema dict (Ollama structured
        outputs). ``think`` overrides the instance-level setting for this call.
        See :func:`maki.structured.generate_structured` for the validated wrapper.
        """
        log.debug("chat: %s", prompt[:100])
        if self._rate_limiter:
            self._rate_limiter.acquire()
        payload = self._build_payload(prompt, history, config, stream=False, system=system, images=images,
                                      response_format=response_format, think=think)
        t0 = time.perf_counter()
        r = self._http.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        elapsed = time.perf_counter() - t0
        response = self._parse_response(Connector.json_or_raise(r), elapsed,
                                        thinking_fallback=response_format is None)
        log.info("chat: %.2fs, %d tokens", elapsed, response.total_tokens)
        return response

    def stream(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
        images: Optional[list[str]] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming generation — yields text chunks as they arrive.

        Example:
            for chunk in llm.stream("Write a haiku about Python"):
                print(chunk, end="", flush=True)
        """
        log.debug("stream: %s", prompt[:100])
        if self._rate_limiter:
            self._rate_limiter.acquire()
        payload = self._build_payload(prompt, history, config, stream=True, system=system, images=images)
        response = None
        try:
            response = self._http.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            for line in Connector.iter_lines(response):
                if line:
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
        except Exception as e:
            log.error("stream failed: %s", e)
            raise
        finally:
            Utils.cleanup_response(response)

    def chat_collect(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
        images: Optional[list[str]] = None,
        response_format: Optional[Union[str, dict]] = None,
    ) -> LLMResponse:
        """
        Like chat() but uses HTTP streaming internally.

        The configured timeout applies per-chunk rather than to the total response,
        so long-running generations (e.g. large ranking tasks) complete without
        hitting the global read timeout.  Raises MakiTimeoutError only if the
        model stops producing output for longer than self.timeout seconds.

        ``response_format`` is ``"json"`` or a JSON-schema dict (Ollama structured
        outputs), as in ``chat()``; a structured reply never falls back to the
        thinking trace. Other backends' ``chat_collect`` do not accept it.
        """
        log.debug("chat_collect (stream): %s", prompt[:100])
        if self._rate_limiter:
            self._rate_limiter.acquire()
        payload = self._build_payload(prompt, history, config, stream=True, system=system, images=images,
                                     response_format=response_format)
        t0 = time.perf_counter()
        chunks: list[str] = []
        last_data: dict = {}
        response = None
        try:
            # The configured timeout applies per-chunk on the streaming read,
            # so long generations survive as long as chunks keep arriving.
            response = self._http.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            for line in Connector.iter_lines(response):
                if line:
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        chunks.append(chunk)
                    if data.get("done"):
                        last_data = data
                        break
        finally:
            Utils.cleanup_response(response)

        elapsed = time.perf_counter() - t0
        content = "".join(chunks)
        if response_format is None and not content.strip():
            thinking = last_data.get("message", {}).get("thinking", "")
            if thinking:
                log.debug("chat_collect: content empty, falling back to thinking field")
                content = thinking
        prompt_tokens = last_data.get("prompt_eval_count", 0)
        completion_tokens = last_data.get("eval_count", 0)
        self._report_prompt_size(prompt_tokens)
        log.info("chat_collect: %.2fs, %d tokens", elapsed, prompt_tokens + completion_tokens)
        return LLMResponse(
            content=content,
            model=last_data.get("model", self.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            elapsed_seconds=elapsed,
            done=last_data.get("done", True),
        )

    async def async_chat(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
        images: Optional[list[str]] = None,
        system: Optional[str] = None,
        response_format: Optional[Union[str, dict]] = None,
        think: Optional[bool] = None,
    ) -> LLMResponse:
        """Async variant of chat() for use inside asyncio event loops. Supports vision via images."""
        log.debug("async_chat: %s", prompt[:100])
        if self._rate_limiter:
            await self._rate_limiter.async_acquire()
        payload = self._build_payload(prompt, history, config, stream=False, images=images, system=system,
                                      response_format=response_format, think=think)
        t0 = time.perf_counter()
        r = await self._async_http.post(f"{self.base_url}/api/chat", json=payload)
        elapsed = time.perf_counter() - t0
        response = self._parse_response(Connector.json_or_raise(r), elapsed,
                                        thinking_fallback=response_format is None)
        log.info("async_chat: %.2fs, %d tokens", elapsed, response.total_tokens)
        return response

    def chat_with_image(
        self,
        prompt: str,
        image_b64: str,
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
    ) -> LLMResponse:
        """Single-turn vision chat with a base64-encoded image."""
        return self.chat(prompt, config=config, system=system, images=[image_b64])

    def request(self, prompt: str) -> LLMResponse:
        """Override base request() to route through the chat API."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
        return self.chat(prompt)

    def session(self, system: Optional[str] = None) -> "ChatSession":
        """Create a stateful multi-turn chat session."""
        return ChatSession(llm=self, system=system)

    # ------------------------------------------------------------------
    # Convenience shortcuts
    # ------------------------------------------------------------------

    def __call__(self, prompt: str, **kwargs) -> LLMResponse:
        """llm("your prompt") as a shorthand for llm.chat(...)."""
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")

        # Pass through every kwarg chat() accepts; warn on anything else.
        allowed_params = {'history', 'config', 'system', 'images'}
        filtered_kwargs = {}

        for key, value in kwargs.items():
            if key in allowed_params:
                filtered_kwargs[key] = value
            else:
                log.warning(f"Disallowed parameter '{key}' ignored in __call__")

        log.debug("Calling LLM with prompt: %s", prompt[:100] + "..." if len(prompt) > 100 else prompt)
        return self.chat(prompt, **filtered_kwargs)

    def __repr__(self) -> str:
        return f"LocalLLM(model={self.model!r}, base_url={self.base_url!r})"

    # ------------------------------------------------------------------
    # Native tool-calling (Ollama tools= API)
    # ------------------------------------------------------------------

    def to_tool_schemas(self, tool_specs: list) -> list:
        """Translate backend-agnostic specs to the Ollama/OpenAI-compatible format."""
        schemas = []
        for spec in tool_specs:
            schemas.append({
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            })
        return schemas

    supports_typed_messages = True

    def to_wire_messages(self, messages: list) -> list:
        """Typed messages as Ollama chat messages (tool calls carry their arguments as an object)."""
        wire = []
        for m in messages:
            entry: dict = {"role": m.role, "content": m.content}
            if m.images:
                entry["images"] = m.images
            if m.tool_calls:
                entry["tool_calls"] = [{"function": {"name": tc.name, "arguments": tc.args}} for tc in m.tool_calls]
            if m.role == "tool" and m.name:
                entry["tool_name"] = m.name
            wire.append(entry)
        return wire

    def chat_with_tools(
        self,
        messages: list,
        tools: list,
        *,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ):
        """One round of Ollama native tool-calling.

        Returns ``(LLMResponse, None, messages)`` on a text answer, or
        ``(None, [ToolCall, ...], messages)`` when tool calls are requested.
        """
        response, tool_calls, updated = self._chat_with_tools_full(messages, tools, system=system, config=config)
        return (None, tool_calls, updated) if tool_calls else (response, None, updated)

    @staticmethod
    def _tool_calls_from(raw_tool_calls: list) -> list:
        """Ollama tool calls (arguments as an object or a JSON string) as :class:`ToolCall` objects."""
        tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(id="", name=fn.get("name", ""), args=args))
        return tool_calls

    def _chat_with_tools_full(
        self,
        messages: list,
        tools: list,
        *,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        tool_choice: Optional[str] = None,
    ):
        """:meth:`chat_with_tools` with a real response (token counts included) on tool rounds too."""
        import time as _t
        if tool_choice == "none":
            tools = []  # Ollama takes tool messages without tool definitions
        if self._rate_limiter:
            self._rate_limiter.acquire()
        cfg = config or self.config
        effective_system = system if system is not None else self.system_prompt

        # Build the full messages list (system first, then the conversation).
        full_messages = []
        if effective_system:
            full_messages.append({"role": "system", "content": effective_system})
        full_messages.extend(messages)

        payload: dict = {
            "model": self.model,
            "messages": full_messages,
            "stream": False,
            "options": cfg.to_ollama_options(),
        }
        self._prepare_context(full_messages, payload["options"])
        if tools:
            payload["tools"] = tools
        if self.think is not None:
            payload["think"] = self.think

        t0 = _t.perf_counter()
        r = self._http.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        elapsed = _t.perf_counter() - t0
        data = Connector.json_or_raise(r)

        msg = data.get("message", {})
        raw_tool_calls = msg.get("tool_calls")

        # Append the assistant turn to the conversation.
        updated = list(messages) + [msg]

        if raw_tool_calls:
            tool_calls = self._tool_calls_from(raw_tool_calls)
            log.debug("chat_with_tools: %d tool call(s) requested", len(tool_calls))
            # A tool round's text is not an answer: keep any thinking out of it.
            return self._parse_response(data, elapsed, thinking_fallback=False), tool_calls, updated

        response = self._parse_response(data, elapsed)
        log.info("chat_with_tools (text): %.2fs", elapsed)
        return response, None, updated

    def append_tool_results(self, messages: list, results: list) -> list:
        """Append Ollama-format tool result messages."""
        updated = list(messages)
        for _tc, result_str in results:
            updated.append({"role": "tool", "content": result_str})
        return updated


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def gemma3(system: Optional[str] = None, **kwargs) -> MakiLLama:
    """Pre-configured wrapper for Google Gemma 3."""
    return MakiLLama(model="gemma3", system_prompt=system, **kwargs)


def gemma4(variant: str = "gemma4:26b", system: Optional[str] = None, **kwargs) -> MakiLLama:
    """Pre-configured wrapper for Google Gemma 4 (vision-capable)."""
    return MakiLLama(model=variant, system_prompt=system, **kwargs)


def qwen(variant: str = "qwen2.5:7b", system: Optional[str] = None, **kwargs) -> MakiLLama:
    """Pre-configured wrapper for Alibaba Qwen."""
    return MakiLLama(model=variant, system_prompt=system, **kwargs)


def llama(variant: str = "llama3.2", system: Optional[str] = None, **kwargs) -> MakiLLama:
    """Pre-configured wrapper for Meta Llama 3."""
    return MakiLLama(model=variant, system_prompt=system, **kwargs)


def mistral(system: Optional[str] = None, **kwargs) -> MakiLLama:
    """Pre-configured wrapper for Mistral."""
    return MakiLLama(model="mistral", system_prompt=system, **kwargs)
