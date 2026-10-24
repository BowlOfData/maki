# Plan: OpenAI and Anthropic Backend Integration

## Goal

Add two new LLM backend classes — `MakiOpenAI` and `MakiAnthropic` — that sit alongside the existing `MakiLLama` and `HFBackend` backends. Both must inherit `LLMBackend` from `maki/backend.py` and produce `LLMResponse` objects, making them fully interchangeable from the agent layer.

---

## Decisions

| Question | Decision |
|----------|----------|
| Naming | `MakiOpenAI` / `MakiAnthropic` |
| `LLMResponse.backend` type | Become a `BackendType` enum |
| Vision support | Included in this PR |
| `session()` method | Expose it; move `ChatSession` to a shared module so all backends can reuse it |

---

## Inheritance Tree After This Work

```
LLMBackend (maki/backend.py)
├── MakiLLama     (maki/makiLLama.py)
├── HFBackend     (maki/makiHG.py)
├── MakiOpenAI    (maki/makiOpenAI.py)    ← NEW
└── MakiAnthropic (maki/makiAnthropic.py) ← NEW
```

---

## Step 1 — Add `BackendType` enum and update `LLMResponse` (`maki/objects.py`)

Add an enum before the `LLMResponse` dataclass:

```python
from enum import Enum

class BackendType(str, Enum):
    OLLAMA    = "ollama"
    OPENAI    = "openai"
    ANTHROPIC = "anthropic"
    HF        = "huggingface"
```

Using `str, Enum` means `BackendType.OLLAMA == "ollama"` stays truthy, preserving backwards compatibility for any code that compares against the old plain-string value.

Update `LLMResponse`:

```python
@dataclass
class LLMResponse:
    ...
    backend: BackendType = BackendType.OLLAMA   # was: backend: str = "ollama"
```

Also update `MakiLLama._parse_response()` and `HFBackend` to pass `backend=BackendType.OLLAMA` / `BackendType.HF`.

---

## Step 2 — Extend `GenerationConfig` (`maki/objects.py`)

Add two serialisation helpers mirroring `to_ollama_options()` and `to_hf_kwargs()`:

```python
def to_openai_kwargs(self) -> dict:
    kwargs = {
        "temperature": self.temperature,
        "top_p": self.top_p,
        "max_tokens": self.max_tokens,
    }
    if self.seed != -1:
        kwargs["seed"] = self.seed
    if self.stop:
        kwargs["stop"] = self.stop
    return kwargs

def to_anthropic_kwargs(self) -> dict:
    kwargs = {
        "temperature": self.temperature,
        "top_p": self.top_p,
        "max_tokens": self.max_tokens,
    }
    if self.stop:
        kwargs["stop_sequences"] = self.stop
    return kwargs
```

> **Note:** `top_k` and `repeat_penalty` have no mapping in the OpenAI API and are silently dropped. Anthropic supports `top_k` but not `repeat_penalty`; if we want to expose it later, `to_anthropic_kwargs()` can be extended.

---

## Step 3 — Add config defaults (`maki/config.py`)

```python
# OpenAI
OPENAI_API_KEY_ENV      = "OPENAI_API_KEY"
DEFAULT_OPENAI_MODEL    = os.getenv("MAKI_OPENAI_MODEL", "gpt-4o-mini")

# Anthropic
ANTHROPIC_API_KEY_ENV   = "ANTHROPIC_API_KEY"
DEFAULT_ANTHROPIC_MODEL = os.getenv("MAKI_ANTHROPIC_MODEL", "claude-sonnet-4-6")
```

API keys are **never** stored as module-level values; they are resolved from the environment at instantiation time.

---

## Step 4 — Extract `ChatSession` to `maki/session.py`

`ChatSession` currently lives inside `maki/makiLLama.py`. Both new backends need the same stateful multi-turn pattern (all three APIs accept a `messages` array), so move it to a shared module.

Both OpenAI and Anthropic natively accept a full `messages` array per request — there is no provider-level session concept beyond what `ChatSession` already provides. No provider-native alternative is needed.

**Changes:**
- Move `ChatSession` verbatim from `maki/makiLLama.py` to `maki/session.py`
- `MakiLLama.session()` imports and returns `ChatSession` from the new location
- `MakiOpenAI.session()` and `MakiAnthropic.session()` do the same

`ChatSession` depends only on a backend that exposes `chat()` and `stream()` with the standard signature, so no code changes to the class itself are required.

---

## Step 5 — `MakiOpenAI` (`maki/makiOpenAI.py`)

Uses the `openai` Python SDK (`pip install openai`).

### Class signature

```python
class MakiOpenAI(LLMBackend):
    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: Optional[str] = None,       # falls back to OPENAI_API_KEY env var
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        rate_limit: Optional[int] = None,
    ) -> None: ...
```

### Methods

| Method | Notes |
|--------|-------|
| `request(prompt)` | Required by `LLMBackend`. Routes to `chat()`. |
| `chat(prompt, history, config, system, images)` | `openai.chat.completions.create(stream=False)`. Images passed as `image_url` content parts. |
| `stream(prompt, history, config, system)` | `stream=True`, yields `delta.content` chunks. |
| `async_chat(prompt, history, config, system, images)` | Uses `openai.AsyncOpenAI`. |
| `session(system)` | Returns `ChatSession(llm=self, system=system)`. |

### Vision support

OpenAI accepts images as content parts within the `user` message:

```python
{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
```

`_build_messages()` builds a mixed-content list when `images` is non-empty.

### Response mapping

```
choices[0].message.content  → content
model                       → model
usage.prompt_tokens         → prompt_tokens
usage.completion_tokens     → completion_tokens
usage.total_tokens          → total_tokens
elapsed (perf_counter)      → elapsed_seconds
BackendType.OPENAI          → backend
```

### Convenience factory functions

```python
def gpt4o(system=None, **kwargs) -> MakiOpenAI: ...
def gpt4o_mini(system=None, **kwargs) -> MakiOpenAI: ...
def o3(system=None, **kwargs) -> MakiOpenAI: ...
```

---

## Step 6 — `MakiAnthropic` (`maki/makiAnthropic.py`)

Uses the `anthropic` Python SDK (`pip install anthropic`).

### Class signature

```python
class MakiAnthropic(LLMBackend):
    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        api_key: Optional[str] = None,       # falls back to ANTHROPIC_API_KEY env var
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        rate_limit: Optional[int] = None,
    ) -> None: ...
```

### Methods

| Method | Notes |
|--------|-------|
| `request(prompt)` | Required by `LLMBackend`. Routes to `chat()`. |
| `chat(prompt, history, config, system, images)` | `anthropic.messages.create(stream=False)`. |
| `stream(prompt, history, config, system)` | `with client.messages.stream(...)`, yields `text_delta` events. |
| `async_chat(prompt, history, config, system, images)` | Uses `anthropic.AsyncAnthropic`. |
| `session(system)` | Returns `ChatSession(llm=self, system=system)`. |

### Anthropic-specific: system prompt placement

The Anthropic API takes `system` as a **top-level parameter**, not inside the messages list. `_build_messages()` must never add a `{"role": "system", ...}` entry; the `system` string is passed separately to `messages.create()`.

### Vision support

Anthropic accepts images as content blocks:

```python
{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
```

`_build_messages()` builds a mixed-content block when `images` is non-empty.

### Response mapping

```
content[0].text             → content
model                       → model
usage.input_tokens          → prompt_tokens
usage.output_tokens         → completion_tokens
input + output              → total_tokens
elapsed (perf_counter)      → elapsed_seconds
BackendType.ANTHROPIC       → backend
```

### Convenience factory functions

```python
def claude_sonnet(system=None, **kwargs) -> MakiAnthropic: ...
def claude_haiku(system=None, **kwargs) -> MakiAnthropic: ...
def claude_opus(system=None, **kwargs) -> MakiAnthropic: ...
```

---

## Step 7 — Wire into `maki/__init__.py`

```python
__all__ = [..., "MakiOpenAI", "MakiAnthropic", "BackendType"]

_LAZY_EXPORTS = {
    ...
    "MakiOpenAI":    (".makiOpenAI",    "MakiOpenAI"),
    "MakiAnthropic": (".makiAnthropic", "MakiAnthropic"),
    "BackendType":   (".objects",       "BackendType"),
}
```

---

## Step 8 — Optional dependency handling

Neither `openai` nor `anthropic` is a hard dependency. Each file guards the import:

```python
try:
    import openai
except ImportError:
    openai = None  # type: ignore

# inside __init__:
if openai is None:
    raise ImportError("openai package is required: pip install openai")
```

`pyproject.toml` optional extras:

```toml
[project.optional-dependencies]
openai     = ["openai>=1.0"]
anthropic  = ["anthropic>=0.20"]
all        = ["openai>=1.0", "anthropic>=0.20"]
```

---

## Step 9 — Tests

Two new test files mirroring `maki/test/test_makiLLama_unittest.py`, all API calls mocked with `unittest.mock.patch`:

- `maki/test/test_makiOpenAI.py`
- `maki/test/test_makiAnthropic.py`

Minimum coverage per class:
- `request()` happy path
- `chat()` happy path
- `chat()` with image (vision)
- `stream()` yields chunks correctly
- `async_chat()` happy path
- `session()` multi-turn accumulates history
- API key missing → `ImportError` / `ValueError`
- Network error → `MakiNetworkError`
- HTTP 4xx → `MakiAPIError`
- Rate limiter integration

---

## Files Touched

| File | Change |
|------|--------|
| `maki/objects.py` | Add `BackendType` enum; update `LLMResponse.backend`; add `to_openai_kwargs()` / `to_anthropic_kwargs()` to `GenerationConfig` |
| `maki/config.py` | Add `DEFAULT_OPENAI_MODEL`, `DEFAULT_ANTHROPIC_MODEL`, `OPENAI_API_KEY_ENV`, `ANTHROPIC_API_KEY_ENV` |
| `maki/session.py` | **New file** — `ChatSession` moved here from `makiLLama.py` |
| `maki/makiLLama.py` | Remove `ChatSession` definition; import it from `maki/session.py` |
| `maki/makiHG.py` | Update `LLMResponse` construction to pass `backend=BackendType.HF` |
| `maki/makiOpenAI.py` | **New file** — `MakiOpenAI` + factory functions |
| `maki/makiAnthropic.py` | **New file** — `MakiAnthropic` + factory functions |
| `maki/__init__.py` | Extend `__all__` and `_LAZY_EXPORTS` |
| `maki/test/test_makiOpenAI.py` | **New file** |
| `maki/test/test_makiAnthropic.py` | **New file** |
| `pyproject.toml` | Add optional extras `[openai]`, `[anthropic]`, `[all]` |
