# Agent memory

A stateful agent remembers earlier tasks by carrying them into every new prompt. That works for a short chat. In a long one, or one where tools return big files and web pages, the prompt fills up and the agent starts to forget or to answer badly. A `MemoryPolicy` keeps the agent's memory small and useful, so its answers stay correct as the conversation grows.

Nothing changes unless you pass a policy. Existing agents behave exactly as before.

## What goes wrong without it

- **Old facts fall out silently.** The conversation is capped at a token budget and the oldest turns are dropped first. The first turn is usually the one that states the goal, so it goes first.
- **Pasted history confuses the model.** By default earlier turns are pasted into the prompt as `Task: ... Response: ...` text. Models sometimes copy that pattern instead of answering.
- **One big tool result can crowd out everything.** A file read or a web page can fill a small context window, and some servers then cut the prompt without telling you.
- **The agent's own history grows without limit.** `task_history` keeps every answer in full.

## What you get

The result you care about is the quality of the answers. In a live test on a local model (Ollama, `gemma4:26b`, 4096-token window), the same 13-message chat was run three ways. Message 1 stated a project codename, messages 2 to 12 pasted log excerpts, and message 13 asked for the codename.

| Setting | Sent to the model | Answered message 13 correctly |
|---|---|---|
| No policy (default) | 108,670 characters | **No** |
| `messages` + tiers | 59,462 characters (45% less) | **Yes** |
| `messages` + tiers + pin | 59,862 characters | **Yes** |

- **The default setup got the answer wrong.** Its reply to message 13 was `Response: Response: Response:` followed by a comment on the logs. It never gave the codename.
- **Both policy settings answered `The project codename is KESTREL-7.`** Older messages were shortened, so more of the conversation fit, and the codename message stayed in view.
- **The pin was not needed here**, because the codename message is short and shortening left it intact. Pin a fact when it sits inside a long message, or when it must survive no matter what.
- **About half the text is sent**, which also helps when your window is small.

This is one model and one run per setting, so read it as a strong hint, not proof. In a separate probe on the same model, about 9,600 characters pasted into a single message reached the model as roughly 2,050 tokens, while the same text sent as chat history reached it as roughly 3,400. If your server cuts long single messages like this, `history_mode="messages"` avoids it.

An offline benchmark (`python -m maki.test.memory_bench`, no model needed, token counts estimated at 4 characters per token) shows the other parts:

| Measure | Default | With a policy |
|---|---|---|
| Prompt tokens over a 20-turn chat | 53,887 | 26,578 (-51%) |
| A fact stated in turn 1, visible at turn 15 | no | yes, when pinned |
| Tokens in the last round of a tool run that returns 20,000 characters each time | 20,045 | 2,083 (-90%) |
| Characters held by `task_history` after 50 large answers | 1,000,200 | 102,000 (-90%) |

## Quick start

```python
from maki.agents import Agent
from maki.objects import MemoryPolicy

agent = Agent(
    name="Researcher",
    maki_instance=llm,
    stateful=True,
    memory=MemoryPolicy(
        history_mode="messages",   # earlier turns as chat messages, not pasted text
        token_budget=2048,         # how much conversation to keep (estimated tokens)
        recent_turns=3,            # the last 3 turns stay whole
        older_turn_chars=200,      # older messages are shortened to this many characters
        pinned_chars=400,          # room for facts you pin
        tool_result_chars=2000,    # clip each plugin result
        history_entry_chars=2000,  # clip what task_history stores
    ),
)

agent.remember("goal", "Migrate the billing service without downtime.", pin=True)
```

## Settings

| Field | Default | What it does |
|---|---|---|
| `history_mode` | `"text"` | `"messages"` sends earlier turns as structured chat history when the backend declares `supports_history` (Ollama, OpenAI, Anthropic and everything built on them). Other backends keep the text form and log it once, so memory is never lost silently. |
| `token_budget` | 4096 | How much conversation to keep, in estimated tokens (`len // 4`). The oldest turns are dropped first. |
| `recent_turns`, `older_turn_chars` | none | Set together. The last `recent_turns` turns stay whole and each older message is cut to `older_turn_chars`, keeping its start and end, with an `[... N characters omitted ...]` marker. Shortened turns count at their shortened size against `token_budget`, so more turns fit. |
| `pinned_chars` | none | Size cap for pinned facts. Pinning needs it set. |
| `tool_result_chars` | none | A plugin result longer than this is cut to that many characters (start and end kept) plus the marker. Errors and short results are untouched. |
| `history_entry_chars` | none | The same clipping for what `task_history` and `reasoning_history` store. The conversation itself keeps full answers. |

When older turns are dropped, the first remaining message starts with `[N earlier turn(s) dropped]`, so the model knows the history is partial.

## Pinned facts

A goal or constraint that must outlive the conversation can be pinned. Pinned facts go in the system message of every call and are never dropped.

```python
agent.remember("goal", "Migrate the billing service without downtime.", pin=True)
agent.unpin("goal")          # stop sending it (it stays in agent.memory)
```

- **Only your code can pin.** The model has no tool for it, so scraped text cannot plant a lasting instruction.
- **Pinning is capped.** Pinning without `pinned_chars`, or over it, raises `ValueError`.
- **A pinned key is protected.** It can only be changed with `pin=True` or after `unpin()`. A plain `remember()` on it raises `ValueError`, and the `maki serve` `/memory/set` endpoint answers 409, so a remote client cannot rewrite what you pinned.
- **Pins stay.** They survive `reset_conversation()` (so a `context="fresh"` loop keeps its goal), `clear_memory()`, `DELETE /memory` and `to_dict()`/`from_dict()`. Only `unpin()` removes them.

## Choosing values

- **Keep the budget well under your window.** The token estimate is `len // 4`. For log-like text, code or non-English text, real tokens are closer to 3 characters each, so the estimate runs low. In the live test a budget of 2048 produced prompts of about 4,050 real tokens. Start with `token_budget` at about half of the model's context window (for Ollama, `num_ctx`).
- **Pin what must never be lost**: the goal, hard constraints, names and identifiers.
- **Clip tool output** (`tool_result_chars`) whenever a plugin can return a whole file or web page.
- **Use `history_mode="messages"`** with Ollama, OpenAI, Anthropic, llama.cpp and vLLM. It also gives provider prompt caches a stable start; `Usage.cached_tokens` reports what Anthropic, OpenAI and OpenAI-compatible servers served from cache.

## Limits

- A fact outside a pin that falls out of the budget is still dropped. The `[N earlier turn(s) dropped]` note tells the model, but it cannot get the fact back.
- Shortening keeps the start and end of a message. A key detail in the middle of a long message can be cut.
- The memory layer never calls the model itself, so there are no summaries; everything is a fixed rule, which keeps behaviour reproducible.
- Cache hits against a real provider and the token estimate on other tokenizers have not been measured.

## With `maki serve`

The same fields go under `memory:` in the agent YAML. Unknown keys are an error.

```yaml
name: researcher
stateful: true
memory:
  history_mode: messages
  token_budget: 2048
  recent_turns: 3
  older_turn_chars: 200
  pinned_chars: 400
  tool_result_chars: 2000
```

Pins are set by code on the server side only; there is no endpoint for them.

For notes you want to keep across runs and search by meaning, see [Memory: Obsidian and RAG](memory.md).
