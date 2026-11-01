# Agent Memory Optimization

## Problem Statement
How might we keep in an agent's prompt only what it needs from its past, so Maki agents use fewer
tokens, invent less about earlier turns, and have a bounded memory footprint, on every backend and
with deterministic, reproducible behaviour?

## What the Code Does Today
- Stateful history is pasted as text ("Prior conversation: Task / Response") after the task on every
  call, in full (`Agent._build_history_section`, `ConversationMemory.format_as_text`). Backends already
  accept structured `history=`; `ChatSession` uses it, `Agent` does not.
- The only management is a 4096-token cap (`len//4` estimate) that silently evicts the oldest pair.
  No summary, dedup or pinning. `Agent` cannot set the budget. The first turn (usually the goal) is the
  first to go.
- `agent.remember()`/`recall()` is a dict that is never injected into a prompt; it is only serialized.
- Plugin results are `str(result)` with no size cap on the native path and the `TOOL:` path, for up to
  5 rounds per task.
- `task_history` and `reasoning_history` are bounded by count (1000), not size; `to_dict` serializes
  all of it.
- `obsidian_memory` and `rag_memory` are manual; the documented pattern replays the newest notes, not
  the relevant ones.
- Baseline tooling exists: the `Usage` counter, and the loop scratchpad as prior art.

## Recommended Direction
Opt-in, deterministic, in two steps behind one seam. The default behaviour of `Agent` does not change.

**Step A: make existing memory cheaper.**
1. Send stateful history to the backend as structured messages instead of pasted text, so the prefix
   is append-only and provider caches can hit.
2. Cap each plugin result (head and tail kept, the rest spilled to a handle the model can page), on
   both tool paths.
3. Bound `task_history` and `reasoning_history` by size as well as count; store clipped results.
4. Expose the history token budget on `Agent`.
5. Add a repeatable benchmark on the `Usage` counter: a 20-turn stateful run and a tool-heavy run.

**Step B: tiered memory with a policy object.**
A single `MemoryPolicy` (`budget`, `pin`, `tool_result_cap`, `retain`) passed as `Agent(memory=...)`.
Tiers: pinned facts (goal, constraints) that are never evicted; a recent window verbatim; older turns
reduced mechanically (old tool output stubbed, long answers clipped head and tail, pointer to the full
text on disk). `remember(key, value, pin=True)` feeds the pinned tier under a budget, so the dict is
finally visible to the model. Eviction is no longer silent: a one-line marker says how many turns were
reduced.

The `MemoryPolicy` seam is where a later retrieval tier (RAG/Obsidian, top-k by relevance, for
cross-session recall) would plug in. It is not part of this work.

## Key Assumptions to Validate
- [ ] Structured history yields real cache hits on Anthropic, OpenAI and vLLM. Test: same 20-turn run
      both ways, compare cached vs. uncached prompt tokens.
- [ ] FIFO eviction is what causes wrong claims about earlier turns. Test: small recall eval (plant a
      fact in turn 1, ask in turn 15) with today's eviction vs. pinned goal.
- [ ] Mechanical clipping keeps enough signal. Test: same recall eval with clipped older turns; look for
      lost decisive lines.
- [ ] `len//4` is close enough. Test: compare with a real tokenizer on code and non-English text; keep
      the estimate if the error is tolerable, otherwise allow an injected counter.
- [ ] Structured history works on every backend (Ollama, llama.cpp, vLLM, OpenAI, Anthropic, HF).
      Test: the existing backend test suites plus one stateful run each.

## MVP Scope
In:
- Step A items 1 to 5, shipped on their own. Items 1 to 3 are opt-in (a flag or `memory=` option); the
  default stays as it is until the benchmark justifies changing it, then the change goes in the changelog.
- Success targets, all measured on the benchmark: lower prompt tokens on the 20-turn run, equal-or-better
  recall eval score, and flat memory use over a long run of `maki serve`. Numeric targets are set after
  the baseline is measured, not before.
- Step B only once A's numbers exist: `MemoryPolicy` with pinned and recent tiers and the eviction
  marker, plus `remember(pin=True)`.

Out: anything that calls an LLM, any embedding dependency, any change to the loop scratchpad.

## Not Doing (and Why)
- LLM-written summaries: you chose deterministic first, and they add a call, cost and drift. Possible
  later as an optional tier.
- Retrieval/RAG-backed memory and cross-session recall: valuable, but it needs an embedder. The policy
  seam keeps the door open.
- A shared memory service for `maki serve` agents: premature before one agent's memory is efficient.
- Making "fresh context" the default: too large a change to the API.
- Hard numeric thresholds (e.g. "compact at 60%"): the gap analysis calls them teaching defaults, not
  measured values.
- Changing defaults for existing users in the first release: opt-in only until the benchmark says so.

## Open Questions
- Where do spilled tool results live, and for how long (a `~/.maki/` directory like the file_mover
  journal, with retention)?
- Should `MemoryPolicy` also configure the loop's `context="shared"` mode, which depends on stateful
  history?
- Does `to_dict`/`from_dict` stay backward compatible when entries are clipped?
- Which backends report cached-token counts, so the benchmark can show cache hits?
