# 04. Frontier Harness Designs

The course's per-product breakdowns of Claude Code, Codex, Pi and DeepSeek Harness (`docs/en/harness-designs/`). Some details come from outside analysis (VILA Lab, Claude Code v2.1.88) and may be out of date; verify before relying on them.

| Design lesson | Seen in | Core idea |
|---|---|---|
| Entry file is a directory page (~100 lines) | Codex, Claude Code (scoped CLAUDE.md) | Details in `docs/`, loaded on demand; only invariants and verification commands up front |
| Scope instructions by location | Claude Code, Pi | Subdirectory instructions load only when files there are read; layered global -> parent -> cwd |
| Staged compaction funnel | Claude Code | Lossless pruning first, structured distillation next, lossy summary last; circuit breakers on compaction |
| Pluggable compaction | Pi | Compaction is a replaceable strategy (topic-based, code-aware, different summarizer model) |
| Structured history over summaries | Pi (session tree), Claude Code (append-only jsonl, resume/fork), DeepSeek (append-only event log) | Replayable history is a sturdier state layer than the last summary |
| "Model-visible means logged" | DeepSeek | Anything entering a model request must be reconstructable from the log |
| Hooks for enforcement | Claude Code (PreToolUse, PostToolUse, Stop), Pi (lifecycle extensions), DeepSeek (events like tools/pre-execute) | Deterministic checks at lifecycle points; a Stop hook intercepts premature "done" |
| Subagent context isolation | Claude Code (sidechain files), Codex (own history and tools, inherits AGENTS.md) | Subtask noise stays out of the parent context |
| Environment-enforced boundaries | Codex | Each task in its own git worktree; plus local logs/metrics/traces for verification |
| Environment deltas | Codex | Each turn emits only changed environment fields, not the full context again |
| Capability interfaces over concrete tools | DeepSeek | Service definition / provider / consumer; swap sandboxes without changing the model-visible tool surface |
| Everything is an event point | DeepSeek | Permissions, memory, policy, logging attach as listeners, no privileged core |
| Prompt-cache-aware loading | Pi | Load skills and rules on demand to protect cache and cost |
| Approval and plan mode | Codex, Claude Code (permission modes + classifier) | Risky actions sit behind a plan and human sign-off at runtime |

Trade-offs the course notes: Pi decides little for you (you build or install what other tools ship). DeepSeek's plugin-everything design costs configuration effort and is a developer preview. The shared lesson is restraint: small core, conventions in the repo, disciplined context management.
