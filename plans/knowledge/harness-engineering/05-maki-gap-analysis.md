# 05. Maki Gap Analysis and Roadmap Candidates

**How might we** turn maki from "a framework that can call an LLM in a workflow" into "a framework whose agents verifiably finish things, remember across sessions, and can be audited", using the course's harness ideas as primitives rather than prose advice?

Key framing: maki is a *library for building agents*, not a coding agent. The course targets coding agents working in a repo. So the useful translation is: which of these mechanisms should maki **ship as primitives** so users get a good harness by composition. Repo-hygiene items (entry files, progress files for developing maki itself) are a separate, smaller track.

## What maki already has (verified in code)

| Course idea | Maki today |
|---|---|
| Durable workflow state / checkpoint | `WorkflowState`, `StateStore` (`LocalStateStore`, `RedisStateStore`), atomic JSON writes, resume by `workflow_id` (`maki/distributed/state_store.py`) |
| Dependency DAG, retries | `WorkflowTask` dependencies, topological sort, `max_retries`, parallel option (`maki/agents/workflow.py`) |
| Gates with reasons | `GateResult(ok, reason)` and `conditions` on tasks |
| Human-approval node | `ApprovalSpec` / `ApprovalStore`, status `AWAITING_APPROVAL`, key bound to content hash, TTL (`maki/agents/approvals.py`) |
| Typed handoff between tasks | `TaskOutput.data` passed to dependents; `generate_structured` with validation retries (`maki/structured.py`) |
| Fail-closed tools, dangerous-tool gating | `ALLOWED_METHODS`, `DANGEROUS_METHODS`, `allow_dangerous_tools` |
| Bounded loops/history | `_MAX_TOOL_ROUNDS = 5`, `deque(maxlen=1000)` |
| Self-correction | `ReasoningEngine.self_correct` (same-agent) |

## What is missing (verified by grep: no tracing, hooks/events, compaction, or verifier/evaluator code outside tests)

| # | Gap | Course basis | Where it would plug in |
|---|---|---|---|
| G1 | *(shipped: `verification.py`)* **Independent verifier / maker-checker as a first-class task type** (fresh context, only sees the artifact, returns Accept/Revise/Block with evidence; failure routes back to the maker) | L9, L13, L14 | New `VerifyTask`/checker spec on `WorkflowTask`; routing in `agent_manager._execute_workflow_task` |
| G2 | *(shipped as in-task revision loop; cross-task edges still open)* **Rollback / conditional routing edges** (verify-fail -> back to a named task, bounded) | L14 | Extend workflow engine beyond forward-only DAG; bounded by per-edge attempt cap |
| G3 | **Goal loop with ratchet + stop conditions** (goal, verification callable, turn/budget cap, per-item attempt cap, escalate to approval) | L13 | `AgentManager.run_goal(...)` built on `StateStore` and `ApprovalStore` |
| G4 | *(partial: ordered gates composable via callable verifiers)* **Layered completion gate** (static -> runtime -> system, ordered, short-circuit, repair hints in failures) | L9, L10 | Compose with `GateResult`; verifiers return structured evidence |
| G5 | **Event/hook points in the agent loop** (pre/post tool, pre-step, stop) so policy, logging, memory and verification attach without editing the loop | Claude Code, DeepSeek, Pi | `PluginHandler` tool loop and `Agent.execute_task` |
| G6 | **Tracing and a session event log** (one trace per workflow run, span per task and per verification, append-only, "model-visible means logged") | L11, DeepSeek | Logging layer + `StateStore`; optional OpenTelemetry extra |
| G7 | **Context management** (budget awareness, staged compaction, durable "why" notes; today history is count-bounded, not token-bounded) | L5, Claude Code, Pi | `Agent._build_history_section`, `ChatSession` |
| G8 | **Run handoff artifact** (verified-now / changed / broken / next step, generated from workflow state) | L5, L12, template | Export from `WorkflowState` |
| G9 | **Harness audit** (`maki audit`: checks an agent YAML / workflow for verifier present, stop condition set, tool allowlist scoped, approvals on dangerous tools, checkpointing on) | audit-harness.sh | `maki/__main__.py` CLI, `config_loader.py` |
| G10 | **Recurring-failure-to-check promotion**, VCR-style metric (verified / activated) over workflow runs | L7, L10 | Metrics from persisted `WorkflowState` |

## Candidate directions

**A. Verified workflows (G1, G2, G4, G8).** Make "done means evidence" a workflow property. Smallest sharp version: a task can declare a `verifier`; a failed verification re-queues the maker with the failure message up to N times, then parks as `AWAITING_APPROVAL`. Reuses retry, gates and approvals that already exist. Painkiller: it is the failure mode (agents overclaim) users hit first.

**B. Goal loops (G3, plus G1).** `run_goal` with stop conditions and persisted loop state. Higher ambition and higher risk of runaway cost; needs A first, since a loop without an independent verifier is self-grading.

**C. Observability and audit (G5, G6, G9).** Hooks plus an event log plus `maki audit`. Less flashy, but it is the enabling layer: A and B want hooks and traces to be debuggable, and the audit CLI gives a cheap, demonstrable win.

**D. Context management (G7).** Real but model-and-backend-dependent, and the course's own evidence is thin (numbers are teaching defaults, evidence on instruction files is mixed). Defer until A-C show where context hurts.

Recommended order: **A -> C (hooks and event log) -> B -> D.** Rationale: A is smallest, builds on code that exists, and proves the maker/checker idea; C makes everything after it observable; B composes A and C; D is least validated.

## Assumptions to validate (honest list)

- [ ] A separate verifier agent on the same small local model (Ollama-class) actually catches errors; self-grading bias may persist when maker and checker share a model. Test: run maker-only vs maker+checker on a handful of tasks with known-bad outputs.
- [ ] Users want these as framework primitives rather than writing a second `Agent` themselves. Check against how the examples and the newsletter project use workflows.
- [ ] Deterministic checks (tests, lint, schema validation) cover most of the value; an LLM verifier is the fallback. This suggests starting with callable verifiers and adding an LLM checker second.
- [ ] Course claims about gains are mostly illustrative. Do not justify roadmap items by its percentages; justify them by maki-specific failures.
- [ ] Rollback edges can be added without breaking topological-sort assumptions and checkpoint/resume semantics (needs a design spike on `WorkflowState`).

## Not doing (and why)

- Graph engine rewrite (LangGraph-style) — the course itself says topology is cheap, replayability and recoverability are the value; extend the existing DAG with bounded rollback only.
- Multi-agent fleets and self-feeding loops (ladder levels 4-5) — orchestration and review cost dominate; no evidence maki users need them yet.
- Copying the course's Electron/TS project curriculum or its shell audit script as-is — maki is Python; borrow the checks' ideas only.
- Adopting numeric thresholds (60% context, 100-line files) as hard rules — teaching defaults, not measured.
- Auto-generated `AGENTS.md`/scaffolding tool for end users — separate product (`harness-creator` already exists); at most keep maki's own `CLAUDE.md` short and current.
- Heavy observability stack (mandatory OpenTelemetry) — keep it an optional extra to preserve the small core install.

## Smaller track: maki repo's own harness hygiene

Cheap wins the course points at, relevant to developing maki itself:

- `CLAUDE.md` is already a good map but has drifted before (the earlier `ARCHITECTURE_REVIEW.md` found it referencing a removed `Maki` class). Add a drift check to CI or the review habit.
- Add one aggregate verification command to `CLAUDE.md` (today: separate `pytest` and `flake8`), e.g. a `make check` or documented one-liner.
- Consider a short `docs/decisions/` log; `docs/` already holds plan docs that effectively serve this.

## Open questions

- Should verifier output be a typed model (reuse `generate_structured`) with fields like verdict, evidence, missing, fixes?
- Where do loop and verification state live: extend `WorkflowState` or add a sibling store?
- Is `AgentProxy`/distributed mode in scope for verifier tasks (remote checker agents are a natural fit for fresh-context verification)?
