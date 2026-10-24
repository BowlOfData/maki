# Plan: Verified Workflows (maker/checker + bounded revision)

Status: **approved; implemented on branch feat/verified-workflows** · Origin: [harness-engineering gap analysis, direction A](knowledge/harness-engineering/05-maki-gap-analysis.md) (G1, G2, G4)

## Goal

A `WorkflowTask` can declare a **verifier**. After the task runs, the verifier judges the result with evidence. If it says *revise*, the task re-runs with the feedback, up to a bound. Only an *accepted* result counts as `COMPLETED` and flows downstream. Success criterion: a stub-agent workflow where the maker's first output is bad, the verifier rejects it with feedback, the second output is accepted, and the checkpoint/resume path preserves the verdict, all under `pytest`.

## What the code does today (researched)

- `AgentManager._execute_workflow_task` (`maki/agents/agent_manager.py:531`) runs one task: skip/blocked/approval gates, build context from dependency `data`/`result`, call `agent.execute_task_with_retry` (retries only transient network errors), record `COMPLETED` or `FAILED`.
- A `FAILED` task returns `(name, None)`: **dependents still run**, with that dependency's context simply absent. For unverified-output safety this must differ for verification failures (see D3).
- `WorkflowState.update_task_status` stores a plain dict per task; `to_dict/from_dict` copy entries generically, so one extra key (`verification`) round-trips through `LocalStateStore`/`RedisStateStore` with no store changes.
- `WorkflowTask.to_dict` omits callables (`conditions`, `approval`); a verifier spec is the same kind of thing and is treated the same way (`has_verifier` flag only).
- Agents are non-stateful by default, so a verifier agent's prompt is just task + context: "fresh context" comes for free as long as the verifier agent is a different `Agent` and is not set `stateful`.
- Approval gate sits *before* execution and is keyed on content; it is not reused for after-the-fact escalation (D2).
- Tests use stub agents with `execute_task_with_retry` (see `maki/test/test_workflow_approval.py`); the new tests follow that pattern. Plans live as `plans/PLAN_*.md`.

## Design (proposed)

```python
class Verdict(Enum): ACCEPT, REVISE, BLOCK

@dataclass(frozen=True)
class VerifyResult:
    verdict: Verdict
    evidence: str = ""       # what was checked / observed
    fixes: str = ""          # actionable: what failed, where, how to fix (fed back to the maker)

@dataclass
class VerifySpec:
    verify: Callable[[dict], VerifyResult | bool]   # ctx: task, result, data, context, revision
    max_revisions: int = 2                          # extra maker runs after the first
```

Helpers: `agent_verifier(agent_name, criteria)` returns a callable that asks a verifier agent (via the manager) for a JSON verdict. **Fails closed**: unparsable output, exception, or missing agent means `BLOCK`.

Loop inside `_execute_workflow_task` (after approval gate), per revision `r = 0..max_revisions`:
1. Run the maker. On `r > 0` its context additionally holds `__verification_feedback` (previous `fixes`) and `__previous_result`.
2. Run the verifier with the produced result and data.
3. `ACCEPT` -> `COMPLETED`. `REVISE` and revisions remain -> loop. `REVISE` exhausted or `BLOCK` -> `FAILED` with reason `verification failed: <fixes>`, and dependents are `SKIPPED` ("dependency '<x>' failed verification").
4. State entry gains `verification = {verdict, evidence, revisions, history: [...]}` for audit and for the later handoff/metrics work (G8, G10).

`bool` returns from a callable are accepted for convenience (`True` -> ACCEPT, `False` -> BLOCK with no evidence; use `VerifyResult` to get REVISE).

## Steps

| # | Step | Size | Blocked by | Files | Verification |
|---|---|---|---|---|---|
| S1 | Verification types: `Verdict`, `VerifyResult`, `VerifySpec`, normalizer for `bool`/`VerifyResult`; export from `maki.agents` | S | none | new `maki/agents/verification.py`, `maki/agents/__init__.py`, new `maki/test/test_verification_types.py` | `pytest maki/test/test_verification_types.py` |
| S2 | Engine integration: `WorkflowTask(verify=...)`, revision loop, state `verification` entry, dependent skip on verification failure, `has_verifier` in `to_dict`, `update_task_status(verification=...)` | M | S1 | `maki/agents/workflow.py`, `maki/agents/agent_manager.py`, new `maki/test/test_verified_workflow.py` | `pytest maki/test/test_verified_workflow.py maki/test/test_workflow_approval.py maki/test/test_enhanced_workflow.py` |
| S3 | `agent_verifier(...)` adapter: prompt builder (criteria + artifact only), stdlib JSON verdict parse, fail-closed; reject verifier == maker agent at workflow start | S | S1 | `maki/agents/verification.py`, `maki/agents/agent_manager.py` (validation), tests in `test_verified_workflow.py` | same file, plus `pytest maki/test/test_mixin_contracts.py` |
| S4 | Checkpoint/resume + distributed: resumed run keeps verdicts, a task that failed verification is retried on resume, `LocalStateStore` round trip; confirm `AgentProxy` works as verifier agent via stub | S | S2 | tests only (`test_verified_workflow.py`, `test_distributed_workflow.py` additions) | `pytest maki/test/test_verified_workflow.py maki/test/test_distributed_workflow.py` |
| S5 | Docs: `CLAUDE.md` (workflow section), `CHANGELOG.md`, one example in `examples/`, mark G1/G2/G4 in the gap analysis as shipped | XS | S2, S3 | `CLAUDE.md`, `CHANGELOG.md`, `examples/verified_workflow.py`, `docs/knowledge/harness-engineering/05-maki-gap-analysis.md` | `flake8 .` and example runs against a stub backend |

Frontier at start: S1 only. After S1: S2 and S3 can run in parallel (they share `verification.py` only at the agent-verifier function, so they should be sequenced S2 then S3 if run by one agent). Highest-risk step is S2 (touches the central execution path); do it first after S1 and run the existing workflow suites before anything else.

**Full verification at the end:** `pytest && flake8 .` (413+ tests at last review, CI matrix 3.10-3.13).

## Decision log (D1-D5 confirmed by user; D6-D7 are my defaults, correct them at approval)

- D1. Rollback = **revision loop inside one task**, not cross-task rollback edges. Keeps the topological-sort/checkpoint model untouched; cross-task edges are a later design spike.
- D2. Exhausted/blocked verification = `FAILED` (no human-escalation path in v1). Escalating to approval needs a new post-execution approval semantic; deferred.
- D3. Dependents of a verification failure are `SKIPPED`. Scoped to verification failures only; generic `FAILED` propagation is not changed in this plan.
- D4. v1 supports callable verifiers **and** an agent verifier adapter.
- D5. Verifier agent must differ from the maker agent (checked at workflow start; raises `ValueError`).
- D6. Agent verifier output is JSON parsed with the stdlib; no pydantic dependency; parse failure = `BLOCK`.
- D7. `max_revisions` default 2; per-task value; maker `max_retries` (transient errors) stays independent.

## Always / Never

**Always:** fail closed on any verifier error; keep dict workflows and existing `WorkflowTask` behavior unchanged when `verify` is unset; add tests with the change; keep core install dependency-free.

**Never (this plan):** cross-task rollback edges; LLM-as-judge defaults enabled implicitly; loop/goal runner (direction B); hooks and tracing (direction C); context compaction (direction D); changing generic failure propagation; real LLM calls in tests.

## Not yet specified (fog)

- Surfacing verification evidence in a run handoff document (G8) and a VCR-style metric (G10): depends on this landing; shape of `verification.history` is the contract.
- Whether remote verification (verifier on another host via `AgentProxy`) needs timeouts distinct from the maker's.

## Progress

| Step | Status |
|---|---|
| S1 | done |
| S2 | done |
| S3 | done |
| S4 | done |
| S5 | done |
