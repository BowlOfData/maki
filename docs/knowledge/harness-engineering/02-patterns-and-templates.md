# 02. Patterns, Checklists and Templates

Reworded and condensed from the course's lectures and `resources/templates`. Numbers are adjustable defaults.

## Entry file skeleton (`AGENTS.md` / `CLAUDE.md`)

1. One or two sentences: what the project is.
2. First-run commands, plus the single aggregate verification command.
3. Hard constraints (<= ~15), each with why and removal condition.
4. Links to topic docs with "read when..." conditions.
5. Clock-in / clock-out routine and rules like WIP=1 and "do not edit feature state by hand".

## Feature list item

```json
{
  "id": "F03",
  "behavior": "POST /cart/items with {product_id, quantity} returns 201",
  "verification": "<exact command>",
  "state": "not_started | active | blocked | passing",
  "evidence": "commit hash, test output reference"
}
```

Who may change what (course principles, field ownership inferred): agent proposes behavior/verification text and may move an item to `active`; only the verifier moves to `passing` and attaches evidence; a human approves scope changes and resolves `blocked`. Failed verification keeps the item `active`. Only one item is `active`.

## Three-layer termination gate

| Layer | Checks | Rule |
|---|---|---|
| 1 Static | lint, types, syntax | cheap baseline, must pass first |
| 2 Runtime | tests, app starts, critical paths | only if layer 1 passes |
| 3 System | end-to-end flow, side effects, cleanup | only if layer 2 passes |

Acceptable evidence: app reaches ready state, critical path runs, side effects are correct (DB/file/external), temp resources cleaned, logs show no unexpected errors. "Tests pass" and "code written" are intermediate signals.

## Before accepting "done"

- [ ] Static analysis and types pass
- [ ] Unit and integration tests pass, none skipped to get there
- [ ] Whole app starts without errors
- [ ] Full user flow executed, not described
- [ ] Persisted state verified after the operation
- [ ] Required config and external services present in the target environment
- [ ] Failed or partial migrations detected and reported or rolled back
- [ ] Temporary resources cleaned up
- [ ] No refactoring before core behavior was verified
- [ ] An independent check (not the producing agent) reviewed it

## Clean-state definition (L12)

1. Build passes. 2. All tests pass, pre-existing ones included. 3. Progress recorded in versioned files. 4. No stale artifacts (temp files, debug code, stray TODOs). 5. The standard startup path works for the next session. Make cleanup scripts idempotent.

## Session handoff (template fields)

- **Verified now:** what works, what verification actually ran.
- **Changed this session:** behavior added, harness changes.
- **Broken or unverified:** known defects, unverified paths, risks.
- **Next best step:** the item, why it is next, what counts as passing, what must not change.
- **Commands:** startup, verification, focused debug.

Minimum handoff data (L5): repo state (commit hash), runtime state (test status), blockers, next actions. Clock-in: read progress and decision log, run the check command, resume at top next step. Clock-out: update progress, record verification, log decisions with reasons, run checks, commit.

## Evaluator rubric (template)

Used after implementation, before acceptance. Six categories scored 0-2 with notes: correctness, verification (checks actually ran, with evidence), scope discipline, reliability (survives restart/rerun), maintainability, handoff readiness (fresh session can continue from repo artifacts alone). Verdict: **Accept / Revise / Block**, with required follow-ups: missing evidence, required fixes, next review trigger. A planned check does not count; missing evidence is recorded explicitly.

## Sprint contract (L11)

Agreed before coding: scope, verification standard, explicit exclusions. Judged afterward against the rubric, with scores tied to evidence.

## Architecture rules as data (L10 + audit script)

A rule registry (`.harness/arch-rules.json`) where every rule has `what`, `why`, `fix`; a checker script runs on each commit and prints those fields on violation. Review findings that recur get promoted into this registry.

## Quality document (L12)

Per module, grade: verification, agent-understandability, test stability, boundaries, convention adherence. The next session fixes the lowest-graded module first.

## Audit script (course's `tools/audit-harness.sh`)

~70 checks grouped by subsystem and by lecture; a handful are CRITICAL and drive the exit code (entry file exists with a system description in its first lines and a verification command; dependency lockfile; progress file with a current-state block), the rest RECOMMENDED (pinned runtime, task runner targets like `setup/dev/check/test/e2e/vcr/clean-check`, decision log, feature list with `id/behavior/verification/state/evidence`, WIP=1 rule, three-layer rules, architecture rule registry, trace directory, quality document). Known quirks: it greps patterns loosely, the VCR check counts lines not JSON entries, a module-doc check runs twice. Good as inspiration for a `maki audit`-style checker, not as ground truth.

## harness-creator skill (course's `skills/harness-creator`)

Scaffolds a minimal harness (`AGENTS.md`/`CLAUDE.md`, `feature_list.json`, `progress.md`, `init.sh`, optional `session-handoff.md`); has create / audit / report workflows; ships seven on-demand reference guides (memory persistence, skill runtime, tool registry and safety, context engineering, multi-agent coordination, lifecycle and bootstrap, gotchas). Its own benchmark is structural only; real effectiveness needs before/after agent runs on real tasks.
