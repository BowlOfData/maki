# Harness Engineering: Agent Quick Reference

Load this when designing or changing anything about how maki agents plan, verify, remember or hand off. Details: [01-concepts.md](01-concepts.md), [02-patterns-and-templates.md](02-patterns-and-templates.md).

## Rules of thumb

1. **Diagnose the harness before the model.** Failures sort into: unclear task, missing context, broken environment, no feedback, lost state. Fix the layer, re-run.
2. **Done = observable evidence**, never "code written" or "looks right". The harness decides completion, not the agent's confidence.
3. **Worker != checker.** Self-grading is generous. Verify in a fresh context with a strict role or deterministic check.
4. **Gate in order:** static (lint/types) -> runtime (tests, app starts) -> system (end-to-end). Do not run a layer if the previous failed.
5. **One active unit at a time (WIP=1).** Block new work while something activated is unverified.
6. **Agents may not mark their own work `passing`.** A verification run sets that state and attaches evidence.
7. **State lives on disk** (progress file, decision log, feature list, git), not in the conversation. Start by reading it, end by updating it.
8. **Short entry file, topics on demand.** A map (~100 lines) with links, not a manual. Put critical rules at the top or bottom.
9. **Failure messages must say what failed, why, and how to fix it.** That lets the agent self-correct.
10. **Every loop needs three things:** a concrete goal, a machine-checkable verification, a stop condition (budget/turn cap, attempt cap per item, human escalation).
11. **Enforce boundaries by environment** (isolation, permissions, hooks), not by asking nicely in a prompt.
12. **Turn recurring review comments into automated checks.**

## Session close-out (clean state)

Build passes; all tests pass; progress/feature state updated with evidence; no debug leftovers or temp files; the standard start path still works; handoff written (verified now / changed / broken or unverified / next best step / commands).

## Graph vs loop vs workflow

Linear and deterministic -> workflow. Retries until a check passes -> loop. Real branches, rollback edges, parallel branches, human approval, checkpoint worth keeping -> graph. A long pipeline is not a graph by length alone.

## Known failure modes

Self-grading; verification debt; Goodhart (optimizing the closure metric); runaway retries; token blowout from growing context; conflicting loops; orchestration tax; comprehension rot (nobody understands what the loop built).
