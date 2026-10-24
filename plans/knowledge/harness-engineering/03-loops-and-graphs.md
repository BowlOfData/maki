# 03. Loops and Graphs (Lectures 13-14)

## Loops

A loop replaces you as the person prompting: it decides when to start, runs, checks, saves state and decides what is next. Leverage moves from prompt writing to loop design.

| Type | Trigger | Stops when | Fit |
|---|---|---|---|
| Turn-based | you type | agent claims done / you interrupt | small, exploratory |
| Goal-based | stated goal | independent evaluator confirms, or turn/budget cap | clear finish line |
| Time-based | interval | you stop it | polling, periodic checks; runs carry no cumulative progress |
| Event-driven | PR, CI failure, new issue | handled, or retry cap | reactive pipelines |

Test: does the work have an end? Yes -> goal loop. No -> time loop. Misuse ("keep building payments" on a timer) accumulates nothing.

**Three required parts:** concrete goal; verification a machine or independent judge can run; stop condition including a budget. "Looks about right" is not a stop condition.

**Ratchet pattern:** propose one change, snapshot, run a fixed-budget evaluation, keep only if the metric improved, otherwise revert; log every attempt including failures. Fixed budgets make iterations comparable.

**Maker/checker:** explorer finds the issue, implementer fixes, verifier checks independently with cited evidence (ideally a different model or fresh session). Fail goes back to the implementer. Some teams use several verifiers and require a majority.

**Loop state file:** last run timestamp, task attempted, changes, verification command + result + evidence, status (pass/fail/blocked), blockers needing a human, ordered next-up queue. Each run reads it first and updates it last. A results log has one row per iteration (time, commit, metric, resource, description).

**Goal template:** goal; allowed and forbidden scope; exact verification commands; per-item rules; stop when all items done, a human decision is needed, or N attempts fail on the same item; where to log.

**Failure modes:** self-grading; verification debt (output outpaces review); comprehension rot (you stop understanding what was built, so require short written summaries); cognitive surrender; token blowout (cost grows faster than turns, so keep state in files and restart sessions from them); runaway retries (cap per item, route blocked items to a human inbox); wrong primitive.

**Maturity ladder:** (1) goal runner, (2) scheduled single task, (3) multi-agent maker/checker with isolated worktrees, (4) self-feeding loop, (5) fleet sharing a memory layer. The course says level 1 gives the fastest return and most teams sit at 2-3.

## Graphs

Four parts: **nodes** (deterministic code, a model call, a tool, or a full agent), **edges** (handoffs), **shared state** (with defined merge semantics for concurrent writes, e.g. summing a failure counter), **routing rules**. A workflow is the fully deterministic special case; a graph allows agent nodes and dynamic routing on outputs or verification results.

**Edge patterns:** parallel branches that converge at a merge node; conditional routing (verify pass -> merge, fail -> implement); **rollback edges** (failed verification -> implement, insufficient info -> research; in a single loop these return paths live only implicitly in the context window, a graph makes them explicit); bounded self-loops (implement retries up to 3); **human approval nodes** (checkpoint, pause before a sensitive step, resume after sign-off).

**Maker-checker graph build order:** define shared state and merge rules; list nodes (each with private prompt, context, tools); wire the spine research -> implement -> verify -> merge; write routing; checkpoint after every step; run under a thread/run ID. Key design choice: the verify node gets a **fresh context** and sees only the artifact in shared state, not the implementer's reasoning.

**Use a graph when at least 3 of 5 hold:** decomposes into independent parallel units; has branch or rollback paths; intermediate state is worth checkpointing; each node has an automatically checkable done; coordination benefit exceeds cost. Length alone does not justify one.

**Graph failure modes:** self-review (checker shares producer context); Goodhart (support bot optimized for ticket closure, closure up, churn doubled); blindness upward (no loop questions the goal itself); conflicting loops; missing anchors (no ground truth or human spot checks, so the network drifts together); orchestration tax (reviewing and reconciling still bottlenecks on one person); unverified numbers; shape over substance (replayability, observability and recoverability carry the engineering value, not the diagram).
