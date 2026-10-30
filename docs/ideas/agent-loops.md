# Agent Loops

## Problem Statement
How might we let a user assign a goal plus requirements to an agent, and have it iterate
until a judge confirms every requirement is met, within hard limits on iterations and time?

## Recommended Direction
A standalone loop engine (`maki/agents/loop.py`) beside `verify=`. It shares the verdict
vocabulary of `verification.py` (`Verdict`, `VerifyResult`, last-JSON-object parsing, fail
closed) but judges per requirement and resolves its judge agent directly, so
`Agent.run_loop` works with no manager.

**What the judge sees.** The artifact is the agent's result plus an optional
`artifact_fn()` (for code: the `git diff` or file contents) plus the output of command
checks. The judge sees the requirements and artifact only, never the agent's reasoning.

**Requirements.** Written by the user or drafted by an agent. A draft is a proposal until
the user approves it, then it is frozen. Each requirement is judged by an agent or by a
command (argv list, no shell, fixed cwd, minimal env, timeout, exit code is the verdict).
Commands are user-authored only and are shown in full at requirements approval.

**Command-check risk.** A command such as `pytest` executes the code the agent just
wrote. That is inherent to coding loops, so it is gated instead of blocked: file-writing
tools need `allow_dangerous_tools=True`, the first write waits for approval, and commands
run in a fixed working directory with a minimal environment and a timeout. This is
documented as unsandboxed.

**Judging.** One batched judge call per iteration returns pass/fail plus feedback per
agent-judged requirement. Command checks run first and feed their output to the judge.
An error or unparsable reply fails all agent-judged requirements for that iteration. BLOCK
means the judge declares the whole goal unachievable and ends the loop.

**Stopping.** Success, iteration cap, wall-clock cap (checked between iterations; each
model call has its own timeout), BLOCK, or 3 consecutive iterations with no improvement.
An iteration improves if more requirements pass than in the previous iteration, so a
recovery after a regression resets the counter and a swap does not. Regressions are
logged, and the iteration cap is the backstop against a flapping judge.

**Context.** Each iteration starts fresh with the goal, failing requirements with
feedback, and the scratchpad. The scratchpad has three owned sections: engine log
(engine-only), judge feedback (replaced each round), agent notes (appended). It is
checkpointed in `state_store` and written as an atomically replaced file that a human may
edit. Edits to the engine section are reported and reverted.

**Approval.** Phase 1 uses a synchronous hook, `on_iteration(event) -> approve | reject(note)`,
called each iteration. A rejection's note is fed to the agent as feedback. Phase 2 adds
durable approval through `ApprovalStore` (key from iteration number and rejection count),
with suspend and resume through the checkpoint.

**Events.** One structured event per iteration, which is the seam for `POST /loop` later.

## Key Assumptions to Validate (Phase 0 spike, before any engine code)
- [ ] Per-requirement judging is stable enough for stall detection. Test: re-judge the
      same output 5 times across 3 tasks and measure the flip rate.
- [ ] A fresh context with notes and feedback converges as well as a shared conversation.
      Test: one coding and one writing task, both modes, compare iterations to success.
- [ ] A batched judge call stays accurate with 3 vs 10 requirements.
- [ ] A command check plus the judge catches what the judge alone misses on code.
- [ ] One `execute_task` can chain several plugin calls, and `LLMResponse` exposes usage
      for cheap per-iteration cost logging.
- [ ] A stall threshold of 3 yields few false stalls on replayed runs.

## Plan
Phase 0 (spike, throwaway script on the existing `verify=`): run the assumption tests above.
Go/no-go on the design before building. Adjust the thresholds and context mode to what the
data shows.
Phase 1 (engine): `LoopSpec`, engine, command and agent-judge requirements, artifact_fn,
drafted requirements with approval, caps, stall detection, scratchpad with human edits,
events, synchronous approval hook, `Agent.run_loop`, tests, and one example (fix failing
tests until green).
Phase 2: durable approval with suspend and resume, checkpoint resume after restart.
Phase 3 (separate change): `POST /loop` with streaming and cancel, `WorkflowTask` integration.

## Success Criteria
- On a fixed set of 5 tasks (2 coding, 2 writing, 1 research), the loop meets all
  requirements within the cap on at least 4 of them.
- Stall detection stops hopeless runs before the iteration cap.
- A loop with file writes pauses for approval before its first write.

## Not Doing (and Why)
- Monitoring and interval loops: a scheduler problem with no judge or feedback.
- Parallel makers and best-of-N: expensive, and hides whether the base loop works.
- Mid-loop requirement rewriting: criteria stay fixed so progress stays comparable.
- Token and cost budgets: needs per-backend accounting. Caps cover v1; usage is logged.
- Shared-conversation mode as a feature: only tested in the spike.
- Commands from agent-drafted requirements: an agent-written command is an injection risk.
- Sandboxing command checks: out of scope. The risk is documented and gated instead.
- Replacing `verify=`: keep it for single-shot checks.

## Open Questions
- None blocking. Phase 0 results may change the context mode and the stall threshold.
