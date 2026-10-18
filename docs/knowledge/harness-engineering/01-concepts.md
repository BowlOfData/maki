# 01. Core Concepts

## What a harness is

Everything outside the model weights that determines how much of the model's capability is actually used. A prompt file alone is not a harness. Course framing (Lectures 1-2): **instructions, tools, environment, state, feedback**; a harness missing any of the five is incomplete. Later lectures add supporting mechanisms: scope control, initialization, observability and session lifecycle.

| Subsystem | Question it answers | Typical artifacts |
|---|---|---|
| Instructions | What do I do, in what order, under which constraints? | `AGENTS.md`/`CLAUDE.md` (short), topic docs, module docs |
| Tools | What can I act with, and with what permission? | Shell, test runner, MCP, scoped permissions, hooks |
| Environment | Does it start the same way every time? | Pinned deps and runtime, `make setup`, containers |
| State | Where are we, what was decided? | Progress file, decision log, feature list, git history |
| Feedback | How do I (and the harness) know it works? | Test/lint/type commands, e2e, logs, rubrics |

Feedback is described as usually the highest-return subsystem. The course's illustrative ladder (assumed numbers, not data): README only 20% -> add entry file 60% -> add verification commands 80% -> add progress file 80-100%, same model throughout.

## Failure taxonomy (L1)

1. Task specification: vague requirements force guessing.
2. Context provision: conventions exist only in someone's head.
3. Execution environment: setup trouble eats the context window.
4. Verification feedback: no checks, agent declares done "by feel".
5. State management: progress lost between sessions.

Plus **context anxiety**: as context fills, agents rush, skip verification and pick simpler solutions. Model-dependent (Anthropic reported one model showing strong anxiety that compaction did not fix, a newer one much less).

## Principles by lecture

- **L3 Repo is the system of record.** If the agent cannot see it in the repo it does not exist (chat, wikis, tickets are invisible). Test: start a fresh session with no verbal context and ask five questions: what is this, how is it organized, how do I run it, how do I verify it, where are we now? Each unanswered one is a gap. Keep module docs next to code; update docs with code. Apply ACID to agent state: atomic commits of verified changes, consistency checks after each step, isolation between concurrent agents, durability via tracked files.
- **L4 One giant instruction file fails.** It burns context, buries rules mid-file (models use start and end of long text better than the middle), goes stale and contradicts itself. Use progressive disclosure: entry file as a router with links and "read when..." conditions, facts live in code (types, interface comments). Record source, applicability and expiry for each rule. Evidence is mixed: one study found repo context files did not improve success and raised cost over 20%, another found lower runtime/tokens with no correctness measurement. Defaults: entry 50-200 lines (under 100 preferred), at most ~15 hard constraints, topic docs 50-150 lines.
- **L5 Long tasks lose continuity.** Final code records *what*; intermediate reasoning (*why*) is what compaction drops. Consequences: re-deciding settled questions, duplicate work, compounding drift, rushed endings. Persist progress, decisions and verification notes; use small atomic commits. Compaction (summarize in-session) vs reset (new session from artifacts): reset depends on handoff quality. Defaults: short tasks (<~30 min) in one session; start preparing handoff past ~60% of the window.
- **L6 Initialization is its own phase.** An initializer run builds no features; it produces a runnable environment, one passing example test, a start/test/verify doc, an ordered task list with acceptance criteria and a clean commit. Coding sessions then start from that checkpoint. Startup readiness = can start, can test, progress visible, next steps clear.
- **L7 Overreach and under-finish.** Agents start many things and finish none. Enforce WIP=1; atomic units each with one behavior, one verification command and dependencies (DAG with states not_started / active / blocked / passing); track **Verified Completion Rate** = verified / activated and block new activations while below 1.0.
- **L8 Feature lists are primitives.** Machine-readable, in the repo, each item has behavior + verification command + state (+ evidence). The agent cannot set `passing`; only a successful verification run can. One item fits one session.
- **L9 Premature "done".** Unit tests with mocks hide cross-component faults, handoffs lose information, self-evaluation is biased. Evidence-based completion and a three-layer gate (see [02](02-patterns-and-templates.md)). Verify correctness before performance before style; do not refactor before core behavior is verified.
- **L10 End-to-end changes results.** Agents that know e2e will run reason more about interactions. Define architecture boundaries early (agents copy existing patterns, including bad ones) as executable rules, enforce invariants not implementations, write agent-oriented failure messages, and promote recurring review comments into checks.
- **L11 Observability belongs in the harness.** Two layers: runtime (lifecycle logs, one trace per session with a span per task and verification step, resource metrics, critical-path records) and process (sprint contract agreed before coding, evaluator rubric with evidence, versioned progress files). Agents cannot reliably instrument themselves.
- **L12 Every session leaves a clean state.** Entropy is the default; "clean up later" means never. See the clean-state definition in [02](02-patterns-and-templates.md). Two cadences: immediate cleanup each session and periodic maintenance (quality document grading modules, golden rules encoded as tooling). Also: monthly, disable one harness component and keep it removed if benchmarks do not degrade (harness debt is real).
- **L13-14 Loops and graphs.** See [03-loops-and-graphs.md](03-loops-and-graphs.md).

## Diagnosing with ablation (L2)

Hold the model fixed, remove one subsystem at a time, measure the drop. The biggest drop shows highest marginal value now. Near-zero does not prove uselessness (could be redundant or unexercised). Ablation shows value, not bottlenecks; for bottlenecks, log failures and attribute causes.
