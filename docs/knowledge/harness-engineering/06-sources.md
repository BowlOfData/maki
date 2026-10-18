# 06. Sources

Base: https://github.com/walkinglabs/learn-harness-engineering (MIT). Raw paths below are relative to `docs/en/` on `main`.

| Topic | Source |
|---|---|
| Overview, syllabus | `README.md`; site https://walkinglabs.github.io/learn-harness-engineering/ |
| L1 Why capable agents fail | `lectures/lecture-01-why-capable-agents-still-fail` |
| L2 What a harness is | `lectures/lecture-02-what-a-harness-actually-is` |
| L3 Repo as system of record | `lectures/lecture-03-why-the-repository-must-become-the-system-of-record` |
| L4 Giant instruction files | `lectures/lecture-04-why-one-giant-instruction-file-fails` |
| L5 Continuity | `lectures/lecture-05-why-long-running-tasks-lose-continuity` |
| L6 Initialization | `lectures/lecture-06-why-initialization-needs-its-own-phase` |
| L7 Overreach / under-finish | `lectures/lecture-07-why-agents-overreach-and-under-finish` |
| L8 Feature lists | `lectures/lecture-08-why-feature-lists-are-harness-primitives` |
| L9 Premature done | `lectures/lecture-09-why-agents-declare-victory-too-early` |
| L10 End-to-end testing | `lectures/lecture-10-why-end-to-end-testing-changes-results` |
| L11 Observability | `lectures/lecture-11-why-observability-belongs-inside-the-harness` |
| L12 Clean state | `lectures/lecture-12-why-every-session-must-leave-a-clean-state` |
| L13 Loops, L14 Graphs | `lectures/lecture-13-loop-engineering`, `lectures/lecture-14-graph-engineering` |
| Product breakdowns | `harness-designs/{claude-code,codex,pi,deepseek}` |
| Templates | `resources/templates/` (AGENTS.md, CLAUDE.md, claude-progress.md, clean-state-checklist, evaluator-rubric, feature_list.json, init.sh, quality-document, session-handoff) |
| Scaffold skill | `skills/harness-creator/` (repo root) |
| Audit script | `tools/audit-harness.sh` (repo root) |

Primary works the course cites (not independently read here): OpenAI's Codex harness write-up ("the repo is the spec"), Anthropic's long-running agent harness posts, LangChain's harness-change results on Terminal Bench 2.0, Liu et al. 2023 (lost in the middle), Lulla et al. on AGENTS.md efficiency, VILA Lab's Claude Code analysis.

Not yet read: project-01..08 starter/solution code, `resources/openai-advanced`, `resources/reference`.
