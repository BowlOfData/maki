# Harness Engineering Knowledge Base

A distilled reference on **harness engineering** (making AI agents reliable through the environment around the model), built from the course
[walkinglabs/learn-harness-engineering](https://github.com/walkinglabs/learn-harness-engineering) (MIT, 14 lectures, 8 projects, product breakdowns, templates, an audit script).
Written to serve two readers: **the maker of maki** (roadmap input) and **coding agents working on maki** (on-demand reference).

## How to use

| If you want to... | Read |
|---|---|
| Load the short version into an agent's context | [AGENT_QUICKREF.md](AGENT_QUICKREF.md) (~1 page) |
| Understand the model and vocabulary | [01-concepts.md](01-concepts.md) |
| Copy a checklist, schema or template | [02-patterns-and-templates.md](02-patterns-and-templates.md) |
| Design loops, maker/checker, graphs | [03-loops-and-graphs.md](03-loops-and-graphs.md) |
| See how Claude Code, Codex, Pi, DeepSeek do it | [04-frontier-harness-designs.md](04-frontier-harness-designs.md) |
| Decide what to build in maki | [05-maki-gap-analysis.md](05-maki-gap-analysis.md) |
| Trace a claim to its lecture | [06-sources.md](06-sources.md) |

## One-paragraph thesis

The same model performs very differently depending on its harness: instructions, tools, environment, state and feedback. Most "the model failed" cases are really "nobody told it what done means", "it could not run the checks", or "it forgot what the last session decided". The highest-leverage fixes are cheap and file-based: a short map-style entry file, runnable verification commands, a machine-readable feature list the agent cannot self-certify, a progress log with a clean-state exit routine, and an independent checker.

## Provenance and caveats (read before trusting a number)

- Content was gathered on 2026-10-18 by fetching the course pages and having a fast model summarize each one; lectures were **not** read verbatim line by line. Treat details as faithful-in-spirit, and consult the source link for anything you will quote or build on.
- The course itself flags most numbers (line limits, 60% context handoff, success-rate tables, "3x" gains) as **illustrative teaching defaults, not measured results**. They are marked "default" here. Do not cite them as evidence.
- Some claims come from third-party research the course cites (OpenAI Codex harness write-up, Anthropic long-running-agent posts, LangChain, VILA Lab's analysis of Claude Code v2.1.88). Those are second-hand here.
- The course has internal count inconsistencies (13 vs 14 lectures, 7 vs 8 projects); the detailed listings (14 and 8) are used.
- Not covered: the 8 hands-on projects' starter/solution code, the `openai-advanced` and `reference` resource folders, and the non-English translations.
