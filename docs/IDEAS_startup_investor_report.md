# Startup Investor Report — Maki Ecosystem

> **Scope:** maki · maki_newsletter · maki_story · maki_trading
> **Original date:** 2026-05-31 · **Updated:** 2026-06-01
> **Reviewer role:** Early-stage investor, pre-seed / seed focus

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Product Assessments](#2-product-assessments)
   - 2.1 [Maki — Multi-Agent LLM Framework](#21-maki--multi-agent-llm-framework)
   - 2.2 [Maki Newsletter — AI-Automated Technical Newsletter](#22-maki-newsletter--ai-automated-technical-newsletter)
   - 2.3 [Maki Story — AI Comic Storyboard Generator](#23-maki-story--ai-comic-storyboard-generator)
   - 2.4 [Maki Trading — Multi-Agent Paper Trading System](#24-maki-trading--multi-agent-paper-trading-system)
3. [Startup Structure Options](#3-startup-structure-options)
4. [Prioritised Fix Roadmap](#4-prioritised-fix-roadmap)
5. [Overall Verdict](#5-overall-verdict)

---

## 1. Executive Summary

The four projects form a coherent ecosystem: **maki** is the AI framework layer, and the other three are vertical applications built on top of it. This is a strong structural signal — the builder has proven the framework by dogfooding it across three distinct domains. The portfolio has one high-commercial-potential product (trading), one solid content-automation product (newsletter), one early-stage creative AI play (story), and one excellent but over-competitive infrastructure piece (the framework itself).

**Recommendation:** One startup, two tracks — maki_trading as the primary commercial bet, maki as the open-source moat and distribution engine.

---

## 2. Product Assessments

---

### 2.1 Maki — Multi-Agent LLM Framework

**What it is:** A Python framework for building multi-agent LLM applications. Supports Ollama, OpenAI, Anthropic, and HuggingFace backends, a plugin system (18+ plugins including RAG, Obsidian memory, trading data, web search, image classification), a workflow engine with dependency resolution, retry logic, and parallel execution. As of 2026-06-01, a full distributed-deployment layer has been implemented.

#### ✅ Implemented since original report (2026-06-01)

| Item | Status |
|------|--------|
| **OpenAI and Anthropic backends** (`MakiOpenAI`, `MakiAnthropic`) | ✅ Shipped. `LLMBackend` ABC now has four concrete implementations. |
| **Distributed agent deployment** (`maki[distributed]`) | ✅ Shipped. Full 5-phase implementation — see detail below. |
| **State serialisation** (`Agent.to_dict / from_dict`, `WorkflowTask`, `WorkflowState`) | ✅ Shipped. All agent and workflow state is JSON-serialisable and fully round-trippable. |
| **`AgentServer`** (FastAPI HTTP wrapper per agent) | ✅ Shipped. `maki serve --config agent.yaml` starts a production-ready HTTP service with Bearer auth, SSE streaming, memory CRUD, and history endpoints. |
| **`AgentProxy`** (drop-in remote `Agent` replacement) | ✅ Shipped. `AgentManager` is unmodified — it calls `.execute_task()` and the proxy transparently dispatches over HTTP. |
| **`DistributedAgentManager`** | ✅ Shipped. `register_remote(name, endpoint, api_key)` adds a proxy under the same agents dict. `assign_task`, `coordinate_agents`, `collaborative_task`, and `run_workflow` all work across local+remote agents. |
| **Distributed workflow checkpointing** (`StateStore`) | ✅ Shipped. `LocalStateStore` (JSON files, zero deps) and `RedisStateStore` (TTL-aware). `run_workflow(workflow_id=..., state_store=...)` resumes mid-workflow after restarts; completed tasks are skipped, failed tasks are retried. |
| **Circuit breaker** | ✅ Shipped. `CircuitBreaker` (CLOSED → OPEN → HALF_OPEN) with configurable failure threshold and recovery timeout. `AgentProxy` fails fast without HTTP calls when the circuit is open; retry loops abort early. |
| **Distributed tracing** | ✅ Shipped. UUID `trace_id` generated per request, propagated via `X-Maki-Trace-Id` header, echoed in the response body, and logged on both proxy and server. `proxy.last_trace_id` available after each call. |
| **mTLS support** | ✅ Shipped. `AgentProxy(ssl_verify=..., cert=...)` passed through to `httpx.Client`. `maki serve --tls-cert FILE --tls-key FILE` enables HTTPS via uvicorn. |
| **154 new tests** (5 new test files) | ✅ 565 total tests passing (up from 411). |

**Distributed architecture summary:**

```
Orchestrator node                    Worker nodes (any machine)
──────────────────────────           ──────────────────────────
DistributedAgentManager              maki serve --config agent.yaml
  ├── local Agent (in-process)         FastAPI (AgentServer)
  ├── AgentProxy → HTTP ─────────────► POST /execute
  │     circuit breaker                GET  /stream  (SSE)
  │     trace_id header                GET/POST/DELETE /memory
  │     ssl_verify / cert              GET/DELETE /history
  └── StateStore (LocalStore/Redis)    GET /health  /info
        checkpoint per task
        resume after restart
```

#### Pros

- **Proven by dogfooding.** Three separate production-grade apps are built on it. This is the best evidence a framework works.
- **Rich plugin ecosystem.** 18+ plugins including domain-specific ones (Alpaca trading, Obsidian memory, RAG backends) that generic frameworks lack.
- **Security-conscious.** SSRF protection, validated URL handling, clean exception hierarchy, circuit breaker, Bearer auth — rare in the space.
- **Backend-agnostic design.** `LLMBackend` ABC has four implementations: Ollama, OpenAI, Anthropic, HuggingFace.
- **MIT licensed.** Correct licensing choice for community adoption.
- **Production-deployment ready.** `maki serve` turns any agent into a networked service. Agents on different machines coordinate transparently.

#### Remaining Gaps

| # | Gap | Fix |
|---|-----|-----|
| C1 | ~~**Ollama-first = local-compute dependency.**~~ **RESOLVED.** OpenAI and Anthropic backends now shipped. | — |
| C2 | **Not on PyPI.** `pip install -e .` only; no release artifact. Kills adoption. | Publish to PyPI. Add a `build.yml` GitHub Action to push on tag. |
| C3 | **No documentation site.** CLAUDE.md is for internal use only. There is no website, no API reference, no tutorials. | Generate docs with MkDocs + mkdocstrings from existing docstrings. Deploy to GitHub Pages. Cost: one afternoon. |
| C4 | **Competing in a crowded space.** LangChain, LlamaIndex, CrewAI, AutoGen, and Pydantic AI all have larger communities and VC backing. | Differentiate on **distributed-first, privacy-preserving** angle — agents run on any machines, state survives restarts, no vendor lock-in. This is now a genuine architectural differentiator, not just positioning. |
| C5 | **No versioning or changelog.** No SemVer releases, though a `CHANGELOG.md` exists. | Tag a v0.2.0 release covering the distributed layer. Prerequisite for any partner or enterprise adoption. |
| C6 | ~~**No multi-tenant or auth layer.**~~ **RESOLVED.** `AgentServer` ships with Bearer token auth per instance. | Multi-tenancy (multiple agents per server, user isolation) remains future work. |

---

### 2.2 Maki Newsletter — AI-Automated Technical Newsletter

**What it is:** A fully automated multi-agent pipeline that ingests RSS, HackerNews, GitHub Trending, Reddit, and Lobste.rs; ranks and summarises articles; generates an editorial digest in the founder's voice; and publishes to Substack and WordPress weekly. Already running in production.

#### Pros

- **End-to-end working product.** This is not a demo — it runs weekly and publishes. Rare at this stage.
- **Human-in-the-loop design.** The pipeline stops for review before committing. Correct design for editorial quality control.
- **Two-channel output.** Substack (subscriber community) + WordPress (SEO). Covers both monetization paths.
- **Voice preservation.** The Substack step validates word count, forbidden phrases, and editorial style — shows quality discipline.
- **Trend-aware ranking.** Google Trends + GitHub + Reddit signals before ranking is genuinely differentiated vs. simple RSS readers.

#### Cons & Fixes

| # | Con | Fix |
|---|-----|-----|
| C1 | **Heavy local compute required.** gemma4:26b needs ~20 GB VRAM. Most potential customers don't have this. Kills self-hosting adoption. | Support cloud LLMs via the `CloudBackend` fix above. Gemini Flash / GPT-4o-mini would cost <$0.50 per newsletter run and unlock 99% of the market. |
| C2 | **Single newsletter, single voice.** The pipeline is hardcoded to one author's style. Not multi-tenant. | Parameterise the voice/style via a config file or brief. This transforms a personal tool into a product others can use. |
| C3 | **Published on Altervista (free hosting).** A credibility problem for a product showcasing AI capabilities. | Move to a custom domain + Vercel/Netlify for the web output. Cost: ~$12/year for the domain. |
| C4 | **No subscriber analytics loop.** The pipeline doesn't ingest open rates, click rates, or reader feedback to improve future curation. | Add a feedback stage: pull Substack stats or a simple "reader picks" form. Close the loop for the learning system. |
| C5 | **No monetization layer.** No sponsorship tracking, no premium tier, no affiliate link management. | Define a monetization model before scaling. Sponsorship slots are the easiest path ($500–2,000/slot for technical newsletters at 5K+ subscribers). |

#### Commercial Verdict

This is a **strong internal tool**, not a standalone startup. Best repositioned as: (a) a showcase for the maki framework, or (b) a feature inside a future "AI content studio" SaaS. The TAM for AI newsletter automation is real but requires cloud LLM support and multi-tenancy to address it.

---

### 2.3 Maki Story — AI Comic Storyboard Generator

**What it is:** A multi-agent system that takes a structured story brief and generates a full comic storyboard — panel-by-panel visual descriptions, dialogue, camera shots — with hybrid memory (Obsidian vault + vector RAG) to maintain character and world continuity across issues.

#### Pros

- **Continuity problem solved.** Hybrid Obsidian + RAG memory for character consistency across panels and issues is the correct technical approach and a genuine differentiator vs. generic LLM prompting.
- **Structured output.** `storyboard.json` + Markdown is pipeline-ready for downstream image generation.
- **Creative AI is a large market.** Midjourney is at $200M+ ARR. Comic/manga content creation is a real underserved niche.
- **Format flexibility.** Strip vs. book mode, configurable panels per page.

#### Cons & Fixes

| # | Con | Fix |
|---|-----|-----|
| C1 | **No image generation.** The product outputs panel *descriptions*, not actual images. This is the critical missing step for any paying customer. | Integrate Stable Diffusion (local, via a maki plugin) or Replicate/Fal.ai APIs for panel image generation from the structured descriptions. The `storyboard.json` schema already has the right fields. |
| C2 | **Early-stage skeleton.** 18 tests passing, but the pipeline is not production-proven like newsletter or trading. | Prioritise end-to-end pipeline completion before any commercial push. Run 5 real briefs to production output and document them as demos. |
| C3 | **Very niche addressable market if B2B.** Professional comic writers are few. | Pivot the positioning to **game/animation pre-production** (concept art storyboards) or **social content creators** (short-form visual story threads). Both are larger markets. |
| C4 | **No collaboration features.** Single-user, local-only. | A web interface where writers upload a brief and receive a storyboard is the minimum viable product. |
| C5 | **No visual style consistency system.** The `STYLE_BIBLE` env var exists for text, but there is no visual consistency mechanism yet. | Build a LoRA/style-prompt system that carries the style bible into image generation prompts — the core quality differentiator vs. just asking ChatGPT. |

#### Commercial Verdict

**Highest optionality, lowest readiness.** If image generation is added and the pipeline matures, this could be the most differentiated product (AI-native comic studio). Current state is a proof-of-concept. Estimated 12 months away from commercialisation.

---

### 2.4 Maki Trading — Multi-Agent Paper Trading System

**What it is:** A multi-asset (crypto/forex/equities) paper-trading system with 8 specialised agents, an event-driven architecture, a contextual-bandit RL loop (Thompson sampling), Obsidian vault memory, and deterministic Python risk management. 316 tests. Ships as a macOS launchd daemon.

#### Pros

- **Most technically sophisticated product in the portfolio.** The RL loop, reward shaping, and Obsidian-backed policy management are production-grade design choices.
- **Deterministic risk path.** The LLM is never in the risk or execution path — only in analysis and proposal. This is the correct and rare architectural choice, and crucial for regulatory credibility.
- **316 tests.** By far the best-tested product. Sizing math, circuit breakers, and risk rules are all covered.
- **Multi-asset from day one.** Crypto + forex + equities with per-class risk rules is a significant scope that most quant startups don't tackle early.
- **Explainability by design.** Every decision has a vault note with YAML frontmatter. Full audit trail. Regulators value this.
- **Thoughtful reward shaping.** `drawdown_penalty + time_penalty + mfe_zero_penalty` goes well beyond simple P&L optimisation.

#### Cons & Fixes

| # | Con | Fix |
|---|-----|-----|
| C1 | **Paper trading only — no live track record.** Without verified live performance, this cannot raise institutional capital or charge for signals. | Run paper trading for 6 months with audited logs. Track Sharpe ratio, max drawdown, and alpha vs. benchmark. This is the single most important milestone before any commercial move. |
| C2 | **Local LLM latency on the critical path.** gemma4:latest on CPU/low-end GPU can take 3–15 seconds per inference. For intraday setups, this can be too slow. | Profile end-to-end tick latency. For latency-sensitive agents (MarketDataAnalyst), explore smaller quantised models (qwen2.5:4b) or offload to a cloud API. The `CloudBackend` fix on maki unblocks this. |
| C3 | **macOS-only.** launchd is not portable. Limits deployment to a single laptop. | Containerise with Docker + a `docker-compose.yml`. Remove the launchd hard dependency. This enables cloud deployment (EC2, VPS) for 24/7 uptime. |
| C4 | **No backtesting framework.** The RL loop learns forward only. There is no way to validate a strategy change against historical data before deploying. | Add a `maki_trading backtest` CLI command that replays a saved `trade_outcomes.jsonl` against historical bar data from Alpaca's historical API. Critical for strategy development iteration speed. |
| C5 | **Single-user, single-account architecture.** All paths are hardcoded to one Alpaca account. | Abstract the account layer behind an interface. This is the prerequisite for managing multiple accounts (family office, fund-of-funds model). |
| C6 | **No regulatory consideration.** Automated trading advice triggers RIA registration in the US if offered to others. | Consult a fintech lawyer before any commercial launch. Likely paths: (a) sell the *software*, not the signals; (b) operate as a proprietary trading firm on your own capital; (c) obtain RIA/investment advisor registration. |
| C7 | **LLM alpha generation is unproven at scale.** The agents produce qualitative analysis, but whether this systematically beats a simple momentum baseline is unknown. | Run an A/B comparison: deploy the system alongside a benchmark rule-based strategy (e.g., simple MA crossover with the same risk rules). Measure alpha. If there is no alpha over 3 months, the LLM analysis step may be decorative. |

#### Commercial Verdict

**Highest commercial potential in the portfolio.** Fintech is high-value, the technical architecture is sound, and the explainability angle is a real differentiator. The path to revenue is clear: proven paper performance → live proprietary trading → licensed signal SaaS or fund management.

---

## 3. Startup Structure Options

### Option A — One Company, Two Tracks (Recommended)

**Track 1 — maki_trading** (commercial product, revenue target)

- 6-month paper trading run with public audited dashboard
- Live trading on proprietary capital
- Subscription SaaS: "AI trading assistant" for individual investors ($49–199/month)
- Requires regulatory review before any customer-facing launch

**Track 2 — maki framework** (open-source, distribution moat)

- Publish to PyPI, launch docs site
- maki_newsletter and maki_story become showcase demos
- Community → enterprise support and consulting revenue
- Positions the startup as an "AI infrastructure company," which is more fundable than a pure fintech play

**Why one company:** The maki framework IS the defensible moat of maki_trading. If a competitor wants to replicate the trading system, they have to rebuild the framework. Keeping them together preserves this advantage. Splitting them creates a fragile dependency between two separate cap tables.

---

### Option B — maki_story as a Separate Spin-out (Conditional)

Only worth considering **if** image generation is added and a proper web UX is built. At that point, "AI comic studio" is a different buyer, different distribution, and different fundraising story than fintech. The creative AI market is large enough to justify a standalone seed round ($1–3M).

**Condition to trigger:** A working web demo with brief-to-storyboard-with-images in under 5 minutes, validated by 20+ external users.

---

## 4. Prioritised Fix Roadmap

| Priority | Fix | Effort | Status |
|----------|-----|--------|--------|
| P0 | ~~Add `CloudBackend` (OpenAI-compatible) to maki~~ | ~~1 day~~ | ✅ **Done.** OpenAI + Anthropic backends shipped. |
| P0 | ~~Add distributed deployment layer to maki~~ | ~~3–5 weeks~~ | ✅ **Done.** Full 5-phase implementation: server, proxy, state store, circuit breaker, tracing, mTLS. |
| P0 | ~~Add auth layer to maki (token-gated API)~~ | ~~1 day~~ | ✅ **Done.** `AgentServer` ships with Bearer token auth. |
| P0 | Run maki_trading paper for 6 months, audit logs | Ongoing | 🔄 In progress — start the clock |
| P1 | Publish maki to PyPI + launch docs site | 3 days | ⬜ Not started — highest remaining leverage |
| P1 | Containerise maki_trading with Docker | 2 days | ⬜ Not started |
| P1 | Add backtesting CLI to maki_trading | 1 week | ⬜ Not started |
| P2 | Parameterise maki_newsletter voice/style config | 2 days | ⬜ Not started |
| P2 | Fintech regulatory consultation | — | ⬜ Not started — mandatory before customer launch |
| P3 | Add image generation plugin to maki (SD / Replicate) | 3 days | ⬜ Not started |
| P3 | maki_story web interface (brief → storyboard) | 2 weeks | ⬜ Not started |

---

## 5. Overall Verdict

| Product | Readiness | Commercial Potential | Recommended Action |
|---------|-----------|---------------------|--------------------|
| maki (framework) | **High** *(was: Medium)* | Medium–High (B2B infra) | Cloud backends + distributed layer now shipped. Next: PyPI publish and docs site. |
| maki_newsletter | High (personal use) | Low (as-is) | Showcase app + future SaaS feature; cloud LLM now unblocked. |
| maki_story | Low | High (if images added) | Conditional spin-out after image integration. |
| maki_trading | Medium | Very High | Primary commercial bet — start the paper trading clock now; cloud LLM latency concern resolved. |

### What changed since the original report

The original report's two P0 technical blockers for maki have been resolved:

1. **Cloud LLM dependency** — `MakiOpenAI` and `MakiAnthropic` are now first-class backends. Every product that was blocked on local-only Ollama (trading latency, newsletter cost, story compute) can now route to cloud models.

2. **No deployment story** — The `maki[distributed]` package lets any agent run as a networked service on any machine, with the orchestrator treating remote agents identically to local ones. This is the architectural foundation for a multi-tenant SaaS product built on maki.

The framework has moved from "impressive personal tool" to "deployable infrastructure." The remaining gap before serious commercial traction is visibility: **PyPI and documentation.** Without those, the distributed layer ships to nobody.

**The recommendation is unchanged: one company, two tracks.** Trading is the commercial bet; the framework (now substantially more capable) is the moat. The distributed layer means maki_trading can be cloud-deployed for 24/7 operation without waiting for the Docker containerisation work — `maki serve` covers it already.

The ecosystem is technically impressive and unusually coherent for a solo or small-team build. The main risk is spreading attention across four products simultaneously.

**Pick one commercial bet (trading), invest in the framework as a moat, and let newsletter and story mature at lower priority.** The technical foundation is now solid enough, and production-deployment-ready, to build a real company on.
