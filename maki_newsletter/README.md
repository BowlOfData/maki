# Maki Newsletter

A multi-agent application that automatically generates a weekly technical newsletter.
It fetches current-week articles from RSS feeds and HackerNews, ranks them with an LLM,
produces per-article summaries, and writes a curated Markdown newsletter.

---

## How it works

The pipeline runs in two separate steps to give you a chance to review articles before
the final newsletter is assembled.

```
Step 1 — pipeline                     Step 2 — generate
─────────────────────────────         ─────────────────────────────
Search RSS + HackerNews           →   Load summaries JSON
Download articles as Markdown     →   Filter by score
LLM: extract metadata             →   LLM: write newsletter
LLM: rank, select top 10          →
LLM: summarise each article       →
Write evaluation file  ← STOP         Write news_<WW>_<YYYY>.md
```

### Pipeline stages

| # | Stage | Type | Output |
|---|-------|------|--------|
| 1 | Search | function | Article candidates from RSS + HackerNews |
| 2 | Download | function | Markdown files in `output/articles/<month>/<week>/` |
| 3 | Read | LLM agent | Metadata: topic, key points, technologies, quality score |
| 4 | Rank | LLM agent | Top 10 articles selected |
| 5 | Summarise | LLM agent | ≤200-word summary per article |
| 6 | Evaluate | function | `evaluate_<week>.md` + `summaries_<WW>_<YYYY>.json` |
| 7 | Write | LLM agent | `news_<WW>_<YYYY>.md` (triggered separately) |

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running locally with the configured model pulled
- The `maki` package installed (see below)

### Install

From the repository root:

```bash
pip install -e .
```

### Pull the LLM model

```bash
ollama pull gemma4:26b
```

---

## Usage

### Step 1 — Run the pipeline

```bash
python -m maki_newsletter.main
```

This fetches articles, ranks them, summarises each one, and writes two files:

- **Evaluation file** — `output/articles/<month>/<week>/evaluate_<week>.md`
- **Summaries data** — `output/summaries_<WW>_<YYYY>.json`

The pipeline stops here so you can review the evaluation file before committing
to a final newsletter.

### Step 2 — Review the evaluation file

Open the evaluation file. Articles are grouped into three categories:

| Score | Category | Default action |
|-------|----------|----------------|
| 6–10 | ✅ Good match | Included by default |
| 4–5 | ⚠️ Needs review | Excluded by default (use `--min-score 4` to include) |
| 0–3 | ❌ Discard | Excluded |

Example path: `output/articles/april/14/evaluate_14.md`

### Step 3 — Generate the newsletter

```bash
# Include only good match articles (score >= 6, default)
python -m maki_newsletter.generate

# Also include "needs review" articles (score >= 4)
python -m maki_newsletter.generate --min-score 4
```

Output: `output/news_<WW>_<YYYY>.md`

---

## Output structure

```
maki_newsletter/
└── output/
    ├── news_14_2026.md                  ← final newsletter
    ├── summaries_14_2026.json           ← intermediate data for generate.py
    └── articles/
        └── april/
            └── 14/
                ├── evaluate_14.md       ← evaluation file (review this)
                ├── techcrunch_com_…md   ← downloaded article
                └── …
```

---

## Configuration

All settings are in `maki_newsletter/config.py`.

### RSS feeds

```python
RSS_FEEDS = {
    "TechCrunch":   "https://techcrunch.com/feed/",
    "Wired":        "https://www.wired.com/feed/rss",
    "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
    "TechRadar":    "https://www.techradar.com/rss",
    "Gizmodo":      "https://gizmodo.com/rss",
    "HackerNews":   "https://news.ycombinator.com/rss",
}
```

Add or remove feeds by editing this dict. Any RSS/Atom URL works.

### Topic queries

```python
SEARCH_QUERIES = [
    "artificial intelligence latest research 2026",
    "cyber security vulnerabilities",
    ...
]
```

Used to search the HackerNews Algolia API. Edit to match your interests.

### LLM settings

```python
LLM_MODEL   = "gemma4:26b"           # any model available in Ollama
OLLAMA_HOST = "http://localhost:11434"
```

### Volume controls

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_CANDIDATES` | 30 | Max article candidates before ranking |
| `TOP_N` | 10 | Articles selected for the newsletter |
| `MAX_PER_FEED` | 10 | Max articles fetched per RSS feed |
| `MAX_HN_PER_QUERY` | 5 | Max HackerNews results per query |
| `MAX_ARTICLE_CHARS` | 6000 | Characters sent to LLM per article |
| `SUMMARY_MAX_WORDS` | 200 | Max words per article summary |

---

## Article filtering

Only articles published **during the current ISO calendar week** (Monday 00:00 UTC
through Sunday) are included. Articles from previous weeks are discarded before
any LLM processing.

---

## Troubleshooting

**No candidates found**
Verify network access and that at least one RSS feed URL is reachable.

**LLM 404 error**
The configured model is not available in Ollama. Run `ollama list` to see
installed models and update `LLM_MODEL` in `config.py`.

**All articles discarded (content too short)**
The article page returned a paywall or redirect. This is expected for some
sources — the pipeline continues with the remaining articles.

**No summaries JSON found (generate.py)**
The main pipeline has not been run yet, or the `output/` directory was cleared.
Run `python -m maki_newsletter.main` first.
