# Maki Newsletter

A multi-agent application that automatically generates a weekly technical newsletter.
It fetches current-week articles from RSS feeds, HackerNews, GitHub trending, and
Lobste.rs, ranks them with an LLM guided by trend signals, produces per-article
summaries and long-form resumes, writes a curated Markdown newsletter, and publishes
it to an Altervista WordPress blog.

---

## How it works

The pipeline runs in three steps: generate content, review, then publish.

```
Step 1 — pipeline                           Step 2 — generate         Step 3 — publish
───────────────────────────────────         ──────────────────         ────────────────────────────
Fetch trends (Google, Reddit,           →   Load summaries JSON    →   Build WordPress page HTML
  GitHub, Lobste.rs)                    →   Filter by score        →   POST to Altervista via REST API
Search RSS + HackerNews                 →   LLM: write newsletter  →   Back-fill altervista_url in JSON
Download articles as Markdown           →
LLM: extract metadata                   →
LLM: rank, select top articles          →
LLM: summarise each article (2s)        →
LLM: write long-form resume per article →
Fetch Pexels cover image                →
Write evaluation file  ← STOP              Write news_<WW>_<YYYY>.md
```

### Pipeline stages

| # | Stage | Type | Output |
|---|-------|------|--------|
| 1b | Trends | function | Keywords from Google Trends, Reddit, GitHub trending, Lobste.rs |
| 1 | Search | function | Article candidates from RSS + HackerNews, guided by trend keywords |
| 2 | Download | function | Markdown files + incremental manifest per article |
| 3 | Read | LLM agent | Metadata: topic, key points, technologies, quality score |
| 4 | Rank | LLM agent | Top articles selected, boosted by trending keywords |
| 5 | Summarise | LLM agent | 2-sentence summary (35–48 words) + 3-paragraph long resume per article |
| 6 | Evaluate | function | `evaluate_<week>.md` + `summaries_<WW>_<YYYY>.json` |
| 7 | Write | LLM agent | `news_<WW>_<YYYY>.md` with Pexels cover image (triggered separately) |
| 8 | Publish | function | WordPress page on Altervista (triggered separately) |

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running locally with the configured model pulled
- The `maki` package installed (see below)
- A `.env` file in the repository root (see [Environment variables](#environment-variables))

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

## Environment variables

Create a `.env` file in the repository root (`maki/.env`). All keys are loaded
automatically by `config.py` via `python-dotenv`.

```env
# Pexels — free cover image per newsletter (https://www.pexels.com/api/)
PEXELS_API_KEY=your_pexels_api_key

# Altervista WordPress publishing
ALTERVISTA_SITE_URL=https://yourblog.altervista.org
ALTERVISTA_USERNAME=your_wordpress_username
ALTERVISTA_APP_PASSWORD=xxxx xxxx xxxx xxxx xxxx xxxx

# Optional: pin the Newsletter parent page ID to skip auto-discovery
# ALTERVISTA_NEWSLETTER_PAGE_ID=123
```

The **Application Password** is generated in WordPress under
*Users → Profile → Application Passwords*. Spaces in the password are fine.

---

## Usage

### Step 1 — Run the pipeline

```bash
python -m maki_newsletter.main
```

This fetches trend signals, discovers articles, ranks them, produces a 2-sentence
summary and a long-form 3-paragraph resume for each, fetches a thematic cover image
from Pexels, and writes two files:

- **Evaluation file** — `output/articles/<month>/<week>/evaluate_<week>.md`
- **Summaries data** — `output/summaries_<WW>_<YYYY>.json`

The pipeline stops here so you can review the evaluation file before committing
to a final newsletter.

#### Resume from stage 3

If downloads completed but the pipeline was interrupted during LLM stages (read, rank,
summarise, evaluate), you can resume without re-downloading:

```bash
python -m maki_newsletter.resume

# Re-fetch trend signals before ranking (adds network calls)
python -m maki_newsletter.resume --trends
```

The resume command reads the incremental manifest written by stage 2 and runs
stages 3–6 against the current week's already-downloaded articles.

### Step 2 — Review the evaluation file

Open the evaluation file. Articles are grouped into three categories:

| Score | Category | Default action |
|-------|----------|----------------|
| 6–10 | ✅ Good match | Included by default |
| 4–5 | ⚠️ Needs review | Excluded by default (use `--min-score 4` to include) |
| 0–3 | ❌ Discard | Excluded |

To **remove an article permanently**, delete it from `summaries_<WW>_<YYYY>.json`.
The pipeline will detect the removal, delete all related files, and never re-add
it for the same week.

### Step 3 — Generate the newsletter

```bash
# Include only good match articles (score >= 6, default)
python -m maki_newsletter.generate

# Also include "needs review" articles (score >= 4)
python -m maki_newsletter.generate --min-score 4
```

Output: `output/news_<WW>_<YYYY>.md`

### Step 4 — Publish to Altervista

```bash
# Publish the latest week
python -m maki_newsletter.publish

# Publish a specific week
python -m maki_newsletter.publish --week 18 --year 2026

# Preview without publishing
python -m maki_newsletter.publish --dry-run
```

This creates (or updates) a WordPress page under the **Newsletter** menu on Altervista.
Each article on the page includes:

- Title linked to the original source, with an anchor for direct linking
- Source badge and quality score
- Main topic
- **TL;DR** — the 2-sentence pipeline summary
- **Full article resume** — the 3-paragraph long-form resume generated by the pipeline
- Technology tags
- Prominent **Read full article →** button

After publishing, `altervista_url` is back-filled into each entry in
`summaries_<WW>_<YYYY>.json` and used as the display URL in the newsletter Markdown.

A `published_<WW>_<YYYY>.json` sidecar is saved alongside the summaries to record the
WordPress page ID; subsequent runs update the existing page instead of creating a duplicate.

---

## Article removal

If you manually remove an article from `summaries_<WW>_<YYYY>.json`:

1. The next pipeline run detects the missing URL by comparing against the checkpoint.
2. All related files are deleted (`.md`, `_meta.json`, `_summary.txt`, `_long_resume.txt`).
3. The URL is saved to `removed_<WW>_<YYYY>.json` and permanently excluded from
   future runs for that week.

---

## Caching

Each article accumulates sidecar files next to its downloaded `.md`:

| Sidecar | Content | Reused when |
|---------|---------|-------------|
| `_meta.json` | LLM-extracted metadata (topic, score, …) | Always — skips stage 3 LLM call |
| `_summary.txt` | 2-sentence short summary | Valid + uniform — skips stage 5 short-summary call |
| `_long_resume.txt` | 3-paragraph long-form resume | Present — skips stage 5 long-resume call |

A cached summary is reused only if it passes the uniformity check (2 complete
sentences, 35–48 words). If it fails, the cache is deleted and the LLM is called again.

---

## Trend sources

Stage 1b fetches signals from four sources before article discovery:

| Source | What is fetched | How it is used |
|--------|----------------|----------------|
| Google Trends | Rising queries for seed keywords | Boost matching articles in ranking |
| Reddit | Hot posts from tech subreddits | Added to candidate pool + keyword list |
| GitHub | Trending repos (rolling 7-day window) | Topics + titles added to keyword list |
| Lobste.rs | Current-week tech articles | Titles added to keyword list |

---

## Cover image

The pipeline fetches a thematic cover image from [Pexels](https://www.pexels.com) using
the most frequent technology keywords across the week's articles. The image is embedded
between the introduction and the first article in the generated Markdown newsletter.
Requires `PEXELS_API_KEY` in `.env`.

---

## Volume controls

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_CANDIDATES` | 20 | Max article candidates before ranking |
| `TOP_N` | 20 | Articles selected for the newsletter |
| `MAX_PER_FEED` | 5 | Max articles fetched per RSS feed |
| `MAX_HN_PER_QUERY` | 5 | Max HackerNews results per query |
| `MAX_REDDIT_PER_SUB` | 5 | Max Reddit posts per subreddit |
| `MAX_GITHUB_REPOS` | 10 | Max GitHub trending repos fetched |
| `MAX_LOBSTERS_ARTICLES` | 10 | Max Lobste.rs articles fetched |
| `MAX_ARTICLE_CHARS` | 20000 | Characters sent to LLM per article |
| `SUMMARY_MIN_WORDS` | 35 | Minimum words in a 2-sentence article summary |
| `SUMMARY_MAX_WORDS` | 48 | Maximum words in a 2-sentence article summary |

---

## Article filtering

Only articles published **during the current ISO calendar week** (Monday 00:00 UTC
through Sunday) are included from RSS and Lobste.rs. GitHub trending uses a rolling
7-day window. Articles from previous weeks are discarded before any LLM processing.
