# Maki Newsletter

A multi-agent application that automatically generates a weekly technical newsletter.
It fetches current-week articles from RSS feeds and HackerNews, ranks them with an LLM,
produces per-article summaries, writes a curated Markdown newsletter, and publishes it
to an Altervista WordPress blog.

---

## How it works

The pipeline runs in three steps: generate content, review, then publish.

```
Step 1 — pipeline                     Step 2 — generate         Step 3 — publish
─────────────────────────────         ──────────────────         ────────────────────────────
Search RSS + HackerNews           →   Load summaries JSON    →   Build WordPress page HTML
Download articles as Markdown     →   Filter by score        →   POST to Altervista via REST API
LLM: extract metadata             →   LLM: write newsletter  →   Back-fill altervista_url in JSON
LLM: rank, select top articles    →
LLM: summarise each article (2s)  →
Fetch Pexels cover image          →
Write evaluation file  ← STOP         Write news_<WW>_<YYYY>.md
```

### Pipeline stages

| # | Stage | Type | Output |
|---|-------|------|--------|
| 1 | Search | function | Article candidates from RSS + HackerNews |
| 2 | Download | function | Markdown files in `output/articles/<month>/<week>/` |
| 3 | Read | LLM agent | Metadata: topic, key points, technologies, quality score |
| 4 | Rank | LLM agent | Top articles selected |
| 5 | Summarise | LLM agent | 2-sentence summary per article (cached in `_summary.txt`) |
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

This fetches articles, ranks them, summarises each one (2 sentences), fetches a
thematic cover image from Pexels, and writes two files:

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

To **remove an article permanently**, delete it from `summaries_<WW>_<YYYY>.json`.
The pipeline will detect the removal, delete all related files, and never re-add
it for the same week.

Example path: `output/articles/april/14/evaluate_14.md`

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

- Title linked to the original source, with an anchor (`#article-title-slug`) for direct linking
- Source badge and quality score
- Main topic
- **TL;DR** — the 2-sentence pipeline summary
- **Full article resume** — up to 8 paragraphs extracted from the downloaded Markdown file
- Technology tags
- Prominent **Read full article →** button

After publishing, `altervista_url` is back-filled into each entry in
`summaries_<WW>_<YYYY>.json`, so every article carries a direct deep-link to its
section on the Altervista page.

A `published_<WW>_<YYYY>.json` sidecar is saved alongside the summaries to record the
WordPress page ID; subsequent runs update the existing page instead of creating a duplicate.

---

## Output structure

```
maki_newsletter/
└── output/
    ├── news_18_2026.md                  ← final newsletter (Markdown)
    ├── summaries_18_2026.json           ← article data (enriched with altervista_url after publish)
    ├── published_18_2026.json           ← WordPress page ID/URL sidecar
    ├── checkpoint_18_2026.json          ← URL set for removal detection
    ├── removed_18_2026.json             ← permanently removed article URLs
    └── articles/
        └── may/
            └── 18/
                ├── evaluate_18.md               ← evaluation file (review this)
                ├── techcrunch_com_…md           ← downloaded article
                ├── techcrunch_com_…_meta.json   ← extracted metadata
                ├── techcrunch_com_…_summary.txt ← cached 2-sentence summary
                └── …
```

---

## Article removal

If you manually remove an article from `summaries_<WW>_<YYYY>.json`:

1. The next pipeline run detects the missing URL by comparing against the checkpoint.
2. All related files are deleted (`.md`, `_meta.json`, `_summary.txt`).
3. The URL is saved to `removed_<WW>_<YYYY>.json` and permanently excluded from
   future runs for that week.

---

## Summary caching

Each article's 2-sentence summary is cached in a `_summary.txt` sidecar next to its
`.md` file. On re-runs, the cache is used directly — the LLM is only called for
articles that are new or whose cache is empty.

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
| `MAX_CANDIDATES` | 50 | Max article candidates before ranking |
| `TOP_N` | 50 | Articles selected for the newsletter |
| `MAX_PER_FEED` | 10 | Max articles fetched per RSS feed |
| `MAX_HN_PER_QUERY` | 5 | Max HackerNews results per query |
| `MAX_ARTICLE_CHARS` | 10000 | Characters sent to LLM per article |
| `SUMMARY_MIN_WORDS` | 35 | Preferred minimum words per 2-sentence article summary |
| `SUMMARY_MAX_WORDS` | 48 | Preferred maximum words per 2-sentence article summary |

---

## Article filtering

Only articles published **during the current ISO calendar week** (Monday 00:00 UTC
through Sunday) are included. Articles from previous weeks are discarded before
any LLM processing.
