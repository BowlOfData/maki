# Web Search Plugin for Maki Framework

Aggregates articles and trend signals from multiple sources without requiring
API keys (except the optional Pexels image search).

## Features

- **RSS/Atom feeds** — current-week articles filtered by keyword
- **HackerNews** — full-text search via the Algolia public API
- **GitHub Trending** — most-starred repositories created in the last 7 days
- **Lobste.rs** — curated developer link-aggregator, current-week stories
- **Reddit** — hot posts from chosen subreddits, no authentication needed
- **Google Trends** — rising queries for seed keywords (`pytrends`)
- **Pexels** — landscape cover image search (API key required)

All date filtering uses the current ISO calendar week (Monday 00:00 UTC through now).
Articles with an unparseable date are included by default (benefit of the doubt).

## Installation

```bash
pip install feedparser pytrends requests
# Optional: for Pexels image search, obtain a free key at https://www.pexels.com/api/
```

## Usage

```python
from maki.plugins.web_search.web_search import WebSearch

ws = WebSearch()
```

### `search_rss(feeds, max_per_feed=10, keywords=None)`

Fetches articles from RSS/Atom feeds filtered to the current ISO week.

```python
articles = ws.search_rss(
    feeds={
        "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
        "Wired":        "https://www.wired.com/feed/rss",
    },
    max_per_feed=5,
    keywords=["AI", "machine learning"],  # None = no keyword filter
)
```

Returns a list of dicts: `title`, `url`, `snippet`, `source`, `published`.

---

### `search_hackernews(query, max_results=10)`

Searches HackerNews stories from the current week via the Algolia API.

```python
articles = ws.search_hackernews("open source security", max_results=10)
```

Returns a list of dicts: `title`, `url`, `snippet`, `source`, `published`.

---

### `fetch_github_trending(max_results=10)`

Fetches repositories created in the last 7 days with the most stars,
using the public GitHub Search API (60 unauthenticated requests/hour).

```python
repos = ws.fetch_github_trending(max_results=10)
```

Returns a list of dicts: `title`, `url`, `snippet`, `source`, `published`, `topics`.

---

### `fetch_lobsters(max_results=10)`

Fetches hot stories from the Lobste.rs RSS feed, filtered to the current week.
No authentication required; all content is already developer-curated.

```python
stories = ws.fetch_lobsters(max_results=10)
```

Returns a list of dicts: `title`, `url`, `snippet`, `source`, `published`.

---

### `fetch_reddit_hot(subreddits, max_per_sub=10)`

Fetches hot posts from a list of subreddits using Reddit's public JSON API.
Filters to the current ISO week; skips self-posts, media URLs, and posts with
fewer than 10 upvotes.

```python
posts = ws.fetch_reddit_hot(
    subreddits=["MachineLearning", "netsec", "technology"],
    max_per_sub=5,
)
```

Returns a list of dicts: `title`, `url`, `snippet`, `source`, `published`.

---

### `fetch_google_trends(seed_keywords, timeframe="now 7-d", geo="")`

Retrieves rising related queries for each seed keyword from Google Trends
via `pytrends`. Keywords are queried one at a time to stay within rate limits.

```python
trends = ws.fetch_google_trends(
    seed_keywords=["artificial intelligence", "cybersecurity"],
    timeframe="now 7-d",
    geo="",          # empty = worldwide; "US" = United States only
)
# {"artificial intelligence": ["gpt-4o", "claude 3", ...], "cybersecurity": [...]}
```

Returns a dict mapping each seed keyword to a list of rising query strings.

---

### `fetch_pexels_image(query, api_key)`

Searches Pexels for a landscape photo matching `query` and returns its URL.
Requires a free API key from [pexels.com/api](https://www.pexels.com/api/).

```python
url = ws.fetch_pexels_image("technology innovation", api_key="your_key_here")
```

Returns a URL string, or `None` if no image is found or the key is missing.

---

## Error Handling

Every method catches exceptions internally and logs a warning, returning an
empty list (or empty dict / `None`) on failure. This allows the pipeline to
continue gracefully when a single source is unavailable.

## Integration with Maki Agents

```python
from maki.plugins.web_search.web_search import WebSearch
from maki.agents.agent_manager import AgentManager
from maki.makiLLama import MakiLLama

llm = MakiLLama(model="gemma4:26b", base_url="http://localhost:11434")
manager = AgentManager(llm)
ws = WebSearch()

# Collect candidates from all sources
rss_articles   = ws.search_rss({"Wired": "https://www.wired.com/feed/rss"})
hn_articles    = ws.search_hackernews("AI security")
github_repos   = ws.fetch_github_trending(max_results=5)
lobsters       = ws.fetch_lobsters(max_results=5)

candidates = rss_articles + hn_articles + github_repos + lobsters
```
