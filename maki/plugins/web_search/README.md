# Web Search Plugin for Maki Framework

Focuses on article, community, and repository discovery from public web sources.

## Features

- **RSS/Atom feeds** — current-week articles filtered by keyword
- **HackerNews** — full-text search via the Algolia public API
- **GitHub Trending** — most-starred repositories created in the last 7 days
- **Lobste.rs** — curated developer link-aggregator, current-week stories
- **Reddit** — hot posts from chosen subreddits, no authentication needed

All date filtering uses the current ISO calendar week (Monday 00:00 UTC through now).
Articles with an unparseable date are included by default (benefit of the doubt).

## Installation

```bash
pip install feedparser requests
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
