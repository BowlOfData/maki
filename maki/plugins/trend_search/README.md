# Trend Search Plugin

`trend_search` is responsible for trend-intelligence retrieval rather than article discovery.

## Features

- Google Trends rising query retrieval via `pytrends`
- Per-keyword rate-limit-aware querying
- Fallback to top queries when rising data is unavailable

## Usage

```python
from maki.plugins.trend_search import TrendSearch

plugin = TrendSearch()
trends = plugin.fetch_google_trends(
    seed_keywords=["artificial intelligence", "cybersecurity"],
    timeframe="now 7-d",
)
print(trends)
```
