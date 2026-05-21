# Alpaca News Plugin

Fetches crypto news from Alpaca's news API and from free public RSS feeds (CoinDesk, The Block, CryptoPanic, CoinTelegraph). No API key is required for RSS-only usage.

## Requirements

```
APCA_API_KEY_ID=<your alpaca api key>
APCA_API_SECRET_KEY=<your alpaca secret key>
```

Dependency: `feedparser`

## Usage

```python
from maki.plugins.alpaca_news.alpaca_news import AlpacaNews

plugin = AlpacaNews()

# Alpaca news for specific symbols
articles = plugin.get_news(symbols=["BTC/USD", "ETH/USD"], since_hours=6, limit=10)

# Free RSS feeds only
rss = plugin.get_rss_news(since_hours=12, limit=20)

# Merged, deduplicated from both sources
all_news = plugin.get_all_news(symbols=["BTC/USD"], since_hours=6, limit=15)
```

## Methods

### `get_news(symbols=None, since_hours=6, limit=20)`

Fetches recent articles from the Alpaca news API, optionally filtered by symbol list.

**Returns:** list of dicts with keys `source`, `id`, `headline`, `summary`, `symbols`, `published_at`, `url`.

### `get_rss_news(symbols=None, since_hours=6, limit=20)`

Fetches from free RSS feeds, filtering by keyword match when symbols are provided.

**Returns:** same schema as `get_news`.

### `get_all_news(symbols=None, since_hours=6, limit=20)`

Merges Alpaca + RSS results, deduplicates by headline, and sorts newest-first.

**Returns:** same schema as `get_news`.
