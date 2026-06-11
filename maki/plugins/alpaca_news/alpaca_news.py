"""
Alpaca News plugin for Maki.

Fetches news articles from Alpaca's news endpoint.
Also aggregates free RSS feeds (CoinDesk, The Block, CryptoPanic, CoinTelegraph).
No CryptoPanic API key required — uses the public RSS feed.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    import feedparser
except ImportError as e:
    raise ImportError(
        'feedparser is not installed. Run: pip install "maki[alpaca]"'
    ) from e

logger = logging.getLogger(__name__)

ALLOWED_METHODS = ["get_news", "get_rss_news", "get_all_news"]

FREE_RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "TheBlock": "https://www.theblock.co/rss.xml",
    "CryptoPanic": "https://cryptopanic.com/news/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
}


class AlpacaNews:
    def __init__(self, maki_instance=None):
        try:
            from alpaca.data.historical import NewsClient
        except ImportError as e:
            raise ImportError(
                'alpaca-py is not installed. Run: pip install "maki[alpaca]"'
            ) from e

        api_key = os.environ.get("APCA_API_KEY_ID")
        api_secret = os.environ.get("APCA_API_SECRET_KEY")
        self._client = NewsClient(api_key=api_key, secret_key=api_secret)
        logger.info("AlpacaNews plugin initialised")

    def get_news(
        self,
        symbols: Optional[List[str]] = None,
        since_hours: int = 6,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Fetch recent news from Alpaca for the given symbols."""
        from alpaca.data.requests import NewsRequest

        start = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        symbols_param = ",".join(symbols) if symbols else None
        req = NewsRequest(symbols=symbols_param, start=start, limit=limit, sort="desc")
        try:
            response = self._client.get_news(req)
            articles = []
            all_items = [item for items in response.data.values() for item in items]
            for item in all_items:
                articles.append({
                    "source": "alpaca",
                    "id": str(item.id),
                    "headline": item.headline,
                    "summary": item.summary or "",
                    "symbols": list(item.symbols or []),
                    "published_at": item.created_at.isoformat() if item.created_at else "",
                    "url": item.url or "",
                })
            return articles
        except Exception as e:
            logger.warning(f"Alpaca news fetch failed: {e}")
            return []

    def get_rss_news(
        self,
        symbols: Optional[List[str]] = None,
        since_hours: int = 6,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Fetch news from free RSS feeds, optionally filtered by keyword match."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        keywords = _symbol_keywords(symbols or [])
        articles = []
        for source_name, url in FREE_RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    title = getattr(entry, "title", "")
                    summary = getattr(entry, "summary", "")
                    text = (title + " " + summary).lower()
                    if keywords and not any(kw in text for kw in keywords):
                        continue
                    published = _parse_feed_date(entry)
                    if published and published < cutoff:
                        continue
                    articles.append({
                        "source": source_name,
                        "id": getattr(entry, "id", entry.get("link", "")),
                        "headline": title,
                        "summary": summary[:300],
                        "symbols": _match_symbols(text, symbols or []),
                        "published_at": published.isoformat() if published else "",
                        "url": getattr(entry, "link", ""),
                    })
                    if len(articles) >= limit * len(FREE_RSS_FEEDS):
                        break
            except Exception as e:
                logger.warning(f"RSS feed {source_name} failed: {e}")
        articles.sort(key=lambda a: a["published_at"], reverse=True)
        return articles[:limit]

    def get_all_news(
        self,
        symbols: Optional[List[str]] = None,
        since_hours: int = 6,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Merge Alpaca news + RSS news, deduplicated and sorted by date."""
        alpaca = self.get_news(symbols=symbols, since_hours=since_hours, limit=limit)
        rss = self.get_rss_news(symbols=symbols, since_hours=since_hours, limit=limit)
        seen = {a["headline"] for a in alpaca}
        merged = alpaca + [a for a in rss if a["headline"] not in seen]
        merged.sort(key=lambda a: a["published_at"], reverse=True)
        return merged[:limit]


def register_plugin(maki_instance=None):
    return AlpacaNews(maki_instance)


def _symbol_keywords(symbols: List[str]) -> List[str]:
    keywords = []
    for s in symbols:
        base = s.split("/")[0].lower()
        keywords.append(base)
        _CRYPTO_NAMES = {"btc": "bitcoin", "eth": "ethereum", "sol": "solana",
                         "bnb": "binance", "ada": "cardano", "dot": "polkadot"}
        if base in _CRYPTO_NAMES:
            keywords.append(_CRYPTO_NAMES[base])
    return keywords


def _match_symbols(text: str, symbols: List[str]) -> List[str]:
    return [s for s in symbols if any(kw in text for kw in _symbol_keywords([s]))]


def _parse_feed_date(entry) -> Optional[datetime]:
    import email.utils
    for attr in ("published_parsed", "updated_parsed"):
        val = getattr(entry, attr, None)
        if val:
            try:
                import time as _time
                return datetime.fromtimestamp(_time.mktime(val), tz=timezone.utc)
            except Exception:
                pass
    return None
