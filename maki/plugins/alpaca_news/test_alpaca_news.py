import pytest
from unittest.mock import MagicMock, patch
from maki.plugins.alpaca_news.alpaca_news import _symbol_keywords, _match_symbols, _parse_feed_date


def test_symbol_keywords_crypto():
    kws = _symbol_keywords(["BTC/USD", "ETH/USD"])
    assert "btc" in kws
    assert "bitcoin" in kws
    assert "eth" in kws
    assert "ethereum" in kws


def test_symbol_keywords_unknown():
    kws = _symbol_keywords(["XYZ/USD"])
    assert "xyz" in kws


def test_symbol_keywords_empty():
    assert _symbol_keywords([]) == []


def test_match_symbols_hit():
    matched = _match_symbols("bitcoin price surges", ["BTC/USD"])
    assert "BTC/USD" in matched


def test_match_symbols_miss():
    matched = _match_symbols("ethereum network update", ["BTC/USD"])
    assert matched == []


def test_parse_feed_date_none():
    entry = MagicMock(spec=[])
    result = _parse_feed_date(entry)
    assert result is None


def test_get_all_news_deduplicates():
    with patch("maki.plugins.alpaca_news.alpaca_news.AlpacaNews.__init__", lambda self, *a, **kw: None):
        from maki.plugins.alpaca_news.alpaca_news import AlpacaNews
        plugin = AlpacaNews.__new__(AlpacaNews)

    shared_headline = "BTC hits new high"
    article = {"source": "alpaca", "id": "1", "headline": shared_headline,
               "summary": "", "symbols": [], "published_at": "2024-01-01T00:00:00+00:00", "url": ""}
    dup = dict(article, source="CoinDesk")

    plugin.get_news = MagicMock(return_value=[article])
    plugin.get_rss_news = MagicMock(return_value=[dup])

    result = plugin.get_all_news()
    headlines = [a["headline"] for a in result]
    assert headlines.count(shared_headline) == 1
