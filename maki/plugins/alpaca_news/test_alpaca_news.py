import pytest
from unittest.mock import MagicMock, patch

pytest.importorskip("feedparser")

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


def test_get_rss_news_fetches_via_connector():
    """RSS feeds must be fetched through the hardened Connector (not feedparser's
    own unbounded network fetch) so timeouts/SSRF validation apply."""
    with patch("maki.plugins.alpaca_news.alpaca_news.AlpacaNews.__init__", lambda self, *a, **kw: None):
        from maki.plugins.alpaca_news.alpaca_news import AlpacaNews
        plugin = AlpacaNews.__new__(AlpacaNews)

    rss_xml = """<?xml version="1.0"?>
    <rss version="2.0"><channel>
        <item>
            <title>Bitcoin surges past new high</title>
            <summary>Market update</summary>
            <link>https://example.com/a</link>
        </item>
    </channel></rss>"""
    mock_resp = MagicMock()
    mock_resp.text = rss_xml

    with patch("maki.plugins.alpaca_news.alpaca_news._cdn_get", return_value=mock_resp) as mock_get:
        results = plugin.get_rss_news(symbols=["BTC/USD"], since_hours=999999, limit=5)

    assert mock_get.called
    fetched_urls = {call.args[0] for call in mock_get.call_args_list}
    from maki.plugins.alpaca_news.alpaca_news import FREE_RSS_FEEDS
    assert fetched_urls == set(FREE_RSS_FEEDS.values())
    assert any("Bitcoin" in a["headline"] for a in results)


def test_get_rss_news_feed_failure_is_skipped_not_raised():
    with patch("maki.plugins.alpaca_news.alpaca_news.AlpacaNews.__init__", lambda self, *a, **kw: None):
        from maki.plugins.alpaca_news.alpaca_news import AlpacaNews
        plugin = AlpacaNews.__new__(AlpacaNews)

    with patch("maki.plugins.alpaca_news.alpaca_news._cdn_get", side_effect=Exception("network down")):
        results = plugin.get_rss_news(symbols=None, since_hours=6, limit=5)

    assert results == []


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
