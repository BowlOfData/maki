"""
Example usage of the AlpacaNews plugin with Maki agents.

Requires environment variables: APCA_API_KEY_ID, APCA_API_SECRET_KEY
feedparser must be installed: pip install feedparser
"""

from maki.plugins.alpaca_news.alpaca_news import AlpacaNews


def main():
    plugin = AlpacaNews()

    print("AlpacaNews plugin example usage")
    print("================================")

    # Example 1: Alpaca news for BTC/USD
    print("\nExample 1: Alpaca news for BTC/USD (last 6 hours)")
    articles = plugin.get_news(symbols=["BTC/USD"], since_hours=6, limit=5)
    for a in articles:
        print(f"  [{a['source']}] {a['headline'][:80]}")

    # Example 2: Free RSS feeds only
    print("\nExample 2: RSS news (no API key required)")
    rss = plugin.get_rss_news(since_hours=12, limit=5)
    for a in rss:
        print(f"  [{a['source']}] {a['headline'][:80]}")

    # Example 3: Merged from all sources
    print("\nExample 3: All news merged and deduplicated")
    all_news = plugin.get_all_news(symbols=["ETH/USD"], since_hours=24, limit=5)
    for a in all_news:
        print(f"  [{a['source']}] {a['published_at']}  {a['headline'][:60]}")


if __name__ == "__main__":
    main()
