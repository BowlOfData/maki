"""
Example usage of the TrendSearch plugin.
"""

from maki.plugins.trend_search import TrendSearch


plugin = TrendSearch()
print(
    plugin.fetch_google_trends(
        seed_keywords=["artificial intelligence", "developer tools"],
        timeframe="now 7-d",
    )
)
