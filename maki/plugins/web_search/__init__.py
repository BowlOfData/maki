"""
Web Search Plugin for Maki Framework

Aggregates articles and trend signals from multiple sources:
RSS/Atom feeds, HackerNews (Algolia API), Reddit, GitHub Trending,
Lobste.rs, and Google Trends. No API keys are required except for
the optional Pexels image search.
"""

from .web_search import WebSearch, register_plugin

__all__ = ["WebSearch", "register_plugin"]
