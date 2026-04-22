"""
Web Search Plugin for Maki Framework

This plugin provides news and article search via DuckDuckGo and the
HackerNews Algolia API. No external API keys are required.
"""

from .web_search import WebSearch, register_plugin

__all__ = ["WebSearch", "register_plugin"]
