"""
Web to Markdown Plugin for Maki Framework

This module provides the WebToMd plugin that fetches web pages and converts them to markdown format.
"""

from .web_to_md import WebToMd, register_plugin

__all__ = ['WebToMd', 'register_plugin']