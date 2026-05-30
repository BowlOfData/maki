"""
Plugins for Maki Framework

This package contains all available plugins for the Maki framework.
"""

from __future__ import annotations

from importlib import import_module

from maki.config import PLUGIN_PACKAGE_PREFIX

PLUGIN_REGISTRY = {
    "directory_reader": "DirectoryReader",
    "file_reader": "FileReader",
    "file_writer": "FileWriter",
    "ftp_client": "FTPClient",
    "image_classifier": "ImageClassifier",
    "json_reader": "JsonReader",
    "media_search": "MediaSearch",
    "provider_updates": "ProviderUpdates",
    "trend_search": "TrendSearch",
    "web_search": "WebSearch",
    "web_to_md": "WebToMd",
    # Tranding plugins
    "alpaca_data": "AlpacaData",
    "alpaca_news": "AlpacaNews",
    "alpaca_trading": "AlpacaTrading",
    "alpaca_stream": "AlpacaStream",
    "obsidian_memory": "ObsidianMemory",
    "rag_memory": "RagMemory",
}

__all__ = ["PLUGIN_REGISTRY", "get_plugin_class", "list_plugins"]


def list_plugins() -> list[str]:
    """Return the built-in plugin names in a stable order."""
    return sorted(PLUGIN_REGISTRY)


def get_plugin_class(plugin_name: str):
    """Resolve and return a built-in plugin class by name."""
    if plugin_name not in PLUGIN_REGISTRY:
        raise KeyError(f"Unknown built-in plugin: {plugin_name}")
    module = import_module(f"{PLUGIN_PACKAGE_PREFIX}.{plugin_name}")
    return getattr(module, PLUGIN_REGISTRY[plugin_name])
