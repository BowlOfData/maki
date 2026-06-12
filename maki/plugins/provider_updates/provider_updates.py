"""
Provider Updates Plugin for Maki Framework.

Focuses on scraping provider announcement pages and extracting update text.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List

import requests

from maki.config import (
    DEFAULT_BROWSER_ACCEPT_LANGUAGE,
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_WEB_USER_AGENT,
)
from maki.plugins._web_utils import strip_html


ALLOWED_METHODS = ["fetch_model_releases"]


class ProviderUpdates:
    """Fetch provider announcement pages and return cleaned content."""

    # Mirror the module-level whitelist on the class: tool-call validation
    # reads ALLOWED_METHODS from the plugin instance, not the module.
    ALLOWED_METHODS = ALLOWED_METHODS

    def __init__(self, maki_instance=None):
        self.maki = maki_instance
        self.logger = logging.getLogger(__name__)
        self.logger.info("ProviderUpdates plugin initialized")

    def fetch_model_releases(
        self,
        sources: Dict[str, str],
        max_chars: int = 8000,
    ) -> List[Dict[str, Any]]:
        """Fetch each AI provider's news/announcement page and return its text content."""
        headers = {
            "User-Agent": DEFAULT_WEB_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": DEFAULT_BROWSER_ACCEPT_LANGUAGE,
        }
        results: List[Dict[str, Any]] = []

        for provider, url in sources.items():
            try:
                resp = requests.get(url, headers=headers, timeout=DEFAULT_HTTP_TIMEOUT)
                if resp.status_code != 200:
                    self.logger.warning(
                        "fetch_model_releases: HTTP %d for %s (%s)",
                        resp.status_code, provider, url,
                    )
                    continue
                text = strip_html(resp.text)
                # For GitHub releases pages, skip the navigation boilerplate
                # that precedes the first version tag.
                if "github.com" in url and "/releases" in url:
                    match = re.search(r"\bv\d+\.\d+", text)
                    if match:
                        text = text[match.start():]
                elif text.count("Loading") > 5:
                    match = re.search(
                        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2}|\b20\d{2}-\d{2}-\d{2}\b",
                        text,
                    )
                    if match:
                        text = text[match.start():]
                text = text[:max_chars]
                results.append({"provider": provider, "url": url, "content": text})
                self.logger.info(
                    "fetch_model_releases: fetched %s (%d chars)", provider, len(text)
                )
            except Exception as exc:
                self.logger.warning(
                    "fetch_model_releases: failed for %s: %s", provider, exc
                )
            time.sleep(0.5)

        return results


def register_plugin(maki_instance=None) -> ProviderUpdates:
    return ProviderUpdates(maki_instance)
