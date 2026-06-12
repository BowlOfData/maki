"""
Media Search Plugin for Maki Framework.

Focuses on external asset lookup rather than article discovery.
"""

from __future__ import annotations

import logging
from typing import Optional

from maki.config import DEFAULT_HTTP_TIMEOUT, PEXELS_API_KEY_ENV
from maki.connector import Connector


ALLOWED_METHODS = ["fetch_pexels_image"]

# All outbound HTTP goes through the hardened Connector layer
# (SSRF validation + DNS pinning + error classification).
_connector = Connector(timeout=DEFAULT_HTTP_TIMEOUT)


def _http_get(url, **kwargs):
    """Fetch *url* through the shared Connector; raises Maki errors on failure."""
    return _connector.get(url, **kwargs)


class MediaSearch:
    """Lookup external media assets from dedicated asset providers."""

    # Mirror the module-level whitelist on the class: tool-call validation
    # reads ALLOWED_METHODS from the plugin instance, not the module.
    ALLOWED_METHODS = ALLOWED_METHODS

    def __init__(self, maki_instance=None):
        self.maki = maki_instance
        self.logger = logging.getLogger(__name__)
        self.logger.info("MediaSearch plugin initialized")

    def fetch_pexels_image(self, query: str, api_key: str) -> Optional[str]:
        """Search Pexels for a landscape photo matching *query* and return its URL."""
        if not api_key:
            self.logger.warning("fetch_pexels_image: %s not set; skipping image", PEXELS_API_KEY_ENV)
            return None

        try:
            resp = _http_get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": api_key},
                params={"query": query, "per_page": 1, "orientation": "landscape"},
            )
            photos = resp.json().get("photos", [])
            if not photos:
                self.logger.warning("fetch_pexels_image: no results for query '%s'", query)
                return None
            return photos[0]["src"]["large"]
        except Exception as exc:
            self.logger.warning("fetch_pexels_image: request failed: %s", exc)
            return None


def register_plugin(maki_instance=None) -> MediaSearch:
    return MediaSearch(maki_instance)
