"""
Trend Search Plugin for Maki Framework.

Focuses on trend-intelligence retrieval rather than article discovery.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional


ALLOWED_METHODS = ["fetch_google_trends"]


class TrendSearch:
    """Fetch trend signals from external trend-oriented sources."""

    # Mirror the module-level whitelist on the class: tool-call validation
    # reads ALLOWED_METHODS from the plugin instance, not the module.
    ALLOWED_METHODS = ALLOWED_METHODS

    def __init__(self, maki_instance=None):
        self.maki = maki_instance
        self.logger = logging.getLogger(__name__)
        self.logger.info("TrendSearch plugin initialized")

    def fetch_google_trends(
        self,
        seed_keywords: List[str],
        timeframe: str = "now 7-d",
        geo: str = "",
    ) -> Dict[str, List[str]]:
        """Retrieve rising related queries for each seed keyword from Google Trends."""
        try:
            from pytrends.request import TrendReq
        except ImportError:
            self.logger.error(
                'pytrends is not installed. Run: pip install "maki[trends]"'
            )
            return {}

        results: Dict[str, List[str]] = {kw: [] for kw in seed_keywords}
        retry_delays = (30.0, 60.0, 120.0)

        try:
            pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
        except Exception as exc:
            self.logger.warning("fetch_google_trends: TrendReq init failed: %s", exc)
            return results

        for kw in seed_keywords:
            related = self._google_trends_query(
                pytrends, [kw], timeframe, geo, retry_delays
            )
            if related is None:
                continue

            kw_data = related.get(kw, {}) or {}
            rising_df = kw_data.get("rising")
            if rising_df is not None and not rising_df.empty:
                results[kw] = rising_df["query"].tolist()
                self.logger.debug(
                    "fetch_google_trends: '%s' → %d rising queries", kw, len(results[kw])
                )
            else:
                top_df = kw_data.get("top")
                if top_df is not None and not top_df.empty:
                    results[kw] = top_df["query"].head(10).tolist()
                    self.logger.debug(
                        "fetch_google_trends: '%s' → %d top queries (no rising data)",
                        kw, len(results[kw]),
                    )
            time.sleep(5.0)

        total = sum(len(v) for v in results.values())
        self.logger.info(
            "fetch_google_trends: %d total trending queries across %d keywords",
            total, len(seed_keywords),
        )
        return results

    def _google_trends_query(
        self,
        pytrends,
        keywords: List[str],
        timeframe: str,
        geo: str,
        retry_delays: tuple,
    ) -> Optional[Dict]:
        """Call pytrends with retry handling for 429 rate-limit responses."""
        for attempt, delay in enumerate((*retry_delays, None), start=1):
            try:
                pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
                return pytrends.related_queries()
            except Exception as exc:
                is_429 = "429" in str(exc)
                if is_429 and delay is not None:
                    self.logger.warning(
                        "fetch_google_trends: 429 rate-limited for %s "
                        "(attempt %d/%d) — waiting %.0fs before retry",
                        keywords, attempt, len(retry_delays) + 1, delay,
                    )
                    time.sleep(delay)
                else:
                    self.logger.warning(
                        "fetch_google_trends: query failed for %s: %s", keywords, exc
                    )
                    return None
        return None


def register_plugin(maki_instance=None) -> TrendSearch:
    return TrendSearch(maki_instance)
