"""
Web Search Plugin for Maki Framework

This plugin provides functionality to search for recent tech articles by
querying RSS/Atom feeds directly (no search engine, no captcha), the
HackerNews Algolia API, Google Trends (via pytrends), and Reddit's public
JSON API. No API keys are required.

Dependencies:
    pip install feedparser>=6.0 pytrends>=4.9.0
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote as urlquote, urlparse

import requests

from maki.config import DEFAULT_HTTP_TIMEOUT, DEFAULT_WEB_USER_AGENT
from maki.plugins._web_utils import (
    is_current_week as _is_current_week,
    is_media_url as _is_media_url,
    now_utc as _now_utc,
    parse_published as _parse_published,
    struct_time_to_datetime as _struct_time_to_datetime,
    week_start_utc as _week_start_utc,
)

logger = logging.getLogger(__name__)

ALLOWED_METHODS = [
    "search_rss",
    "search_hackernews",
    "fetch_reddit_hot",
    "fetch_github_trending",
    "fetch_lobsters",
]

_DEFAULT_HEADERS = {"User-Agent": DEFAULT_WEB_USER_AGENT}


# ---------------------------------------------------------------------------
# WebSearch plugin
# ---------------------------------------------------------------------------

class WebSearch:
    """
    A plugin class for searching recent tech articles.

    Fetches articles directly from RSS/Atom feeds (no search engine required)
    and from HackerNews via the public Algolia API.
    """

    # Mirror the module-level whitelist on the class: tool-call validation
    # reads ALLOWED_METHODS from the plugin instance, not the module.
    ALLOWED_METHODS = ALLOWED_METHODS

    def __init__(self, maki_instance=None):
        """
        Initialize the WebSearch plugin.

        Args:
            maki_instance: Optional Maki instance (unused; kept for plugin contract).
        """
        self.maki = maki_instance
        self.logger = logging.getLogger(__name__)
        self.logger.info("WebSearch plugin initialized")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def search_rss(
        self,
        feeds: Dict[str, str],
        max_per_feed: int = 10,
        keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch articles from RSS/Atom feeds, filtered to the current ISO week.

        Feeds are fetched via ``requests`` (plain HTTP) and parsed with
        ``feedparser``, bypassing any search-engine captcha.

        Args:
            feeds:        Mapping of ``{source_name: feed_url}``.
            max_per_feed: Maximum articles to keep per feed (default 10).
            keywords:     Optional list of keywords; when provided only articles
                          whose title or summary contains at least one keyword
                          (case-insensitive) are returned. Pass ``None`` (default)
                          to return all current-week articles.

        Returns:
            List of article dicts with keys:
            ``title``, ``url``, ``snippet``, ``source``, ``published``.
        """
        try:
            import feedparser
        except ImportError:
            self.logger.error(
                "feedparser is not installed. "
                'Run: pip install "maki[web]"'
            )
            return []

        now = _now_utc()
        week_start = _week_start_utc(now)
        kw_lower = [k.lower() for k in keywords] if keywords else None
        results: List[Dict[str, Any]] = []

        for source, feed_url in feeds.items():
            try:
                resp = requests.get(
                    feed_url,
                    timeout=DEFAULT_HTTP_TIMEOUT,
                    headers=_DEFAULT_HEADERS,
                )
                resp.raise_for_status()
                feed = feedparser.parse(resp.text)
            except Exception as exc:
                self.logger.warning("search_rss: failed to fetch '%s' (%s): %s", source, feed_url, exc)
                continue

            count = 0
            for entry in feed.entries:
                if count >= max_per_feed:
                    break

                url = entry.get("link", "")
                if not url:
                    continue

                # Prefer feedparser's pre-parsed UTC struct_time for reliability
                dt = _struct_time_to_datetime(entry.get("published_parsed") or entry.get("updated_parsed"))
                if dt is None:
                    # Fall back to string parsing
                    raw_date = entry.get("published", "") or entry.get("updated", "")
                    dt = _parse_published(raw_date)

                published_str = entry.get("published", "") or entry.get("updated", "")

                # Drop articles published before this ISO week
                if dt is not None and not (week_start <= dt <= now):
                    continue

                title = entry.get("title", "")
                summary = entry.get("summary", "")

                # Optional keyword pre-filter
                if kw_lower:
                    text = (title + " " + summary).lower()
                    if not any(kw in text for kw in kw_lower):
                        continue

                results.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": summary[:400].strip() if summary else "",
                        "source": source,
                        "published": published_str,
                    }
                )
                count += 1

            self.logger.debug(
                "search_rss: '%s' → %d current-week articles", source, count
            )
            time.sleep(0.5)  # polite pause between feed fetches

        self.logger.debug(
            "search_rss: %d total articles from %d feeds (since %s)",
            len(results), len(feeds), week_start.strftime("%Y-%m-%d"),
        )
        return results

    def search_hackernews(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search HackerNews stories published in the current ISO calendar week
        (Monday 00:00:00 UTC through now) via the Algolia API.

        Uses the public endpoint ``https://hn.algolia.com/api/v1/search`` — no
        authentication required.

        Args:
            query:       Search terms.
            max_results: Maximum number of stories to return (default 10).

        Returns:
            List of article dicts
            (``title``, ``url``, ``snippet``, ``source``, ``published``).
        """
        now = _now_utc()
        week_start_ts = int(_week_start_utc(now).timestamp())
        api_url = (
            "https://hn.algolia.com/api/v1/search"
            f"?query={urlquote(query)}"
            "&tags=story"
            f"&numericFilters=created_at_i>{week_start_ts}"
            f"&hitsPerPage={max_results}"
        )

        results: List[Dict[str, Any]] = []
        try:
            resp = requests.get(api_url, timeout=DEFAULT_HTTP_TIMEOUT)
            resp.raise_for_status()
            for hit in resp.json().get("hits", []):
                article_url = hit.get("url", "")
                if not article_url:
                    continue
                results.append(
                    {
                        "title": hit.get("title", ""),
                        "url": article_url,
                        "snippet": (
                            f"HackerNews · {hit.get('points', 0)} points "
                            f"· {hit.get('num_comments', 0)} comments"
                        ),
                        "source": "HackerNews",
                        "published": hit.get("created_at", ""),
                    }
                )
        except Exception as exc:
            self.logger.warning("search_hackernews('%s') failed: %s", query, exc)

        return results

    def fetch_github_trending(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch recently created GitHub repositories gaining the most stars.

        Uses the public GitHub Search API (no authentication required).
        A rolling 7-day window is used instead of the ISO week boundary so that
        early-week runs (Monday morning) still return meaningful results.
        Repositories with at least 10 stars are returned, sorted by star count.

        Returns:
            List of article dicts with keys:
            ``title``, ``url``, ``snippet``, ``source``, ``published``, ``topics``.
        """
        now = _now_utc()
        since_date = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        query = f"created:>{since_date} stars:>10"
        api_url = (
            "https://api.github.com/search/repositories"
            f"?q={urlquote(query)}"
            f"&sort=stars&order=desc&per_page={max_results}"
        )
        headers = {
            "User-Agent": DEFAULT_WEB_USER_AGENT,
            "Accept": "application/vnd.github.v3+json",
        }

        results: List[Dict[str, Any]] = []
        try:
            resp = requests.get(api_url, headers=headers, timeout=DEFAULT_HTTP_TIMEOUT)
            resp.raise_for_status()
            for item in resp.json().get("items", []):
                url = item.get("html_url", "")
                if not url:
                    continue
                description = item.get("description", "") or ""
                topics = item.get("topics", [])
                stars = item.get("stargazers_count", 0)
                snippet_parts = [description] if description else []
                if topics:
                    snippet_parts.append(f"Topics: {', '.join(topics[:5])}")
                if stars:
                    snippet_parts.append(f"({stars} stars this week)")
                results.append({
                    "title": item.get("full_name", url),
                    "url": url,
                    "snippet": "  ".join(snippet_parts)[:400],
                    "source": "GitHub Trending",
                    "published": item.get("created_at", ""),
                    "topics": topics,
                })
        except Exception as exc:
            self.logger.warning("fetch_github_trending: request failed: %s", exc)

        self.logger.info("fetch_github_trending: %d repos found", len(results))
        return results

    def fetch_lobsters(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch hot stories from Lobste.rs RSS feed, filtered to the current ISO week.

        Lobste.rs is a curated link-aggregator for software developers — all
        content is already tech-relevant so no keyword filtering is applied.

        Returns:
            List of article dicts with keys:
            ``title``, ``url``, ``snippet``, ``source``, ``published``.
        """
        try:
            import feedparser
        except ImportError:
            self.logger.error(
                'feedparser is not installed. Run: pip install "maki[web]"'
            )
            return []

        now = _now_utc()
        week_start = _week_start_utc(now)

        try:
            resp = requests.get(
                "https://lobste.rs/rss",
                timeout=DEFAULT_HTTP_TIMEOUT,
                headers=_DEFAULT_HEADERS,
            )
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
        except Exception as exc:
            self.logger.warning("fetch_lobsters: request failed: %s", exc)
            return []

        results: List[Dict[str, Any]] = []
        for entry in feed.entries:
            if len(results) >= max_results:
                break

            url = entry.get("link", "")
            if not url or _is_media_url(url):
                continue

            dt = _struct_time_to_datetime(
                entry.get("published_parsed") or entry.get("updated_parsed")
            )
            if dt is None:
                raw_date = entry.get("published", "") or entry.get("updated", "")
                dt = _parse_published(raw_date)

            if dt is not None and not (week_start <= dt <= now):
                continue

            published_str = entry.get("published", "") or entry.get("updated", "")
            summary = entry.get("summary", "")
            results.append({
                "title": entry.get("title", ""),
                "url": url,
                "snippet": summary[:400].strip() if summary else "",
                "source": "Lobste.rs",
                "published": published_str,
            })

        self.logger.info("fetch_lobsters: %d articles found", len(results))
        return results

    def fetch_reddit_hot(
        self,
        subreddits: List[str],
        max_per_sub: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Fetch hot posts from a list of subreddits using Reddit's public JSON API.

        No authentication is required. Posts are filtered to the current ISO
        calendar week (Monday 00:00 UTC through now).

        Args:
            subreddits:   List of subreddit names (without the ``r/`` prefix).
            max_per_sub:  Maximum posts to return per subreddit (default 10).

        Returns:
            List of article dicts with keys:
            ``title``, ``url``, ``snippet``, ``source``, ``published``.
            Self-posts (no external URL) and very-low-score posts (< 10) are
            excluded.
        """
        import feedparser
        import re as _re

        now = _now_utc()
        week_start = _week_start_utc(now)
        results: List[Dict[str, Any]] = []
        headers = {
            "User-Agent": os.getenv(
                "MAKI_REDDIT_USER_AGENT",
                "python:maki_newsletter:v1.0 (by /u/maki_bot)",
            )
        }

        for sub in subreddits:
            # Reddit's .json API returns 403 — use the RSS feed instead
            url = f"https://www.reddit.com/r/{sub}/hot.rss?limit={max_per_sub + 5}"
            try:
                resp = requests.get(url, headers=headers, timeout=DEFAULT_HTTP_TIMEOUT)
                resp.raise_for_status()
                feed = feedparser.parse(resp.text)
                entries = feed.entries
            except Exception as exc:
                self.logger.warning("fetch_reddit_hot: r/%s failed: %s", sub, exc)
                time.sleep(0.5)
                continue

            count = 0
            for entry in entries:
                if count >= max_per_sub:
                    break

                # Extract the external URL from the entry content HTML
                html = ""
                if entry.get("content"):
                    html = entry.content[0].get("value", "")
                elif entry.get("summary"):
                    html = entry.summary

                hrefs = _re.findall(r'href="(https?://[^"]+)"', html)
                external_url = next(
                    (h for h in hrefs if "reddit.com" not in h), ""
                )
                if not external_url:
                    continue

                # Skip direct media/image URLs — these are not articles
                if _is_media_url(external_url):
                    self.logger.debug("fetch_reddit_hot: skipping media URL %s", external_url)
                    continue

                published_str = ""
                if entry.get("published_parsed"):
                    post_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    if not (week_start <= post_dt <= now):
                        continue
                    published_str = post_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

                title = entry.get("title", "")
                snippet = title.strip()

                results.append({
                    "title": title,
                    "url": external_url,
                    "snippet": snippet,
                    "source": f"Reddit r/{sub}",
                    "published": published_str,
                })
                count += 1

            self.logger.debug("fetch_reddit_hot: r/%s → %d posts", sub, count)
            time.sleep(0.75)  # polite pause between subreddit requests

        self.logger.info(
            "fetch_reddit_hot: %d total posts from %d subreddits",
            len(results), len(subreddits),
        )
        return results

# ---------------------------------------------------------------------------
# Plugin registration function (maki contract)
# ---------------------------------------------------------------------------

def register_plugin(maki_instance=None) -> WebSearch:
    """
    Register the WebSearch plugin with the Maki framework.

    Args:
        maki_instance: Optional Maki instance.

    Returns:
        WebSearch: An instance of the WebSearch plugin.
    """
    return WebSearch(maki_instance)
