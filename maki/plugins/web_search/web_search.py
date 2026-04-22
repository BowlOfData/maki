"""
Web Search Plugin for Maki Framework

This plugin provides functionality to search for recent tech articles by
querying RSS/Atom feeds directly (no search engine, no captcha), the
HackerNews Algolia API, Google Trends (via pytrends), and Reddit's public
JSON API. No API keys are required.

Dependencies:
    pip install feedparser>=6.0 pytrends>=4.9.0
"""

import calendar
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote as urlquote, urlparse

import requests

logger = logging.getLogger(__name__)

ALLOWED_METHODS = ["search_rss", "search_hackernews", "fetch_google_trends", "fetch_reddit_hot"]


# ---------------------------------------------------------------------------
# Week-boundary helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def _week_start_utc(now: Optional[datetime] = None) -> datetime:
    """Return Monday 00:00:00 UTC of the current ISO calendar week."""
    now = now or _now_utc()
    return now - timedelta(
        days=now.weekday(),
        hours=now.hour,
        minutes=now.minute,
        seconds=now.second,
        microseconds=now.microsecond,
    )


def _parse_published(date_str: str) -> Optional[datetime]:
    """
    Parse a published-date string into a timezone-aware datetime.

    Handles ISO 8601 (DuckDuckGo / HackerNews) and RFC 2822
    (RSS standard: ``"Fri, 17 Apr 2026 02:55:06 +0000"``).
    Returns ``None`` when the string cannot be parsed.
    """
    if not date_str:
        return None
    # ISO 8601
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        pass
    # RFC 2822 (standard RSS date format)
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str)
    except Exception:
        return None


def _struct_time_to_datetime(st) -> Optional[datetime]:
    """Convert a ``time.struct_time`` (UTC, as returned by feedparser) to datetime."""
    if st is None:
        return None
    try:
        return datetime.fromtimestamp(calendar.timegm(st), tz=timezone.utc)
    except Exception:
        return None


def _is_current_week(date_str: str, now: Optional[datetime] = None) -> bool:
    """
    Return True if *date_str* falls within the current ISO calendar week
    (Monday 00:00:00 UTC through now).

    Articles with an unparseable date are included (benefit of the doubt).
    """
    dt = _parse_published(date_str)
    if dt is None:
        return True
    now = now or _now_utc()
    return _week_start_utc(now) <= dt <= now


# ---------------------------------------------------------------------------
# WebSearch plugin
# ---------------------------------------------------------------------------

class WebSearch:
    """
    A plugin class for searching recent tech articles.

    Fetches articles directly from RSS/Atom feeds (no search engine required)
    and from HackerNews via the public Algolia API.
    """

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
                "Run: pip install 'feedparser>=6.0'"
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
                    timeout=15,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; MakiNewsletter/1.0)"},
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
            resp = requests.get(api_url, timeout=10)
            resp.raise_for_status()
            for hit in resp.json().get("hits", []):
                article_url = hit.get("url", "")
                if not article_url:
                    continue
                published = hit.get("created_at", "")
                if not _is_current_week(published, now=now):
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
                        "published": published,
                    }
                )
        except Exception as exc:
            self.logger.warning("search_hackernews('%s') failed: %s", query, exc)

        return results

    def fetch_google_trends(
        self,
        seed_keywords: List[str],
        timeframe: str = "now 7-d",
        geo: str = "",
    ) -> Dict[str, List[str]]:
        """
        Retrieve rising related queries for each seed keyword from Google Trends.

        Uses the unofficial ``pytrends`` library (no API key required).
        Rising queries represent topics gaining momentum this week — more
        useful for trend cross-checking than the static "top" queries.

        Args:
            seed_keywords: Broad topic keywords, e.g.
                           ["artificial intelligence", "cybersecurity"].
                           Google Trends accepts up to 5 at a time; if more
                           are provided they are batched automatically.
            timeframe:     Google Trends timeframe string (default ``"now 7-d"``
                           = last 7 days). Other useful values: ``"now 1-d"``,
                           ``"today 1-m"``.
            geo:           Country code (e.g. ``"US"``).  Empty string = worldwide.

        Returns:
            Dict mapping each seed keyword to a list of rising query strings.
            Keywords that returned no rising data map to an empty list.
            Returns an empty dict if ``pytrends`` is not installed or if the
            request fails.
        """
        try:
            from pytrends.request import TrendReq
        except ImportError:
            self.logger.error(
                "pytrends is not installed. Run: pip install 'pytrends>=4.9.0'"
            )
            return {}

        results: Dict[str, List[str]] = {kw: [] for kw in seed_keywords}

        # Google Trends payload accepts max 5 keywords at once — batch them.
        batch_size = 5
        try:
            pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
        except Exception as exc:
            self.logger.warning("fetch_google_trends: TrendReq init failed: %s", exc)
            return results

        for batch_start in range(0, len(seed_keywords), batch_size):
            batch = seed_keywords[batch_start: batch_start + batch_size]
            try:
                pytrends.build_payload(batch, timeframe=timeframe, geo=geo)
                related = pytrends.related_queries()
            except Exception as exc:
                self.logger.warning(
                    "fetch_google_trends: related_queries failed for batch %s: %s",
                    batch, exc,
                )
                time.sleep(2)
                continue

            for kw in batch:
                kw_data = related.get(kw, {}) or {}
                rising_df = kw_data.get("rising")
                if rising_df is not None and not rising_df.empty:
                    results[kw] = rising_df["query"].tolist()
                    self.logger.debug(
                        "fetch_google_trends: '%s' → %d rising queries",
                        kw, len(results[kw]),
                    )
                else:
                    # Fall back to top queries when no rising data is available
                    top_df = kw_data.get("top")
                    if top_df is not None and not top_df.empty:
                        results[kw] = top_df["query"].head(10).tolist()
                        self.logger.debug(
                            "fetch_google_trends: '%s' → %d top queries (no rising data)",
                            kw, len(results[kw]),
                        )

            # Polite pause between batches to avoid rate limiting
            time.sleep(1.5)

        total = sum(len(v) for v in results.values())
        self.logger.info(
            "fetch_google_trends: %d total trending queries across %d keywords",
            total, len(seed_keywords),
        )
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
        now = _now_utc()
        week_start = _week_start_utc(now)
        results: List[Dict[str, Any]] = []
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MakiNewsletter/1.0)"}

        for sub in subreddits:
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit={max_per_sub + 5}"
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                posts = resp.json().get("data", {}).get("children", [])
            except Exception as exc:
                self.logger.warning("fetch_reddit_hot: r/%s failed: %s", sub, exc)
                time.sleep(0.5)
                continue

            count = 0
            for child in posts:
                if count >= max_per_sub:
                    break
                post = child.get("data", {})

                # Skip self-posts (no external link) and low-engagement posts
                external_url = post.get("url", "")
                if post.get("is_self") or not external_url:
                    continue
                if post.get("score", 0) < 10:
                    continue

                created_utc = post.get("created_utc")
                if created_utc:
                    post_dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                    if not (week_start <= post_dt <= now):
                        continue
                    published_str = post_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    published_str = ""

                title = post.get("title", "")
                selftext = post.get("selftext", "") or ""

                results.append({
                    "title": title,
                    "url": external_url,
                    "snippet": selftext[:400].strip() if selftext else f"r/{sub} · {post.get('score', 0)} upvotes",
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
