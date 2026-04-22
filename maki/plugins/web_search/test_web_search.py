"""
Unit tests for the WebSearch plugin.
"""

import sys
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from maki.plugins.web_search.web_search import (
    WebSearch,
    register_plugin,
    ALLOWED_METHODS,
    _week_start_utc,
    _parse_published,
    _struct_time_to_datetime,
    _is_current_week,
)

import time as _time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_feed_response(entries):
    """Return a mock requests.Response whose .text parses to a feed with given entries."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    # Build minimal RSS XML from the entry list
    items = ""
    for e in entries:
        items += f"""
        <item>
            <title>{e.get('title','')}</title>
            <link>{e.get('link','')}</link>
            <pubDate>{e.get('pubDate','')}</pubDate>
            <description>{e.get('description','')}</description>
        </item>"""
    mock_resp.text = f"""<?xml version="1.0"?>
    <rss version="2.0"><channel><title>Feed</title>{items}</channel></rss>"""
    return mock_resp


class TestWeekHelpers(unittest.TestCase):

    def test_week_start_is_monday(self):
        ws = _week_start_utc()
        self.assertEqual(ws.weekday(), 0, "week start must be a Monday")

    def test_week_start_is_midnight(self):
        ws = _week_start_utc()
        self.assertEqual(ws.hour, 0)
        self.assertEqual(ws.minute, 0)
        self.assertEqual(ws.second, 0)
        self.assertEqual(ws.microsecond, 0)

    def test_week_start_is_utc(self):
        ws = _week_start_utc()
        self.assertEqual(ws.tzinfo, timezone.utc)

    def test_parse_published_iso8601(self):
        dt = _parse_published("2026-04-14T09:42:06+00:00")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 4)
        self.assertEqual(dt.day, 14)

    def test_parse_published_rfc2822(self):
        dt = _parse_published("Fri, 17 Apr 2026 02:55:06 +0000")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 4)
        self.assertEqual(dt.day, 17)

    def test_parse_published_naive_date(self):
        dt = _parse_published("2026-04-14")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.tzinfo, timezone.utc)

    def test_parse_published_empty(self):
        self.assertIsNone(_parse_published(""))
        self.assertIsNone(_parse_published(None))

    def test_parse_published_invalid(self):
        self.assertIsNone(_parse_published("not-a-date"))

    def test_struct_time_to_datetime(self):
        st = _time.gmtime(0)  # Unix epoch
        dt = _struct_time_to_datetime(st)
        self.assertEqual(dt, datetime(1970, 1, 1, tzinfo=timezone.utc))

    def test_struct_time_to_datetime_none(self):
        self.assertIsNone(_struct_time_to_datetime(None))

    def test_is_current_week_today(self):
        today = datetime.now(timezone.utc).isoformat()
        self.assertTrue(_is_current_week(today))

    def test_is_current_week_last_week(self):
        last_week = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        self.assertFalse(_is_current_week(last_week))

    def test_is_current_week_monday_this_week(self):
        monday = _week_start_utc().isoformat()
        self.assertTrue(_is_current_week(monday))

    def test_is_current_week_unparseable_included(self):
        self.assertTrue(_is_current_week("unknown"))
        self.assertTrue(_is_current_week(""))


class TestWebSearchInit(unittest.TestCase):

    def test_instantiation_no_maki(self):
        ws = WebSearch()
        self.assertIsNone(ws.maki)

    def test_instantiation_with_maki(self):
        mock_maki = MagicMock()
        ws = WebSearch(mock_maki)
        self.assertEqual(ws.maki, mock_maki)

    def test_register_plugin_returns_instance(self):
        ws = register_plugin()
        self.assertIsInstance(ws, WebSearch)

    def test_allowed_methods(self):
        self.assertIn("search_rss", ALLOWED_METHODS)
        self.assertIn("search_hackernews", ALLOWED_METHODS)

    def test_all_allowed_methods_exist(self):
        ws = WebSearch()
        for method in ALLOWED_METHODS:
            self.assertTrue(hasattr(ws, method), f"Missing method: {method}")


class TestSearchRss(unittest.TestCase):

    def setUp(self):
        self.ws = WebSearch()
        self.today_rfc = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
        self.last_week_rfc = (datetime.now(timezone.utc) - timedelta(days=8)).strftime(
            "%a, %d %b %Y %H:%M:%S +0000"
        )

    def _patch_requests(self, entries_by_url):
        """Patch requests.get to return different entries per feed URL."""
        def fake_get(url, **kwargs):
            entries = entries_by_url.get(url, [])
            return _mock_feed_response(entries)
        return patch("maki.plugins.web_search.web_search.requests.get", side_effect=fake_get)

    def test_returns_list(self):
        with patch("maki.plugins.web_search.web_search.requests.get",
                   return_value=_mock_feed_response([])):
            results = self.ws.search_rss({"TestFeed": "https://example.com/rss"})
        self.assertIsInstance(results, list)

    def test_returns_empty_on_import_error(self):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "feedparser":
                raise ImportError
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            results = self.ws.search_rss({"F": "https://x.com/rss"})
        self.assertEqual(results, [])

    def test_result_structure(self):
        entries = [{"title": "T", "link": "https://x.com/a", "pubDate": self.today_rfc, "description": "desc"}]
        with self._patch_requests({"https://x.com/rss": entries}):
            with patch("maki.plugins.web_search.web_search.time.sleep"):
                results = self.ws.search_rss({"X": "https://x.com/rss"})

        if results:
            expected_keys = {"title", "url", "snippet", "source", "published"}
            self.assertEqual(set(results[0].keys()), expected_keys)

    def test_filters_old_articles(self):
        entries = [
            {"title": "Old",     "link": "https://x.com/old",  "pubDate": self.last_week_rfc, "description": ""},
            {"title": "Current", "link": "https://x.com/new",  "pubDate": self.today_rfc,     "description": ""},
        ]
        with self._patch_requests({"https://x.com/rss": entries}):
            with patch("maki.plugins.web_search.web_search.time.sleep"):
                results = self.ws.search_rss({"X": "https://x.com/rss"})

        urls = [r["url"] for r in results]
        self.assertNotIn("https://x.com/old", urls)
        self.assertIn("https://x.com/new", urls)

    def test_keyword_filter(self):
        entries = [
            {"title": "Python release", "link": "https://x.com/py", "pubDate": self.today_rfc, "description": ""},
            {"title": "Football match", "link": "https://x.com/ft", "pubDate": self.today_rfc, "description": ""},
        ]
        with self._patch_requests({"https://x.com/rss": entries}):
            with patch("maki.plugins.web_search.web_search.time.sleep"):
                results = self.ws.search_rss({"X": "https://x.com/rss"}, keywords=["python"])

        urls = [r["url"] for r in results]
        self.assertIn("https://x.com/py", urls)
        self.assertNotIn("https://x.com/ft", urls)

    def test_respects_max_per_feed(self):
        entries = [
            {"title": f"Article {i}", "link": f"https://x.com/{i}", "pubDate": self.today_rfc, "description": ""}
            for i in range(20)
        ]
        with self._patch_requests({"https://x.com/rss": entries}):
            with patch("maki.plugins.web_search.web_search.time.sleep"):
                results = self.ws.search_rss({"X": "https://x.com/rss"}, max_per_feed=5)

        self.assertLessEqual(len(results), 5)

    def test_skips_failed_feed_continues_others(self):
        def fake_get(url, **kwargs):
            if "bad" in url:
                raise ConnectionError("unreachable")
            return _mock_feed_response([
                {"title": "OK", "link": "https://good.com/a", "pubDate": self.today_rfc, "description": ""}
            ])

        with patch("maki.plugins.web_search.web_search.requests.get", side_effect=fake_get):
            with patch("maki.plugins.web_search.web_search.time.sleep"):
                results = self.ws.search_rss({
                    "Bad":  "https://bad.com/rss",
                    "Good": "https://good.com/rss",
                })

        urls = [r["url"] for r in results]
        self.assertIn("https://good.com/a", urls)

    def test_source_name_in_result(self):
        entries = [{"title": "T", "link": "https://tc.com/a", "pubDate": self.today_rfc, "description": ""}]
        with self._patch_requests({"https://tc.com/rss": entries}):
            with patch("maki.plugins.web_search.web_search.time.sleep"):
                results = self.ws.search_rss({"TechCrunch": "https://tc.com/rss"})

        if results:
            self.assertEqual(results[0]["source"], "TechCrunch")


class TestSearchHackerNews(unittest.TestCase):

    def setUp(self):
        self.ws = WebSearch()

    def test_returns_list_on_http_error(self):
        with patch("maki.plugins.web_search.web_search.requests.get", side_effect=Exception("timeout")):
            results = self.ws.search_hackernews("python")
        self.assertIsInstance(results, list)
        self.assertEqual(results, [])

    def test_skips_hits_without_url(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "hits": [
                {"title": "No URL", "url": "", "points": 10, "num_comments": 5, "created_at": "2026-04-17"},
                {"title": "Has URL", "url": "https://example.com", "points": 20, "num_comments": 2, "created_at": "2026-04-17"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("maki.plugins.web_search.web_search.requests.get", return_value=mock_resp):
            results = self.ws.search_hackernews("test")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "HackerNews")

    def test_result_structure(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "hits": [{"title": "T", "url": "https://x.com", "points": 5, "num_comments": 1, "created_at": "2026-04-17"}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("maki.plugins.web_search.web_search.requests.get", return_value=mock_resp):
            results = self.ws.search_hackernews("test")

        self.assertEqual(len(results), 1)
        for key in ("title", "url", "snippet", "source", "published"):
            self.assertIn(key, results[0])

    def test_uses_week_start_timestamp(self):
        captured = []

        def fake_get(url, **kwargs):
            captured.append(url)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"hits": []}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("maki.plugins.web_search.web_search.requests.get", side_effect=fake_get):
            self.ws.search_hackernews("test")

        week_start_ts = int(_week_start_utc().timestamp())
        self.assertIn(f"created_at_i>{week_start_ts}", captured[0])


if __name__ == "__main__":
    unittest.main()
