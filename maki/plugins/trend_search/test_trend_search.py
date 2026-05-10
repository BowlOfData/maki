"""
Tests for the TrendSearch plugin.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from maki.plugins.trend_search.trend_search import ALLOWED_METHODS, TrendSearch, register_plugin


class _FakeSeries:
    def __init__(self, values):
        self._values = values

    def head(self, n):
        return _FakeSeries(self._values[:n])

    def tolist(self):
        return list(self._values)


class _FakeDataFrame:
    def __init__(self, values):
        self.empty = len(values) == 0
        self._values = values

    def __getitem__(self, key):
        if key != "query":
            raise KeyError(key)
        return _FakeSeries(self._values)

    def head(self, n):
        return _FakeDataFrame(self._values[:n])


class TestTrendSearch(unittest.TestCase):
    def setUp(self):
        self.plugin = TrendSearch()

    def test_register_plugin_returns_instance(self):
        self.assertIsInstance(register_plugin(), TrendSearch)

    def test_allowed_methods(self):
        self.assertEqual(ALLOWED_METHODS, ["fetch_google_trends"])

    def test_returns_empty_dict_when_pytrends_missing(self):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pytrends.request":
                raise ImportError
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = self.plugin.fetch_google_trends(["ai"])
        self.assertEqual(result, {})

    def test_uses_rising_queries_when_available(self):
        trend_req = MagicMock()
        trend_req.related_queries.return_value = {
            "ai": {"rising": _FakeDataFrame(["gpt", "agents"]), "top": None}
        }
        fake_request_module = types.SimpleNamespace(TrendReq=MagicMock(return_value=trend_req))
        with patch.dict(sys.modules, {"pytrends.request": fake_request_module}, clear=False), \
             patch("maki.plugins.trend_search.trend_search.time.sleep"):
            result = self.plugin.fetch_google_trends(["ai"])
        self.assertEqual(result["ai"], ["gpt", "agents"])

    def test_falls_back_to_top_queries(self):
        trend_req = MagicMock()
        trend_req.related_queries.return_value = {
            "ai": {"rising": _FakeDataFrame([]), "top": _FakeDataFrame(["openai", "llm"])}
        }
        fake_request_module = types.SimpleNamespace(TrendReq=MagicMock(return_value=trend_req))
        with patch.dict(sys.modules, {"pytrends.request": fake_request_module}, clear=False), \
             patch("maki.plugins.trend_search.trend_search.time.sleep"):
            result = self.plugin.fetch_google_trends(["ai"])
        self.assertEqual(result["ai"], ["openai", "llm"])
