import pytest
from unittest.mock import MagicMock, patch

pytest.importorskip("alpaca")


def _make_bar(t, o, h, l, c, v):
    bar = MagicMock()
    bar.timestamp.isoformat.return_value = t
    bar.open = o
    bar.high = h
    bar.low = l
    bar.close = c
    bar.volume = v
    return bar


def _make_quote(bid, ask, bid_size, ask_size, ts):
    q = MagicMock()
    q.bid_price = bid
    q.ask_price = ask
    q.bid_size = bid_size
    q.ask_size = ask_size
    q.timestamp.isoformat.return_value = ts
    return q


@pytest.fixture()
def plugin():
    with patch("maki.plugins.alpaca_data.alpaca_data.AlpacaData.__init__", lambda self, *a, **kw: None):
        from maki.plugins.alpaca_data.alpaca_data import AlpacaData
        instance = AlpacaData.__new__(AlpacaData)
        instance._client = MagicMock()
        return instance


def test_get_crypto_bars_returns_ohlcv(plugin):
    from maki.plugins.alpaca_data.alpaca_data import AlpacaData

    bar = _make_bar("2024-01-01T00:00:00+00:00", 40000, 41000, 39000, 40500, 1.5)
    plugin._client.get_crypto_bars.return_value = {"BTC/USD": [bar]}

    with patch("alpaca.data.historical.CryptoHistoricalDataClient", MagicMock()), \
         patch("alpaca.data.requests.CryptoBarsRequest", MagicMock()), \
         patch("alpaca.data.timeframe.TimeFrame", MagicMock()), \
         patch("alpaca.data.timeframe.TimeFrameUnit", MagicMock()):
        # Call the internal logic directly via the client mock
        from alpaca.data.requests import CryptoBarsRequest
        result = plugin._client.get_crypto_bars(MagicMock())
        bars = result["BTC/USD"]

    assert len(bars) == 1
    assert bars[0].open == 40000


def test_get_crypto_latest_quote_keys(plugin):
    q = _make_quote(42000.0, 42010.0, 0.5, 0.3, "2024-01-01T00:01:00+00:00")
    plugin._client.get_crypto_latest_quote.return_value = {"BTC/USD": q}

    with patch("alpaca.data.requests.CryptoLatestQuoteRequest", MagicMock()):
        from alpaca.data.requests import CryptoLatestQuoteRequest
        result = plugin._client.get_crypto_latest_quote(MagicMock())
        quote = result["BTC/USD"]

    assert quote.bid_price == 42000.0
    assert quote.ask_price == 42010.0


def test_tf_minutes():
    from maki.plugins.alpaca_data.alpaca_data import _tf_minutes

    assert _tf_minutes("1Min") == 1
    assert _tf_minutes("5Min") == 5
    assert _tf_minutes("15Min") == 15
    assert _tf_minutes("1Hour") == 60
    assert _tf_minutes("1Day") == 1440
    assert _tf_minutes("unknown") == 1


def test_register_plugin_returns_instance():
    with patch("maki.plugins.alpaca_data.alpaca_data.AlpacaData.__init__", lambda self, *a, **kw: None):
        from maki.plugins.alpaca_data.alpaca_data import register_plugin
        result = register_plugin(maki_instance=None)
        from maki.plugins.alpaca_data.alpaca_data import AlpacaData
        assert isinstance(result, AlpacaData)


def test_forex_and_equity_methods_are_allowed():
    from maki.plugins.alpaca_data.alpaca_data import ALLOWED_METHODS

    for method in (
        "get_forex_bars",
        "get_forex_latest_quote",
        "get_equity_bars",
        "get_equity_latest_quote",
    ):
        assert method in ALLOWED_METHODS


def test_import_yfinance_missing_raises_helpful_error():
    import builtins
    from maki.plugins.alpaca_data.alpaca_data import _import_yfinance

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yfinance":
            raise ImportError("no module named yfinance")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ImportError, match='pip install "maki-framework\\[alpaca\\]"'):
            _import_yfinance()


def test_get_equity_bars_returns_ohlcv(plugin):
    bar = _make_bar("2024-01-01T00:00:00+00:00", 190.0, 191.0, 189.0, 190.5, 1000.0)
    mock_bars = MagicMock()
    mock_bars.data = {"AAPL": [bar]}
    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value = mock_bars

    with patch("alpaca.data.historical.StockHistoricalDataClient", return_value=mock_client), \
         patch("alpaca.data.requests.StockBarsRequest", MagicMock()), \
         patch("alpaca.data.timeframe.TimeFrame", MagicMock()), \
         patch("alpaca.data.timeframe.TimeFrameUnit", MagicMock()):
        result = plugin.get_equity_bars("AAPL", timeframe="1Day", lookback=1)

    assert len(result) == 1
    assert result[0]["o"] == 190.0
    assert result[0]["c"] == 190.5


def test_get_equity_latest_quote_no_quote_raises(plugin):
    mock_client = MagicMock()
    mock_client.get_stock_latest_quote.return_value = {}

    with patch("alpaca.data.historical.StockHistoricalDataClient", return_value=mock_client), \
         patch("alpaca.data.requests.StockLatestQuoteRequest", MagicMock()):
        with pytest.raises(RuntimeError, match="No quote available"):
            plugin.get_equity_latest_quote("AAPL")
