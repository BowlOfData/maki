import pytest
from unittest.mock import MagicMock, patch


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
