import asyncio
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture()
def plugin():
    with patch("alpaca.data.live.CryptoDataStream"):
        from maki.plugins.alpaca_stream.alpaca_stream import AlpacaStream
        return AlpacaStream()


def test_subscribe_adds_symbols(plugin):
    plugin.subscribe(["BTC/USD", "ETH/USD"])
    assert "BTC/USD" in plugin._subscribed_symbols
    assert "ETH/USD" in plugin._subscribed_symbols


def test_subscribe_no_duplicates(plugin):
    plugin.subscribe(["BTC/USD"])
    plugin.subscribe(["BTC/USD"])
    assert plugin._subscribed_symbols.count("BTC/USD") == 1


def test_unsubscribe_removes_symbol(plugin):
    plugin._subscribed_symbols = ["BTC/USD", "ETH/USD"]
    plugin.unsubscribe(["BTC/USD"])
    assert "BTC/USD" not in plugin._subscribed_symbols
    assert "ETH/USD" in plugin._subscribed_symbols


def test_get_status_not_running(plugin):
    status = plugin.get_status()
    assert status["running"] is False
    assert status["queue_size"] == 0
    assert isinstance(status["subscribed_symbols"], list)


@pytest.mark.asyncio
async def test_on_bar_enqueues_event(plugin):
    bar = MagicMock()
    bar.symbol = "BTC/USD"
    bar.timestamp.isoformat.return_value = "2024-01-01T00:00:00+00:00"
    bar.open = 40000.0
    bar.high = 41000.0
    bar.low = 39000.0
    bar.close = 40500.0
    bar.volume = 1.5

    await plugin._on_bar(bar)
    event = plugin.queue.get_nowait()
    assert event["type"] == "bar"
    assert event["symbol"] == "BTC/USD"
    assert event["close"] == 40500.0


@pytest.mark.asyncio
async def test_on_quote_enqueues_event(plugin):
    quote = MagicMock()
    quote.symbol = "ETH/USD"
    quote.timestamp.isoformat.return_value = "2024-01-01T00:00:01+00:00"
    quote.bid_price = 2000.0
    quote.ask_price = 2001.0

    await plugin._on_quote(quote)
    event = plugin.queue.get_nowait()
    assert event["type"] == "quote"
    assert event["bid"] == 2000.0


@pytest.mark.asyncio
async def test_queue_full_drops_bar(plugin):
    # Fill the queue to capacity
    for _ in range(plugin._queue.maxsize):
        await plugin._queue.put({"type": "bar"})

    bar = MagicMock()
    bar.symbol = "BTC/USD"
    bar.timestamp.isoformat.return_value = "2024-01-01T00:00:00+00:00"
    bar.open = bar.high = bar.low = bar.close = bar.volume = 1.0
    # Should not raise even when queue is full
    await plugin._on_bar(bar)
