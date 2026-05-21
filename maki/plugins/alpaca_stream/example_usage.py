"""
Example usage of the AlpacaStream plugin.

Requires environment variables: APCA_API_KEY_ID, APCA_API_SECRET_KEY
"""

import asyncio
from unittest.mock import MagicMock, patch


async def demo_with_mock():
    """Demonstrate the plugin API without a live Alpaca connection."""
    print("AlpacaStream plugin example usage")
    print("===================================")

    with patch("alpaca.data.live.CryptoDataStream"):
        from maki.plugins.alpaca_stream.alpaca_stream import AlpacaStream
        plugin = AlpacaStream()

    plugin.subscribe(["BTC/USD", "ETH/USD"], channels=["bars"])
    status = plugin.get_status()
    print(f"\nSubscribed symbols: {status['subscribed_symbols']}")
    print(f"Running: {status['running']}")
    print(f"Queue size: {status['queue_size']}")

    # Simulate receiving a bar event
    mock_bar = MagicMock()
    mock_bar.symbol = "BTC/USD"
    mock_bar.timestamp.isoformat.return_value = "2024-01-01T00:00:00+00:00"
    mock_bar.open = 40000.0
    mock_bar.high = 41000.0
    mock_bar.low = 39000.0
    mock_bar.close = 40500.0
    mock_bar.volume = 1.5

    await plugin._on_bar(mock_bar)
    event = await plugin.queue.get()
    print(f"\nReceived bar event: {event}")


if __name__ == "__main__":
    asyncio.run(demo_with_mock())
