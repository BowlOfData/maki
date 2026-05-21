# Alpaca Stream Plugin

Live WebSocket price streaming for crypto assets via Alpaca's `CryptoDataStream`. Pushes bar and quote events into an `asyncio.Queue` that the application event bus can drain.

## Requirements

```
APCA_API_KEY_ID=<your alpaca api key>
APCA_API_SECRET_KEY=<your alpaca secret key>
```

## Usage

```python
import asyncio
from maki.plugins.alpaca_stream.alpaca_stream import AlpacaStream

async def main():
    plugin = AlpacaStream()

    # Subscribe to bar and quote channels
    plugin.subscribe(["BTC/USD", "ETH/USD"], channels=["bars", "quotes"])

    # Start the stream in the background
    await plugin.start()

    # Drain events from the queue
    while True:
        event = await plugin.queue.get()
        print(event)

asyncio.run(main())
```

## Methods

### `subscribe(symbols, channels=None)`

Registers handlers for the given symbols. Default channel: `["bars"]`. Call before `start()`.

### `unsubscribe(symbols)`

Removes symbols from the active subscription list.

### `start()`

Async. Launches the WebSocket stream as a background asyncio task with exponential-backoff reconnect on connection-limit errors.

### `stop()`

Async. Gracefully shuts down the stream and cancels the background task.

### `get_status()`

Returns a dict with keys `subscribed_symbols`, `running`, `queue_size`. This is the only method exposed via `TOOL:` directives to prevent accidental LLM-triggered stream operations.

### `queue`

An `asyncio.Queue(maxsize=1000)` of bar/quote event dicts.
