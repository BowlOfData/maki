"""
Alpaca WebSocket streaming plugin for Maki.

Wraps alpaca-py's CryptoDataStream for live price events.
Exposes an async subscribe/unsubscribe interface and an asyncio.Queue
that the event bus drains.
"""

import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ALLOWED_METHODS = ["get_status"]


class AlpacaStream:
    """
    Manages a CryptoDataStream subscription.

    Not used via TOOL: directives — the event bus drives it directly
    through subscribe() and unsubscribe(). ALLOWED_METHODS exposes only
    get_status so the LLM cannot accidentally trigger stream operations.
    """

    def __init__(self, maki_instance=None):
        from alpaca.data.live import CryptoDataStream

        api_key = os.environ.get("APCA_API_KEY_ID")
        api_secret = os.environ.get("APCA_API_SECRET_KEY")
        self._stream = CryptoDataStream(api_key=api_key, secret_key=api_secret)
        self._subscribed_symbols: List[str] = []
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._task: Optional[asyncio.Task] = None
        logger.info("AlpacaStream plugin initialised")

    @property
    def queue(self) -> asyncio.Queue:
        return self._queue

    def subscribe(self, symbols: List[str], channels: Optional[List[str]] = None) -> None:
        """Register bar handlers for *symbols*. Call before start()."""
        channels = channels or ["bars"]
        for symbol in symbols:
            if symbol in self._subscribed_symbols:
                continue
            if "bars" in channels:
                self._stream.subscribe_bars(self._on_bar, symbol)
            if "quotes" in channels:
                self._stream.subscribe_quotes(self._on_quote, symbol)
            self._subscribed_symbols.append(symbol)
        logger.info(f"AlpacaStream subscribed: {self._subscribed_symbols}")

    def unsubscribe(self, symbols: List[str]) -> None:
        for symbol in symbols:
            self._subscribed_symbols = [s for s in self._subscribed_symbols if s != symbol]

    async def start(self) -> None:
        """Run the stream in a background asyncio task with reconnect backoff."""
        self._task = asyncio.create_task(self._run_with_backoff())

    async def _run_with_backoff(self) -> None:
        """Wrap _run_forever with exponential backoff on connection-limit errors."""
        from alpaca.data.live import CryptoDataStream
        import os

        delay = 10
        while True:
            try:
                await self._stream._run_forever()
                return  # clean stop via _should_run=False
            except Exception as e:
                if "connection limit" in str(e).lower():
                    logger.warning(
                        f"AlpacaStream: connection limit exceeded — waiting {delay}s then recreating stream"
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 120)
                    # Recreate the underlying stream so the old connection is dropped
                    api_key = os.environ.get("APCA_API_KEY_ID")
                    api_secret = os.environ.get("APCA_API_SECRET_KEY")
                    self._stream = CryptoDataStream(api_key=api_key, secret_key=api_secret)
                    for symbol in self._subscribed_symbols:
                        self._stream.subscribe_bars(self._on_bar, symbol)
                else:
                    logger.error(f"AlpacaStream: unhandled error: {e}", exc_info=True)
                    await asyncio.sleep(delay)

    async def stop(self) -> None:
        # Signal alpaca-py to close the WebSocket cleanly before cancelling the task,
        # so close_connection() runs while the event loop is still live.
        try:
            self._stream.stop()
        except Exception:
            pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def get_status(self) -> Dict[str, Any]:
        return {
            "subscribed_symbols": list(self._subscribed_symbols),
            "running": self._task is not None and not self._task.done(),
            "queue_size": self._queue.qsize(),
        }

    async def _on_bar(self, bar) -> None:
        event = {
            "type": "bar",
            "symbol": bar.symbol,
            "timestamp": bar.timestamp.isoformat(),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        }
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("AlpacaStream queue full — dropping bar event")

    async def _on_quote(self, quote) -> None:
        event = {
            "type": "quote",
            "symbol": quote.symbol,
            "timestamp": quote.timestamp.isoformat(),
            "bid": float(quote.bid_price),
            "ask": float(quote.ask_price),
        }
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass


def register_plugin(maki_instance=None):
    return AlpacaStream(maki_instance)
