"""
Alpaca market-data plugin for Maki.

Wraps alpaca-py's CryptoHistoricalDataClient for crypto bar/quote fetching,
and StockHistoricalDataClient for forex bar/quote fetching.
Equity methods raise NotImplementedError (enabled in v2).
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ALLOWED_METHODS = [
    "get_crypto_bars",
    "get_crypto_latest_quote",
    "get_forex_bars",
    "get_forex_latest_quote",
    "list_crypto_assets",
]

_ALPACA_DATA_URL = "https://data.alpaca.markets"


class AlpacaData:
    def __init__(self, maki_instance=None):
        from alpaca.data.historical import CryptoHistoricalDataClient

        api_key = os.environ.get("APCA_API_KEY_ID")
        api_secret = os.environ.get("APCA_API_SECRET_KEY")
        self._client = CryptoHistoricalDataClient(api_key=api_key, secret_key=api_secret)
        logger.info("AlpacaData plugin initialised (crypto + forex)")

    def get_crypto_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        lookback: int = 60,
    ) -> List[Dict[str, Any]]:
        """Return the last *lookback* OHLCV bars for *symbol* (e.g. 'BTC/USD')."""
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }
        tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Minute))
        start = datetime.now(timezone.utc) - timedelta(minutes=lookback * _tf_minutes(timeframe))
        req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start)
        bars = self._client.get_crypto_bars(req)
        result = []
        for bar in bars[symbol]:
            result.append({
                "t": bar.timestamp.isoformat(),
                "o": float(bar.open),
                "h": float(bar.high),
                "l": float(bar.low),
                "c": float(bar.close),
                "v": float(bar.volume),
            })
        return result

    def get_crypto_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Return the latest bid/ask for *symbol*."""
        from alpaca.data.requests import CryptoLatestQuoteRequest

        req = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = self._client.get_crypto_latest_quote(req)
        q = quotes[symbol]
        return {
            "symbol": symbol,
            "bid": float(q.bid_price),
            "ask": float(q.ask_price),
            "bid_size": float(q.bid_size),
            "ask_size": float(q.ask_size),
            "timestamp": q.timestamp.isoformat(),
        }

    def list_crypto_assets(self) -> List[str]:
        """Return all tradable crypto symbols available on Alpaca."""
        from alpaca.trading import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass

        api_key = os.environ.get("APCA_API_KEY_ID")
        api_secret = os.environ.get("APCA_API_SECRET_KEY")
        tc = TradingClient(api_key=api_key, secret_key=api_secret, paper=True)
        req = GetAssetsRequest(asset_class=AssetClass.CRYPTO)
        assets = tc.get_all_assets(req)
        return [a.symbol for a in assets if a.tradable]

    # ------------------------------------------------------------------
    # Forex — Alpaca serves FX via the stock single-symbol REST endpoints.
    # The SDK's multi-symbol endpoint requires a SIP subscription; the
    # per-symbol endpoints (/v2/stocks/{sym}/bars) work on free accounts.
    # Symbol format: EUR/USD → EURUSD (strip the slash).
    # ------------------------------------------------------------------

    @staticmethod
    def _fx_symbol(symbol: str) -> str:
        """Convert 'EUR/USD' → 'EURUSD' for Alpaca's market data endpoints."""
        return symbol.replace("/", "")

    def get_forex_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        lookback: int = 60,
    ) -> List[Dict[str, Any]]:
        """Return the last *lookback* OHLCV bars for a forex pair (e.g. 'EUR/USD').

        Returns an empty list when the market is closed (weekends) or no bars
        are available yet — the caller already handles this gracefully.
        """
        import httpx

        _TF_MAP = {
            "1Min": "1Min", "5Min": "5Min", "15Min": "15Min",
            "1Hour": "1Hour", "1Day": "1Day",
        }
        tf = _TF_MAP.get(timeframe, "1Min")
        fx_sym = self._fx_symbol(symbol)
        start = (
            datetime.now(timezone.utc) - timedelta(minutes=lookback * _tf_minutes(timeframe))
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        url = f"{_ALPACA_DATA_URL}/v2/stocks/{fx_sym}/bars"
        params = {"timeframe": tf, "start": start, "limit": lookback, "feed": "iex"}
        headers = {
            "APCA-API-KEY-ID": os.environ.get("APCA_API_KEY_ID", ""),
            "APCA-API-SECRET-KEY": os.environ.get("APCA_API_SECRET_KEY", ""),
        }
        resp = httpx.get(url, params=params, headers=headers, timeout=10.0)
        resp.raise_for_status()
        raw = resp.json().get("bars") or []
        return [
            {
                "t": b["t"],
                "o": float(b["o"]),
                "h": float(b["h"]),
                "l": float(b["l"]),
                "c": float(b["c"]),
                "v": float(b.get("v", 0.0)),
            }
            for b in raw
        ]

    def get_forex_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Return the latest bid/ask for a forex pair (e.g. 'EUR/USD').

        Raises RuntimeError when the market is closed or no quote is available —
        the caller (safety tick, tick workflow) should treat this as a skip.
        """
        import httpx

        fx_sym = self._fx_symbol(symbol)
        url = f"{_ALPACA_DATA_URL}/v2/stocks/{fx_sym}/quotes/latest"
        headers = {
            "APCA-API-KEY-ID": os.environ.get("APCA_API_KEY_ID", ""),
            "APCA-API-SECRET-KEY": os.environ.get("APCA_API_SECRET_KEY", ""),
        }
        resp = httpx.get(url, params={"feed": "iex"}, headers=headers, timeout=10.0)
        if resp.status_code == 404:
            raise RuntimeError(f"No quote available for {symbol} (market may be closed)")
        resp.raise_for_status()
        q = resp.json().get("quote") or {}
        if not q or not q.get("bp"):
            raise RuntimeError(f"Empty quote for {symbol} (market may be closed)")
        return {
            "symbol": symbol,
            "bid": float(q["bp"]),
            "ask": float(q["ap"]),
            "bid_size": float(q.get("bs", 0)),
            "ask_size": float(q.get("as", 0)),
            "timestamp": q.get("t", ""),
        }

    # ------------------------------------------------------------------
    # Equities stubs — v2
    # ------------------------------------------------------------------

    def get_bars(self, symbol: str, timeframe: str = "1Min", lookback: int = 60):
        raise NotImplementedError("Equities bar fetch enabled in v2")

    def get_latest_quote(self, symbol: str):
        raise NotImplementedError("Equities quote fetch enabled in v2")


def register_plugin(maki_instance=None):
    return AlpacaData(maki_instance)


def _tf_minutes(tf: str) -> int:
    return {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60, "1Day": 1440}.get(tf, 1)
