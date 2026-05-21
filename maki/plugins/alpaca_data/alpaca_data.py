"""
Alpaca market-data plugin for Maki.

Wraps alpaca-py's CryptoHistoricalDataClient for bar and quote fetching.
Equities methods raise NotImplementedError (enabled in v2).
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ALLOWED_METHODS = [
    "get_crypto_bars",
    "get_crypto_latest_quote",
    "list_crypto_assets",
]

_ALPACA_DATA_URL = "https://data.alpaca.markets"


class AlpacaData:
    def __init__(self, maki_instance=None):
        from alpaca.data.historical import CryptoHistoricalDataClient

        api_key = os.environ.get("APCA_API_KEY_ID")
        api_secret = os.environ.get("APCA_API_SECRET_KEY")
        self._client = CryptoHistoricalDataClient(api_key=api_key, secret_key=api_secret)
        logger.info("AlpacaData plugin initialised (crypto)")

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
