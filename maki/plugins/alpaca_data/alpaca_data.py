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
    # Mirror the module-level whitelist on the class: tool-call validation
    # reads ALLOWED_METHODS from the plugin instance, not the module.
    ALLOWED_METHODS = ALLOWED_METHODS

    def __init__(self, maki_instance=None):
        try:
            from alpaca.data.historical import CryptoHistoricalDataClient
        except ImportError as e:
            raise ImportError(
                'alpaca-py is not installed. Run: pip install "maki[alpaca]"'
            ) from e

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

    @staticmethod
    def _yf_symbol(symbol: str) -> str:
        """Convert 'EUR/USD' → 'EURUSD=X' for Yahoo Finance."""
        return symbol.replace("/", "") + "=X"

    @staticmethod
    def _yf_interval(timeframe: str) -> str:
        return {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "1h", "1Day": "1d"}.get(timeframe, "1m")

    def get_forex_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        lookback: int = 60,
    ) -> List[Dict[str, Any]]:
        """Return the last *lookback* OHLCV bars for a forex pair via Yahoo Finance."""
        import yfinance as yf

        yf_sym = self._yf_symbol(symbol)
        interval = self._yf_interval(timeframe)
        period_minutes = lookback * _tf_minutes(timeframe)
        # yfinance period string: use days for longer windows, else intraday
        if period_minutes <= 1440:
            period = "1d"
        elif period_minutes <= 7 * 1440:
            period = "5d"
        else:
            period = "1mo"

        ticker = yf.Ticker(yf_sym)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return []
        df = df.tail(lookback)
        return [
            {
                "t": idx.isoformat(),
                "o": float(row["Open"]),
                "h": float(row["High"]),
                "l": float(row["Low"]),
                "c": float(row["Close"]),
                "v": float(row.get("Volume", 0.0)),
            }
            for idx, row in df.iterrows()
        ]

    def get_forex_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Return the latest bid/ask for a forex pair via Yahoo Finance.

        Raises RuntimeError when no quote is available.
        """
        import yfinance as yf

        yf_sym = self._yf_symbol(symbol)
        ticker = yf.Ticker(yf_sym)
        info = ticker.fast_info
        price = getattr(info, "last_price", None)
        if not price:
            raise RuntimeError(f"No quote available for {symbol} (market may be closed)")
        # Yahoo Finance doesn't provide a real spread for FX; synthesise a 1-pip spread.
        pip = 0.0001 if "JPY" not in symbol else 0.01
        return {
            "symbol": symbol,
            "bid": round(price - pip / 2, 6),
            "ask": round(price + pip / 2, 6),
            "bid_size": 0.0,
            "ask_size": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Equities stubs — v2
    # ------------------------------------------------------------------

    def get_equity_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        lookback: int = 60,
    ) -> List[Dict[str, Any]]:
        """Return the last *lookback* OHLCV bars for a US equity (e.g. 'AAPL')."""
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }
        tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Minute))
        api_key = os.environ.get("APCA_API_KEY_ID")
        api_secret = os.environ.get("APCA_API_SECRET_KEY")
        client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
        start = datetime.now(timezone.utc) - timedelta(minutes=lookback * _tf_minutes(timeframe))
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start, limit=lookback)
        bars = client.get_stock_bars(req)
        symbol_bars = bars.data.get(symbol) or []
        return [
            {
                "t": bar.timestamp.isoformat(),
                "o": float(bar.open),
                "h": float(bar.high),
                "l": float(bar.low),
                "c": float(bar.close),
                "v": float(bar.volume),
            }
            for bar in symbol_bars
        ]

    def get_equity_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Return the latest bid/ask for a US equity (e.g. 'AAPL')."""
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest

        api_key = os.environ.get("APCA_API_KEY_ID")
        api_secret = os.environ.get("APCA_API_SECRET_KEY")
        client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = client.get_stock_latest_quote(req)
        q = quotes.get(symbol) if isinstance(quotes, dict) else quotes.data.get(symbol)
        if not q or not q.bid_price or not q.ask_price:
            raise RuntimeError(f"No quote available for {symbol} (market may be closed)")
        return {
            "symbol": symbol,
            "bid": float(q.bid_price),
            "ask": float(q.ask_price),
            "bid_size": float(q.bid_size),
            "ask_size": float(q.ask_size),
            "timestamp": q.timestamp.isoformat(),
        }


def register_plugin(maki_instance=None):
    return AlpacaData(maki_instance)


def _tf_minutes(tf: str) -> int:
    return {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60, "1Day": 1440}.get(tf, 1)
