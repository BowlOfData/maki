"""
Alpaca trading plugin for Maki.

Wraps alpaca-py's TradingClient. Paper-only by default.
Live trading requires TRANDING_ALLOW_LIVE=1 AND runtime CLI confirmation.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ALLOWED_METHODS = [
    "submit_order",
    "list_positions",
    "get_account",
    "cancel_order",
    "close_position",
    "get_order",
]

_PAPER_URL = "https://paper-api.alpaca.markets"
_LIVE_URL = "https://api.alpaca.markets"


class AlpacaTrading:
    def __init__(self, maki_instance=None):
        try:
            from alpaca.trading import TradingClient
        except ImportError as e:
            raise ImportError(
                'alpaca-py is not installed. Run: pip install "maki[alpaca]"'
            ) from e

        api_key = os.environ.get("APCA_API_KEY_ID")
        api_secret = os.environ.get("APCA_API_SECRET_KEY")

        allow_live = os.environ.get("TRANDING_ALLOW_LIVE", "0").strip() == "1"
        if allow_live:
            logger.warning("LIVE trading mode enabled — real money at risk")
            self._paper = False
        else:
            self._paper = True

        self._client = TradingClient(api_key=api_key, secret_key=api_secret, paper=self._paper)
        mode = "paper" if self._paper else "LIVE"
        logger.info(f"AlpacaTrading plugin initialised ({mode})")

    def get_account(self) -> Dict[str, Any]:
        """Return account equity, cash, and buying power."""
        acct = self._client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "currency": acct.currency,
            "paper": self._paper,
        }

    def list_positions(self) -> List[Dict[str, Any]]:
        """Return all open positions."""
        positions = self._client.get_all_positions()
        return [
            {
                "symbol": _normalize_symbol(p.symbol),
                "qty": float(p.qty),
                "side": p.side.value,
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price) if p.current_price else None,
                "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl else None,
                "market_value": float(p.market_value) if p.market_value else None,
            }
            for p in positions
        ]

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "gtc",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit a trading order. Returns order dict."""
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif_enum = TimeInForce.GTC if time_in_force.lower() == "gtc" else TimeInForce.DAY

        if order_type == "market":
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=tif_enum,
                client_order_id=client_order_id,
            )
        elif order_type == "limit" and limit_price:
            req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=tif_enum,
                limit_price=limit_price,
                client_order_id=client_order_id,
            )
        elif order_type == "stop" and stop_price:
            req = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                type=OrderType.STOP,
                time_in_force=tif_enum,
                stop_price=round(stop_price, 8),
                client_order_id=client_order_id,
            )
        else:
            raise ValueError(f"Unsupported order_type='{order_type}'")

        order = self._client.submit_order(req)
        logger.info(f"Order submitted: {side} {qty} {symbol} ({order_type}) → {order.id}")
        return _order_to_dict(order)

    def get_order(self, order_id: str) -> Dict[str, Any]:
        order = self._client.get_order_by_id(order_id)
        return _order_to_dict(order)

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.warning(f"Cancel order {order_id} failed: {e}")
            return False

    def close_position(self, symbol: str) -> Dict[str, Any]:
        order = self._client.close_position(symbol)
        return _order_to_dict(order)


def register_plugin(maki_instance=None):
    return AlpacaTrading(maki_instance)


# Alpaca may return crypto symbols without a slash (e.g. "BTCUSD" instead of "BTC/USD").
# Normalize to slash-separated format so all consumers can match consistently.
_CRYPTO_QUOTES = ("USDT", "USDC", "USD")  # longest first to avoid partial matches


def _normalize_symbol(symbol: str) -> str:
    if "/" in symbol:
        return symbol
    for quote in _CRYPTO_QUOTES:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return f"{symbol[:-len(quote)]}/{quote}"
    return symbol


def _order_to_dict(order) -> Dict[str, Any]:
    return {
        "id": str(order.id),
        "client_order_id": str(order.client_order_id or ""),
        "symbol": order.symbol,
        "qty": str(order.qty or ""),
        "filled_qty": str(order.filled_qty or ""),
        "filled_avg_price": str(order.filled_avg_price or ""),
        "side": order.side.value,
        "type": order.type.value,
        "status": order.status.value,
        "submitted_at": order.submitted_at.isoformat() if order.submitted_at else "",
        "filled_at": order.filled_at.isoformat() if order.filled_at else "",
    }
