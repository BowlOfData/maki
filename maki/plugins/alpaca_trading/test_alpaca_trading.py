import pytest
from .alpaca_trading import _normalize_symbol


@pytest.mark.parametrize("raw, expected", [
    # Already normalized — leave alone
    ("BTC/USD",  "BTC/USD"),
    ("ETH/USDT", "ETH/USDT"),
    ("SOL/USDC", "SOL/USDC"),
    # Alpaca slash-less format — normalize
    ("BTCUSD",   "BTC/USD"),
    ("ETHUSD",   "ETH/USD"),
    ("SOLUSD",   "SOL/USD"),
    ("DOGEUSD",  "DOGE/USD"),
    ("LINKUSD",  "LINK/USD"),
    ("BTCUSDT",  "BTC/USDT"),
    ("ETHUSDC",  "ETH/USDC"),
    # Equity symbols — untouched
    ("AAPL",     "AAPL"),
    ("TSLA",     "TSLA"),
    ("SPY",      "SPY"),
])
def test_normalize_symbol(raw, expected):
    assert _normalize_symbol(raw) == expected
