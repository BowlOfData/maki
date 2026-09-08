# Alpaca Data Plugin

Fetches crypto and equity market data from Alpaca via the `alpaca-py` SDK.
Forex data is served via Yahoo Finance (`yfinance`) since Alpaca does not
offer FX data.

## Requirements

Set the following environment variables:

```
APCA_API_KEY_ID=<your alpaca api key>
APCA_API_SECRET_KEY=<your alpaca secret key>
```

Forex methods additionally require `yfinance` (installed automatically with
`pip install "maki[alpaca]"`).

## Usage

```python
from maki.plugins.alpaca_data.alpaca_data import AlpacaData

plugin = AlpacaData()

# Crypto OHLCV bars
bars = plugin.get_crypto_bars("BTC/USD", timeframe="1Hour", lookback=24)

# Crypto latest bid/ask
quote = plugin.get_crypto_latest_quote("ETH/USD")

# All tradable crypto symbols
symbols = plugin.list_crypto_assets()

# Forex OHLCV bars (via Yahoo Finance)
fx_bars = plugin.get_forex_bars("EUR/USD", timeframe="1Hour", lookback=24)

# Forex latest bid/ask (synthesised spread, via Yahoo Finance)
fx_quote = plugin.get_forex_latest_quote("EUR/USD")

# Equity OHLCV bars
eq_bars = plugin.get_equity_bars("AAPL", timeframe="1Day", lookback=30)

# Equity latest bid/ask
eq_quote = plugin.get_equity_latest_quote("AAPL")
```

## Methods

### `get_crypto_bars(symbol, timeframe="1Min", lookback=60)`

Returns the last `lookback` OHLCV bars for `symbol` (e.g. `"BTC/USD"`).

**Supported timeframes:** `1Min`, `5Min`, `15Min`, `1Hour`, `1Day`

**Returns:** list of dicts with keys `t`, `o`, `h`, `l`, `c`, `v`.

### `get_crypto_latest_quote(symbol)`

Returns the latest bid/ask quote for `symbol`.

**Returns:** dict with keys `symbol`, `bid`, `ask`, `bid_size`, `ask_size`, `timestamp`.

### `list_crypto_assets()`

Returns all tradable crypto symbols available on Alpaca.

**Returns:** list of symbol strings.

### `get_forex_bars(symbol, timeframe="1Min", lookback=60)`

Returns the last `lookback` OHLCV bars for a forex pair (e.g. `"EUR/USD"`) via
Yahoo Finance. Same return shape as `get_crypto_bars`.

### `get_forex_latest_quote(symbol)`

Returns the latest bid/ask for a forex pair via Yahoo Finance. Yahoo doesn't
provide a real spread for FX, so a 1-pip spread is synthesised around the
last price. Raises `RuntimeError` when no quote is available (e.g. market
closed).

### `get_equity_bars(symbol, timeframe="1Min", lookback=60)`

Returns the last `lookback` OHLCV bars for a US equity (e.g. `"AAPL"`). Same
return shape as `get_crypto_bars`.

### `get_equity_latest_quote(symbol)`

Returns the latest bid/ask for a US equity. Raises `RuntimeError` when no
quote is available (e.g. market closed).
