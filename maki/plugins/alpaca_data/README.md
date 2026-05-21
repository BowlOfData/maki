# Alpaca Data Plugin

Fetches crypto market data from Alpaca via the `alpaca-py` SDK. Equities support is planned for v2.

## Requirements

Set the following environment variables:

```
APCA_API_KEY_ID=<your alpaca api key>
APCA_API_SECRET_KEY=<your alpaca secret key>
```

## Usage

```python
from maki.plugins.alpaca_data.alpaca_data import AlpacaData

plugin = AlpacaData()

# OHLCV bars
bars = plugin.get_crypto_bars("BTC/USD", timeframe="1Hour", lookback=24)

# Latest bid/ask
quote = plugin.get_crypto_latest_quote("ETH/USD")

# All tradable crypto symbols
symbols = plugin.list_crypto_assets()
```

## Methods

### `get_crypto_bars(symbol, timeframe="1Min", lookback=60)`

Returns the last `lookback` OHLCV bars for `symbol`.

**Supported timeframes:** `1Min`, `5Min`, `15Min`, `1Hour`, `1Day`

**Returns:** list of dicts with keys `t`, `o`, `h`, `l`, `c`, `v`.

### `get_crypto_latest_quote(symbol)`

Returns the latest bid/ask quote for `symbol`.

**Returns:** dict with keys `symbol`, `bid`, `ask`, `bid_size`, `ask_size`, `timestamp`.

### `list_crypto_assets()`

Returns all tradable crypto symbols available on Alpaca.

**Returns:** list of symbol strings.
