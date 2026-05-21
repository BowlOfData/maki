# Alpaca Trading Plugin

Submits and manages orders via Alpaca's `TradingClient`. Runs in **paper mode by default**; live trading requires `TRANDING_ALLOW_LIVE=1`.

## Requirements

```
APCA_API_KEY_ID=<your alpaca api key>
APCA_API_SECRET_KEY=<your alpaca secret key>

# Optional — enable live trading (real money at risk)
TRANDING_ALLOW_LIVE=1
```

## Usage

```python
from maki.plugins.alpaca_trading.alpaca_trading import AlpacaTrading

plugin = AlpacaTrading()  # paper mode

# Account info
acct = plugin.get_account()
print(acct["equity"], acct["buying_power"])

# Place a market order
order = plugin.submit_order("BTC/USD", qty=0.001, side="buy")

# List open positions
positions = plugin.list_positions()

# Cancel an order
plugin.cancel_order(order["id"])

# Close an entire position
plugin.close_position("BTC/USD")
```

## Methods

### `get_account()`

Returns account equity, cash, buying power, portfolio value, currency, and paper flag.

### `list_positions()`

Returns all open positions as a list of dicts.

### `submit_order(symbol, qty, side, order_type="market", time_in_force="gtc", limit_price=None, stop_price=None, client_order_id=None)`

Submits a market or limit order. Returns an order dict.

### `get_order(order_id)`

Fetches a single order by ID.

### `cancel_order(order_id)`

Cancels an order. Returns `True` on success.

### `close_position(symbol)`

Closes the entire position for `symbol`. Returns the resulting order dict.
