"""
Example usage of the AlpacaTrading plugin.

Requires environment variables: APCA_API_KEY_ID, APCA_API_SECRET_KEY
Runs in paper mode by default. Set TRANDING_ALLOW_LIVE=1 for live trading.
"""

from maki.plugins.alpaca_trading.alpaca_trading import AlpacaTrading


def main():
    plugin = AlpacaTrading()  # paper mode by default

    print("AlpacaTrading plugin example usage")
    print("===================================")

    # Example 1: Account info
    print("\nExample 1: Account info")
    acct = plugin.get_account()
    print(f"  equity={acct['equity']}  cash={acct['cash']}  paper={acct['paper']}")

    # Example 2: Open positions
    print("\nExample 2: Open positions")
    positions = plugin.list_positions()
    if positions:
        for p in positions:
            print(f"  {p['symbol']}  qty={p['qty']}  side={p['side']}")
    else:
        print("  No open positions")

    # Example 3: Submit a paper market order
    print("\nExample 3: Submit a BTC/USD market buy (paper)")
    order = plugin.submit_order("BTC/USD", qty=0.001, side="buy")
    print(f"  order id={order['id']}  status={order['status']}")

    # Example 4: Cancel the order
    print("\nExample 4: Cancel the order")
    cancelled = plugin.cancel_order(order["id"])
    print(f"  cancelled={cancelled}")


if __name__ == "__main__":
    main()
