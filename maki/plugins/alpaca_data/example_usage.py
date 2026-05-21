"""
Example usage of the AlpacaData plugin with Maki agents.

Requires environment variables: APCA_API_KEY_ID, APCA_API_SECRET_KEY
"""

from maki.plugins.alpaca_data.alpaca_data import AlpacaData


def main():
    plugin = AlpacaData()

    print("AlpacaData plugin example usage")
    print("================================")

    # Example 1: latest BTC/USD quote
    print("\nExample 1: Latest BTC/USD quote")
    quote = plugin.get_crypto_latest_quote("BTC/USD")
    print(f"  bid={quote['bid']}  ask={quote['ask']}  @ {quote['timestamp']}")

    # Example 2: last 5 hourly bars for ETH/USD
    print("\nExample 2: Last 5 hourly ETH/USD bars")
    bars = plugin.get_crypto_bars("ETH/USD", timeframe="1Hour", lookback=5)
    for bar in bars:
        print(f"  {bar['t']}  o={bar['o']}  c={bar['c']}  v={bar['v']}")

    # Example 3: list tradable crypto assets
    print("\nExample 3: Tradable crypto assets (first 10)")
    symbols = plugin.list_crypto_assets()
    print("  " + ", ".join(symbols[:10]))


if __name__ == "__main__":
    main()
