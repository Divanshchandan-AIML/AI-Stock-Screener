from api.broker import connect
from utils.token_map import TOKENS


def get_live_market_data(symbol):
    """
    Get full live market data for an NSE stock.

    Returns:
        dict: Full broker market-data response.
        None: If the request fails or the symbol/token is unavailable.
    """

    try:
        # Check whether symbol exists
        if symbol not in TOKENS:
            print(f"{symbol}: Token not found")
            return None

        token = TOKENS[symbol]

        # Connect to broker
        api = connect()

        # Request FULL market data
        data = api.getMarketData(
            "FULL",
            {
                "NSE": [str(token)]
            }
        )

        if not data:
            print(f"{symbol}: No market data returned")
            return None

        return data

    except Exception as e:
        print(f"Error getting live data for {symbol}: {e}")
        return None


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    symbol = "SBIN-EQ"

    data = get_live_market_data(symbol)

    print("\n========== FULL LIVE MARKET DATA ==========\n")

    if data:
        print(data)
    else:
        print("No data received.")