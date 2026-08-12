from api.live_data import get_live_market_data
from utils.live_parser import parse_live_market_data
from utils.token_map import TOKENS


# ============================================================
# STOCK LIST
# ============================================================
from utils.token_map import TOKENS


# ============================================================
# NSE STOCK UNIVERSE
# ============================================================

STOCKS = list(TOKENS.keys())

# ============================================================
# SCREEN ONE STOCK
# ============================================================

def screen_stock(symbol):

    try:

        # ----------------------------------------------------
        # GET LIVE MARKET DATA
        # ----------------------------------------------------

        raw_data = get_live_market_data(symbol)

        if raw_data is None:
            print(f"{symbol}: No live data")
            return None


        # ----------------------------------------------------
        # PARSE API RESPONSE
        # ----------------------------------------------------

        data = parse_live_market_data(
            symbol,
            raw_data
        )

        if data is None:
            print(f"{symbol}: Unable to parse data")
            return None


        # ----------------------------------------------------
        # GET VALUES
        # ----------------------------------------------------

        ltp = data["LTP"]
        bid_quantity = data["Bid Quantity"]
        ask_quantity = data["Ask Quantity"]


        # ----------------------------------------------------
        # PRICE FILTER
        # Requirement: ₹30 - ₹500
        # ----------------------------------------------------

        if not (30 <= ltp <= 500):

            print(
                f"{symbol}: REJECTED - "
                f"LTP ₹{ltp} outside ₹30-₹500"
            )

            return None


        # ----------------------------------------------------
        # BID QUANTITY FILTER
        # Requirement: > 10,00,000
        # ----------------------------------------------------

        if bid_quantity <= 1_000_000:

            print(
                f"{symbol}: REJECTED - "
                f"Bid Quantity = {bid_quantity:,}"
            )

            return None


        # ----------------------------------------------------
        # ASK QUANTITY FILTER
        # Requirement: > 10,00,000
        # ----------------------------------------------------

        if ask_quantity <= 1_000_000:

            print(
                f"{symbol}: REJECTED - "
                f"Ask Quantity = {ask_quantity:,}"
            )

            return None


        # ----------------------------------------------------
        # STOCK PASSED ALL FILTERS
        # ----------------------------------------------------

        print(
            f"{symbol}: PASSED FILTER"
        )

        return data


    except Exception as e:

        print(
            f"{symbol}: ERROR - {e}"
        )

        return None


# ============================================================
# SCREEN ALL STOCKS
# ============================================================

def screen_all_stocks():

    results = []

    stocks = list(TOKENS.keys())

    print(f"\nTotal NSE stocks available: {len(stocks)}")
    print("Starting screening...\n")

    for symbol in stocks:

        result = screen_stock(symbol)

        if result is not None:
            results.append(result)

    return results
# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n========== LIVE STOCK SCREENER ==========\n")

    # Test only first 20 NSE stocks
    test_stocks = list(TOKENS.keys())[:20]

    print(f"Testing {len(test_stocks)} stocks...\n")

    results = []

    for symbol in test_stocks:

        result = screen_stock(symbol)

        if result is not None:
            results.append(result)

    print("\n========== QUALIFIED STOCKS ==========\n")

    if not results:

        print("No stocks passed the filters.")

    else:

        for result in results:

            print(
                f"{result['Stock']:<15}"
                f"LTP: ₹{result['LTP']:<10}"
                f"Bid Qty: {result['Bid Quantity']:<12}"
                f"Ask Qty: {result['Ask Quantity']}"
            )

    print(
        f"\nStocks passed filter: {len(results)}"
    )