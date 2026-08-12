# ============================================================
# STAGE 18 - MARKET DEPTH VALIDATOR
# stage18/depth_validator.py
# ============================================================

import time
import traceback

from api.historical_data import get_smartapi_object
from utils.token_map import TOKENS


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SYMBOLS = [
    "SBIN-EQ",
    "RELIANCE-EQ",
    "ITC-EQ",
    "SUZLON-EQ",
    "IRFC-EQ",
    "TATAMOTORS-EQ",
]

BATCH_SIZE = 50
DELAY = 1.2


# ============================================================
# NORMALIZE SYMBOL
# ============================================================

def normalize_symbol(symbol):

    if symbol is None:
        return ""

    symbol = str(symbol).strip().upper()

    if symbol.endswith("-EQ"):
        return symbol

    return f"{symbol}-EQ"


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        return float(
            str(value).replace(",", "").strip()
        )

    except Exception:

        return default


# ============================================================
# EXTRACT DEPTH
# ============================================================

def extract_depth(quote):

    bid_price = 0.0
    bid_quantity = 0.0

    ask_price = 0.0
    ask_quantity = 0.0

    if not isinstance(quote, dict):
        return (
            bid_price,
            bid_quantity,
            ask_price,
            ask_quantity,
        )

    # --------------------------------------------------------
    # Direct fields
    # --------------------------------------------------------

    bid_price = safe_float(
        quote.get(
            "bestBidPrice",
            quote.get(
                "bidPrice",
                0
            )
        )
    )

    bid_quantity = safe_float(
        quote.get(
            "bestBidQty",
            quote.get(
                "bestBidQuantity",
                quote.get(
                    "bidQty",
                    quote.get(
                        "bidQuantity",
                        0
                    )
                )
            )
        )
    )

    ask_price = safe_float(
        quote.get(
            "bestAskPrice",
            quote.get(
                "askPrice",
                0
            )
        )
    )

    ask_quantity = safe_float(
        quote.get(
            "bestAskQty",
            quote.get(
                "bestAskQuantity",
                quote.get(
                    "askQty",
                    quote.get(
                        "askQuantity",
                        0
                    )
                )
            )
        )
    )

    # --------------------------------------------------------
    # Angel One depth
    # --------------------------------------------------------

    depth = quote.get(
        "depth",
        {}
    )

    if isinstance(depth, dict):

        buy = depth.get(
            "buy",
            []
        )

        sell = depth.get(
            "sell",
            []
        )

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        if isinstance(buy, list) and buy:

            first_buy = buy[0]

            if isinstance(first_buy, dict):

                price = safe_float(
                    first_buy.get(
                        "price",
                        0
                    )
                )

                quantity = safe_float(
                    first_buy.get(
                        "quantity",
                        first_buy.get(
                            "qty",
                            0
                        )
                    )
                )

                if price > 0:
                    bid_price = price

                if quantity > 0:
                    bid_quantity = quantity

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------

        if isinstance(sell, list) and sell:

            first_sell = sell[0]

            if isinstance(first_sell, dict):

                price = safe_float(
                    first_sell.get(
                        "price",
                        0
                    )
                )

                quantity = safe_float(
                    first_sell.get(
                        "quantity",
                        first_sell.get(
                            "qty",
                            0
                        )
                    )
                )

                if price > 0:
                    ask_price = price

                if quantity > 0:
                    ask_quantity = quantity

    return (
        bid_price,
        bid_quantity,
        ask_price,
        ask_quantity,
    )


# ============================================================
# FETCH MARKET DATA
# ============================================================

def fetch_market_data():

    print()
    print("=" * 80)
    print("STAGE 18 - MARKET DEPTH VALIDATION")
    print("=" * 80)

    api = get_smartapi_object()

    if api is None:

        print()
        print("ERROR: SmartAPI connection unavailable.")

        return {}

    resolved = []

    for symbol in TEST_SYMBOLS:

        symbol = normalize_symbol(symbol)

        token = TOKENS.get(symbol)

        if token:

            resolved.append(
                (
                    symbol,
                    str(token)
                )
            )

        else:

            print(
                f"Token not found: {symbol}"
            )

    if not resolved:

        print("No tokens resolved.")

        return {}

    results = {}

    # ========================================================
    # BATCH REQUEST
    # ========================================================

    for start in range(
        0,
        len(resolved),
        BATCH_SIZE
    ):

        batch = resolved[
            start:start + BATCH_SIZE
        ]

        exchange_tokens = {
            "NSE": [
                token
                for _, token in batch
            ]
        }

        print()
        print(
            f"Requesting FULL market data "
            f"for {len(batch)} stocks..."
        )

        try:

            response = api.getMarketData(
                "FULL",
                exchange_tokens
            )

            if not isinstance(
                response,
                dict
            ):

                print(
                    "Invalid response."
                )

                continue

            print(
                f"API status: "
                f"{response.get('status')}"
            )

            if not response.get(
                "status",
                False
            ):

                print(
                    "API error:"
                )

                print(response)

                continue

            data = response.get(
                "data",
                {}
            )

            fetched = data.get(
                "fetched",
                []
            )

            unfetched = data.get(
                "unfetched",
                []
            )

            print(
                f"Fetched : {len(fetched)}"
            )

            print(
                f"Unfetched : {len(unfetched)}"
            )

            # ------------------------------------------------
            # Save by token
            # ------------------------------------------------

            for item in fetched:

                if not isinstance(
                    item,
                    dict
                ):

                    continue

                token = str(
                    item.get(
                        "symbolToken",
                        item.get(
                            "symboltoken",
                            ""
                        )
                    )
                ).strip()

                matched_symbol = None

                for symbol, known_token in batch:

                    if token == str(
                        known_token
                    ).strip():

                        matched_symbol = symbol

                        break

                if matched_symbol:

                    results[
                        matched_symbol
                    ] = item

        except Exception as e:

            print()
            print(
                "Market-data request failed:"
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

        time.sleep(DELAY)

    return results


# ============================================================
# DISPLAY ONE QUOTE
# ============================================================

def display_quote(
    symbol,
    quote
):

    print()
    print("-" * 70)
    print(symbol)
    print("-" * 70)

    if not isinstance(
        quote,
        dict
    ):

        print(
            "No valid quote."
        )

        return

    # --------------------------------------------------------
    # Basic fields
    # --------------------------------------------------------

    print(
        f"LTP                  : "
        f"{safe_float(quote.get('ltp'))}"
    )

    print(
        f"Trade Volume         : "
        f"{safe_float(quote.get('tradeVolume'))}"
    )

    print(
        f"Average Price        : "
        f"{safe_float(quote.get('avgPrice'))}"
    )

    print(
        f"Total Buy Quantity   : "
        f"{safe_float(quote.get('totBuyQuan'))}"
    )

    print(
        f"Total Sell Quantity  : "
        f"{safe_float(quote.get('totSellQuan'))}"
    )

    # --------------------------------------------------------
    # Depth
    # --------------------------------------------------------

    (
        bid_price,
        bid_quantity,
        ask_price,
        ask_quantity,
    ) = extract_depth(quote)

    print()

    print(
        f"Best Bid Price       : "
        f"{bid_price}"
    )

    print(
        f"Best Bid Quantity    : "
        f"{bid_quantity}"
    )

    print(
        f"Best Ask Price       : "
        f"{ask_price}"
    )

    print(
        f"Best Ask Quantity    : "
        f"{ask_quantity}"
    )

    # --------------------------------------------------------
    # Liquidity test
    # --------------------------------------------------------

    buy_pass = (
        bid_quantity > 1_000_000
    )

    sell_pass = (
        ask_quantity > 1_000_000
    )

    print()

    print(
        f"Buy > 1,000,000      : "
        f"{'PASS' if buy_pass else 'FAIL'}"
    )

    print(
        f"Sell > 1,000,000     : "
        f"{'PASS' if sell_pass else 'FAIL'}"
    )

    print(
        f"Liquidity            : "
        f"{'PASS' if buy_pass and sell_pass else 'FAIL'}"
    )

    # --------------------------------------------------------
    # RAW DEPTH
    # --------------------------------------------------------

    depth = quote.get(
        "depth"
    )

    print()

    print(
        "RAW DEPTH:"
    )

    print(
        depth
    )


# ============================================================
# MAIN
# ============================================================

def main():

    quotes = fetch_market_data()

    print()
    print("=" * 80)
    print("DEPTH VALIDATION RESULTS")
    print("=" * 80)

    if not quotes:

        print()
        print(
            "No market quotes received."
        )

        return

    for symbol in TEST_SYMBOLS:

        symbol = normalize_symbol(
            symbol
        )

        quote = quotes.get(
            symbol
        )

        display_quote(
            symbol,
            quote
        )

    print()
    print("=" * 80)
    print("STAGE 18 DEPTH VALIDATION COMPLETE")
    print("=" * 80)


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    main()