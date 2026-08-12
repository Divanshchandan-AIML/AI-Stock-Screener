from api.historical_data import get_historical_data
from indicators.smma import add_smma_indicators
from indicators.crossover import get_crossover_signals


# ============================================================
# TRACKED STOCKS
# ============================================================

STOCKS = [
    "SBIN-EQ",
    "SUZLON-EQ",
    "IRFC-EQ",
    "TATAMOTORS-EQ",
    "RELIANCE-EQ",
    "ITC-EQ",
]


# ============================================================
# CHECK ONE STOCK
# ============================================================

def check_stock(symbol):
    """
    Fetch historical data, calculate SMMA indicators,
    detect crossover signals and return the latest result.
    """

    print(f"\nChecking {symbol}...")

    # --------------------------------------------------------
    # 1. Get historical data
    # --------------------------------------------------------

    try:
        df = get_historical_data(
            symbol,
            days=250
        )
    except Exception as e:
        print(f"{symbol}: Historical data error - {e}")
        return None

    if df is None or df.empty:
        print(f"{symbol}: No historical data")
        return None

    # --------------------------------------------------------
    # 2. Calculate SMMA20 and SMMA120
    # --------------------------------------------------------

    try:
        df = add_smma_indicators(df)
    except Exception as e:
        print(f"{symbol}: SMMA calculation error - {e}")
        return None

    if df is None or df.empty:
        print(f"{symbol}: No data after SMMA calculation")
        return None

    # --------------------------------------------------------
    # 3. Detect crossover signals
    # --------------------------------------------------------

    try:
        df = get_crossover_signals(df)
    except Exception as e:
        print(f"{symbol}: Crossover calculation error - {e}")
        return None

    if df is None or df.empty:
        print(f"{symbol}: No data after crossover calculation")
        return None

    # --------------------------------------------------------
    # 4. Check required columns
    # --------------------------------------------------------

    required_columns = [
        "close",
        "SMMA20",
        "SMMA120",
        "Signal",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print(
            f"{symbol}: Missing columns: "
            f"{', '.join(missing_columns)}"
        )
        return None

    # --------------------------------------------------------
    # 5. Get latest row
    # --------------------------------------------------------

    latest = df.iloc[-1]

    # --------------------------------------------------------
    # 6. Validate indicator values
    # --------------------------------------------------------

    if (
        latest["close"] is None
        or latest["SMMA20"] is None
        or latest["SMMA120"] is None
    ):
        print(f"{symbol}: Invalid latest indicator values")
        return None

    # --------------------------------------------------------
    # 7. Create result
    # --------------------------------------------------------

    result = {
        "Stock": symbol.replace("-EQ", ""),
        "Symbol": symbol,
        "Price": float(latest["close"]),
        "SMMA20": float(latest["SMMA20"]),
        "SMMA120": float(latest["SMMA120"]),
        "Signal": str(latest["Signal"]),
    }

    return result


# ============================================================
# CHECK ALL STOCKS
# ============================================================

def check_all_stocks():
    """
    Check all tracked stocks and return successful results.
    """

    results = []

    print("\n" + "=" * 60)
    print("             SMMA CROSSOVER SCANNER")
    print("=" * 60)

    for symbol in STOCKS:

        try:

            result = check_stock(symbol)

            if result is not None:
                results.append(result)

        except Exception as e:

            print(
                f"{symbol}: UNEXPECTED ERROR - {e}"
            )

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):
    """
    Print scanner results in a clean table.
    """

    print("\n" + "=" * 90)
    print("                              RESULTS")
    print("=" * 90)

    if not results:
        print("No stocks returned valid results.")
        print("=" * 90)
        return

    print(
        f"{'STOCK':<12}"
        f"{'PRICE':>12}"
        f"{'SMMA20':>14}"
        f"{'SMMA120':>14}"
        f"{'SIGNAL':>18}"
    )

    print("-" * 90)

    for result in results:

        print(
            f"{result['Stock']:<12}"
            f"₹{result['Price']:>10.2f}"
            f"{result['SMMA20']:>14.2f}"
            f"{result['SMMA120']:>14.2f}"
            f"{result['Signal']:>18}"
        )

    print("-" * 90)

    # --------------------------------------------------------
    # Signal summary
    # --------------------------------------------------------

    buy_count = sum(
        1
        for result in results
        if result["Signal"].upper() == "BUY"
    )

    sell_count = sum(
        1
        for result in results
        if result["Signal"].upper() == "SELL"
    )

    no_signal_count = sum(
        1
        for result in results
        if result["Signal"].upper() == "NO SIGNAL"
    )

    print(f"BUY signals      : {buy_count}")
    print(f"SELL signals     : {sell_count}")
    print(f"NO SIGNAL        : {no_signal_count}")
    print(f"Stocks checked   : {len(results)}")

    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = check_all_stocks()

    print_results(results)