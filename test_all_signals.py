from api.historical_data import get_historical_data
from indicators.smma import add_smma_indicators
from indicators.crossover import get_crossover_signals


STOCKS = [
    "SBIN-EQ",
    "SUZLON-EQ",
    "IRFC-EQ",
    "TATAMOTORS-EQ",
    "RELIANCE-EQ",
    "ITC-EQ",
]


def test_stock(symbol):

    print("\n" + "=" * 70)
    print(f"TESTING {symbol}")
    print("=" * 70)

    try:

        # --------------------------------------------------
        # 1. Historical data
        # --------------------------------------------------

        df = get_historical_data(
            symbol,
            days=500
        )

        if df is None or df.empty:
            print("No historical data")
            return None

        print(f"Historical rows: {len(df)}")

        # --------------------------------------------------
        # 2. SMMA
        # --------------------------------------------------

        df = add_smma_indicators(df)

        # --------------------------------------------------
        # 3. Crossover
        # --------------------------------------------------

        df = get_crossover_signals(df)

        # --------------------------------------------------
        # 4. Remove invalid SMMA rows
        # --------------------------------------------------

        df = df.dropna(
            subset=[
                "SMMA20",
                "SMMA120"
            ]
        ).copy()

        # --------------------------------------------------
        # 5. Count signals
        # --------------------------------------------------

        buy_count = (
            df["Signal"] == "BUY"
        ).sum()

        sell_count = (
            df["Signal"] == "SELL"
        ).sum()

        # --------------------------------------------------
        # 6. Latest row
        # --------------------------------------------------

        latest = df.iloc[-1]

        print(f"Usable rows : {len(df)}")
        print(f"BUY signals : {buy_count}")
        print(f"SELL signals: {sell_count}")

        print("\nLatest:")
        print(
            f"Price={latest['close']:.2f} | "
            f"SMMA20={latest['SMMA20']:.2f} | "
            f"SMMA120={latest['SMMA120']:.2f} | "
            f"Signal={latest['Signal']}"
        )

        # --------------------------------------------------
        # 7. Show actual signals
        # --------------------------------------------------

        signals = df[
            df["Signal"].isin(
                ["BUY", "SELL"]
            )
        ]

        if not signals.empty:

            print("\nCROSSOVERS FOUND:")

            print(
                signals[
                    [
                        "date",
                        "close",
                        "SMMA20",
                        "SMMA120",
                        "Signal"
                    ]
                ].to_string(index=False)
            )

        else:

            print("\nNo crossover found.")

        return {
            "symbol": symbol,
            "rows": len(df),
            "buy": int(buy_count),
            "sell": int(sell_count),
            "latest_signal": latest["Signal"]
        }

    except Exception as e:

        print(
            f"ERROR for {symbol}: {e}"
        )

        return None


def main():

    results = []

    print("\n")
    print("=" * 70)
    print("       ALL STOCK CROSSOVER TEST")
    print("=" * 70)

    for symbol in STOCKS:

        result = test_stock(symbol)

        if result is not None:
            results.append(result)

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("                  FINAL SUMMARY")
    print("=" * 70)

    print(
        f"{'STOCK':<15}"
        f"{'ROWS':<10}"
        f"{'BUY':<10}"
        f"{'SELL':<10}"
        f"{'LATEST SIGNAL'}"
    )

    print("-" * 70)

    total_buy = 0
    total_sell = 0

    for result in results:

        total_buy += result["buy"]
        total_sell += result["sell"]

        print(
            f"{result['symbol']:<15}"
            f"{result['rows']:<10}"
            f"{result['buy']:<10}"
            f"{result['sell']:<10}"
            f"{result['latest_signal']}"
        )

    print("-" * 70)

    print(f"Total BUY signals : {total_buy}")
    print(f"Total SELL signals: {total_sell}")
    print(f"Stocks tested     : {len(results)}")

    print("=" * 70)


if __name__ == "__main__":
    main()