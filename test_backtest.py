import pandas as pd

from api.historical_data import get_historical_data
from indicators.smma import add_smma_indicators
from indicators.crossover import get_crossover_signals


SYMBOL = "SBIN-EQ"


def run_test():
    print("=" * 70)
    print(f"BACKTEST TEST: {SYMBOL}")
    print("=" * 70)

    # 1. Get historical data
    print("\nLoading historical data...")

    df = get_historical_data(
        SYMBOL,
        days=500
    )

    if df is None or df.empty:
        print("ERROR: No historical data")
        return

    print(f"Historical rows: {len(df)}")

    # 2. Add SMMA
    df = add_smma_indicators(df)

    print("SMMA indicators calculated.")

    # 3. Add crossover signals
    df = get_crossover_signals(df)

    print("Crossover signals calculated.")

    # 4. Remove rows where indicators are unavailable
    df = df.dropna(
        subset=["SMMA20", "SMMA120"]
    ).copy()

    # 5. Find signals
    signals = df[
        df["Signal"].isin(["BUY", "SELL"])
    ]

    print("\n" + "=" * 70)
    print("BACKTEST DATA")
    print("=" * 70)

    print(f"Rows available for backtest: {len(df)}")
    print(f"BUY signals: {(df['Signal'] == 'BUY').sum()}")
    print(f"SELL signals: {(df['Signal'] == 'SELL').sum()}")

    # 6. Show signals
    print("\n" + "=" * 70)
    print("SIGNALS")
    print("=" * 70)

    if signals.empty:
        print("No BUY/SELL signals found.")

    else:
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

    # 7. Latest data
    latest = df.iloc[-1]

    print("\n" + "=" * 70)
    print("LATEST")
    print("=" * 70)

    print(f"Date    : {latest['date']}")
    print(f"Price   : ₹{latest['close']:.2f}")
    print(f"SMMA20  : {latest['SMMA20']:.2f}")
    print(f"SMMA120 : {latest['SMMA120']:.2f}")
    print(f"Signal  : {latest['Signal']}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_test()