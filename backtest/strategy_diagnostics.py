import pandas as pd

from api.historical_data import get_historical_data
from indicators.smma import add_smma_indicators
from indicators.crossover import get_crossover_signals


# ============================================================
# STOCKS
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
# PREPARE DATA
# ============================================================

def prepare_data(symbol):

    print("\n")
    print("=" * 80)
    print(f"PREPARING {symbol}")
    print("=" * 80)

    # --------------------------------------------------------
    # GET HISTORICAL DATA
    # --------------------------------------------------------

    df = get_historical_data(
        symbol,
        days=500
    )

    if df is None or df.empty:

        print(f"{symbol}: No historical data")

        return None

    print(
        f"Historical rows before sorting: {len(df)}"
    )

    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "date",
        "close"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print(
            f"{symbol}: Missing columns: "
            f"{missing_columns}"
        )

        return None

    # --------------------------------------------------------
    # CONVERT DATE
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Remove invalid dates
    df = df.dropna(
        subset=["date"]
    ).copy()

    # --------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # IMPORTANT
    # --------------------------------------------------------

    df = df.sort_values(
        "date",
        ascending=True
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATE DATES
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["date"],
        keep="last"
    ).reset_index(
        drop=True
    )

    print(
        f"Rows after chronological sorting: {len(df)}"
    )

    # --------------------------------------------------------
    # ADD SMMA INDICATORS
    # --------------------------------------------------------

    df = add_smma_indicators(df)

    if df is None or df.empty:

        print(
            f"{symbol}: SMMA calculation failed"
        )

        return None

    print(
        "SMMA indicators added successfully."
    )

    # --------------------------------------------------------
    # GENERATE CROSSOVER SIGNALS
    # --------------------------------------------------------

    df = get_crossover_signals(df)

    if df is None or df.empty:

        print(
            f"{symbol}: Crossover calculation failed"
        )

        return None

    print(
        "Crossover detection completed."
    )

    # --------------------------------------------------------
    # REMOVE INVALID SMMA ROWS
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "SMMA20",
            "SMMA120",
            "close"
        ]
    ).copy()

    # --------------------------------------------------------
    # MAKE SURE SIGNAL COLUMN EXISTS
    # --------------------------------------------------------

    if "Signal" not in df.columns:

        print(
            f"{symbol}: Signal column missing."
        )

        df["Signal"] = "NO SIGNAL"

    # --------------------------------------------------------
    # CLEAN SIGNAL COLUMN
    # --------------------------------------------------------

    df["Signal"] = (
        df["Signal"]
        .fillna("NO SIGNAL")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # SORT AGAIN AFTER ALL CALCULATIONS
    # --------------------------------------------------------

    df = df.sort_values(
        "date",
        ascending=True
    ).reset_index(
        drop=True
    )

    print(
        f"Usable rows: {len(df)}"
    )

    return df


# ============================================================
# ANALYZE ONE STOCK
# ============================================================

def analyze_stock(symbol):

    df = prepare_data(symbol)

    if df is None or df.empty:

        return

    # ========================================================
    # SIGNAL COUNTS
    # ========================================================

    buy_count = (
        df["Signal"] == "BUY"
    ).sum()

    sell_count = (
        df["Signal"] == "SELL"
    ).sum()

    no_signal_count = (
        df["Signal"] == "NO SIGNAL"
    ).sum()

    print("\n")
    print("=" * 80)
    print("SIGNAL COUNTS")
    print("=" * 80)

    print(
        f"BUY       : {buy_count}"
    )

    print(
        f"SELL      : {sell_count}"
    )

    print(
        f"NO SIGNAL : {no_signal_count}"
    )

    # ========================================================
    # RAW SMMA CROSSOVER CHECK
    # ========================================================

    debug_df = df[
        [
            "date",
            "close",
            "SMMA20",
            "SMMA120",
            "Signal"
        ]
    ].copy()

    debug_df["difference"] = (
        debug_df["SMMA20"]
        - debug_df["SMMA120"]
    )

    # --------------------------------------------------------
    # Previous difference
    # --------------------------------------------------------

    previous_difference = (
        debug_df["difference"].shift(1)
    )

    current_difference = (
        debug_df["difference"]
    )

    # --------------------------------------------------------
    # BUY crossover
    #
    # Previous:
    # SMMA20 <= SMMA120
    #
    # Current:
    # SMMA20 > SMMA120
    # --------------------------------------------------------

    buy_crosses = (
        (previous_difference <= 0)
        &
        (current_difference > 0)
    )

    # --------------------------------------------------------
    # SELL crossover
    #
    # Previous:
    # SMMA20 >= SMMA120
    #
    # Current:
    # SMMA20 < SMMA120
    # --------------------------------------------------------

    sell_crosses = (
        (previous_difference >= 0)
        &
        (current_difference < 0)
    )

    raw_buy_count = buy_crosses.sum()

    raw_sell_count = sell_crosses.sum()

    # ========================================================
    # CROSSOVER EVENTS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("CROSSOVER EVENTS")
    print("=" * 80)

    print(
        f"Raw BUY crossovers  : {raw_buy_count}"
    )

    print(
        f"Raw SELL crossovers : {raw_sell_count}"
    )

    # --------------------------------------------------------
    # BUY EVENTS
    # --------------------------------------------------------

    if raw_buy_count > 0:

        print("\nBUY CROSSOVER ROWS")
        print("-" * 80)

        buy_rows = debug_df.loc[
            buy_crosses,
            [
                "date",
                "close",
                "SMMA20",
                "SMMA120",
                "difference"
            ]
        ]

        print(
            buy_rows.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # SELL EVENTS
    # --------------------------------------------------------

    if raw_sell_count > 0:

        print("\nSELL CROSSOVER ROWS")
        print("-" * 80)

        sell_rows = debug_df.loc[
            sell_crosses,
            [
                "date",
                "close",
                "SMMA20",
                "SMMA120",
                "difference"
            ]
        ]

        print(
            sell_rows.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Existing signal rows
    # --------------------------------------------------------

    signal_rows = debug_df[
        debug_df["Signal"].isin(
            [
                "BUY",
                "SELL"
            ]
        )
    ]

    if not signal_rows.empty:

        print("\n")
        print("=" * 80)
        print("GENERATED SIGNAL ROWS")
        print("=" * 80)

        print(
            signal_rows.to_string(
                index=False
            )
        )

    # ========================================================
    # LATEST DATA
    # ========================================================

    latest = df.iloc[-1]

    latest_price = float(
        latest["close"]
    )

    latest_smma20 = float(
        latest["SMMA20"]
    )

    latest_smma120 = float(
        latest["SMMA120"]
    )

    latest_difference = (
        latest_smma20
        - latest_smma120
    )

    latest_signal = str(
        latest["Signal"]
    )

    print("\n")
    print("=" * 80)
    print("LATEST DATA")
    print("=" * 80)

    print(
        f"Date       : {latest['date']}"
    )

    print(
        f"Close      : ₹{latest_price:.2f}"
    )

    print(
        f"SMMA20     : {latest_smma20:.2f}"
    )

    print(
        f"SMMA120    : {latest_smma120:.2f}"
    )

    print(
        f"Signal     : {latest_signal}"
    )

    print(
        f"Difference : {latest_difference:.2f}"
    )

    if latest_difference > 0:

        print(
            "SMMA20 is ABOVE SMMA120"
        )

    elif latest_difference < 0:

        print(
            "SMMA20 is BELOW SMMA120"
        )

    else:

        print(
            "SMMA20 is EQUAL TO SMMA120"
        )

    # ========================================================
    # LAST 10 ROWS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("LAST 10 DATA ROWS")
    print("=" * 80)

    print(
        debug_df.tail(10).to_string(
            index=False
        )
    )

    # ========================================================
    # DIAGNOSTIC RESULT
    # ========================================================

    print("\n")
    print("=" * 80)
    print(f"{symbol} DIAGNOSTIC COMPLETE")
    print("=" * 80)


# ============================================================
# RUN ALL STOCKS
# ============================================================

def run_diagnostics():

    print("\n")
    print("=" * 80)
    print("SMMA CROSSOVER STRATEGY DIAGNOSTICS")
    print("=" * 80)

    print(
        f"Stocks to test: {len(STOCKS)}"
    )

    total_buy = 0
    total_sell = 0

    successful_stocks = 0

    # --------------------------------------------------------
    # TEST EACH STOCK
    # --------------------------------------------------------

    for symbol in STOCKS:

        try:

            df = prepare_data(symbol)

            if df is None or df.empty:

                print(
                    f"{symbol}: Unable to analyze."
                )

                continue

            successful_stocks += 1

            # ------------------------------------------------
            # Signal counts
            # ------------------------------------------------

            buy_count = (
                df["Signal"] == "BUY"
            ).sum()

            sell_count = (
                df["Signal"] == "SELL"
            ).sum()

            total_buy += buy_count
            total_sell += sell_count

            # ------------------------------------------------
            # Raw crossover check
            # ------------------------------------------------

            difference = (
                df["SMMA20"]
                - df["SMMA120"]
            )

            previous_difference = (
                difference.shift(1)
            )

            buy_crosses = (
                (previous_difference <= 0)
                &
                (difference > 0)
            )

            sell_crosses = (
                (previous_difference >= 0)
                &
                (difference < 0)
            )

            raw_buy = buy_crosses.sum()
            raw_sell = sell_crosses.sum()

            print("\n")
            print("-" * 80)
            print(f"{symbol} SUMMARY")
            print("-" * 80)

            print(
                f"Rows        : {len(df)}"
            )

            print(
                f"BUY signals : {buy_count}"
            )

            print(
                f"SELL signals: {sell_count}"
            )

            print(
                f"Raw BUY     : {raw_buy}"
            )

            print(
                f"Raw SELL    : {raw_sell}"
            )

            # ------------------------------------------------
            # Latest values
            # ------------------------------------------------

            latest = df.iloc[-1]

            print(
                f"Latest Price: ₹{float(latest['close']):.2f}"
            )

            print(
                f"Latest SMMA20: "
                f"{float(latest['SMMA20']):.2f}"
            )

            print(
                f"Latest SMMA120: "
                f"{float(latest['SMMA120']):.2f}"
            )

            print(
                f"Latest Signal: "
                f"{latest['Signal']}"
            )

        except Exception as e:

            print("\n")
            print("-" * 80)
            print(f"{symbol}: ERROR")
            print("-" * 80)

            print(
                f"{type(e).__name__}: {e}"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("=" * 80)

    print(
        f"Stocks tested       : {len(STOCKS)}"
    )

    print(
        f"Stocks successful   : {successful_stocks}"
    )

    print(
        f"Total BUY signals   : {total_buy}"
    )

    print(
        f"Total SELL signals  : {total_sell}"
    )

    print("\n")

    if total_buy == 0 and total_sell == 0:

        print(
            "RESULT: No BUY/SELL signals detected."
        )

        print(
            "The next step is to inspect the "
            "crossover calculation."
        )

    else:

        print(
            "RESULT: Crossover signals detected."
        )

        print(
            "The strategy can now proceed to "
            "backtesting."
        )

    print("\n")
    print("=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_diagnostics()