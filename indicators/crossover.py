import pandas as pd

from indicators.smma import calculate_smma


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

def _normalize_columns(df):
    """
    Convert common historical-data column names into the
    standard names used by the project:

        date
        close
    """

    df = df.copy()

    column_map = {}

    for column in df.columns:

        lower_column = str(column).strip().lower()

        if lower_column == "date":
            column_map[column] = "date"

        elif lower_column == "close":
            column_map[column] = "close"

    df = df.rename(columns=column_map)

    return df


# ============================================================
# PREPARE CROSSOVER DATA
# ============================================================

def prepare_crossover_data(df):
    """
    Prepare historical data and calculate SMMA20/SMMA120.

    Detects:

        BUY:
            Previous SMMA20 <= Previous SMMA120
            AND
            Current SMMA20 > Current SMMA120

        SELL:
            Previous SMMA20 >= Previous SMMA120
            AND
            Current SMMA20 < Current SMMA120
    """

    if df is None or df.empty:
        return None

    df = _normalize_columns(df)

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "date",
        "close"
    ]

    for column in required_columns:

        if column not in df.columns:
            print(
                f"Crossover error: missing column '{column}'"
            )

            return None

    # --------------------------------------------------------
    # Clean date
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Clean close
    # --------------------------------------------------------

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "date",
            "close"
        ]
    ).copy()

    if df.empty:
        return None

    # --------------------------------------------------------
    # Remove duplicate dates
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset=["date"],
            keep="last"
        )
        .sort_values(
            "date",
            ascending=True
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # CALCULATE SMMA20
    # ========================================================

    df["SMMA20"] = calculate_smma(
        df["close"],
        20
    )

    # ========================================================
    # CALCULATE SMMA120
    # ========================================================

    df["SMMA120"] = calculate_smma(
        df["close"],
        120
    )

    # --------------------------------------------------------
    # Make sure SMMA values are numeric
    # --------------------------------------------------------

    df["SMMA20"] = pd.to_numeric(
        df["SMMA20"],
        errors="coerce"
    )

    df["SMMA120"] = pd.to_numeric(
        df["SMMA120"],
        errors="coerce"
    )

    # ========================================================
    # PREVIOUS SMMA VALUES
    # ========================================================

    df["Previous_SMMA20"] = (
        df["SMMA20"].shift(1)
    )

    df["Previous_SMMA120"] = (
        df["SMMA120"].shift(1)
    )

    # ========================================================
    # BUY CROSSOVER
    # ========================================================

    df["BUY_CROSSOVER"] = (
        (df["Previous_SMMA20"] <= df["Previous_SMMA120"])
        &
        (df["SMMA20"] > df["SMMA120"])
    )

    # ========================================================
    # SELL CROSSOVER
    # ========================================================

    df["SELL_CROSSOVER"] = (
        (df["Previous_SMMA20"] >= df["Previous_SMMA120"])
        &
        (df["SMMA20"] < df["SMMA120"])
    )

    # ========================================================
    # SIGNAL COLUMN
    # ========================================================

    df["Signal"] = "NO SIGNAL"

    df.loc[
        df["BUY_CROSSOVER"],
        "Signal"
    ] = "BUY"

    df.loc[
        df["SELL_CROSSOVER"],
        "Signal"
    ] = "SELL"

    # ========================================================
    # REMOVE INVALID SMMA ROWS
    # ========================================================

    df = df.dropna(
        subset=[
            "SMMA20",
            "SMMA120"
        ]
    ).copy()

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# GET CROSSOVER SIGNALS
# ============================================================

def get_crossover_signals(df):
    """
    Main function used by the backtest.

    Returns the original dataframe with:

        SMMA20
        SMMA120
        Previous_SMMA20
        Previous_SMMA120
        BUY_CROSSOVER
        SELL_CROSSOVER
        Signal
    """

    df = prepare_crossover_data(df)

    if df is None or df.empty:
        return df

    # --------------------------------------------------------
    # Diagnostic information
    # --------------------------------------------------------

    buy_count = int(
        (df["Signal"] == "BUY").sum()
    )

    sell_count = int(
        (df["Signal"] == "SELL").sum()
    )

    no_signal_count = int(
        (df["Signal"] == "NO SIGNAL").sum()
    )

    print(
        "Crossover detection completed."
    )

    print(
        f"BUY signals : {buy_count}"
    )

    print(
        f"SELL signals: {sell_count}"
    )

    print(
        f"NO SIGNAL   : {no_signal_count}"
    )

    return df


# ============================================================
# CALCULATE TRADE RESULT
# ============================================================

def calculate_trade_result(
    df,
    crossover_index,
    signal,
    holding_period=20
):
    """
    Evaluate what happened after a crossover.

    BUY:
        Profit if future price > entry price.

    SELL:
        Profit if future price < entry price.

    holding_period:
        Number of future trading bars used for evaluation.
    """

    if df is None or df.empty:
        return None

    if signal not in [
        "BUY",
        "SELL"
    ]:
        return None

    # --------------------------------------------------------
    # Check index
    # --------------------------------------------------------

    if crossover_index < 0:
        return None

    if crossover_index >= len(df):
        return None

    # --------------------------------------------------------
    # Calculate exit index
    # --------------------------------------------------------

    exit_index = (
        crossover_index
        + holding_period
    )

    # --------------------------------------------------------
    # Make sure enough future data exists
    # --------------------------------------------------------

    if exit_index >= len(df):
        return None

    # --------------------------------------------------------
    # Entry
    # --------------------------------------------------------

    entry_price = float(
        df["close"].iloc[
            crossover_index
        ]
    )

    entry_date = df["date"].iloc[
        crossover_index
    ]

    if entry_price <= 0:
        return None

    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    exit_price = float(
        df["close"].iloc[
            exit_index
        ]
    )

    exit_date = df["date"].iloc[
        exit_index
    ]

    # --------------------------------------------------------
    # Calculate return
    # --------------------------------------------------------

    if signal == "BUY":

        return_percent = (
            (
                exit_price
                - entry_price
            )
            / entry_price
        ) * 100

    else:

        return_percent = (
            (
                entry_price
                - exit_price
            )
            / entry_price
        ) * 100

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if return_percent > 0:

        result = "PROFITABLE"

    elif return_percent < 0:

        result = "FAILED"

    else:

        result = "BREAKEVEN"

    # --------------------------------------------------------
    # Future prices
    # --------------------------------------------------------

    future_prices = (
        df["close"]
        .iloc[
            crossover_index + 1:
            exit_index + 1
        ]
    )

    # --------------------------------------------------------
    # No future prices
    # --------------------------------------------------------

    if future_prices.empty:

        max_favorable = 0.0
        max_adverse = 0.0

    else:

        # ----------------------------------------------------
        # Maximum Favorable Excursion
        # ----------------------------------------------------

        if signal == "BUY":

            max_favorable = (
                (
                    future_prices.max()
                    - entry_price
                )
                / entry_price
            ) * 100

        else:

            max_favorable = (
                (
                    entry_price
                    - future_prices.min()
                )
                / entry_price
            ) * 100

        # ----------------------------------------------------
        # Maximum Adverse Excursion
        # ----------------------------------------------------

        if signal == "BUY":

            max_adverse = (
                (
                    future_prices.min()
                    - entry_price
                )
                / entry_price
            ) * 100

        else:

            max_adverse = (
                (
                    entry_price
                    - future_prices.max()
                )
                / entry_price
            ) * 100

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "Entry Date": entry_date,
        "Signal": signal,
        "Entry Price": round(
            entry_price,
            2
        ),
        "Exit Date": exit_date,
        "Exit Price": round(
            exit_price,
            2
        ),
        "Return %": round(
            return_percent,
            2
        ),
        "Result": result,
        "Max Favorable %": round(
            max_favorable,
            2
        ),
        "Max Adverse %": round(
            max_adverse,
            2
        ),
        "Holding Bars": holding_period
    }


# ============================================================
# ANALYZE CROSSOVERS
# ============================================================

def analyze_crossovers(
    df,
    holding_period=20
):
    """
    Analyze every BUY and SELL crossover.
    """

    df = prepare_crossover_data(df)

    if df is None or df.empty:
        return []

    results = []

    # --------------------------------------------------------
    # Find every crossover
    # --------------------------------------------------------

    for i in range(len(df)):

        signal = None

        if bool(
            df["BUY_CROSSOVER"].iloc[i]
        ):

            signal = "BUY"

        elif bool(
            df["SELL_CROSSOVER"].iloc[i]
        ):

            signal = "SELL"

        # ----------------------------------------------------
        # No crossover
        # ----------------------------------------------------

        if signal is None:
            continue

        # ----------------------------------------------------
        # Evaluate crossover
        # ----------------------------------------------------

        trade_result = calculate_trade_result(
            df=df,
            crossover_index=i,
            signal=signal,
            holding_period=holding_period
        )

        if trade_result is not None:

            results.append(
                trade_result
            )

    return results


# ============================================================
# CROSSOVER SUMMARY
# ============================================================

def get_crossover_summary(
    crossover_results
):
    """
    Calculate overall crossover statistics.
    """

    if not crossover_results:

        return {
            "Total Crossovers": 0,
            "Profitable": 0,
            "Failed": 0,
            "Breakeven": 0,
            "Win Rate %": 0.0,
            "Average Return %": 0.0
        }

    total = len(
        crossover_results
    )

    profitable = sum(
        1
        for result in crossover_results
        if result["Result"] == "PROFITABLE"
    )

    failed = sum(
        1
        for result in crossover_results
        if result["Result"] == "FAILED"
    )

    breakeven = sum(
        1
        for result in crossover_results
        if result["Result"] == "BREAKEVEN"
    )

    average_return = (
        sum(
            result["Return %"]
            for result in crossover_results
        )
        / total
    )

    win_rate = (
        profitable
        / total
    ) * 100

    return {
        "Total Crossovers": total,
        "Profitable": profitable,
        "Failed": failed,
        "Breakeven": breakeven,
        "Win Rate %": round(
            win_rate,
            2
        ),
        "Average Return %": round(
            average_return,
            2
        )
    }


# ============================================================
# PRINT CROSSOVER SUMMARY
# ============================================================

def print_crossover_summary(
    df,
    symbol=None
):
    """
    Print a simple diagnostic summary for one stock.
    """

    if df is None or df.empty:

        print(
            "No data available."
        )

        return

    df = get_crossover_signals(df)

    if df is None or df.empty:

        print(
            "No crossover data available."
        )

        return

    buy_signals = df[
        df["Signal"] == "BUY"
    ]

    sell_signals = df[
        df["Signal"] == "SELL"
    ]

    latest = df.iloc[-1]

    print("\n")
    print("=" * 70)

    if symbol:
        print(
            f"{symbol} CROSSOVER SUMMARY"
        )
    else:
        print(
            "CROSSOVER SUMMARY"
        )

    print("=" * 70)

    print(
        f"Rows          : {len(df)}"
    )

    print(
        f"BUY signals   : {len(buy_signals)}"
    )

    print(
        f"SELL signals  : {len(sell_signals)}"
    )

    print(
        f"Latest Price  : ₹{float(latest['close']):.2f}"
    )

    print(
        f"Latest SMMA20 : {float(latest['SMMA20']):.2f}"
    )

    print(
        f"Latest SMMA120: {float(latest['SMMA120']):.2f}"
    )

    print(
        f"Latest Signal : {latest['Signal']}"
    )

    print("=" * 70)