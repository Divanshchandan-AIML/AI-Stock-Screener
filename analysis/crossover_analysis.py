import pandas as pd

from indicators.smma import calculate_smma


def prepare_crossover_data(df):
    """
    Prepare historical data and detect SMMA20/SMMA120 crossovers.
    """

    if df is None or df.empty:
        return None

    df = df.copy()

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------

    required_columns = ["Date", "Close"]

    for column in required_columns:
        if column not in df.columns:
            return None

    # ---------------------------------------------------------
    # Clean data
    # ---------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date", "Close"]
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Calculate SMMA
    # ---------------------------------------------------------

    df["SMMA20"] = calculate_smma(
        df["Close"],
        20
    )

    df["SMMA120"] = calculate_smma(
        df["Close"],
        120
    )

    # ---------------------------------------------------------
    # Previous SMMA values
    # ---------------------------------------------------------

    df["Previous_SMMA20"] = (
        df["SMMA20"].shift(1)
    )

    df["Previous_SMMA120"] = (
        df["SMMA120"].shift(1)
    )

    # ---------------------------------------------------------
    # BUY CROSSOVER
    # ---------------------------------------------------------

    df["BUY_CROSSOVER"] = (
        (df["Previous_SMMA20"] <= df["Previous_SMMA120"])
        &
        (df["SMMA20"] > df["SMMA120"])
    )

    # ---------------------------------------------------------
    # SELL CROSSOVER
    # ---------------------------------------------------------

    df["SELL_CROSSOVER"] = (
        (df["Previous_SMMA20"] >= df["Previous_SMMA120"])
        &
        (df["SMMA20"] < df["SMMA120"])
    )

    return df


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

    # ---------------------------------------------------------
    # Make sure enough future data exists
    # ---------------------------------------------------------

    exit_index = crossover_index + holding_period

    if exit_index >= len(df):

        return None

    # ---------------------------------------------------------
    # Entry
    # ---------------------------------------------------------

    entry_price = float(
        df["Close"].iloc[crossover_index]
    )

    entry_date = df["Date"].iloc[
        crossover_index
    ]

    # ---------------------------------------------------------
    # Exit
    # ---------------------------------------------------------

    exit_price = float(
        df["Close"].iloc[exit_index]
    )

    exit_date = df["Date"].iloc[
        exit_index
    ]

    # ---------------------------------------------------------
    # Calculate return
    # ---------------------------------------------------------

    if signal == "BUY":

        return_percent = (
            (exit_price - entry_price)
            / entry_price
        ) * 100

    elif signal == "SELL":

        return_percent = (
            (entry_price - exit_price)
            / entry_price
        ) * 100

    else:

        return None

    # ---------------------------------------------------------
    # PROFIT / LOSS
    # ---------------------------------------------------------

    if return_percent > 0:

        result = "PROFITABLE"

    elif return_percent < 0:

        result = "FAILED"

    else:

        result = "BREAKEVEN"

    # ---------------------------------------------------------
    # Future prices
    # ---------------------------------------------------------

    future_prices = (
        df["Close"]
        .iloc[
            crossover_index + 1:
            exit_index + 1
        ]
    )

    # ---------------------------------------------------------
    # Maximum Favorable Excursion
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Maximum Adverse Excursion
    # ---------------------------------------------------------

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


def analyze_crossovers(
    df,
    holding_period=20
):
    """
    Analyze every BUY and SELL crossover.

    Returns a list containing the profitability
    of each crossover.
    """

    df = prepare_crossover_data(df)

    if df is None or df.empty:
        return []

    results = []

    # ---------------------------------------------------------
    # Find every crossover
    # ---------------------------------------------------------

    for i in range(len(df)):

        signal = None

        if bool(df["BUY_CROSSOVER"].iloc[i]):

            signal = "BUY"

        elif bool(df["SELL_CROSSOVER"].iloc[i]):

            signal = "SELL"

        # No crossover
        if signal is None:
            continue

        # -----------------------------------------------------
        # Evaluate trade
        # -----------------------------------------------------

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

    average_return = sum(
        result["Return %"]
        for result in crossover_results
    ) / total

    win_rate = (
        profitable / total
    ) * 100

    return {
        "Total Crossovers": total,
        "Profitable": profitable,
        "Failed": failed,
        "Win Rate %": round(
            win_rate,
            2
        ),
        "Average Return %": round(
            average_return,
            2
        )
    }