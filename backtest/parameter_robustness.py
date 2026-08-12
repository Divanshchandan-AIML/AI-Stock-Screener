# ============================================================
# STAGE 5 - SMMA PARAMETER ROBUSTNESS TEST
# ============================================================
#
# Purpose:
# Test multiple SMMA FAST/SLOW combinations and compare
# crossover performance.
#
# Current project:
# AI Stock Screener
#
# ============================================================

import pandas as pd

from api.historical_data import get_historical_data
from indicators.smma import calculate_smma


# ============================================================
# SETTINGS
# ============================================================

INITIAL_CAPITAL = 100000

# Holding period selected from Stage 4
HOLDING_PERIOD = 60

# Historical data
HISTORICAL_DAYS = 500

# Stocks used in previous stages
STOCKS = [
    "SBIN-EQ",
    "SUZLON-EQ",
    "IRFC-EQ",
    "TATAMOTORS-EQ",
    "RELIANCE-EQ",
    "ITC-EQ",
]


# ============================================================
# SMMA COMBINATIONS TO TEST
# ============================================================

SMMA_COMBINATIONS = [
    (10, 50),
    (10, 100),
    (20, 100),
    (20, 120),
    (20, 150),
    (30, 120),
    (30, 150),
    (50, 200),
]


# ============================================================
# NORMALIZE HISTORICAL DATA
# ============================================================

def normalize_data(df):
    """
    Convert historical data into standard columns:

        Date
        Close
    """

    if df is None or df.empty:
        return None

    df = df.copy()

    # --------------------------------------------------------
    # Find Date column
    # --------------------------------------------------------

    if "Date" not in df.columns:

        if "date" in df.columns:
            df["Date"] = df["date"]

        elif "datetime" in df.columns:
            df["Date"] = df["datetime"]

        elif "Datetime" in df.columns:
            df["Date"] = df["Datetime"]

        else:
            print("Date column not found.")
            return None

    # --------------------------------------------------------
    # Find Close column
    # --------------------------------------------------------

    if "Close" not in df.columns:

        if "close" in df.columns:
            df["Close"] = df["close"]

        elif "CLOSE" in df.columns:
            df["Close"] = df["CLOSE"]

        else:
            print("Close column not found.")
            return None

    # --------------------------------------------------------
    # Convert types
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "Date",
            "Close"
        ]
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# PREPARE DATA FOR ONE SMMA COMBINATION
# ============================================================

def prepare_parameter_data(
    df,
    fast_period,
    slow_period
):
    """
    Calculate SMMA fast/slow values and detect crossovers.
    """

    if df is None or df.empty:
        return None

    data = normalize_data(df)

    if data is None or data.empty:
        return None

    # --------------------------------------------------------
    # Calculate SMMA
    # --------------------------------------------------------

    data["SMMA_FAST"] = calculate_smma(
        data["Close"],
        fast_period
    )

    data["SMMA_SLOW"] = calculate_smma(
        data["Close"],
        slow_period
    )

    # --------------------------------------------------------
    # Previous values
    # --------------------------------------------------------

    data["PREV_FAST"] = (
        data["SMMA_FAST"].shift(1)
    )

    data["PREV_SLOW"] = (
        data["SMMA_SLOW"].shift(1)
    )

    # --------------------------------------------------------
    # BUY CROSSOVER
    #
    # Fast SMMA moves from below/equal
    # to above slow SMMA.
    # --------------------------------------------------------

    data["BUY_CROSSOVER"] = (
        (data["PREV_FAST"] <= data["PREV_SLOW"])
        &
        (data["SMMA_FAST"] > data["SMMA_SLOW"])
    )

    # --------------------------------------------------------
    # SELL CROSSOVER
    #
    # Fast SMMA moves from above/equal
    # to below slow SMMA.
    # --------------------------------------------------------

    data["SELL_CROSSOVER"] = (
        (data["PREV_FAST"] >= data["PREV_SLOW"])
        &
        (data["SMMA_FAST"] < data["SMMA_SLOW"])
    )

    # --------------------------------------------------------
    # Remove rows where indicators unavailable
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            "SMMA_FAST",
            "SMMA_SLOW"
        ]
    ).reset_index(drop=True)

    return data


# ============================================================
# ANALYZE ONE STOCK
# ============================================================

def analyze_stock(
    symbol,
    fast_period,
    slow_period,
    holding_period
):
    """
    Test one stock using one SMMA combination.
    """

    print(
        f"Testing {symbol} | "
        f"SMMA {fast_period}/{slow_period}"
    )

    try:

        # ----------------------------------------------------
        # Load historical data
        # ----------------------------------------------------

        df = get_historical_data(
            symbol,
            days=HISTORICAL_DAYS
        )

        if df is None or df.empty:

            print(
                f"{symbol}: No historical data"
            )

            return []

        # ----------------------------------------------------
        # Prepare indicators
        # ----------------------------------------------------

        df = prepare_parameter_data(
            df,
            fast_period,
            slow_period
        )

        if df is None or df.empty:
            return []

        results = []

        # ----------------------------------------------------
        # Find crossovers
        # ----------------------------------------------------

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

            if signal is None:
                continue

            # ------------------------------------------------
            # Make sure enough future data exists
            # ------------------------------------------------

            exit_index = (
                i + holding_period
            )

            if exit_index >= len(df):
                continue

            # ------------------------------------------------
            # Entry
            # ------------------------------------------------

            entry_price = float(
                df["Close"].iloc[i]
            )

            entry_date = df["Date"].iloc[i]

            # ------------------------------------------------
            # Exit
            # ------------------------------------------------

            exit_price = float(
                df["Close"].iloc[exit_index]
            )

            exit_date = (
                df["Date"].iloc[exit_index]
            )

            # ------------------------------------------------
            # Calculate return
            # ------------------------------------------------

            if entry_price <= 0:
                continue

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

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            if return_percent > 0:

                result = "PROFITABLE"

            elif return_percent < 0:

                result = "FAILED"

            else:

                result = "BREAKEVEN"

            results.append({

                "Stock": symbol.replace(
                    "-EQ",
                    ""
                ),

                "Fast": fast_period,

                "Slow": slow_period,

                "Signal": signal,

                "Entry Date": entry_date,

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

                "Holding": holding_period
            })

        return results

    except Exception as e:

        print(
            f"{symbol}: ERROR - {e}"
        )

        return []


# ============================================================
# ANALYZE ONE SMMA COMBINATION
# ============================================================

def analyze_combination(
    fast_period,
    slow_period
):
    """
    Analyze all stocks for one SMMA combination.
    """

    all_trades = []

    print("\n")
    print("=" * 80)

    print(
        f"SMMA {fast_period}/{slow_period}"
    )

    print("=" * 80)

    for symbol in STOCKS:

        trades = analyze_stock(
            symbol=symbol,
            fast_period=fast_period,
            slow_period=slow_period,
            holding_period=HOLDING_PERIOD
        )

        all_trades.extend(trades)

    return all_trades


# ============================================================
# CALCULATE COMBINATION STATISTICS
# ============================================================

def calculate_statistics(
    trades,
    fast_period,
    slow_period
):
    """
    Calculate statistics for one SMMA combination.
    """

    total_trades = len(trades)

    profitable = sum(
        1
        for trade in trades
        if trade["Result"] == "PROFITABLE"
    )

    failed = sum(
        1
        for trade in trades
        if trade["Result"] == "FAILED"
    )

    breakeven = sum(
        1
        for trade in trades
        if trade["Result"] == "BREAKEVEN"
    )

    # --------------------------------------------------------
    # Win rate
    # --------------------------------------------------------

    if total_trades > 0:

        win_rate = (
            profitable
            / total_trades
        ) * 100

    else:

        win_rate = 0.0

    # --------------------------------------------------------
    # Average return
    # --------------------------------------------------------

    if total_trades > 0:

        average_return = (
            sum(
                trade["Return %"]
                for trade in trades
            )
            / total_trades
        )

    else:

        average_return = 0.0

    # --------------------------------------------------------
    # Total return
    # --------------------------------------------------------

    total_return = sum(
        trade["Return %"]
        for trade in trades
    )

    # --------------------------------------------------------
    # Best trade
    # --------------------------------------------------------

    if trades:

        best_trade = max(
            trades,
            key=lambda x: x["Return %"]
        )

        worst_trade = min(
            trades,
            key=lambda x: x["Return %"]
        )

        best_return = (
            best_trade["Return %"]
        )

        worst_return = (
            worst_trade["Return %"]
        )

    else:

        best_return = 0.0
        worst_return = 0.0

    return {

        "Fast": fast_period,

        "Slow": slow_period,

        "Trades": total_trades,

        "Wins": profitable,

        "Losses": failed,

        "Breakeven": breakeven,

        "Win Rate %": round(
            win_rate,
            2
        ),

        "Avg Return %": round(
            average_return,
            2
        ),

        "Total Return %": round(
            total_return,
            2
        ),

        "Best Trade %": round(
            best_return,
            2
        ),

        "Worst Trade %": round(
            worst_return,
            2
        )
    }


# ============================================================
# RUN PARAMETER ROBUSTNESS TEST
# ============================================================

def run_parameter_robustness():

    statistics = []

    all_trade_results = []

    print("\n")
    print("=" * 80)
    print("        STAGE 5 - SMMA PARAMETER ROBUSTNESS")
    print("=" * 80)

    print(
        f"Stocks tested      : {len(STOCKS)}"
    )

    print(
        f"Historical days    : {HISTORICAL_DAYS}"
    )

    print(
        f"Holding period     : {HOLDING_PERIOD} trading days"
    )

    print(
        f"SMMA combinations  : {len(SMMA_COMBINATIONS)}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Test every combination
    # --------------------------------------------------------

    for fast_period, slow_period in SMMA_COMBINATIONS:

        trades = analyze_combination(
            fast_period,
            slow_period
        )

        all_trade_results.extend(
            trades
        )

        stats = calculate_statistics(
            trades,
            fast_period,
            slow_period
        )

        statistics.append(
            stats
        )

        print(
            f"\nSMMA {fast_period}/{slow_period}"
        )

        print(
            f"Trades       : {stats['Trades']}"
        )

        print(
            f"Wins         : {stats['Wins']}"
        )

        print(
            f"Losses       : {stats['Losses']}"
        )

        print(
            f"Win Rate     : {stats['Win Rate %']:.2f}%"
        )

        print(
            f"Avg Return   : {stats['Avg Return %']:.2f}%"
        )

    return statistics, all_trade_results


# ============================================================
# PRINT FINAL COMPARISON
# ============================================================

def print_parameter_comparison(
    statistics
):

    print("\n")
    print("=" * 100)
    print("                 SMMA PARAMETER COMPARISON")
    print("=" * 100)

    print(
        f"{'SMMA':<12}"
        f"{'Trades':>10}"
        f"{'Wins':>10}"
        f"{'Losses':>10}"
        f"{'Win Rate %':>14}"
        f"{'Avg Return %':>16}"
        f"{'Total Return %':>18}"
    )

    print("-" * 100)

    for result in statistics:

        smma_name = (
            f"{result['Fast']}/{result['Slow']}"
        )

        print(
            f"{smma_name:<12}"
            f"{result['Trades']:>10}"
            f"{result['Wins']:>10}"
            f"{result['Losses']:>10}"
            f"{result['Win Rate %']:>14.2f}"
            f"{result['Avg Return %']:>16.2f}"
            f"{result['Total Return %']:>18.2f}"
        )

    print("-" * 100)


# ============================================================
# FIND BEST PARAMETER
# ============================================================

def print_best_parameters(
    statistics
):

    if not statistics:
        return

    # --------------------------------------------------------
    # Only consider combinations with at least 3 trades
    # --------------------------------------------------------

    valid_results = [
        result
        for result in statistics
        if result["Trades"] >= 3
    ]

    if not valid_results:

        print("\n")
        print(
            "Not enough trades to select a robust parameter."
        )

        return

    # --------------------------------------------------------
    # Best by average return
    # --------------------------------------------------------

    best_return = max(
        valid_results,
        key=lambda x: x["Avg Return %"]
    )

    # --------------------------------------------------------
    # Best by win rate
    # --------------------------------------------------------

    best_win_rate = max(
        valid_results,
        key=lambda x: x["Win Rate %"]
    )

    # --------------------------------------------------------
    # Best balanced result
    #
    # Score:
    # Win Rate + Avg Return
    #
    # This avoids selecting a parameter solely because
    # it has a high return with very few winning trades.
    # --------------------------------------------------------

    for result in valid_results:

        result["Score"] = (
            result["Win Rate %"]
            +
            (
                result["Avg Return %"]
                * 10
            )
        )

    best_balanced = max(
        valid_results,
        key=lambda x: x["Score"]
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("                 BEST PARAMETERS")
    print("=" * 80)

    print("\nBEST BY AVERAGE RETURN")
    print("-" * 40)

    print(
        f"SMMA         : "
        f"{best_return['Fast']}/{best_return['Slow']}"
    )

    print(
        f"Trades       : "
        f"{best_return['Trades']}"
    )

    print(
        f"Win Rate     : "
        f"{best_return['Win Rate %']:.2f}%"
    )

    print(
        f"Avg Return   : "
        f"{best_return['Avg Return %']:.2f}%"
    )

    print("\nBEST BY WIN RATE")
    print("-" * 40)

    print(
        f"SMMA         : "
        f"{best_win_rate['Fast']}/{best_win_rate['Slow']}"
    )

    print(
        f"Trades       : "
        f"{best_win_rate['Trades']}"
    )

    print(
        f"Win Rate     : "
        f"{best_win_rate['Win Rate %']:.2f}%"
    )

    print(
        f"Avg Return   : "
        f"{best_win_rate['Avg Return %']:.2f}%"
    )

    print("\nBEST BALANCED PARAMETER")
    print("-" * 40)

    print(
        f"SMMA         : "
        f"{best_balanced['Fast']}/{best_balanced['Slow']}"
    )

    print(
        f"Trades       : "
        f"{best_balanced['Trades']}"
    )

    print(
        f"Win Rate     : "
        f"{best_balanced['Win Rate %']:.2f}%"
    )

    print(
        f"Avg Return   : "
        f"{best_balanced['Avg Return %']:.2f}%"
    )

    print(
        f"Score        : "
        f"{best_balanced['Score']:.2f}"
    )


# ============================================================
# PRINT INDIVIDUAL TRADES
# ============================================================

def print_trade_details(
    all_trade_results
):

    print("\n")
    print("=" * 100)
    print("                    CROSSOVER TRADE DETAILS")
    print("=" * 100)

    if not all_trade_results:

        print(
            "No crossover trades found."
        )

        return

    print(
        f"{'Stock':<12}"
        f"{'SMMA':<10}"
        f"{'Signal':<10}"
        f"{'Entry':>12}"
        f"{'Exit':>12}"
        f"{'Return %':>12}"
        f"{'Result':>16}"
    )

    print("-" * 100)

    for trade in all_trade_results:

        smma_name = (
            f"{trade['Fast']}/{trade['Slow']}"
        )

        print(
            f"{trade['Stock']:<12}"
            f"{smma_name:<10}"
            f"{trade['Signal']:<10}"
            f"₹{trade['Entry Price']:>10.2f}"
            f"₹{trade['Exit Price']:>10.2f}"
            f"{trade['Return %']:>11.2f}"
            f"{trade['Result']:>16}"
        )

    print("-" * 100)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    statistics, all_trade_results = (
        run_parameter_robustness()
    )

    print_parameter_comparison(
        statistics
    )

    print_best_parameters(
        statistics
    )

    print_trade_details(
        all_trade_results
    )

    print("\n")
    print("=" * 80)
    print("              STAGE 5 COMPLETE")
    print("=" * 80)