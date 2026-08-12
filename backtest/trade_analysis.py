import pandas as pd

from api.historical_data import get_historical_data
from indicators.crossover import analyze_crossovers


# ============================================================
# SETTINGS
# ============================================================

HOLDING_PERIODS = [
    5,
    10,
    20,
    40,
    60
]

STOCKS = [
    "SBIN-EQ",
    "SUZLON-EQ",
    "IRFC-EQ",
    "TATAMOTORS-EQ",
    "RELIANCE-EQ",
    "ITC-EQ",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_stock_data(symbol):

    print()
    print("=" * 70)
    print(f"LOADING {symbol}")
    print("=" * 70)

    try:

        df = get_historical_data(
            symbol,
            days=500
        )

    except Exception as e:

        print(
            f"{symbol}: ERROR - {e}"
        )

        return None

    if df is None or df.empty:

        print(
            f"{symbol}: No historical data"
        )

        return None

    print(
        f"{symbol}: {len(df)} historical rows"
    )

    return df


# ============================================================
# ANALYZE ONE STOCK
# ============================================================

def analyze_stock(
    symbol,
    holding_period
):

    df = load_stock_data(symbol)

    if df is None:

        return []

    try:

        results = analyze_crossovers(
            df,
            holding_period=holding_period
        )

    except Exception as e:

        print(
            f"{symbol}: crossover error - {e}"
        )

        return []

    return results


# ============================================================
# RUN ONE HOLDING PERIOD
# ============================================================

def run_holding_period(
    stock_data,
    holding_period
):

    all_trades = []

    print()
    print("=" * 90)
    print(
        f"HOLDING PERIOD: {holding_period} TRADING DAYS"
    )
    print("=" * 90)

    for symbol, df in stock_data.items():

        if df is None or df.empty:

            continue

        try:

            trades = analyze_crossovers(
                df,
                holding_period=holding_period
            )

        except Exception as e:

            print(
                f"{symbol}: ERROR - {e}"
            )

            continue

        for trade in trades:

            trade_copy = trade.copy()

            trade_copy["Stock"] = (
                symbol.replace(
                    "-EQ",
                    ""
                )
            )

            all_trades.append(
                trade_copy
            )

    # ========================================================
    # STATISTICS
    # ========================================================

    total = len(
        all_trades
    )

    profitable = sum(
        1
        for trade in all_trades
        if trade["Result"] == "PROFITABLE"
    )

    failed = sum(
        1
        for trade in all_trades
        if trade["Result"] == "FAILED"
    )

    breakeven = sum(
        1
        for trade in all_trades
        if trade["Result"] == "BREAKEVEN"
    )

    if total > 0:

        win_rate = (
            profitable
            / total
        ) * 100

        average_return = (
            sum(
                trade["Return %"]
                for trade in all_trades
            )
            / total
        )

    else:

        win_rate = 0.0
        average_return = 0.0

    # ========================================================
    # PRINT INDIVIDUAL TRADES
    # ========================================================

    if all_trades:

        print()
        print(
            "CROSSOVER TRADES"
        )

        print(
            "-" * 110
        )

        print(
            f"{'Stock':<12}"
            f"{'Signal':<10}"
            f"{'Entry':>12}"
            f"{'Exit':>12}"
            f"{'Return %':>12}"
            f"{'Result':>16}"
        )

        print(
            "-" * 110
        )

        for trade in all_trades:

            print(
                f"{trade['Stock']:<12}"
                f"{trade['Signal']:<10}"
                f"{trade['Entry Price']:>12.2f}"
                f"{trade['Exit Price']:>12.2f}"
                f"{trade['Return %']:>11.2f}%"
                f"{trade['Result']:>16}"
            )

        print(
            "-" * 110
        )

    else:

        print(
            "No usable crossover trades."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "PERIOD SUMMARY"
    )

    print(
        "-" * 50
    )

    print(
        f"Holding Period : {holding_period} days"
    )

    print(
        f"Total Trades   : {total}"
    )

    print(
        f"Profitable     : {profitable}"
    )

    print(
        f"Failed         : {failed}"
    )

    print(
        f"Breakeven      : {breakeven}"
    )

    print(
        f"Win Rate       : {win_rate:.2f}%"
    )

    print(
        f"Average Return : {average_return:.2f}%"
    )

    print(
        "-" * 50
    )

    return {
        "Holding Period": holding_period,
        "Trades": total,
        "Profitable": profitable,
        "Failed": failed,
        "Breakeven": breakeven,
        "Win Rate %": win_rate,
        "Average Return %": average_return,
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_analysis():

    # --------------------------------------------------------
    # Load data only once
    # --------------------------------------------------------

    stock_data = {}

    print()
    print("=" * 90)
    print(
        "          SMMA CROSSOVER ROBUSTNESS TEST"
    )
    print("=" * 90)

    for symbol in STOCKS:

        df = load_stock_data(
            symbol
        )

        stock_data[symbol] = df

    # --------------------------------------------------------
    # Test all holding periods
    # --------------------------------------------------------

    results = []

    for holding_period in HOLDING_PERIODS:

        result = run_holding_period(
            stock_data,
            holding_period
        )

        results.append(
            result
        )

    return results


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(results):

    print()
    print("=" * 100)
    print(
        "                 HOLDING PERIOD COMPARISON"
    )
    print("=" * 100)

    print(
        f"{'Period':>10}"
        f"{'Trades':>12}"
        f"{'Wins':>12}"
        f"{'Losses':>12}"
        f"{'Win Rate %':>15}"
        f"{'Avg Return %':>17}"
    )

    print(
        "-" * 100
    )

    for result in results:

        print(
            f"{result['Holding Period']:>10}"
            f"{result['Trades']:>12}"
            f"{result['Profitable']:>12}"
            f"{result['Failed']:>12}"
            f"{result['Win Rate %']:>14.2f}"
            f"{result['Average Return %']:>16.2f}"
        )

    print(
        "-" * 100
    )

    # ========================================================
    # BEST PERIOD
    # ========================================================

    valid_results = [
        result
        for result in results
        if result["Trades"] > 0
    ]

    if valid_results:

        best = max(
            valid_results,
            key=lambda x: x["Average Return %"]
        )

        print()
        print(
            "BEST HOLDING PERIOD"
        )

        print(
            "-" * 50
        )

        print(
            f"Period         : "
            f"{best['Holding Period']} days"
        )

        print(
            f"Trades         : "
            f"{best['Trades']}"
        )

        print(
            f"Profitable     : "
            f"{best['Profitable']}"
        )

        print(
            f"Failed         : "
            f"{best['Failed']}"
        )

        print(
            f"Win Rate       : "
            f"{best['Win Rate %']:.2f}%"
        )

        print(
            f"Average Return : "
            f"{best['Average Return %']:.2f}%"
        )

        print(
            "-" * 50
        )

    # ========================================================
    # COMPLETION
    # ========================================================

    print()
    print("=" * 90)
    print(
        "             ROBUSTNESS TEST COMPLETE"
    )
    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = run_analysis()

    print_final_summary(
        results
    )