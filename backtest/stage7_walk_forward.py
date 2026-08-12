"""
STAGE 7
OUT-OF-SAMPLE / WALK-FORWARD VALIDATION

SMMA:
    Fast = 20
    Slow = 120

Purpose:
    Test the already-selected SMMA 20/120 strategy
    on data that was not used for parameter selection.

Important:
    No parameter optimization is performed here.

    The complete historical dataset is used to calculate
    the indicators, but ONLY crossover signals occurring
    inside the test period are evaluated.

    Trade exits are taken from the full chronological
    dataset so that a test-period signal can still have
    its required 60 future trading bars.
"""

import pandas as pd

from api.historical_data import get_historical_data
from indicators.smma import calculate_smma


# ============================================================
# CONFIGURATION
# ============================================================

STOCKS = [
    "SBIN-EQ",
    "SUZLON-EQ",
    "IRFC-EQ",
    "TATAMOTORS-EQ",
    "RELIANCE-EQ",
    "ITC-EQ",
]

# Selected during Stage 5
FAST_SMMA = 20
SLOW_SMMA = 120

# Historical data requested
HISTORICAL_DAYS = 1000

# Chronological train/test split
TRAIN_RATIO = 0.60

# Stage 6 holding period
HOLDING_PERIOD = 60

# Stage 6 slippage assumptions
ENTRY_SLIPPAGE_PERCENT = 0.10
EXIT_SLIPPAGE_PERCENT = 0.10

# Stage 6 transaction cost per side
TRANSACTION_COST_PERCENT = 0.05


# ============================================================
# NORMALIZE COLUMNS
# ============================================================

def normalize_columns(df):

    if df is None or df.empty:
        return None

    df = df.copy()

    rename_map = {}

    for column in df.columns:

        name = str(column).strip().lower()

        if name == "date":
            rename_map[column] = "Date"

        elif name == "datetime":
            rename_map[column] = "Date"

        elif name == "close":
            rename_map[column] = "Close"

    df = df.rename(columns=rename_map)

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(symbol):

    print()
    print("=" * 80)
    print(f"GETTING HISTORICAL DATA FOR {symbol}")
    print("=" * 80)

    try:

        df = get_historical_data(
            symbol,
            days=HISTORICAL_DAYS
        )

    except Exception as error:

        print(
            f"{symbol}: Error loading data: {error}"
        )

        return None

    if df is None or df.empty:

        print(
            f"{symbol}: No historical data"
        )

        return None

    print(
        f"{symbol}: Historical rows loaded: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Normalize columns
    # --------------------------------------------------------

    df = normalize_columns(df)

    if df is None:

        return None

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "Date",
        "Close"
    ]

    for column in required_columns:

        if column not in df.columns:

            print(
                f"{symbol}: Missing required column "
                f"'{column}'"
            )

            return None

    # --------------------------------------------------------
    # Clean Date
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Clean Close
    # --------------------------------------------------------

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Date",
            "Close"
        ]
    ).copy()

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Calculate SMMA
    # --------------------------------------------------------

    df["SMMA20"] = calculate_smma(
        df["Close"],
        FAST_SMMA
    )

    df["SMMA120"] = calculate_smma(
        df["Close"],
        SLOW_SMMA
    )

    # --------------------------------------------------------
    # Previous values
    # --------------------------------------------------------

    df["Previous_SMMA20"] = (
        df["SMMA20"].shift(1)
    )

    df["Previous_SMMA120"] = (
        df["SMMA120"].shift(1)
    )

    # --------------------------------------------------------
    # BUY crossover
    # --------------------------------------------------------

    df["BUY_CROSSOVER"] = (
        (df["Previous_SMMA20"] <= df["Previous_SMMA120"])
        &
        (df["SMMA20"] > df["SMMA120"])
    )

    # --------------------------------------------------------
    # SELL crossover
    # --------------------------------------------------------

    df["SELL_CROSSOVER"] = (
        (df["Previous_SMMA20"] >= df["Previous_SMMA120"])
        &
        (df["SMMA20"] < df["SMMA120"])
    )

    # --------------------------------------------------------
    # Remove rows where SMMA cannot be calculated
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "SMMA20",
            "SMMA120"
        ]
    ).reset_index(drop=True)

    print(
        f"{symbol}: Usable rows: "
        f"{len(df)}"
    )

    print(
        f"{symbol}: BUY crossovers: "
        f"{int(df['BUY_CROSSOVER'].sum())}"
    )

    print(
        f"{symbol}: SELL crossovers: "
        f"{int(df['SELL_CROSSOVER'].sum())}"
    )

    return df


# ============================================================
# SPLIT TRAIN / TEST
# ============================================================

def get_split_index(df):

    if df is None or df.empty:

        return None

    split_index = int(
        len(df) * TRAIN_RATIO
    )

    if split_index <= 0:
        return None

    if split_index >= len(df):
        return None

    return split_index


# ============================================================
# GET SIGNALS
# ============================================================

def get_signals(
    df,
    start_index,
    end_index
):

    signals = []

    if df is None or df.empty:

        return signals

    start_index = max(
        0,
        start_index
    )

    end_index = min(
        len(df),
        end_index
    )

    for i in range(
        start_index,
        end_index
    ):

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

        signals.append({
            "index": i,
            "signal": signal
        })

    return signals


# ============================================================
# CALCULATE TRADE
# ============================================================

def calculate_trade(
    df,
    signal_index,
    signal
):

    exit_index = (
        signal_index +
        HOLDING_PERIOD
    )

    # Need enough future bars
    if exit_index >= len(df):

        return None

    raw_entry_price = float(
        df["Close"].iloc[signal_index]
    )

    raw_exit_price = float(
        df["Close"].iloc[exit_index]
    )

    entry_date = df["Date"].iloc[
        signal_index
    ]

    exit_date = df["Date"].iloc[
        exit_index
    ]

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if signal == "BUY":

        entry_price = (
            raw_entry_price *
            (
                1 +
                ENTRY_SLIPPAGE_PERCENT / 100
            )
        )

        exit_price = (
            raw_exit_price *
            (
                1 -
                EXIT_SLIPPAGE_PERCENT / 100
            )
        )

        gross_return = (
            (
                exit_price -
                entry_price
            )
            /
            entry_price
        ) * 100

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    elif signal == "SELL":

        entry_price = (
            raw_entry_price *
            (
                1 -
                ENTRY_SLIPPAGE_PERCENT / 100
            )
        )

        exit_price = (
            raw_exit_price *
            (
                1 +
                EXIT_SLIPPAGE_PERCENT / 100
            )
        )

        gross_return = (
            (
                entry_price -
                exit_price
            )
            /
            entry_price
        ) * 100

    else:

        return None

    # --------------------------------------------------------
    # Transaction cost
    # --------------------------------------------------------

    transaction_cost = (
        TRANSACTION_COST_PERCENT * 2
    )

    # --------------------------------------------------------
    # Net return
    # --------------------------------------------------------

    net_return = (
        gross_return -
        transaction_cost
    )

    if net_return > 0:

        result = "PROFITABLE"

    elif net_return < 0:

        result = "FAILED"

    else:

        result = "BREAKEVEN"

    return {
        "Entry Date": entry_date,
        "Exit Date": exit_date,
        "Signal": signal,
        "Entry Price": round(
            entry_price,
            2
        ),
        "Exit Price": round(
            exit_price,
            2
        ),
        "Gross Return %": round(
            gross_return,
            2
        ),
        "Transaction Cost %": round(
            transaction_cost,
            2
        ),
        "Net Return %": round(
            net_return,
            2
        ),
        "Result": result,
        "Holding Bars": HOLDING_PERIOD
    }


# ============================================================
# ANALYZE TRAINING PERIOD
# ============================================================

def analyze_training(
    df,
    split_index
):

    signals = get_signals(
        df,
        0,
        split_index
    )

    trades = []

    for signal_data in signals:

        trade = calculate_trade(
            df=df,
            signal_index=signal_data["index"],
            signal=signal_data["signal"]
        )

        if trade is not None:

            # Only accept trades whose
            # crossover occurred in training.
            trades.append(trade)

    return calculate_statistics(
        trades
    )


# ============================================================
# ANALYZE TEST PERIOD
# ============================================================

def analyze_test(
    df,
    split_index
):

    signals = get_signals(
        df,
        split_index,
        len(df)
    )

    trades = []

    for signal_data in signals:

        trade = calculate_trade(
            df=df,
            signal_index=signal_data["index"],
            signal=signal_data["signal"]
        )

        if trade is not None:

            # Signal must occur in test period.
            trades.append(trade)

    return calculate_statistics(
        trades
    )


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(trades):

    total_trades = len(
        trades
    )

    wins = sum(
        1
        for trade in trades
        if trade["Net Return %"] > 0
    )

    losses = sum(
        1
        for trade in trades
        if trade["Net Return %"] < 0
    )

    if total_trades > 0:

        win_rate = (
            wins /
            total_trades
        ) * 100

        gross_return = sum(
            trade["Gross Return %"]
            for trade in trades
        )

        net_return = sum(
            trade["Net Return %"]
            for trade in trades
        )

        average_return = (
            net_return /
            total_trades
        )

    else:

        win_rate = 0.0
        gross_return = 0.0
        net_return = 0.0
        average_return = 0.0

    return {
        "Trades": total_trades,
        "Wins": wins,
        "Losses": losses,
        "Win Rate %": round(
            win_rate,
            2
        ),
        "Gross Return %": round(
            gross_return,
            2
        ),
        "Net Return %": round(
            net_return,
            2
        ),
        "Average Return %": round(
            average_return,
            2
        ),
        "Trades Data": trades
    }


# ============================================================
# PRINT PERIOD
# ============================================================

def print_period(
    title,
    result
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"Trades              : "
        f"{result['Trades']}"
    )

    print(
        f"Wins                : "
        f"{result['Wins']}"
    )

    print(
        f"Losses              : "
        f"{result['Losses']}"
    )

    print(
        f"Win Rate            : "
        f"{result['Win Rate %']:.2f}%"
    )

    print(
        f"Gross Return        : "
        f"{result['Gross Return %']:.2f}%"
    )

    print(
        f"Net Return          : "
        f"{result['Net Return %']:.2f}%"
    )

    print(
        f"Average Net Return  : "
        f"{result['Average Return %']:.2f}%"
    )

    # --------------------------------------------------------
    # Trade details
    # --------------------------------------------------------

    if result["Trades Data"]:

        print()
        print(
            "TRADE DETAILS"
        )

        print(
            "-" * 90
        )

        print(
            f"{'Signal':<8}"
            f"{'Entry':>12}"
            f"{'Exit':>12}"
            f"{'Gross %':>12}"
            f"{'Net %':>12}"
            f"{'Result':>15}"
        )

        print(
            "-" * 90
        )

        for trade in result["Trades Data"]:

            print(
                f"{trade['Signal']:<8}"
                f"{trade['Entry Price']:>12.2f}"
                f"{trade['Exit Price']:>12.2f}"
                f"{trade['Gross Return %']:>11.2f}%"
                f"{trade['Net Return %']:>11.2f}%"
                f"{trade['Result']:>15}"
            )


# ============================================================
# RUN ONE STOCK
# ============================================================

def run_stock(symbol):

    df = prepare_data(
        symbol
    )

    if df is None:

        return None

    split_index = get_split_index(
        df
    )

    if split_index is None:

        print(
            f"{symbol}: "
            f"Unable to create train/test split."
        )

        return None

    train_df = df.iloc[
        :split_index
    ]

    test_df = df.iloc[
        split_index:
    ]

    print()
    print(
        f"{symbol}: TRAIN rows = "
        f"{len(train_df)}"
    )

    print(
        f"{symbol}: TEST rows = "
        f"{len(test_df)}"
    )

    print(
        f"{symbol}: TRAIN period = "
        f"{train_df['Date'].iloc[0].date()} "
        f"to "
        f"{train_df['Date'].iloc[-1].date()}"
    )

    print(
        f"{symbol}: TEST period = "
        f"{test_df['Date'].iloc[0].date()} "
        f"to "
        f"{test_df['Date'].iloc[-1].date()}"
    )

    # --------------------------------------------------------
    # Count crossover signals
    # --------------------------------------------------------

    train_signals = get_signals(
        df,
        0,
        split_index
    )

    test_signals = get_signals(
        df,
        split_index,
        len(df)
    )

    print()
    print(
        f"{symbol}: TRAIN crossover signals = "
        f"{len(train_signals)}"
    )

    print(
        f"{symbol}: TEST crossover signals = "
        f"{len(test_signals)}"
    )

    # --------------------------------------------------------
    # Training analysis
    # --------------------------------------------------------

    train_result = analyze_training(
        df,
        split_index
    )

    # --------------------------------------------------------
    # Test analysis
    # --------------------------------------------------------

    test_result = analyze_test(
        df,
        split_index
    )

    print_period(
        f"{symbol} - TRAINING",
        train_result
    )

    print_period(
        f"{symbol} - OUT-OF-SAMPLE TEST",
        test_result
    )

    return {
        "Stock": symbol.replace(
            "-EQ",
            ""
        ),
        "Train": train_result,
        "Test": test_result
    }


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    results
):

    print()
    print("=" * 110)
    print(
        "                         STAGE 7 - FINAL RESULTS"
    )
    print("=" * 110)

    print(
        f"{'Stock':<15}"
        f"{'Train Trades':>14}"
        f"{'Train Win%':>13}"
        f"{'Train Net%':>13}"
        f"{'Test Trades':>13}"
        f"{'Test Win%':>12}"
        f"{'Test Net%':>12}"
    )

    print(
        "-" * 110
    )

    total_train_trades = 0
    total_train_wins = 0
    total_train_net = 0.0

    total_test_trades = 0
    total_test_wins = 0
    total_test_net = 0.0

    stocks_tested = 0

    for result in results:

        if result is None:
            continue

        stocks_tested += 1

        train = result["Train"]
        test = result["Test"]

        total_train_trades += (
            train["Trades"]
        )

        total_train_wins += (
            train["Wins"]
        )

        total_train_net += (
            train["Net Return %"]
        )

        total_test_trades += (
            test["Trades"]
        )

        total_test_wins += (
            test["Wins"]
        )

        total_test_net += (
            test["Net Return %"]
        )

        print(
            f"{result['Stock']:<15}"
            f"{train['Trades']:>14}"
            f"{train['Win Rate %']:>12.2f}%"
            f"{train['Net Return %']:>12.2f}%"
            f"{test['Trades']:>13}"
            f"{test['Win Rate %']:>11.2f}%"
            f"{test['Net Return %']:>11.2f}%"
        )

    print(
        "-" * 110
    )

    # --------------------------------------------------------
    # Overall training
    # --------------------------------------------------------

    if total_train_trades > 0:

        train_win_rate = (
            total_train_wins /
            total_train_trades
        ) * 100

    else:

        train_win_rate = 0.0

    # --------------------------------------------------------
    # Overall test
    # --------------------------------------------------------

    if total_test_trades > 0:

        test_win_rate = (
            total_test_wins /
            total_test_trades
        ) * 100

    else:

        test_win_rate = 0.0

    print()
    print(
        "OVERALL TRAINING"
    )

    print(
        "-" * 55
    )

    print(
        f"Stocks Tested       : "
        f"{stocks_tested}"
    )

    print(
        f"Total Trades        : "
        f"{total_train_trades}"
    )

    print(
        f"Total Wins          : "
        f"{total_train_wins}"
    )

    print(
        f"Win Rate            : "
        f"{train_win_rate:.2f}%"
    )

    print(
        f"Net Return          : "
        f"{total_train_net:.2f}%"
    )

    print()
    print(
        "OVERALL OUT-OF-SAMPLE TEST"
    )

    print(
        "-" * 55
    )

    print(
        f"Stocks Tested       : "
        f"{stocks_tested}"
    )

    print(
        f"Total Trades        : "
        f"{total_test_trades}"
    )

    print(
        f"Total Wins          : "
        f"{total_test_wins}"
    )

    print(
        f"Win Rate            : "
        f"{test_win_rate:.2f}%"
    )

    print(
        f"Net Return          : "
        f"{total_test_net:.2f}%"
    )

    # --------------------------------------------------------
    # Verdict
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "                         STAGE 7 VALIDATION"
    )
    print("=" * 80)

    if total_test_trades == 0:

        print(
            "RESULT: INCONCLUSIVE"
        )

        print()
        print(
            "There are still no completed crossover "
            "trades in the unseen test period."
        )

        print(
            "More historical data is required before "
            "making a conclusion."
        )

    elif total_test_net > 0:

        print(
            "RESULT: POSITIVE OUT-OF-SAMPLE"
        )

        print()
        print(
            "The locked SMMA 20/120 strategy "
            "remained profitable after costs "
            "and slippage on unseen data."
        )

    else:

        print(
            "RESULT: NEGATIVE OUT-OF-SAMPLE"
        )

        print()
        print(
            "The locked SMMA 20/120 strategy "
            "was not profitable on unseen data."
        )

    print(
        "=" * 80
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print(
        "                 STAGE 7 - OUT-OF-SAMPLE VALIDATION"
    )
    print("=" * 110)

    print()
    print(
        "LOCKED STRATEGY PARAMETERS"
    )

    print(
        f"Fast SMMA           : "
        f"{FAST_SMMA}"
    )

    print(
        f"Slow SMMA           : "
        f"{SLOW_SMMA}"
    )

    print(
        f"Train Ratio         : "
        f"{TRAIN_RATIO * 100:.0f}%"
    )

    print(
        f"Test Ratio          : "
        f"{(1 - TRAIN_RATIO) * 100:.0f}%"
    )

    print(
        f"Holding Period      : "
        f"{HOLDING_PERIOD} trading days"
    )

    print(
        f"Entry Slippage      : "
        f"{ENTRY_SLIPPAGE_PERCENT:.2f}%"
    )

    print(
        f"Exit Slippage       : "
        f"{EXIT_SLIPPAGE_PERCENT:.2f}%"
    )

    print(
        f"Transaction Cost    : "
        f"{TRANSACTION_COST_PERCENT:.2f}% per side"
    )

    print()
    print(
        "SMMA 20/120 IS LOCKED."
    )

    print(
        "NO PARAMETER OPTIMIZATION IS PERFORMED "
        "IN THE TEST PERIOD."
    )

    results = []

    # --------------------------------------------------------
    # Run stocks
    # --------------------------------------------------------

    for symbol in STOCKS:

        try:

            result = run_stock(
                symbol
            )

            results.append(
                result
            )

        except Exception as error:

            print()
            print(
                f"{symbol}: ERROR"
            )

            print(
                str(error)
            )

            results.append(
                None
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_final_summary(
        results
    )

    print()
    print("=" * 110)
    print(
        "                         STAGE 7 COMPLETE"
    )
    print("=" * 110)


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()