"""
============================================================
STAGE 8 - RISK MANAGEMENT
============================================================

Locked strategy:
    SMMA 20 / 120

Purpose:
    Evaluate the already validated SMMA 20/120 strategy
    with explicit position sizing and risk-management rules.

Stage 7 assumptions retained:
    Holding period       = 60 trading days
    Entry slippage       = 0.10%
    Exit slippage        = 0.10%
    Transaction cost      = 0.05% per side

Stage 8 risk rules:
    Initial capital       = Rs. 100,000
    Risk per trade        = 1.00%
    Stop loss             = 5.00%
    Take profit           = 10.00%
    Maximum position      = 20.00% of capital

IMPORTANT:
    SMMA parameters remain LOCKED.
    No optimization is performed in Stage 8.
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


# ============================================================
# LOCKED STRATEGY PARAMETERS
# ============================================================

FAST_SMMA = 20
SLOW_SMMA = 120

HISTORICAL_DAYS = 1000

HOLDING_PERIOD = 60


# ============================================================
# STAGE 6 COST / SLIPPAGE ASSUMPTIONS
# ============================================================

ENTRY_SLIPPAGE_PERCENT = 0.10
EXIT_SLIPPAGE_PERCENT = 0.10

TRANSACTION_COST_PERCENT = 0.05


# ============================================================
# STAGE 8 RISK PARAMETERS
# ============================================================

INITIAL_CAPITAL = 100000.0

RISK_PER_TRADE_PERCENT = 1.00

STOP_LOSS_PERCENT = 5.00

TAKE_PROFIT_PERCENT = 10.00

MAX_POSITION_PERCENT = 20.00


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

        elif name == "open":
            rename_map[column] = "Open"

        elif name == "high":
            rename_map[column] = "High"

        elif name == "low":
            rename_map[column] = "Low"

        elif name == "volume":
            rename_map[column] = "Volume"

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
            f"{symbol}: Error loading historical data: "
            f"{error}"
        )

        return None

    if df is None or df.empty:

        print(
            f"{symbol}: No historical data."
        )

        return None

    print(
        f"{symbol}: Historical rows loaded: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    df = normalize_columns(df)

    if df is None:

        return None

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "Date",
        "Close",
    ]

    for column in required_columns:

        if column not in df.columns:

            print(
                f"{symbol}: Missing required column "
                f"'{column}'"
            )

            return None

    # --------------------------------------------------------
    # Convert Date
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Convert Close
    # --------------------------------------------------------

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
    ).copy()

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(df) < SLOW_SMMA + 2:

        print(
            f"{symbol}: Not enough data for "
            f"SMMA {SLOW_SMMA}."
        )

        return None

    # --------------------------------------------------------
    # Calculate locked SMMA indicators
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
    # Previous SMMA values
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

        (
            df["Previous_SMMA20"]
            <=
            df["Previous_SMMA120"]
        )

        &

        (
            df["SMMA20"]
            >
            df["SMMA120"]
        )
    )

    # --------------------------------------------------------
    # SELL crossover
    # --------------------------------------------------------

    df["SELL_CROSSOVER"] = (

        (
            df["Previous_SMMA20"]
            >=
            df["Previous_SMMA120"]
        )

        &

        (
            df["SMMA20"]
            <
            df["SMMA120"]
        )
    )

    # --------------------------------------------------------
    # Remove rows where SMMA is unavailable
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
# POSITION SIZING
# ============================================================

def calculate_position_size(
    capital,
    entry_price
):

    if capital <= 0:
        return 0

    if entry_price <= 0:
        return 0

    # --------------------------------------------------------
    # Maximum capital allowed in one position
    # --------------------------------------------------------

    maximum_position_value = (
        capital
        *
        MAX_POSITION_PERCENT
        /
        100.0
    )

    # --------------------------------------------------------
    # Maximum money allowed to lose
    # --------------------------------------------------------

    maximum_risk_amount = (
        capital
        *
        RISK_PER_TRADE_PERCENT
        /
        100.0
    )

    # --------------------------------------------------------
    # Money risked per share
    # --------------------------------------------------------

    risk_per_share = (
        entry_price
        *
        STOP_LOSS_PERCENT
        /
        100.0
    )

    if risk_per_share <= 0:

        return 0

    # --------------------------------------------------------
    # Number of shares based on risk
    # --------------------------------------------------------

    shares_by_risk = int(
        maximum_risk_amount
        /
        risk_per_share
    )

    # --------------------------------------------------------
    # Number of shares based on maximum position
    # --------------------------------------------------------

    shares_by_position = int(
        maximum_position_value
        /
        entry_price
    )

    # --------------------------------------------------------
    # Use the smaller number
    # --------------------------------------------------------

    shares = min(
        shares_by_risk,
        shares_by_position
    )

    return max(
        0,
        shares
    )


# ============================================================
# APPLY SLIPPAGE
# ============================================================

def apply_entry_slippage(
    price,
    signal
):

    if signal == "BUY":

        return (
            price
            *
            (
                1
                +
                ENTRY_SLIPPAGE_PERCENT / 100
            )
        )

    elif signal == "SELL":

        return (
            price
            *
            (
                1
                -
                ENTRY_SLIPPAGE_PERCENT / 100
            )
        )

    return price


def apply_exit_slippage(
    price,
    signal
):

    if signal == "BUY":

        return (
            price
            *
            (
                1
                -
                EXIT_SLIPPAGE_PERCENT / 100
            )
        )

    elif signal == "SELL":

        return (
            price
            *
            (
                1
                +
                EXIT_SLIPPAGE_PERCENT / 100
            )
        )

    return price


# ============================================================
# CALCULATE COST
# ============================================================

def calculate_transaction_cost(
    entry_value,
    exit_value
):

    entry_cost = (
        entry_value
        *
        TRANSACTION_COST_PERCENT
        /
        100.0
    )

    exit_cost = (
        exit_value
        *
        TRANSACTION_COST_PERCENT
        /
        100.0
    )

    return (
        entry_cost
        +
        exit_cost
    )


# ============================================================
# FIND RISK-MANAGED EXIT
# ============================================================

def find_exit(
    df,
    entry_index,
    signal,
    entry_price
):

    # --------------------------------------------------------
    # Stop / target
    # --------------------------------------------------------

    if signal == "BUY":

        stop_price = (
            entry_price
            *
            (
                1
                -
                STOP_LOSS_PERCENT / 100
            )
        )

        target_price = (
            entry_price
            *
            (
                1
                +
                TAKE_PROFIT_PERCENT / 100
            )
        )

    elif signal == "SELL":

        stop_price = (
            entry_price
            *
            (
                1
                +
                STOP_LOSS_PERCENT / 100
            )
        )

        target_price = (
            entry_price
            *
            (
                1
                -
                TAKE_PROFIT_PERCENT / 100
            )
        )

    else:

        return None

    # --------------------------------------------------------
    # Maximum holding period
    # --------------------------------------------------------

    exit_index = (
        entry_index
        +
        HOLDING_PERIOD
    )

    if exit_index >= len(df):

        return None

    # --------------------------------------------------------
    # Check each future trading bar
    # --------------------------------------------------------

    for i in range(
        entry_index + 1,
        exit_index + 1
    ):

        close_price = float(
            df["Close"].iloc[i]
        )

        # ====================================================
        # BUY
        # ====================================================

        if signal == "BUY":

            if close_price <= stop_price:

                return {
                    "index": i,
                    "price": close_price,
                    "reason": "STOP_LOSS"
                }

            if close_price >= target_price:

                return {
                    "index": i,
                    "price": close_price,
                    "reason": "TAKE_PROFIT"
                }

        # ====================================================
        # SELL
        # ====================================================

        elif signal == "SELL":

            if close_price >= stop_price:

                return {
                    "index": i,
                    "price": close_price,
                    "reason": "STOP_LOSS"
                }

            if close_price <= target_price:

                return {
                    "index": i,
                    "price": close_price,
                    "reason": "TAKE_PROFIT"
                }

    # --------------------------------------------------------
    # Time exit
    # --------------------------------------------------------

    return {
        "index": exit_index,
        "price": float(
            df["Close"].iloc[exit_index]
        ),
        "reason": "TIME_EXIT"
    }


# ============================================================
# CALCULATE ONE TRADE
# ============================================================

def calculate_trade(
    df,
    signal_index,
    signal,
    capital
):

    raw_entry_price = float(
        df["Close"].iloc[signal_index]
    )

    # --------------------------------------------------------
    # Apply entry slippage
    # --------------------------------------------------------

    entry_price = apply_entry_slippage(
        raw_entry_price,
        signal
    )

    # --------------------------------------------------------
    # Position size
    # --------------------------------------------------------

    shares = calculate_position_size(
        capital,
        entry_price
    )

    if shares <= 0:

        return None

    # --------------------------------------------------------
    # Position value
    # --------------------------------------------------------

    entry_value = (
        shares
        *
        entry_price
    )

    # --------------------------------------------------------
    # Find exit
    # --------------------------------------------------------

    exit_data = find_exit(
        df,
        signal_index,
        signal,
        entry_price
    )

    if exit_data is None:

        return None

    exit_index = exit_data["index"]

    raw_exit_price = (
        exit_data["price"]
    )

    exit_reason = (
        exit_data["reason"]
    )

    # --------------------------------------------------------
    # Apply exit slippage
    # --------------------------------------------------------

    exit_price = apply_exit_slippage(
        raw_exit_price,
        signal
    )

    exit_value = (
        shares
        *
        exit_price
    )

    # ========================================================
    # GROSS P&L
    # ========================================================

    if signal == "BUY":

        gross_pnl = (
            exit_value
            -
            entry_value
        )

    else:

        gross_pnl = (
            entry_value
            -
            exit_value
        )

    # --------------------------------------------------------
    # Gross return
    # --------------------------------------------------------

    gross_return = (
        gross_pnl
        /
        entry_value
    ) * 100

    # ========================================================
    # TRANSACTION COST
    # ========================================================

    transaction_cost = (
        calculate_transaction_cost(
            entry_value,
            exit_value
        )
    )

    # ========================================================
    # NET P&L
    # ========================================================

    net_pnl = (
        gross_pnl
        -
        transaction_cost
    )

    net_return = (
        net_pnl
        /
        entry_value
    ) * 100

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if net_return > 0:

        result = "PROFITABLE"

    elif net_return < 0:

        result = "FAILED"

    else:

        result = "BREAKEVEN"

    return {

        "Entry Date":
            df["Date"].iloc[
                signal_index
            ],

        "Exit Date":
            df["Date"].iloc[
                exit_index
            ],

        "Signal":
            signal,

        "Raw Entry":
            round(
                raw_entry_price,
                2
            ),

        "Entry Price":
            round(
                entry_price,
                2
            ),

        "Raw Exit":
            round(
                raw_exit_price,
                2
            ),

        "Exit Price":
            round(
                exit_price,
                2
            ),

        "Shares":
            shares,

        "Position Value":
            round(
                entry_value,
                2
            ),

        "Gross Return %":
            round(
                gross_return,
                2
            ),

        "Transaction Cost":
            round(
                transaction_cost,
                4
            ),

        "Net PnL":
            round(
                net_pnl,
                2
            ),

        "Net Return %":
            round(
                net_return,
                2
            ),

        "Result":
            result,

        "Exit Reason":
            exit_reason,

        "Holding Bars":
            exit_index - signal_index
    }


# ============================================================
# RUN ONE STOCK
# ============================================================

def run_stock(
    symbol,
    starting_capital
):

    df = prepare_data(
        symbol
    )

    if df is None:

        return {
            "Stock": symbol.replace(
                "-EQ",
                ""
            ),
            "Trades": [],
            "Ending Capital": starting_capital
        }

    trades = []

    capital = starting_capital

    i = 0

    # --------------------------------------------------------
    # Walk forward through data
    # --------------------------------------------------------

    while i < len(df):

        signal = None

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        if bool(
            df["BUY_CROSSOVER"].iloc[i]
        ):

            signal = "BUY"

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------

        elif bool(
            df["SELL_CROSSOVER"].iloc[i]
        ):

            signal = "SELL"

        # ----------------------------------------------------
        # No signal
        # ----------------------------------------------------

        if signal is None:

            i += 1

            continue

        # ----------------------------------------------------
        # Calculate trade
        # ----------------------------------------------------

        trade = calculate_trade(
            df=df,
            signal_index=i,
            signal=signal,
            capital=capital
        )

        if trade is None:

            i += 1

            continue

        # ----------------------------------------------------
        # Add stock name
        # ----------------------------------------------------

        trade["Stock"] = (
            symbol.replace(
                "-EQ",
                ""
            )
        )

        trades.append(
            trade
        )

        # ----------------------------------------------------
        # Update capital
        # ----------------------------------------------------

        capital += (
            trade["Net PnL"]
        )

        # ----------------------------------------------------
        # Move after completed trade
        # ----------------------------------------------------

        exit_date = trade["Exit Date"]

        exit_positions = df.index[
            df["Date"] == exit_date
        ].tolist()

        if exit_positions:

            i = (
                exit_positions[0]
                +
                1
            )

        else:

            i += 1

    return {
        "Stock": symbol.replace(
            "-EQ",
            ""
        ),
        "Trades": trades,
        "Ending Capital": capital
    }


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(
    trades
):

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

    gross_return = sum(
        trade["Gross Return %"]
        for trade in trades
    )

    net_return = sum(
        trade["Net Return %"]
        for trade in trades
    )

    total_pnl = sum(
        trade["Net PnL"]
        for trade in trades
    )

    total_cost = sum(
        trade["Transaction Cost"]
        for trade in trades
    )

    if total_trades > 0:

        win_rate = (
            wins
            /
            total_trades
        ) * 100

        average_net_return = (
            net_return
            /
            total_trades
        )

    else:

        win_rate = 0.0

        average_net_return = 0.0

    return {

        "Trades":
            total_trades,

        "Wins":
            wins,

        "Losses":
            losses,

        "Win Rate %":
            round(
                win_rate,
                2
            ),

        "Gross Return %":
            round(
                gross_return,
                2
            ),

        "Net Return %":
            round(
                net_return,
                2
            ),

        "Average Net Return %":
            round(
                average_net_return,
                2
            ),

        "Total PnL":
            round(
                total_pnl,
                2
            ),

        "Total Costs":
            round(
                total_cost,
                4
            )
    }


# ============================================================
# PRINT STOCK RESULTS
# ============================================================

def print_stock_results(
    result
):

    stock = result["Stock"]

    trades = result["Trades"]

    statistics = calculate_statistics(
        trades
    )

    print()
    print("=" * 110)
    print(
        f"{stock} - RISK MANAGED RESULTS"
    )
    print("=" * 110)

    print(
        f"Trades              : "
        f"{statistics['Trades']}"
    )

    print(
        f"Wins                : "
        f"{statistics['Wins']}"
    )

    print(
        f"Losses              : "
        f"{statistics['Losses']}"
    )

    print(
        f"Win Rate            : "
        f"{statistics['Win Rate %']:.2f}%"
    )

    print(
        f"Gross Return        : "
        f"{statistics['Gross Return %']:.2f}%"
    )

    print(
        f"Net Return          : "
        f"{statistics['Net Return %']:.2f}%"
    )

    print(
        f"Average Net Return  : "
        f"{statistics['Average Net Return %']:.2f}%"
    )

    print(
        f"Total PnL           : "
        f"Rs. {statistics['Total PnL']:.2f}"
    )

    print(
        f"Total Costs         : "
        f"Rs. {statistics['Total Costs']:.4f}"
    )

    print(
        f"Ending Capital      : "
        f"Rs. {result['Ending Capital']:.2f}"
    )

    # --------------------------------------------------------
    # Trade details
    # --------------------------------------------------------

    if not trades:

        print()
        print(
            "No completed risk-managed trades."
        )

        return

    print()
    print(
        "TRADE DETAILS"
    )

    print("-" * 130)

    print(
        f"{'Signal':<8}"
        f"{'Entry':>11}"
        f"{'Exit':>11}"
        f"{'Shares':>9}"
        f"{'Gross %':>11}"
        f"{'Costs':>11}"
        f"{'Net %':>11}"
        f"{'Result':>15}"
        f"{'Exit Reason':>15}"
    )

    print("-" * 130)

    for trade in trades:

        print(
            f"{trade['Signal']:<8}"
            f"{trade['Entry Price']:>11.2f}"
            f"{trade['Exit Price']:>11.2f}"
            f"{trade['Shares']:>9}"
            f"{trade['Gross Return %']:>10.2f}%"
            f"{trade['Transaction Cost']:>11.2f}"
            f"{trade['Net Return %']:>10.2f}%"
            f"{trade['Result']:>15}"
            f"{trade['Exit Reason']:>15}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print(
        "STAGE 8 - RISK MANAGEMENT"
    )
    print("=" * 100)

    print()
    print(
        "LOCKED STRATEGY PARAMETERS"
    )

    print("-" * 60)

    print(
        f"Fast SMMA             : {FAST_SMMA}"
    )

    print(
        f"Slow SMMA             : {SLOW_SMMA}"
    )

    print(
        f"Holding Period        : "
        f"{HOLDING_PERIOD} trading days"
    )

    print(
        f"Historical Days       : "
        f"{HISTORICAL_DAYS}"
    )

    print(
        f"Entry Slippage        : "
        f"{ENTRY_SLIPPAGE_PERCENT:.2f}%"
    )

    print(
        f"Exit Slippage         : "
        f"{EXIT_SLIPPAGE_PERCENT:.2f}%"
    )

    print(
        f"Transaction Cost      : "
        f"{TRANSACTION_COST_PERCENT:.2f}% per side"
    )

    print()
    print(
        "RISK MANAGEMENT PARAMETERS"
    )

    print("-" * 60)

    print(
        f"Initial Capital       : "
        f"Rs. {INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Risk Per Trade        : "
        f"{RISK_PER_TRADE_PERCENT:.2f}%"
    )

    print(
        f"Stop Loss             : "
        f"{STOP_LOSS_PERCENT:.2f}%"
    )

    print(
        f"Take Profit           : "
        f"{TAKE_PROFIT_PERCENT:.2f}%"
    )

    print(
        f"Maximum Position      : "
        f"{MAX_POSITION_PERCENT:.2f}%"
    )

    print()
    print(
        "SMMA 20/120 IS LOCKED."
    )

    print(
        "NO PARAMETER OPTIMIZATION IS PERFORMED."
    )

    print()
    print("=" * 100)

    # ========================================================
    # RUN ALL STOCKS
    # ========================================================

    results = []

    for symbol in STOCKS:

        try:

            result = run_stock(
                symbol=symbol,
                starting_capital=INITIAL_CAPITAL
            )

            results.append(
                result
            )

            print_stock_results(
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

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    all_trades = []

    for result in results:

        all_trades.extend(
            result["Trades"]
        )

    overall = calculate_statistics(
        all_trades
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print("=" * 110)
    print(
        "STAGE 8 - FINAL RESULTS"
    )
    print("=" * 110)

    print(
        f"{'Stock':<15}"
        f"{'Trades':>10}"
        f"{'Wins':>10}"
        f"{'Losses':>10}"
        f"{'Win Rate':>12}"
        f"{'Gross %':>12}"
        f"{'Net %':>12}"
    )

    print("-" * 110)

    for result in results:

        stock = result["Stock"]

        statistics = calculate_statistics(
            result["Trades"]
        )

        print(
            f"{stock:<15}"
            f"{statistics['Trades']:>10}"
            f"{statistics['Wins']:>10}"
            f"{statistics['Losses']:>10}"
            f"{statistics['Win Rate %']:>11.2f}%"
            f"{statistics['Gross Return %']:>11.2f}%"
            f"{statistics['Net Return %']:>11.2f}%"
        )

    print("-" * 110)

    # ========================================================
    # OVERALL
    # ========================================================

    print()
    print(
        "OVERALL RISK-MANAGED RESULTS"
    )

    print("-" * 60)

    print(
        f"Stocks Tested        : "
        f"{len(results)}"
    )

    print(
        f"Total Trades         : "
        f"{overall['Trades']}"
    )

    print(
        f"Total Wins           : "
        f"{overall['Wins']}"
    )

    print(
        f"Total Losses         : "
        f"{overall['Losses']}"
    )

    print(
        f"Overall Win Rate     : "
        f"{overall['Win Rate %']:.2f}%"
    )

    print(
        f"Gross Return         : "
        f"{overall['Gross Return %']:.2f}%"
    )

    print(
        f"Net Return           : "
        f"{overall['Net Return %']:.2f}%"
    )

    print(
        f"Average Net Return   : "
        f"{overall['Average Net Return %']:.2f}%"
    )

    print(
        f"Total PnL            : "
        f"Rs. {overall['Total PnL']:.2f}"
    )

    print(
        f"Total Costs          : "
        f"Rs. {overall['Total Costs']:.4f}"
    )

    # ========================================================
    # BEST STOCK
    # ========================================================

    stocks_with_trades = []

    for result in results:

        statistics = calculate_statistics(
            result["Trades"]
        )

        if statistics["Trades"] > 0:

            stocks_with_trades.append(
                (
                    result["Stock"],
                    statistics
                )
            )

    if stocks_with_trades:

        best_stock, best_stats = max(
            stocks_with_trades,
            key=lambda item:
                item[1]["Net Return %"]
        )

        print()
        print("=" * 70)
        print(
            "BEST STOCK AFTER RISK MANAGEMENT"
        )
        print("=" * 70)

        print(
            f"Stock        : {best_stock}"
        )

        print(
            f"Trades       : "
            f"{best_stats['Trades']}"
        )

        print(
            f"Wins         : "
            f"{best_stats['Wins']}"
        )

        print(
            f"Losses       : "
            f"{best_stats['Losses']}"
        )

        print(
            f"Win Rate     : "
            f"{best_stats['Win Rate %']:.2f}%"
        )

        print(
            f"Gross Return : "
            f"{best_stats['Gross Return %']:.2f}%"
        )

        print(
            f"Net Return   : "
            f"{best_stats['Net Return %']:.2f}%"
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STAGE 8 VALIDATION"
    )
    print("=" * 80)

    if overall["Trades"] == 0:

        print(
            "RESULT: INCONCLUSIVE"
        )

        print()
        print(
            "No completed crossover trades were "
            "available for risk-management analysis."
        )

    elif overall["Net Return %"] > 0:

        print(
            "RESULT: POSITIVE"
        )

        print()
        print(
            "The locked SMMA 20/120 strategy remained "
            "profitable after applying the Stage 8 "
            "risk-management rules."
        )

    else:

        print(
            "RESULT: NEGATIVE"
        )

        print()
        print(
            "The risk-managed strategy was not profitable "
            "under the Stage 8 assumptions."
        )

    print()
    print("=" * 100)
    print(
        "STAGE 8 COMPLETE"
    )
    print("=" * 100)


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()