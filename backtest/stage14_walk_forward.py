"""
============================================================
STAGE 14 - WALK-FORWARD / OUT-OF-SAMPLE VALIDATION
============================================================

PURPOSE
-------
Validate the LOCKED SMMA 20/120 strategy on unseen data.

PAPER / HISTORICAL ANALYSIS ONLY
---------------------------------
NO REAL ORDERS ARE PLACED.

LOCKED PRODUCTION STRATEGY
--------------------------
Fast SMMA        = 20
Slow SMMA        = 120
Holding Period   = 60 trading days

RISK / COST MODEL
-----------------
Reference Capital       = Rs. 100,000
Risk Per Trade         = 1%
Stop Loss              = 5%
Take Profit            = 10%
Maximum Position       = 20%

Entry Slippage         = 0.10%
Exit Slippage          = 0.10%
Transaction Cost       = 0.05%

STAGE 14
--------
1. Download historical data once per stock.
2. Split data chronologically into:
       TRAINING DATA
       OUT-OF-SAMPLE TEST DATA
3. Run the locked SMMA 20/120 strategy.
4. Evaluate only trades whose crossover occurs
   inside the out-of-sample period.
5. Calculate:
       - trades
       - wins/losses
       - win rate
       - gross PnL
       - transaction costs
       - net PnL
       - return
       - profit factor
       - drawdown
       - BUY/SELL performance
       - stock performance
       - yearly performance
6. Save CSV reports.

OUTPUT
------
data/stage14/
    walk_forward_summary.csv
    walk_forward_trades.csv
    stock_performance.csv
    yearly_performance.csv
    buy_sell_performance.csv

IMPORTANT
---------
SMMA 20/120 IS NOT CHANGED BY THIS SCRIPT.
============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import os

import pandas as pd

from api.historical_data import get_historical_data
from indicators.smma import calculate_smma


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
# LOCKED STRATEGY
# ============================================================

FAST_SMMA = 20
SLOW_SMMA = 120

HOLDING_PERIOD = 60


# ============================================================
# CAPITAL / RISK
# ============================================================

INITIAL_CAPITAL = 100000.0

RISK_PER_TRADE_PERCENT = 1.0

STOP_LOSS_PERCENT = 5.0

TAKE_PROFIT_PERCENT = 10.0

MAX_POSITION_PERCENT = 20.0


# ============================================================
# COST / SLIPPAGE
# ============================================================

ENTRY_SLIPPAGE_PERCENT = 0.10

EXIT_SLIPPAGE_PERCENT = 0.10

TRANSACTION_COST_PERCENT = 0.05


# ============================================================
# DATA SETTINGS
# ============================================================

HISTORICAL_DAYS = 2000

TRAIN_PERCENT = 70.0


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIRECTORY = os.path.join(
    "data",
    "stage14"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "walk_forward_summary.csv"
)

TRADES_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "walk_forward_trades.csv"
)

STOCK_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "stock_performance.csv"
)

YEAR_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "yearly_performance.csv"
)

DIRECTION_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "buy_sell_performance.csv"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:

        if pd.isna(value):

            return default

        return float(value)

    except Exception:

        return default


# ============================================================
# NORMALIZE DATA
# ============================================================

def normalize_data(df):
    """
    Convert historical data into:

        date
        open
        high
        low
        close
        volume

    Compatible with the existing API module.
    """

    if df is None or df.empty:

        return None

    df = df.copy()

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [
        str(column)
        .strip()
        .lower()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "date",
        "close"
    ]

    for column in required_columns:

        if column not in df.columns:

            print(
                f"❌ Missing required column: {column}"
            )

            return None

    # --------------------------------------------------------
    # Optional OHLC columns
    # --------------------------------------------------------

    for column in [
        "open",
        "high",
        "low",
        "volume"
    ]:

        if column not in df.columns:

            if column == "volume":

                df[column] = 0.0

            else:

                df[column] = df["close"]

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:

        df[column] = pd.to_numeric(
            df[column],
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
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# LOAD STOCK DATA
# ============================================================

def load_stock_data(symbol):

    print()
    print(
        "-" * 90
    )

    print(
        f"Loading historical data: {symbol}"
    )

    print(
        "-" * 90
    )

    try:

        df = get_historical_data(
            symbol,
            days=HISTORICAL_DAYS
        )

    except Exception as error:

        print(
            f"❌ API error for {symbol}: {error}"
        )

        return None

    if df is None or df.empty:

        print(
            f"❌ No historical data for {symbol}"
        )

        return None

    print(
        f"Raw rows: {len(df)}"
    )

    df = normalize_data(
        df
    )

    if df is None or df.empty:

        print(
            f"❌ Could not normalize {symbol}"
        )

        return None

    print(
        f"Usable rows: {len(df)}"
    )

    print(
        f"Date range : "
        f"{df['date'].iloc[0].strftime('%Y-%m-%d')}"
        f" -> "
        f"{df['date'].iloc[-1].strftime('%Y-%m-%d')}"
    )

    return df


# ============================================================
# ADD INDICATORS
# ============================================================

def add_indicators(df):

    data = df.copy()

    # --------------------------------------------------------
    # SMMA
    # --------------------------------------------------------

    data["SMMA20"] = calculate_smma(
        data["close"],
        FAST_SMMA
    )

    data["SMMA120"] = calculate_smma(
        data["close"],
        SLOW_SMMA
    )

    # --------------------------------------------------------
    # Previous values
    # --------------------------------------------------------

    data["Previous_SMMA20"] = (
        data["SMMA20"].shift(1)
    )

    data["Previous_SMMA120"] = (
        data["SMMA120"].shift(1)
    )

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    data["BUY_CROSSOVER"] = (

        (
            data["Previous_SMMA20"]
            <=
            data["Previous_SMMA120"]
        )

        &

        (
            data["SMMA20"]
            >
            data["SMMA120"]
        )

    )

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    data["SELL_CROSSOVER"] = (

        (
            data["Previous_SMMA20"]
            >=
            data["Previous_SMMA120"]
        )

        &

        (
            data["SMMA20"]
            <
            data["SMMA120"]
        )

    )

    # --------------------------------------------------------
    # Remove invalid indicator rows
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            "SMMA20",
            "SMMA120"
        ]
    ).reset_index(
        drop=True
    )

    return data


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_train_test(df):

    if df is None or df.empty:

        return None, None

    split_index = int(
        len(df)
        *
        TRAIN_PERCENT
        /
        100.0
    )

    # Need enough data in both sections
    if split_index <= SLOW_SMMA:

        print(
            "❌ Training period is too short."
        )

        return None, None

    if split_index >= len(df):

        print(
            "❌ Testing period is empty."
        )

        return None, None

    train = (
        df
        .iloc[:split_index]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        df
        .iloc[split_index:]
        .copy()
        .reset_index(drop=True)
    )

    return train, test


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_position_size(
    capital,
    entry_price
):

    if capital <= 0:
        return 0

    if entry_price <= 0:
        return 0

    # Maximum position value
    maximum_position_value = (
        capital
        *
        MAX_POSITION_PERCENT
        /
        100.0
    )

    # Maximum risk
    maximum_risk_amount = (
        capital
        *
        RISK_PER_TRADE_PERCENT
        /
        100.0
    )

    # Risk per share
    risk_per_share = (
        entry_price
        *
        STOP_LOSS_PERCENT
        /
        100.0
    )

    if risk_per_share <= 0:

        return 0

    shares_by_risk = int(
        maximum_risk_amount
        /
        risk_per_share
    )

    shares_by_position = int(
        maximum_position_value
        /
        entry_price
    )

    return max(
        0,
        min(
            shares_by_risk,
            shares_by_position
        )
    )


# ============================================================
# ENTRY PRICE
# ============================================================

def calculate_entry_price(
    close_price,
    signal
):

    if signal == "BUY":

        return (
            close_price
            *
            (
                1
                +
                ENTRY_SLIPPAGE_PERCENT
                /
                100.0
            )
        )

    if signal == "SELL":

        return (
            close_price
            *
            (
                1
                -
                ENTRY_SLIPPAGE_PERCENT
                /
                100.0
            )
        )

    return close_price


# ============================================================
# STOP LOSS
# ============================================================

def calculate_stop_loss(
    entry_price,
    signal
):

    if signal == "BUY":

        return (
            entry_price
            *
            (
                1
                -
                STOP_LOSS_PERCENT
                /
                100.0
            )
        )

    if signal == "SELL":

        return (
            entry_price
            *
            (
                1
                +
                STOP_LOSS_PERCENT
                /
                100.0
            )
        )

    return entry_price


# ============================================================
# TAKE PROFIT
# ============================================================

def calculate_take_profit(
    entry_price,
    signal
):

    if signal == "BUY":

        return (
            entry_price
            *
            (
                1
                +
                TAKE_PROFIT_PERCENT
                /
                100.0
            )
        )

    if signal == "SELL":

        return (
            entry_price
            *
            (
                1
                -
                TAKE_PROFIT_PERCENT
                /
                100.0
            )
        )

    return entry_price


# ============================================================
# EXIT SLIPPAGE
# ============================================================

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
                EXIT_SLIPPAGE_PERCENT
                /
                100.0
            )
        )

    if signal == "SELL":

        return (
            price
            *
            (
                1
                +
                EXIT_SLIPPAGE_PERCENT
                /
                100.0
            )
        )

    return price


# ============================================================
# TRANSACTION COST
# ============================================================

def calculate_transaction_cost(
    entry_price,
    exit_price,
    shares
):

    entry_value = (
        entry_price
        *
        shares
    )

    exit_value = (
        exit_price
        *
        shares
    )

    total_value = (
        entry_value
        +
        exit_value
    )

    return (
        total_value
        *
        TRANSACTION_COST_PERCENT
        /
        100.0
    )


# ============================================================
# GROSS PNL
# ============================================================

def calculate_gross_pnl(
    signal,
    entry_price,
    exit_price,
    shares
):

    if signal == "BUY":

        return (
            exit_price
            -
            entry_price
        ) * shares

    if signal == "SELL":

        return (
            entry_price
            -
            exit_price
        ) * shares

    return 0.0


# ============================================================
# FIND CROSSOVER TRADES IN TEST DATA
# ============================================================

def generate_test_trades(
    symbol,
    full_data,
    test_start_date
):

    trades = []

    # --------------------------------------------------------
    # Only crossover events in test period are evaluated.
    # --------------------------------------------------------

    test_mask = (
        full_data["date"]
        >=
        pd.Timestamp(
            test_start_date
        )
    )

    crossover_mask = (
        full_data["BUY_CROSSOVER"]
        |
        full_data["SELL_CROSSOVER"]
    )

    candidate_rows = full_data[
        test_mask
        &
        crossover_mask
    ]

    print()
    print(
        f"{symbol}: "
        f"Out-of-sample crossover candidates: "
        f"{len(candidate_rows)}"
    )

    # --------------------------------------------------------
    # Process every crossover
    # --------------------------------------------------------

    for index in candidate_rows.index:

        row = full_data.loc[
            index
        ]

        if bool(
            row["BUY_CROSSOVER"]
        ):

            signal = "BUY"

        elif bool(
            row["SELL_CROSSOVER"]
        ):

            signal = "SELL"

        else:

            continue

        # ----------------------------------------------------
        # Entry
        # ----------------------------------------------------

        close_price = safe_float(
            row["close"]
        )

        if close_price <= 0:

            continue

        entry_price = calculate_entry_price(
            close_price,
            signal
        )

        shares = calculate_position_size(
            INITIAL_CAPITAL,
            entry_price
        )

        if shares <= 0:

            continue

        stop_loss = calculate_stop_loss(
            entry_price,
            signal
        )

        take_profit = calculate_take_profit(
            entry_price,
            signal
        )

        entry_date = pd.Timestamp(
            row["date"]
        ).normalize()

        # ----------------------------------------------------
        # IMPORTANT:
        # We start checking exits AFTER crossover candle.
        # ----------------------------------------------------

        future_data = full_data.loc[
            index + 1:
        ].copy()

        # Only use future candles
        if future_data.empty:

            continue

        # ----------------------------------------------------
        # Maximum holding period
        # ----------------------------------------------------

        check_data = future_data.head(
            HOLDING_PERIOD
        )

        exit_date = None
        exit_price = None
        exit_reason = None
        bars_held = 0

        # ----------------------------------------------------
        # Check SL / TP
        # ----------------------------------------------------

        for _, candle in check_data.iterrows():

            bars_held += 1

            high = safe_float(
                candle["high"]
            )

            low = safe_float(
                candle["low"]
            )

            close = safe_float(
                candle["close"]
            )

            candle_date = pd.Timestamp(
                candle["date"]
            ).normalize()

            # =================================================
            # BUY
            # =================================================

            if signal == "BUY":

                hit_stop = (
                    low <= stop_loss
                )

                hit_target = (
                    high >= take_profit
                )

                # Conservative assumption:
                # if both happen on same candle,
                # STOP LOSS happens first.

                if hit_stop:

                    raw_exit = stop_loss

                    exit_price = (
                        apply_exit_slippage(
                            raw_exit,
                            signal
                        )
                    )

                    exit_date = candle_date

                    exit_reason = "STOP_LOSS"

                    break

                if hit_target:

                    raw_exit = take_profit

                    exit_price = (
                        apply_exit_slippage(
                            raw_exit,
                            signal
                        )
                    )

                    exit_date = candle_date

                    exit_reason = "TAKE_PROFIT"

                    break

            # =================================================
            # SELL
            # =================================================

            elif signal == "SELL":

                hit_stop = (
                    high >= stop_loss
                )

                hit_target = (
                    low <= take_profit
                )

                if hit_stop:

                    raw_exit = stop_loss

                    exit_price = (
                        apply_exit_slippage(
                            raw_exit,
                            signal
                        )
                    )

                    exit_date = candle_date

                    exit_reason = "STOP_LOSS"

                    break

                if hit_target:

                    raw_exit = take_profit

                    exit_price = (
                        apply_exit_slippage(
                            raw_exit,
                            signal
                        )
                    )

                    exit_date = candle_date

                    exit_reason = "TAKE_PROFIT"

                    break

        # ----------------------------------------------------
        # Time exit
        # ----------------------------------------------------

        if exit_price is None:

            if len(check_data) >= HOLDING_PERIOD:

                last_candle = check_data.iloc[
                    HOLDING_PERIOD - 1
                ]

                raw_exit = safe_float(
                    last_candle["close"]
                )

                exit_price = (
                    apply_exit_slippage(
                        raw_exit,
                        signal
                    )
                )

                exit_date = pd.Timestamp(
                    last_candle["date"]
                ).normalize()

                exit_reason = "TIME_EXIT"

                bars_held = HOLDING_PERIOD

            else:

                # Not enough future data to complete
                # a historical trade.

                continue

        # ----------------------------------------------------
        # PnL
        # ----------------------------------------------------

        gross_pnl = calculate_gross_pnl(
            signal,
            entry_price,
            exit_price,
            shares
        )

        transaction_cost = (
            calculate_transaction_cost(
                entry_price,
                exit_price,
                shares
            )
        )

        net_pnl = (
            gross_pnl
            -
            transaction_cost
        )

        position_value = (
            entry_price
            *
            shares
        )

        if position_value > 0:

            return_percent = (
                net_pnl
                /
                position_value
                *
                100.0
            )

        else:

            return_percent = 0.0

        if net_pnl > 0:

            result = "WIN"

        elif net_pnl < 0:

            result = "LOSS"

        else:

            result = "BREAKEVEN"

        trades.append({

            "Stock":
                symbol.replace(
                    "-EQ",
                    ""
                ),

            "Symbol":
                symbol,

            "Signal":
                signal,

            "Entry Date":
                entry_date,

            "Entry Price":
                round(
                    entry_price,
                    4
                ),

            "Shares":
                shares,

            "Stop Loss":
                round(
                    stop_loss,
                    4
                ),

            "Take Profit":
                round(
                    take_profit,
                    4
                ),

            "Exit Date":
                exit_date,

            "Exit Price":
                round(
                    exit_price,
                    4
                ),

            "Bars Held":
                bars_held,

            "Exit Reason":
                exit_reason,

            "Gross PnL":
                round(
                    gross_pnl,
                    4
                ),

            "Transaction Cost":
                round(
                    transaction_cost,
                    4
                ),

            "Net PnL":
                round(
                    net_pnl,
                    4
                ),

            "Return %":
                round(
                    return_percent,
                    4
                ),

            "Result":
                result

        })

    return trades


# ============================================================
# CALCULATE DRAWDOWN
# ============================================================

def calculate_drawdown(
    trades
):

    if not trades:

        return 0.0, 0.0

    sorted_trades = sorted(
        trades,
        key=lambda x: pd.Timestamp(
            x["Exit Date"]
        )
    )

    equity = INITIAL_CAPITAL

    peak_equity = equity

    maximum_drawdown = 0.0

    for trade in sorted_trades:

        equity += safe_float(
            trade["Net PnL"]
        )

        if equity > peak_equity:

            peak_equity = equity

        drawdown = (
            equity
            -
            peak_equity
        )

        if drawdown < maximum_drawdown:

            maximum_drawdown = drawdown

    if peak_equity > 0:

        maximum_drawdown_percent = (
            maximum_drawdown
            /
            peak_equity
            *
            100.0
        )

    else:

        maximum_drawdown_percent = 0.0

    return (
        maximum_drawdown,
        maximum_drawdown_percent
    )


# ============================================================
# PROFIT FACTOR
# ============================================================

def calculate_profit_factor(
    trades
):

    if not trades:

        return 0.0

    gross_wins = sum(
        safe_float(
            trade["Net PnL"]
        )
        for trade in trades
        if safe_float(
            trade["Net PnL"]
        ) > 0
    )

    gross_losses = abs(
        sum(
            safe_float(
                trade["Net PnL"]
            )
            for trade in trades
            if safe_float(
                trade["Net PnL"]
            ) < 0
        )
    )

    if gross_losses == 0:

        if gross_wins > 0:

            return float("inf")

        return 0.0

    return (
        gross_wins
        /
        gross_losses
    )


# ============================================================
# CALCULATE OVERALL STATISTICS
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
        if trade["Result"] == "WIN"
    )

    losses = sum(
        1
        for trade in trades
        if trade["Result"] == "LOSS"
    )

    breakeven = sum(
        1
        for trade in trades
        if trade["Result"] == "BREAKEVEN"
    )

    gross_pnl = sum(
        safe_float(
            trade["Gross PnL"]
        )
        for trade in trades
    )

    transaction_costs = sum(
        safe_float(
            trade["Transaction Cost"]
        )
        for trade in trades
    )

    net_pnl = sum(
        safe_float(
            trade["Net PnL"]
        )
        for trade in trades
    )

    if total_trades > 0:

        win_rate = (
            wins
            /
            total_trades
            *
            100.0
        )

        average_pnl = (
            net_pnl
            /
            total_trades
        )

        average_return = (
            sum(
                safe_float(
                    trade["Return %"]
                )
                for trade in trades
            )
            /
            total_trades
        )

        average_holding = (
            sum(
                int(
                    trade["Bars Held"]
                )
                for trade in trades
            )
            /
            total_trades
        )

    else:

        win_rate = 0.0

        average_pnl = 0.0

        average_return = 0.0

        average_holding = 0.0

    winning_pnls = [
        safe_float(
            trade["Net PnL"]
        )
        for trade in trades
        if safe_float(
            trade["Net PnL"]
        ) > 0
    ]

    losing_pnls = [
        safe_float(
            trade["Net PnL"]
        )
        for trade in trades
        if safe_float(
            trade["Net PnL"]
        ) < 0
    ]

    average_win = (
        sum(winning_pnls)
        /
        len(winning_pnls)
        if winning_pnls
        else 0.0
    )

    average_loss = (
        sum(losing_pnls)
        /
        len(losing_pnls)
        if losing_pnls
        else 0.0
    )

    largest_win = (
        max(winning_pnls)
        if winning_pnls
        else 0.0
    )

    largest_loss = (
        min(losing_pnls)
        if losing_pnls
        else 0.0
    )

    profit_factor = calculate_profit_factor(
        trades
    )

    maximum_drawdown, maximum_drawdown_percent = (
        calculate_drawdown(
            trades
        )
    )

    net_return = (
        net_pnl
        /
        INITIAL_CAPITAL
        *
        100.0
    )

    return {

        "Initial Capital":
            INITIAL_CAPITAL,

        "Total Trades":
            total_trades,

        "Winning Trades":
            wins,

        "Losing Trades":
            losses,

        "Break-even Trades":
            breakeven,

        "Win Rate %":
            round(
                win_rate,
                4
            ),

        "Average PnL":
            round(
                average_pnl,
                4
            ),

        "Average Return %":
            round(
                average_return,
                4
            ),

        "Average Winning Trade":
            round(
                average_win,
                4
            ),

        "Average Losing Trade":
            round(
                average_loss,
                4
            ),

        "Largest Win":
            round(
                largest_win,
                4
            ),

        "Largest Loss":
            round(
                largest_loss,
                4
            ),

        "Profit Factor":
            round(
                profit_factor,
                4
            )
            if profit_factor != float("inf")
            else profit_factor,

        "Gross PnL":
            round(
                gross_pnl,
                4
            ),

        "Transaction Costs":
            round(
                transaction_costs,
                4
            ),

        "Net PnL":
            round(
                net_pnl,
                4
            ),

        "Net Return %":
            round(
                net_return,
                4
            ),

        "Maximum Drawdown":
            round(
                maximum_drawdown,
                4
            ),

        "Maximum Drawdown %":
            round(
                maximum_drawdown_percent,
                4
            ),

        "Average Holding Bars":
            round(
                average_holding,
                4
            )

    }


# ============================================================
# STOCK PERFORMANCE
# ============================================================

def calculate_stock_performance(
    trades
):

    if not trades:

        return pd.DataFrame()

    df = pd.DataFrame(
        trades
    )

    rows = []

    for stock, group in df.groupby(
        "Stock"
    ):

        net_values = pd.to_numeric(
            group["Net PnL"],
            errors="coerce"
        ).fillna(0)

        wins = (
            net_values > 0
        ).sum()

        losses = (
            net_values < 0
        ).sum()

        total = len(
            group
        )

        rows.append({

            "Stock":
                stock,

            "Trades":
                total,

            "Wins":
                int(wins),

            "Losses":
                int(losses),

            "Win Rate %":
                round(
                    wins
                    /
                    total
                    *
                    100.0,
                    2
                )
                if total > 0
                else 0.0,

            "Net PnL":
                round(
                    net_values.sum(),
                    4
                ),

            "Average PnL":
                round(
                    net_values.mean(),
                    4
                ),

            "Largest Win":
                round(
                    net_values.max(),
                    4
                ),

            "Largest Loss":
                round(
                    net_values.min(),
                    4
                )

        })

    return pd.DataFrame(
        rows
    ).sort_values(
        "Net PnL",
        ascending=False
    ).reset_index(
        drop=True
    )


# ============================================================
# BUY / SELL PERFORMANCE
# ============================================================

def calculate_direction_performance(
    trades
):

    if not trades:

        return pd.DataFrame()

    df = pd.DataFrame(
        trades
    )

    rows = []

    for signal, group in df.groupby(
        "Signal"
    ):

        net_values = pd.to_numeric(
            group["Net PnL"],
            errors="coerce"
        ).fillna(0)

        wins = (
            net_values > 0
        ).sum()

        losses = (
            net_values < 0
        ).sum()

        total = len(
            group
        )

        rows.append({

            "Signal":
                signal,

            "Trades":
                total,

            "Wins":
                int(wins),

            "Losses":
                int(losses),

            "Win Rate %":
                round(
                    wins
                    /
                    total
                    *
                    100.0,
                    2
                )
                if total > 0
                else 0.0,

            "Net PnL":
                round(
                    net_values.sum(),
                    4
                ),

            "Average PnL":
                round(
                    net_values.mean(),
                    4
                ),

            "Largest Win":
                round(
                    net_values.max(),
                    4
                ),

            "Largest Loss":
                round(
                    net_values.min(),
                    4
                )

        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# YEARLY PERFORMANCE
# ============================================================

def calculate_yearly_performance(
    trades
):

    if not trades:

        return pd.DataFrame()

    df = pd.DataFrame(
        trades
    )

    df["Exit Date"] = pd.to_datetime(
        df["Exit Date"],
        errors="coerce"
    )

    df["Year"] = (
        df["Exit Date"]
        .dt.year
    )

    rows = []

    for year, group in df.groupby(
        "Year"
    ):

        net_values = pd.to_numeric(
            group["Net PnL"],
            errors="coerce"
        ).fillna(0)

        wins = (
            net_values > 0
        ).sum()

        losses = (
            net_values < 0
        ).sum()

        total = len(
            group
        )

        rows.append({

            "Year":
                int(year),

            "Trades":
                total,

            "Wins":
                int(wins),

            "Losses":
                int(losses),

            "Win Rate %":
                round(
                    wins
                    /
                    total
                    *
                    100.0,
                    2
                )
                if total > 0
                else 0.0,

            "Gross PnL":
                round(
                    pd.to_numeric(
                        group["Gross PnL"],
                        errors="coerce"
                    )
                    .fillna(0)
                    .sum(),
                    4
                ),

            "Transaction Costs":
                round(
                    pd.to_numeric(
                        group["Transaction Cost"],
                        errors="coerce"
                    )
                    .fillna(0)
                    .sum(),
                    4
                ),

            "Net PnL":
                round(
                    net_values.sum(),
                    4
                )

        })

    return pd.DataFrame(
        rows
    ).sort_values(
        "Year"
    ).reset_index(
        drop=True
    )


# ============================================================
# PRINT TRADE TABLE
# ============================================================

def print_trades(
    trades
):

    print()
    print(
        "=" * 140
    )

    print(
        "OUT-OF-SAMPLE TRADE RESULTS"
    )

    print(
        "=" * 140
    )

    if not trades:

        print(
            "No completed out-of-sample trades."
        )

        return

    display_columns = [

        "Stock",
        "Signal",
        "Entry Date",
        "Entry Price",
        "Shares",
        "Exit Date",
        "Exit Price",
        "Bars Held",
        "Exit Reason",
        "Net PnL",
        "Return %",
        "Result"

    ]

    display_df = pd.DataFrame(
        trades
    )[display_columns].copy()

    print(
        display_df.to_string(
            index=False
        )
    )


# ============================================================
# PRINT STOCK PERFORMANCE
# ============================================================

def print_stock_performance(
    stock_df
):

    print()
    print(
        "=" * 120
    )

    print(
        "STOCK PERFORMANCE - OUT OF SAMPLE"
    )

    print(
        "=" * 120
    )

    if stock_df.empty:

        print(
            "No stock performance available."
        )

        return

    print(
        stock_df.to_string(
            index=False
        )
    )


# ============================================================
# PRINT DIRECTION PERFORMANCE
# ============================================================

def print_direction_performance(
    direction_df
):

    print()
    print(
        "=" * 110
    )

    print(
        "BUY / SELL PERFORMANCE - OUT OF SAMPLE"
    )

    print(
        "=" * 110
    )

    if direction_df.empty:

        print(
            "No BUY/SELL performance available."
        )

        return

    print(
        direction_df.to_string(
            index=False
        )
    )


# ============================================================
# PRINT YEAR PERFORMANCE
# ============================================================

def print_yearly_performance(
    yearly_df
):

    print()
    print(
        "=" * 110
    )

    print(
        "YEARLY PERFORMANCE - OUT OF SAMPLE"
    )

    print(
        "=" * 110
    )

    if yearly_df.empty:

        print(
            "No yearly performance available."
        )

        return

    print(
        yearly_df.to_string(
            index=False
        )
    )


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_final_summary(
    statistics,
    train_start,
    train_end,
    test_start,
    test_end
):

    print()
    print(
        "=" * 100
    )

    print(
        "STAGE 14 - WALK-FORWARD VALIDATION"
    )

    print(
        "=" * 100
    )

    print()
    print(
        "LOCKED STRATEGY"
    )

    print(
        f"SMMA                 : "
        f"{FAST_SMMA}/{SLOW_SMMA}"
    )

    print(
        f"Holding Period       : "
        f"{HOLDING_PERIOD} trading days"
    )

    print(
        f"Stop Loss            : "
        f"{STOP_LOSS_PERCENT:.2f}%"
    )

    print(
        f"Take Profit          : "
        f"{TAKE_PROFIT_PERCENT:.2f}%"
    )

    print()
    print(
        "DATA SPLIT"
    )

    print(
        f"Training             : "
        f"{train_start} -> {train_end}"
    )

    print(
        f"Out-of-Sample Test   : "
        f"{test_start} -> {test_end}"
    )

    print()
    print(
        "OUT-OF-SAMPLE RESULTS"
    )

    print(
        f"Total Trades         : "
        f"{statistics['Total Trades']}"
    )

    print(
        f"Winning Trades       : "
        f"{statistics['Winning Trades']}"
    )

    print(
        f"Losing Trades        : "
        f"{statistics['Losing Trades']}"
    )

    print(
        f"Break-even Trades    : "
        f"{statistics['Break-even Trades']}"
    )

    print(
        f"Win Rate             : "
        f"{statistics['Win Rate %']:.2f}%"
    )

    print(
        f"Average Trade PnL    : "
        f"Rs. {statistics['Average PnL']:,.2f}"
    )

    print(
        f"Average Return       : "
        f"{statistics['Average Return %']:.2f}%"
    )

    print(
        f"Average Winning      : "
        f"Rs. {statistics['Average Winning Trade']:,.2f}"
    )

    print(
        f"Average Losing       : "
        f"Rs. {statistics['Average Losing Trade']:,.2f}"
    )

    print(
        f"Largest Win          : "
        f"Rs. {statistics['Largest Win']:,.2f}"
    )

    print(
        f"Largest Loss         : "
        f"Rs. {statistics['Largest Loss']:,.2f}"
    )

    profit_factor = statistics[
        "Profit Factor"
    ]

    if profit_factor == float("inf"):

        profit_factor_text = "INF"

    else:

        profit_factor_text = (
            f"{profit_factor:.2f}"
        )

    print(
        f"Profit Factor        : "
        f"{profit_factor_text}"
    )

    print()
    print(
        f"Gross PnL            : "
        f"Rs. {statistics['Gross PnL']:,.2f}"
    )

    print(
        f"Transaction Costs    : "
        f"Rs. {statistics['Transaction Costs']:,.2f}"
    )

    print(
        f"Net PnL              : "
        f"Rs. {statistics['Net PnL']:,.2f}"
    )

    print(
        f"Net Return           : "
        f"{statistics['Net Return %']:.2f}%"
    )

    print(
        f"Maximum Drawdown     : "
        f"Rs. {statistics['Maximum Drawdown']:,.2f}"
    )

    print(
        f"Maximum Drawdown %   : "
        f"{statistics['Maximum Drawdown %']:.2f}%"
    )

    print(
        f"Average Holding Bars : "
        f"{statistics['Average Holding Bars']:.2f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    statistics,
    trades,
    stock_df,
    yearly_df,
    direction_df
):

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        [statistics]
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Trades
    # --------------------------------------------------------

    if trades:

        trades_df = pd.DataFrame(
            trades
        )

    else:

        trades_df = pd.DataFrame()

    if not trades_df.empty:

        for column in [
            "Entry Date",
            "Exit Date"
        ]:

            if column in trades_df.columns:

                trades_df[column] = (
                    pd.to_datetime(
                        trades_df[column],
                        errors="coerce"
                    )
                    .dt.strftime(
                        "%Y-%m-%d"
                    )
                )

    trades_df.to_csv(
        TRADES_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Stock
    # --------------------------------------------------------

    stock_df.to_csv(
        STOCK_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Year
    # --------------------------------------------------------

    yearly_df.to_csv(
        YEAR_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    direction_df.to_csv(
        DIRECTION_FILE,
        index=False
    )

    print()
    print(
        "=" * 100
    )

    print(
        "STAGE 14 OUTPUT FILES"
    )

    print(
        "=" * 100
    )

    print(
        f"Summary       : {SUMMARY_FILE}"
    )

    print(
        f"Trades        : {TRADES_FILE}"
    )

    print(
        f"Stock         : {STOCK_FILE}"
    )

    print(
        f"Yearly        : {YEAR_FILE}"
    )

    print(
        f"BUY/SELL      : {DIRECTION_FILE}"
    )


# ============================================================
# FINAL VALIDATION DECISION
# ============================================================

def print_validation_decision(
    statistics
):

    print()
    print(
        "=" * 100
    )

    print(
        "STAGE 14 VALIDATION DECISION"
    )

    print(
        "=" * 100
    )

    trades = statistics[
        "Total Trades"
    ]

    net_pnl = statistics[
        "Net PnL"
    ]

    profit_factor = statistics[
        "Profit Factor"
    ]

    drawdown = abs(
        statistics[
            "Maximum Drawdown %"
        ]
    )

    # --------------------------------------------------------
    # No trades
    # --------------------------------------------------------

    if trades == 0:

        print(
            "RESULT: INCONCLUSIVE"
        )

        print()
        print(
            "No completed out-of-sample trades "
            "were available."
        )

        print(
            "Do not conclude that the strategy "
            "is profitable or unprofitable."
        )

        return

    # --------------------------------------------------------
    # Basic validation
    #
    # This is intentionally conservative.
    # It does NOT automatically approve the strategy.
    # --------------------------------------------------------

    if (
        net_pnl > 0
        and
        profit_factor >= 1.0
        and
        drawdown <= 20.0
        and
        trades >= 5
    ):

        print(
            "RESULT: POSITIVE OUT-OF-SAMPLE EVIDENCE"
        )

        print()
        print(
            "The locked SMMA 20/120 strategy "
            "was profitable during the test period."
        )

        print(
            "This does NOT guarantee future profitability."
        )

    elif net_pnl > 0:

        print(
            "RESULT: POSITIVE BUT INSUFFICIENT EVIDENCE"
        )

        print()
        print(
            "The test period produced positive PnL, "
            "but the sample is not strong enough "
            "for a robust conclusion."
        )

    else:

        print(
            "RESULT: NEGATIVE OUT-OF-SAMPLE EVIDENCE"
        )

        print()
        print(
            "The locked SMMA 20/120 strategy "
            "was not profitable during the test period."
        )

    print()
    print(
        "No strategy parameters were changed."
    )

    print(
        "No real order has been placed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 100
    )

    print(
        "       STAGE 14 - WALK-FORWARD / OUT-OF-SAMPLE VALIDATION"
    )

    print(
        "=" * 100
    )

    print()
    print(
        "PAPER / HISTORICAL ANALYSIS ONLY"
    )

    print(
        "NO REAL ORDERS WILL BE PLACED"
    )

    print()
    print(
        f"Stocks              : {len(STOCKS)}"
    )

    print(
        f"Historical days     : {HISTORICAL_DAYS}"
    )

    print(
        f"Training split      : {TRAIN_PERCENT:.0f}%"
    )

    print(
        f"Testing split       : "
        f"{100.0 - TRAIN_PERCENT:.0f}%"
    )

    print(
        f"Locked SMMA         : "
        f"{FAST_SMMA}/{SLOW_SMMA}"
    )

    print(
        f"Holding period      : "
        f"{HOLDING_PERIOD} bars"
    )

    print(
        "=" * 100
    )

    all_trades = []

    overall_train_start = None
    overall_train_end = None
    overall_test_start = None
    overall_test_end = None

    successful_stocks = 0

    # ========================================================
    # LOAD EACH STOCK ONCE
    # ========================================================

    for symbol in STOCKS:

        try:

            df = load_stock_data(
                symbol
            )

            if df is None or df.empty:

                continue

            # ------------------------------------------------
            # Calculate indicators on FULL history.
            #
            # This is important because SMMA120 requires
            # historical warm-up data before the test period.
            # Only crossover events occurring in the test
            # period are evaluated.
            # ------------------------------------------------

            df = add_indicators(
                df
            )

            if df is None or df.empty:

                print(
                    f"❌ Indicator calculation failed "
                    f"for {symbol}"
                )

                continue

            # ------------------------------------------------
            # Split
            # ------------------------------------------------

            train, test = split_train_test(
                df
            )

            if (
                train is None
                or
                test is None
                or
                train.empty
                or
                test.empty
            ):

                continue

            successful_stocks += 1

            train_start = (
                train["date"].iloc[0]
                .strftime("%Y-%m-%d")
            )

            train_end = (
                train["date"].iloc[-1]
                .strftime("%Y-%m-%d")
            )

            test_start = (
                test["date"].iloc[0]
                .strftime("%Y-%m-%d")
            )

            test_end = (
                test["date"].iloc[-1]
                .strftime("%Y-%m-%d")
            )

            if overall_train_start is None:

                overall_train_start = train_start

            overall_train_end = train_end

            if overall_test_start is None:

                overall_test_start = test_start

            overall_test_end = test_end

            print()
            print(
                "=" * 90
            )

            print(
                f"{symbol} WALK-FORWARD SPLIT"
            )

            print(
                "=" * 90
            )

            print(
                f"Training   : "
                f"{train_start} -> {train_end}"
            )

            print(
                f"Testing    : "
                f"{test_start} -> {test_end}"
            )

            # ------------------------------------------------
            # Generate OOS trades
            # ------------------------------------------------

            stock_trades = generate_test_trades(
                symbol=symbol,
                full_data=df,
                test_start_date=test["date"].iloc[0]
            )

            all_trades.extend(
                stock_trades
            )

            print(
                f"Completed OOS trades: "
                f"{len(stock_trades)}"
            )

        except Exception as error:

            print()
            print(
                f"❌ {symbol}: "
                f"{type(error).__name__}: {error}"
            )

    # ========================================================
    # CALCULATE RESULTS
    # ========================================================

    statistics = calculate_statistics(
        all_trades
    )

    stock_df = calculate_stock_performance(
        all_trades
    )

    yearly_df = calculate_yearly_performance(
        all_trades
    )

    direction_df = calculate_direction_performance(
        all_trades
    )

    # ========================================================
    # PRINT
    # ========================================================

    print_final_summary(
        statistics,
        overall_train_start
        if overall_train_start
        else "N/A",
        overall_train_end
        if overall_train_end
        else "N/A",
        overall_test_start
        if overall_test_start
        else "N/A",
        overall_test_end
        if overall_test_end
        else "N/A"
    )

    print_stock_performance(
        stock_df
    )

    print_direction_performance(
        direction_df
    )

    print_yearly_performance(
        yearly_df
    )

    print_trades(
        all_trades
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_results(
        statistics,
        all_trades,
        stock_df,
        yearly_df,
        direction_df
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    print_validation_decision(
        statistics
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        f"Stocks successfully tested: "
        f"{successful_stocks}/{len(STOCKS)}"
    )

    print(
        f"Out-of-sample trades       : "
        f"{len(all_trades)}"
    )

    print(
        "STAGE 14 COMPLETE"
    )

    print(
        "=" * 100
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()