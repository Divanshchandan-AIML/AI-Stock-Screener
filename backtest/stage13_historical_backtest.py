"""
============================================================
STAGE 13 - HISTORICAL STRATEGY BACKTEST
============================================================

PURPOSE
-------
Stage 13 performs a full historical backtest of the LOCKED
SMMA 20/120 strategy.

Unlike Stage 10, this stage does NOT require a crossover on
today's candle.

It scans historical candles and evaluates every valid
SMMA 20/120 crossover.

PAPER / RESEARCH ONLY
---------------------
NO REAL ORDERS ARE PLACED.

LOCKED STRATEGY
---------------
SMMA FAST          = 20
SMMA SLOW          = 120
HOLDING PERIOD     = 60 trading days

RISK MANAGEMENT
---------------
Initial Capital    = Rs. 100,000
Risk Per Trade     = 1%
Stop Loss          = 5%
Take Profit        = 10%
Maximum Position   = 20%

COST / SLIPPAGE
---------------
Entry Slippage     = 0.10%
Exit Slippage      = 0.10%
Transaction Cost    = 0.05%

STOCKS
-------
SBIN
SUZLON
IRFC
TATAMOTORS
RELIANCE
ITC

OUTPUT
------
data/stage13/
    historical_trades.csv
    equity_curve.csv
    stock_summary.csv
    backtest_report.csv

============================================================
"""


# ============================================================
# IMPORTS
# ============================================================

import os
from datetime import datetime

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
# RISK MANAGEMENT
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
# HISTORICAL DATA
# ============================================================

# Larger than Stage 10 because Stage 13 is a historical test.

HISTORICAL_DAYS = 2000


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIRECTORY = os.path.join(
    "data",
    "stage13"
)

TRADES_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "historical_trades.csv"
)

EQUITY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "equity_curve.csv"
)

STOCK_SUMMARY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "stock_summary.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "backtest_report.csv"
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

def normalize_data(
    df
):

    if df is None:

        return None

    if df.empty:

        return None

    df = df.copy()

    df.columns = [
        str(column)
        .strip()
        .lower()
        for column in df.columns
    ]

    required = [
        "date",
        "open",
        "high",
        "low",
        "close"
    ]

    for column in required:

        if column not in df.columns:

            print(
                f"❌ Missing column: {column}"
            )

            return None

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    for column in [
        "open",
        "high",
        "low",
        "close"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "date",
            "open",
            "high",
            "low",
            "close"
        ]
    )

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
# PREPARE INDICATORS
# ============================================================

def prepare_indicators(
    df
):

    df = df.copy()

    df["SMMA20"] = calculate_smma(
        df["close"],
        FAST_SMMA
    )

    df["SMMA120"] = calculate_smma(
        df["close"],
        SLOW_SMMA
    )

    df["Previous_SMMA20"] = (
        df["SMMA20"].shift(1)
    )

    df["Previous_SMMA120"] = (
        df["SMMA120"].shift(1)
    )

    # --------------------------------------------------------
    # BUY CROSSOVER
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
    # SELL CROSSOVER
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

    return df


# ============================================================
# LOAD ONE STOCK
# ============================================================

def load_stock_data(
    symbol
):

    print()
    print(
        "-" * 90
    )

    print(
        f"Loading historical data: {symbol}"
    )

    try:

        df = get_historical_data(
            symbol,
            days=HISTORICAL_DAYS
        )

    except Exception as error:

        print(
            f"❌ Data error for {symbol}: "
            f"{error}"
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

    if df is None:

        return None

    df = prepare_indicators(
        df
    )

    usable = df.dropna(
        subset=[
            "SMMA20",
            "SMMA120"
        ]
    )

    if usable.empty:

        print(
            f"❌ Not enough SMMA data for {symbol}"
        )

        return None

    print(
        f"Usable rows: {len(usable)}"
    )

    print(
        f"Date range : "
        f"{df['date'].iloc[0].strftime('%Y-%m-%d')}"
        f" -> "
        f"{df['date'].iloc[-1].strftime('%Y-%m-%d')}"
    )

    buy_count = int(
        df["BUY_CROSSOVER"].sum()
    )

    sell_count = int(
        df["SELL_CROSSOVER"].sum()
    )

    print(
        f"BUY crossovers  : {buy_count}"
    )

    print(
        f"SELL crossovers : {sell_count}"
    )

    return df


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
                ENTRY_SLIPPAGE_PERCENT / 100
            )
        )

    if signal == "SELL":

        return (
            close_price
            *
            (
                1
                -
                ENTRY_SLIPPAGE_PERCENT / 100
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
                STOP_LOSS_PERCENT / 100
            )
        )

    if signal == "SELL":

        return (
            entry_price
            *
            (
                1
                +
                STOP_LOSS_PERCENT / 100
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
                TAKE_PROFIT_PERCENT / 100
            )
        )

    if signal == "SELL":

        return (
            entry_price
            *
            (
                1
                -
                TAKE_PROFIT_PERCENT / 100
            )
        )

    return entry_price


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

    # --------------------------------------------------------
    # Maximum position value
    # --------------------------------------------------------

    maximum_position_value = (
        capital
        *
        MAX_POSITION_PERCENT
        /
        100.0
    )

    # --------------------------------------------------------
    # Maximum risk amount
    # --------------------------------------------------------

    maximum_risk_amount = (
        capital
        *
        RISK_PER_TRADE_PERCENT
        /
        100.0
    )

    # --------------------------------------------------------
    # Risk per share
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

    shares = min(
        shares_by_risk,
        shares_by_position
    )

    return max(
        0,
        shares
    )


# ============================================================
# ENTRY COST
# ============================================================

def calculate_entry_cost(
    entry_price,
    shares
):

    entry_value = (
        entry_price
        *
        shares
    )

    return (
        entry_value
        *
        TRANSACTION_COST_PERCENT
        /
        100.0
    )


# ============================================================
# EXIT COST
# ============================================================

def calculate_exit_cost(
    exit_price,
    shares
):

    exit_value = (
        exit_price
        *
        shares
    )

    return (
        exit_value
        *
        TRANSACTION_COST_PERCENT
        /
        100.0
    )


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
                EXIT_SLIPPAGE_PERCENT / 100
            )
        )

    if signal == "SELL":

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
# FIND EXIT
# ============================================================

def find_exit(
    df,
    signal_index,
    signal,
    stop_loss,
    take_profit
):

    future = df.loc[
        signal_index + 1:
    ].copy()

    if future.empty:

        return {

            "status": "OPEN",

            "exit_date": None,

            "exit_price": None,

            "reason":
                "NO_FUTURE_DATA",

            "bars_held": 0,

            "latest_price":
                safe_float(
                    df.iloc[-1]["close"]
                )

        }

    check_data = future.head(
        HOLDING_PERIOD
    )

    bars_held = 0

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

        # ====================================================
        # LONG
        # ====================================================

        if signal == "BUY":

            hit_stop = (
                low <= stop_loss
            )

            hit_target = (
                high >= take_profit
            )

            # Conservative assumption:
            # if both are hit on the same candle,
            # STOP LOSS is assumed first.

            if hit_stop:

                raw_exit = stop_loss

                return {

                    "status": "CLOSED",

                    "exit_date":
                        candle_date,

                    "exit_price":
                        apply_exit_slippage(
                            raw_exit,
                            signal
                        ),

                    "reason":
                        "STOP_LOSS",

                    "bars_held":
                        bars_held,

                    "latest_price":
                        close

                }

            if hit_target:

                raw_exit = take_profit

                return {

                    "status": "CLOSED",

                    "exit_date":
                        candle_date,

                    "exit_price":
                        apply_exit_slippage(
                            raw_exit,
                            signal
                        ),

                    "reason":
                        "TAKE_PROFIT",

                    "bars_held":
                        bars_held,

                    "latest_price":
                        close

                }

        # ====================================================
        # SHORT
        # ====================================================

        elif signal == "SELL":

            hit_stop = (
                high >= stop_loss
            )

            hit_target = (
                low <= take_profit
            )

            if hit_stop:

                raw_exit = stop_loss

                return {

                    "status": "CLOSED",

                    "exit_date":
                        candle_date,

                    "exit_price":
                        apply_exit_slippage(
                            raw_exit,
                            signal
                        ),

                    "reason":
                        "STOP_LOSS",

                    "bars_held":
                        bars_held,

                    "latest_price":
                        close

                }

            if hit_target:

                raw_exit = take_profit

                return {

                    "status": "CLOSED",

                    "exit_date":
                        candle_date,

                    "exit_price":
                        apply_exit_slippage(
                            raw_exit,
                            signal
                        ),

                    "reason":
                        "TAKE_PROFIT",

                    "bars_held":
                        bars_held,

                    "latest_price":
                        close

                }

    # ========================================================
    # TIME EXIT
    # ========================================================

    if len(check_data) >= HOLDING_PERIOD:

        last = check_data.iloc[
            HOLDING_PERIOD - 1
        ]

        raw_exit = safe_float(
            last["close"]
        )

        return {

            "status": "CLOSED",

            "exit_date":
                pd.Timestamp(
                    last["date"]
                ).normalize(),

            "exit_price":
                apply_exit_slippage(
                    raw_exit,
                    signal
                ),

            "reason":
                "TIME_EXIT",

            "bars_held":
                HOLDING_PERIOD,

            "latest_price":
                raw_exit

        }

    # ========================================================
    # OPEN POSITION
    # ========================================================

    latest = df.iloc[-1]

    return {

        "status": "OPEN",

        "exit_date": None,

        "exit_price": None,

        "reason":
            "POSITION_OPEN",

        "bars_held":
            bars_held,

        "latest_price":
            safe_float(
                latest["close"]
            )

    }


# ============================================================
# GENERATE CANDIDATE TRADES
# ============================================================

def generate_candidate_trades(
    symbol,
    df
):

    trades = []

    # --------------------------------------------------------
    # Find every crossover
    # --------------------------------------------------------

    crossover_indexes = df.index[
        df["BUY_CROSSOVER"]
        |
        df["SELL_CROSSOVER"]
    ].tolist()

    if not crossover_indexes:

        return trades

    last_exit_index = -1

    for signal_index in crossover_indexes:

        # ----------------------------------------------------
        # Prevent overlapping positions for same stock.
        # ----------------------------------------------------

        if signal_index <= last_exit_index:

            continue

        row = df.loc[
            signal_index
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

        signal_date = pd.Timestamp(
            row["date"]
        ).normalize()

        close_price = safe_float(
            row["close"]
        )

        entry_price = calculate_entry_price(
            close_price,
            signal
        )

        stop_loss = calculate_stop_loss(
            entry_price,
            signal
        )

        take_profit = calculate_take_profit(
            entry_price,
            signal
        )

        # ----------------------------------------------------
        # Use reference capital for candidate sizing.
        # Final portfolio sizing is calculated later.
        # ----------------------------------------------------

        shares = calculate_position_size(
            INITIAL_CAPITAL,
            entry_price
        )

        if shares <= 0:

            continue

        result = find_exit(
            df,
            signal_index,
            signal,
            stop_loss,
            take_profit
        )

        exit_date = result[
            "exit_date"
        ]

        exit_price = result[
            "exit_price"
        ]

        status = result[
            "status"
        ]

        reason = result[
            "reason"
        ]

        bars_held = result[
            "bars_held"
        ]

        # ----------------------------------------------------
        # Ignore positions that cannot be closed historically.
        # ----------------------------------------------------

        if status != "CLOSED":

            # Historical dataset ends while trade is open.
            # Keep it as an open historical candidate.

            trades.append({

                "Stock":
                    symbol.replace(
                        "-EQ",
                        ""
                    ),

                "Symbol":
                    symbol,

                "Signal Date":
                    signal_date,

                "Signal":
                    signal,

                "Entry Price":
                    entry_price,

                "Stop Loss":
                    stop_loss,

                "Take Profit":
                    take_profit,

                "Shares":
                    shares,

                "Exit Date":
                    None,

                "Exit Price":
                    None,

                "Bars Held":
                    bars_held,

                "Exit Reason":
                    reason,

                "Status":
                    "OPEN"

            })

            break

        # ----------------------------------------------------
        # Closed candidate
        # ----------------------------------------------------

        trades.append({

            "Stock":
                symbol.replace(
                    "-EQ",
                    ""
                ),

            "Symbol":
                symbol,

            "Signal Date":
                signal_date,

            "Signal":
                signal,

            "Entry Price":
                entry_price,

            "Stop Loss":
                stop_loss,

            "Take Profit":
                take_profit,

            "Shares":
                shares,

            "Exit Date":
                exit_date,

            "Exit Price":
                exit_price,

            "Bars Held":
                bars_held,

            "Exit Reason":
                reason,

            "Status":
                "CLOSED"

        })

        # ----------------------------------------------------
        # Prevent another same-stock trade before this trade
        # exits.
        # ----------------------------------------------------

        exit_matches = df.index[
            pd.to_datetime(
                df["date"]
            ).dt.normalize()
            ==
            pd.Timestamp(
                exit_date
            ).normalize()
        ]

        if len(exit_matches) > 0:

            last_exit_index = int(
                exit_matches[0]
            )

        else:

            last_exit_index = signal_index

    return trades


# ============================================================
# CALCULATE TRADE PNL
# ============================================================

def calculate_trade_pnl(
    trade
):

    signal = trade["Signal"]

    entry_price = safe_float(
        trade["Entry Price"]
    )

    exit_price = safe_float(
        trade["Exit Price"]
    )

    shares = int(
        safe_float(
            trade["Shares"]
        )
    )

    if signal == "BUY":

        gross_pnl = (
            exit_price
            -
            entry_price
        ) * shares

    elif signal == "SELL":

        gross_pnl = (
            entry_price
            -
            exit_price
        ) * shares

    else:

        gross_pnl = 0.0

    entry_cost = calculate_entry_cost(
        entry_price,
        shares
    )

    exit_cost = calculate_exit_cost(
        exit_price,
        shares
    )

    total_cost = (
        entry_cost
        +
        exit_cost
    )

    net_pnl = (
        gross_pnl
        -
        total_cost
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
            100
        )

    else:

        return_percent = 0.0

    return {

        "Gross PnL":
            gross_pnl,

        "Transaction Costs":
            total_cost,

        "Net PnL":
            net_pnl,

        "Return %":
            return_percent

    }


# ============================================================
# APPLY PORTFOLIO CAPITAL
# ============================================================

def apply_portfolio_sizing(
    trades
):

    if not trades:

        return pd.DataFrame()

    df = pd.DataFrame(
        trades
    )

    if df.empty:

        return df

    df["Signal Date"] = pd.to_datetime(
        df["Signal Date"]
    )

    df["Exit Date"] = pd.to_datetime(
        df["Exit Date"],
        errors="coerce"
    )

    df = (
        df
        .sort_values(
            [
                "Signal Date",
                "Symbol"
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Portfolio state
    # --------------------------------------------------------

    capital = INITIAL_CAPITAL

    active_positions = []

    final_trades = []

    # --------------------------------------------------------
    # Process trades chronologically
    # --------------------------------------------------------

    for _, candidate in df.iterrows():

        signal_date = pd.Timestamp(
            candidate["Signal Date"]
        ).normalize()

        # ----------------------------------------------------
        # Release capital from trades already closed
        # ----------------------------------------------------

        remaining_positions = []

        for position in active_positions:

            exit_date = position[
                "Exit Date"
            ]

            if (
                pd.notna(exit_date)
                and
                pd.Timestamp(
                    exit_date
                ).normalize()
                <= signal_date
            ):

                capital += position[
                    "Exit Proceeds"
                ]

            else:

                remaining_positions.append(
                    position
                )

        active_positions = (
            remaining_positions
        )

        # ----------------------------------------------------
        # Determine current capital available
        # ----------------------------------------------------

        entry_price = safe_float(
            candidate["Entry Price"]
        )

        shares = calculate_position_size(
            capital,
            entry_price
        )

        if shares <= 0:

            continue

        position_value = (
            entry_price
            *
            shares
        )

        maximum_position_value = (
            capital
            *
            MAX_POSITION_PERCENT
            /
            100.0
        )

        if (
            position_value
            >
            maximum_position_value
        ):

            shares = int(
                maximum_position_value
                /
                entry_price
            )

            position_value = (
                entry_price
                *
                shares
            )

        if shares <= 0:

            continue

        # ----------------------------------------------------
        # Entry transaction cost
        # ----------------------------------------------------

        entry_cost = calculate_entry_cost(
            entry_price,
            shares
        )

        required_cash = (
            position_value
            +
            entry_cost
        )

        # ----------------------------------------------------
        # Do not over-allocate portfolio.
        # ----------------------------------------------------

        if required_cash > capital:

            continue

        capital -= required_cash

        trade = candidate.copy()

        trade["Shares"] = shares

        trade["Position Value"] = (
            position_value
        )

        trade["Entry Transaction Cost"] = (
            entry_cost
        )

        # ----------------------------------------------------
        # CLOSED TRADE
        # ----------------------------------------------------

        if (
            str(
                trade["Status"]
            ).upper()
            ==
            "CLOSED"
            and
            pd.notna(
                trade["Exit Date"]
            )
        ):

            exit_price = safe_float(
                trade["Exit Price"]
            )

            exit_value = (
                exit_price
                *
                shares
            )

            exit_cost = calculate_exit_cost(
                exit_price,
                shares
            )

            if trade["Signal"] == "BUY":

                gross_pnl = (
                    exit_price
                    -
                    entry_price
                ) * shares

                exit_proceeds = (
                    exit_value
                    -
                    exit_cost
                )

            else:

                gross_pnl = (
                    entry_price
                    -
                    exit_price
                ) * shares

                # Short proceeds/capital treatment.
                exit_proceeds = (
                    position_value
                    +
                    gross_pnl
                    -
                    exit_cost
                )

            total_cost = (
                entry_cost
                +
                exit_cost
            )

            net_pnl = (
                gross_pnl
                -
                total_cost
            )

            trade["Gross PnL"] = (
                gross_pnl
            )

            trade["Exit Transaction Cost"] = (
                exit_cost
            )

            trade["Transaction Costs"] = (
                total_cost
            )

            trade["Net PnL"] = (
                net_pnl
            )

            trade["Return %"] = (
                net_pnl
                /
                position_value
                *
                100
            )

            trade["Capital Before"] = (
                capital
                +
                required_cash
            )

            trade["Capital After"] = (
                capital
                +
                exit_proceeds
            )

            # ------------------------------------------------
            # Since this trade closes in the future, its exit
            # proceeds are released when its exit date arrives.
            # ------------------------------------------------

            active_positions.append({

                "Exit Date":
                    trade["Exit Date"],

                "Exit Proceeds":
                    exit_proceeds

            })

            # For report purposes.
            final_trades.append(
                trade.to_dict()
            )

        # ----------------------------------------------------
        # OPEN TRADE
        # ----------------------------------------------------

        else:

            trade["Gross PnL"] = 0.0

            trade["Exit Transaction Cost"] = 0.0

            trade["Transaction Costs"] = (
                entry_cost
            )

            trade["Net PnL"] = 0.0

            trade["Return %"] = 0.0

            trade["Capital Before"] = (
                capital
                +
                required_cash
            )

            trade["Capital After"] = (
                capital
            )

            final_trades.append(
                trade.to_dict()
            )

    result = pd.DataFrame(
        final_trades
    )

    if result.empty:

        return result

    result = (
        result
        .sort_values(
            "Signal Date"
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# CREATE EQUITY CURVE
# ============================================================

def create_equity_curve(
    trades
):

    if trades.empty:

        return pd.DataFrame()

    closed = trades[
        trades["Status"]
        .astype(str)
        .str.upper()
        .eq("CLOSED")
    ].copy()

    if closed.empty:

        return pd.DataFrame()

    closed["Exit Date"] = pd.to_datetime(
        closed["Exit Date"]
    )

    closed["Net PnL"] = pd.to_numeric(
        closed["Net PnL"],
        errors="coerce"
    ).fillna(0.0)

    closed = (
        closed
        .sort_values(
            "Exit Date"
        )
        .reset_index(drop=True)
    )

    running_pnl = 0.0

    rows = []

    peak = INITIAL_CAPITAL

    for _, row in closed.iterrows():

        running_pnl += safe_float(
            row["Net PnL"]
        )

        equity = (
            INITIAL_CAPITAL
            +
            running_pnl
        )

        peak = max(
            peak,
            equity
        )

        drawdown = (
            equity
            -
            peak
        )

        if peak > 0:

            drawdown_percent = (
                drawdown
                /
                peak
                *
                100
            )

        else:

            drawdown_percent = 0.0

        rows.append({

            "Date":
                pd.Timestamp(
                    row["Exit Date"]
                ).strftime(
                    "%Y-%m-%d"
                ),

            "Stock":
                row["Stock"],

            "Signal":
                row["Signal"],

            "Trade Net PnL":
                safe_float(
                    row["Net PnL"]
                ),

            "Equity":
                equity,

            "Peak Equity":
                peak,

            "Drawdown":
                drawdown,

            "Drawdown %":
                drawdown_percent

        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# STOCK SUMMARY
# ============================================================

def create_stock_summary(
    trades
):

    if trades.empty:

        return pd.DataFrame()

    closed = trades[
        trades["Status"]
        .astype(str)
        .str.upper()
        .eq("CLOSED")
    ].copy()

    if closed.empty:

        return pd.DataFrame()

    rows = []

    for stock, group in closed.groupby(
        "Stock"
    ):

        pnl = pd.to_numeric(
            group["Net PnL"],
            errors="coerce"
        ).fillna(0.0)

        wins = int(
            (pnl > 0).sum()
        )

        losses = int(
            (pnl < 0).sum()
        )

        total = len(group)

        win_rate = (
            wins
            /
            total
            *
            100
            if total > 0
            else 0.0
        )

        rows.append({

            "Stock":
                stock,

            "Trades":
                total,

            "Wins":
                wins,

            "Losses":
                losses,

            "Win Rate %":
                win_rate,

            "Net PnL":
                pnl.sum(),

            "Average PnL":
                pnl.mean(),

            "Largest Win":
                pnl.max(),

            "Largest Loss":
                pnl.min(),

        })

    return pd.DataFrame(
        rows
    ).sort_values(
        "Net PnL",
        ascending=False
    ).reset_index(drop=True)


# ============================================================
# CREATE PERFORMANCE REPORT
# ============================================================

def create_report(
    trades,
    equity
):

    if trades.empty:

        return {

            "Initial Capital":
                INITIAL_CAPITAL,

            "Total Trades":
                0,

            "Winning Trades":
                0,

            "Losing Trades":
                0,

            "Break-even Trades":
                0,

            "Win Rate %":
                0.0,

            "Loss Rate %":
                0.0,

            "Gross PnL":
                0.0,

            "Transaction Costs":
                0.0,

            "Net PnL":
                0.0,

            "Net Return %":
                0.0,

            "Peak Equity":
                INITIAL_CAPITAL,

            "Maximum Drawdown":
                0.0,

            "Maximum Drawdown %":
                0.0,

            "Average Trade PnL":
                0.0,

            "Average Winning Trade":
                0.0,

            "Average Losing Trade":
                0.0,

            "Largest Win":
                0.0,

            "Largest Loss":
                0.0,

            "Profit Factor":
                0.0,

            "Average Holding Bars":
                0.0,

        }

    closed = trades[
        trades["Status"]
        .astype(str)
        .str.upper()
        .eq("CLOSED")
    ].copy()

    if closed.empty:

        return {

            "Initial Capital":
                INITIAL_CAPITAL,

            "Total Trades":
                0,

            "Winning Trades":
                0,

            "Losing Trades":
                0,

            "Break-even Trades":
                0,

            "Win Rate %":
                0.0,

            "Loss Rate %":
                0.0,

            "Gross PnL":
                0.0,

            "Transaction Costs":
                0.0,

            "Net PnL":
                0.0,

            "Net Return %":
                0.0,

            "Peak Equity":
                INITIAL_CAPITAL,

            "Maximum Drawdown":
                0.0,

            "Maximum Drawdown %":
                0.0,

            "Average Trade PnL":
                0.0,

            "Average Winning Trade":
                0.0,

            "Average Losing Trade":
                0.0,

            "Largest Win":
                0.0,

            "Largest Loss":
                0.0,

            "Profit Factor":
                0.0,

            "Average Holding Bars":
                0.0,

        }

    pnl = pd.to_numeric(
        closed["Net PnL"],
        errors="coerce"
    ).fillna(0.0)

    gross_pnl = pd.to_numeric(
        closed["Gross PnL"],
        errors="coerce"
    ).fillna(0.0).sum()

    transaction_costs = pd.to_numeric(
        closed["Transaction Costs"],
        errors="coerce"
    ).fillna(0.0).sum()

    total_trades = len(
        closed
    )

    wins = int(
        (pnl > 0).sum()
    )

    losses = int(
        (pnl < 0).sum()
    )

    breakeven = int(
        (pnl == 0).sum()
    )

    win_rate = (
        wins
        /
        total_trades
        *
        100
    )

    loss_rate = (
        losses
        /
        total_trades
        *
        100
    )

    net_pnl = pnl.sum()

    net_return = (
        net_pnl
        /
        INITIAL_CAPITAL
        *
        100
    )

    winning_values = pnl[
        pnl > 0
    ]

    losing_values = pnl[
        pnl < 0
    ]

    average_winner = (
        winning_values.mean()
        if not winning_values.empty
        else 0.0
    )

    average_loser = (
        losing_values.mean()
        if not losing_values.empty
        else 0.0
    )

    gross_profit = winning_values.sum()

    gross_loss = abs(
        losing_values.sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            /
            gross_loss
        )

    else:

        profit_factor = 0.0

    average_holding = pd.to_numeric(
        closed["Bars Held"],
        errors="coerce"
    ).fillna(0).mean()

    peak_equity = INITIAL_CAPITAL

    max_drawdown = 0.0

    max_drawdown_percent = 0.0

    if not equity.empty:

        peak_equity = safe_float(
            equity["Peak Equity"].max(),
            INITIAL_CAPITAL
        )

        max_drawdown = safe_float(
            equity["Drawdown"].min(),
            0.0
        )

        max_drawdown_percent = safe_float(
            equity["Drawdown %"].min(),
            0.0
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
            win_rate,

        "Loss Rate %":
            loss_rate,

        "Gross PnL":
            gross_pnl,

        "Transaction Costs":
            transaction_costs,

        "Net PnL":
            net_pnl,

        "Net Return %":
            net_return,

        "Peak Equity":
            peak_equity,

        "Maximum Drawdown":
            max_drawdown,

        "Maximum Drawdown %":
            max_drawdown_percent,

        "Average Trade PnL":
            pnl.mean(),

        "Average Winning Trade":
            average_winner,

        "Average Losing Trade":
            average_loser,

        "Largest Win":
            pnl.max(),

        "Largest Loss":
            pnl.min(),

        "Profit Factor":
            profit_factor,

        "Average Holding Bars":
            average_holding,

    }


# ============================================================
# PRINT TRADE TABLE
# ============================================================

def print_trade_table(
    trades
):

    print()
    print(
        "=" * 120
    )

    print(
        "HISTORICAL TRADE RESULTS"
    )

    print(
        "=" * 120
    )

    if trades.empty:

        print(
            "No historical trades generated."
        )

        return

    columns = [
        "Stock",
        "Signal",
        "Signal Date",
        "Entry Price",
        "Shares",
        "Exit Date",
        "Exit Price",
        "Bars Held",
        "Exit Reason",
        "Net PnL",
        "Return %"
    ]

    available = [
        column
        for column in columns
        if column in trades.columns
    ]

    display_df = trades[
        available
    ].copy()

    for column in [
        "Signal Date",
        "Exit Date"
    ]:

        if column in display_df.columns:

            display_df[column] = (
                pd.to_datetime(
                    display_df[column],
                    errors="coerce"
                )
                .dt.strftime(
                    "%Y-%m-%d"
                )
            )

    print(
        display_df.to_string(
            index=False
        )
    )


# ============================================================
# PRINT STOCK SUMMARY
# ============================================================

def print_stock_summary(
    summary
):

    print()
    print(
        "=" * 100
    )

    print(
        "STOCK PERFORMANCE"
    )

    print(
        "=" * 100
    )

    if summary.empty:

        print(
            "No stock-level results available."
        )

        return

    print(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(
    report
):

    print()
    print(
        "=" * 100
    )

    print(
        "STAGE 13 - HISTORICAL BACKTEST RESULTS"
    )

    print(
        "=" * 100
    )

    print(
        f"Initial Capital        : "
        f"Rs. {report['Initial Capital']:,.2f}"
    )

    print(
        f"Total Trades           : "
        f"{report['Total Trades']}"
    )

    print(
        f"Winning Trades         : "
        f"{report['Winning Trades']}"
    )

    print(
        f"Losing Trades          : "
        f"{report['Losing Trades']}"
    )

    print(
        f"Break-even Trades      : "
        f"{report['Break-even Trades']}"
    )

    print(
        f"Win Rate               : "
        f"{report['Win Rate %']:.2f}%"
    )

    print(
        f"Loss Rate              : "
        f"{report['Loss Rate %']:.2f}%"
    )

    print()

    print(
        f"Gross PnL              : "
        f"Rs. {report['Gross PnL']:,.2f}"
    )

    print(
        f"Transaction Costs      : "
        f"Rs. {report['Transaction Costs']:,.2f}"
    )

    print(
        f"Net PnL                : "
        f"Rs. {report['Net PnL']:,.2f}"
    )

    print(
        f"Net Return             : "
        f"{report['Net Return %']:.2f}%"
    )

    print()

    print(
        f"Peak Equity            : "
        f"Rs. {report['Peak Equity']:,.2f}"
    )

    print(
        f"Maximum Drawdown       : "
        f"Rs. {report['Maximum Drawdown']:,.2f}"
    )

    print(
        f"Maximum Drawdown %     : "
        f"{report['Maximum Drawdown %']:.2f}%"
    )

    print()

    print(
        f"Average Trade PnL      : "
        f"Rs. {report['Average Trade PnL']:,.2f}"
    )

    print(
        f"Average Winning Trade  : "
        f"Rs. {report['Average Winning Trade']:,.2f}"
    )

    print(
        f"Average Losing Trade   : "
        f"Rs. {report['Average Losing Trade']:,.2f}"
    )

    print(
        f"Largest Win            : "
        f"Rs. {report['Largest Win']:,.2f}"
    )

    print(
        f"Largest Loss           : "
        f"Rs. {report['Largest Loss']:,.2f}"
    )

    print(
        f"Profit Factor          : "
        f"{report['Profit Factor']:.2f}"
    )

    print(
        f"Average Holding Bars   : "
        f"{report['Average Holding Bars']:.2f}"
    )

    print(
        "=" * 100
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    trades,
    equity,
    stock_summary,
    report
):

    # --------------------------------------------------------
    # Trades
    # --------------------------------------------------------

    if not trades.empty:

        output = trades.copy()

        for column in [
            "Signal Date",
            "Exit Date"
        ]:

            if column in output.columns:

                output[column] = (
                    pd.to_datetime(
                        output[column],
                        errors="coerce"
                    )
                    .dt.strftime(
                        "%Y-%m-%d"
                    )
                )

        output.to_csv(
            TRADES_FILE,
            index=False
        )

    else:

        pd.DataFrame().to_csv(
            TRADES_FILE,
            index=False
        )

    # --------------------------------------------------------
    # Equity
    # --------------------------------------------------------

    if not equity.empty:

        equity.to_csv(
            EQUITY_FILE,
            index=False
        )

    else:

        pd.DataFrame().to_csv(
            EQUITY_FILE,
            index=False
        )

    # --------------------------------------------------------
    # Stock summary
    # --------------------------------------------------------

    if not stock_summary.empty:

        stock_summary.to_csv(
            STOCK_SUMMARY_FILE,
            index=False
        )

    else:

        pd.DataFrame().to_csv(
            STOCK_SUMMARY_FILE,
            index=False
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_df = pd.DataFrame(
        [
            report
        ]
    )

    report_df.to_csv(
        REPORT_FILE,
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 110
    )

    print(
        "             STAGE 13 - HISTORICAL STRATEGY BACKTEST"
    )

    print(
        "=" * 110
    )

    print()
    print(
        "⚠️ PAPER / RESEARCH BACKTEST ONLY"
    )

    print(
        "⚠️ NO REAL ORDERS WILL BE PLACED"
    )

    print()

    # ========================================================
    # LOCKED SETTINGS
    # ========================================================

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
        f"Initial Capital      : "
        f"Rs. {INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Risk Per Trade       : "
        f"{RISK_PER_TRADE_PERCENT:.2f}%"
    )

    print(
        f"Stop Loss            : "
        f"{STOP_LOSS_PERCENT:.2f}%"
    )

    print(
        f"Take Profit           : "
        f"{TAKE_PROFIT_PERCENT:.2f}%"
    )

    print(
        f"Maximum Position     : "
        f"{MAX_POSITION_PERCENT:.2f}%"
    )

    print(
        f"Entry Slippage       : "
        f"{ENTRY_SLIPPAGE_PERCENT:.2f}%"
    )

    print(
        f"Exit Slippage        : "
        f"{EXIT_SLIPPAGE_PERCENT:.2f}%"
    )

    print(
        f"Transaction Cost     : "
        f"{TRANSACTION_COST_PERCENT:.2f}%"
    )

    print(
        f"Historical Days      : "
        f"{HISTORICAL_DAYS}"
    )

    print()

    print(
        "=" * 110
    )

    # ========================================================
    # LOAD ALL STOCKS
    # ========================================================

    all_candidates = []

    successful_stocks = 0

    for symbol in STOCKS:

        df = load_stock_data(
            symbol
        )

        if df is None:

            continue

        successful_stocks += 1

        candidates = (
            generate_candidate_trades(
                symbol,
                df
            )
        )

        print(
            f"Historical candidate trades "
            f"for {symbol}: {len(candidates)}"
        )

        all_candidates.extend(
            candidates
        )

    # ========================================================
    # CHECK DATA
    # ========================================================

    print()
    print(
        "=" * 110
    )

    print(
        f"Stocks successfully loaded : "
        f"{successful_stocks}/{len(STOCKS)}"
    )

    print(
        f"Candidate trades generated : "
        f"{len(all_candidates)}"
    )

    print(
        "=" * 110
    )

    # ========================================================
    # APPLY PORTFOLIO SIZING
    # ========================================================

    trades = apply_portfolio_sizing(
        all_candidates
    )

    if trades.empty:

        print()
        print(
            "⚠️ No historical trades could be "
            "generated under the locked strategy."
        )

        equity = pd.DataFrame()

        stock_summary = pd.DataFrame()

        report = create_report(
            trades,
            equity
        )

        save_results(
            trades,
            equity,
            stock_summary,
            report
        )

        print_report(
            report
        )

        print()
        print(
            f"Report file : {REPORT_FILE}"
        )

        print(
            "STAGE 13 COMPLETE"
        )

        return

    # ========================================================
    # EQUITY CURVE
    # ========================================================

    equity = create_equity_curve(
        trades
    )

    # ========================================================
    # STOCK SUMMARY
    # ========================================================

    stock_summary = create_stock_summary(
        trades
    )

    # ========================================================
    # PERFORMANCE REPORT
    # ========================================================

    report = create_report(
        trades,
        equity
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_results(
        trades,
        equity,
        stock_summary,
        report
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print_trade_table(
        trades
    )

    print_stock_summary(
        stock_summary
    )

    print_report(
        report
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print(
        "=" * 110
    )

    if report["Total Trades"] == 0:

        print(
            "RESULT: NO HISTORICAL CLOSED TRADES"
        )

        print()
        print(
            "The selected historical data did not "
            "produce closed trades."
        )

    elif report["Net PnL"] > 0:

        print(
            "RESULT: POSITIVE HISTORICAL BACKTEST"
        )

        print()
        print(
            "The locked SMMA 20/120 strategy produced "
            "a positive historical net PnL after "
            "transaction costs and slippage."
        )

    elif report["Net PnL"] < 0:

        print(
            "RESULT: NEGATIVE HISTORICAL BACKTEST"
        )

        print()
        print(
            "The locked SMMA 20/120 strategy produced "
            "a negative historical net PnL after "
            "transaction costs and slippage."
        )

    else:

        print(
            "RESULT: BREAK-EVEN HISTORICAL BACKTEST"
        )

    print()
    print(
        "No strategy parameters were changed."
    )

    print(
        "No real order has been placed."
    )

    print(
        "=" * 110
    )

    print()
    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 110
    )

    print(
        f"Trades       : {TRADES_FILE}"
    )

    print(
        f"Equity Curve : {EQUITY_FILE}"
    )

    print(
        f"Stock Summary: {STOCK_SUMMARY_FILE}"
    )

    print(
        f"Report       : {REPORT_FILE}"
    )

    print(
        "=" * 110
    )

    print()
    print(
        "STAGE 13 COMPLETE"
    )

    print(
        "=" * 110
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()