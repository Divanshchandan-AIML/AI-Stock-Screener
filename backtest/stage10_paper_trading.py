"""
============================================================
STAGE 10 - PAPER TRADING / LIVE SIGNAL ENGINE
============================================================

LOCKED STRATEGY
---------------
SMMA FAST  = 20
SMMA SLOW  = 120
HOLDING    = 60 trading days

RISK MANAGEMENT
---------------
Initial reference capital = Rs. 100,000
Risk per trade            = 1%
Stop loss                 = 5%
Take profit               = 10%
Maximum position          = 20%

COST / SLIPPAGE
---------------
Entry slippage            = 0.10%
Exit slippage             = 0.10%
Transaction cost          = 0.05%

IMPORTANT
---------
This is PAPER TRADING ONLY.

NO REAL ORDER IS PLACED.
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

REFERENCE_CAPITAL = 100000.0

RISK_PER_TRADE_PERCENT = 1.0

STOP_LOSS_PERCENT = 5.0

TAKE_PROFIT_PERCENT = 10.0

MAX_POSITION_PERCENT = 20.0


# ============================================================
# COSTS / SLIPPAGE
# ============================================================

ENTRY_SLIPPAGE_PERCENT = 0.10

EXIT_SLIPPAGE_PERCENT = 0.10

TRANSACTION_COST_PERCENT = 0.05


# ============================================================
# OUTPUT FILE
# ============================================================

OUTPUT_DIRECTORY = "data"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "paper_signals.csv"
)


# ============================================================
# NORMALIZE DATA
# ============================================================

def normalize_data(df):
    """
    Make historical data compatible with this script.

    Expected columns from your existing API:

        date
        open
        high
        low
        close
        volume
    """

    if df is None:
        return None

    if df.empty:
        return None

    df = df.copy()

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

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
    # Convert values
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

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
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# PREPARE INDICATORS
# ============================================================

def prepare_indicators(df):
    """
    Calculate locked SMMA 20/120.
    """

    df = df.copy()

    df["SMMA20"] = calculate_smma(
        df["close"],
        FAST_SMMA
    )

    df["SMMA120"] = calculate_smma(
        df["close"],
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

    return df


# ============================================================
# FIND LATEST CROSSOVER
# ============================================================

def find_latest_crossover(df):
    """
    Find the most recent BUY or SELL crossover.
    """

    if df is None or df.empty:
        return None

    crossover_rows = df[
        df["BUY_CROSSOVER"]
        |
        df["SELL_CROSSOVER"]
    ]

    if crossover_rows.empty:
        return None

    latest = crossover_rows.iloc[-1]

    if bool(
        latest["BUY_CROSSOVER"]
    ):

        signal = "BUY"

    elif bool(
        latest["SELL_CROSSOVER"]
    ):

        signal = "SELL"

    else:

        return None

    return {

        "date": latest["date"],

        "signal": signal,

        "close": float(
            latest["close"]
        ),

        "smma20": float(
            latest["SMMA20"]
        ),

        "smma120": float(
            latest["SMMA120"]
        ),

    }


# ============================================================
# CHECK IF SIGNAL IS RECENT
# ============================================================

def signal_is_current(
    signal_date,
    latest_date
):
    """
    A crossover is considered actionable only when it
    occurred on the latest available trading candle.

    This prevents repeatedly treating an old crossover
    as a new signal.
    """

    signal_day = pd.Timestamp(
        signal_date
    ).normalize()

    latest_day = pd.Timestamp(
        latest_date
    ).normalize()

    return signal_day == latest_day


# ============================================================
# APPLY ENTRY SLIPPAGE
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
                / 100
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
                / 100
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
                / 100
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
                / 100
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
                / 100
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
                / 100
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
    # Maximum amount allowed for position
    # --------------------------------------------------------

    maximum_position_value = (

        capital
        *
        MAX_POSITION_PERCENT
        /
        100.0

    )

    # --------------------------------------------------------
    # Maximum amount willing to lose
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

    # --------------------------------------------------------
    # Shares based on risk
    # --------------------------------------------------------

    shares_by_risk = int(

        maximum_risk_amount
        /
        risk_per_share

    )

    # --------------------------------------------------------
    # Shares based on max position
    # --------------------------------------------------------

    shares_by_position = int(

        maximum_position_value
        /
        entry_price

    )

    # --------------------------------------------------------
    # Final position
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
# COST CALCULATION
# ============================================================

def calculate_estimated_cost(
    entry_price,
    shares,
    exit_price
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

    # --------------------------------------------------------
    # Transaction cost
    # --------------------------------------------------------

    transaction_cost = (

        entry_value
        +
        exit_value

    ) * (
        TRANSACTION_COST_PERCENT
        / 100.0
    )

    # --------------------------------------------------------
    # Slippage impact
    # --------------------------------------------------------

    entry_slippage = (

        entry_value
        *
        ENTRY_SLIPPAGE_PERCENT
        /
        100.0

    )

    exit_slippage = (

        exit_value
        *
        EXIT_SLIPPAGE_PERCENT
        /
        100.0

    )

    total_cost = (

        transaction_cost
        +
        entry_slippage
        +
        exit_slippage

    )

    return {

        "entry_value":
            entry_value,

        "exit_value":
            exit_value,

        "transaction_cost":
            transaction_cost,

        "entry_slippage":
            entry_slippage,

        "exit_slippage":
            exit_slippage,

        "total_cost":
            total_cost

    }


# ============================================================
# BUILD PAPER SIGNAL
# ============================================================

def build_paper_signal(
    symbol,
    df,
    crossover
):

    signal = crossover["signal"]

    close_price = crossover["close"]

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

    shares = calculate_position_size(

        REFERENCE_CAPITAL,

        entry_price

    )

    if shares <= 0:

        return None

    # --------------------------------------------------------
    # Cost estimate
    # --------------------------------------------------------

    cost_info = calculate_estimated_cost(

        entry_price,

        shares,

        take_profit

    )

    # --------------------------------------------------------
    # Risk amount
    # --------------------------------------------------------

    risk_per_share = abs(

        entry_price
        -
        stop_loss

    )

    risk_amount = (

        risk_per_share
        *
        shares

    )

    # --------------------------------------------------------
    # Position value
    # --------------------------------------------------------

    position_value = (

        entry_price
        *
        shares

    )

    position_percent = (

        position_value
        /
        REFERENCE_CAPITAL
        *
        100

    )

    # --------------------------------------------------------
    # Potential profit
    # --------------------------------------------------------

    if signal == "BUY":

        potential_profit = (

            take_profit
            -
            entry_price

        ) * shares

    else:

        potential_profit = (

            entry_price
            -
            take_profit

        ) * shares

    # --------------------------------------------------------
    # Potential loss
    # --------------------------------------------------------

    potential_loss = (

        risk_per_share
        *
        shares

    )

    return {

        "Generated At":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "Stock":
            symbol.replace(
                "-EQ",
                ""
            ),

        "Symbol":
            symbol,

        "Signal Date":
            crossover["date"].strftime(
                "%Y-%m-%d"
            ),

        "Signal":
            signal,

        "Close Price":
            close_price,

        "Entry Price":
            entry_price,

        "Stop Loss":
            stop_loss,

        "Take Profit":
            take_profit,

        "Shares":
            shares,

        "Position Value":
            position_value,

        "Position %":
            position_percent,

        "Risk Amount":
            risk_amount,

        "Potential Profit":
            potential_profit,

        "Potential Loss":
            potential_loss,

        "SMMA20":
            crossover["smma20"],

        "SMMA120":
            crossover["smma120"],

        "Transaction Cost":
            cost_info[
                "transaction_cost"
            ],

        "Entry Slippage":
            cost_info[
                "entry_slippage"
            ],

        "Estimated Total Cost":
            cost_info[
                "total_cost"
            ],

        "Status":
            "PAPER_SIGNAL"

    }


# ============================================================
# LOAD EXISTING SIGNALS
# ============================================================

def load_existing_signals():

    if not os.path.exists(
        OUTPUT_FILE
    ):

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            OUTPUT_FILE
        )

        return df

    except Exception as error:

        print(
            f"⚠️ Could not read existing "
            f"signals: {error}"
        )

        return pd.DataFrame()


# ============================================================
# DUPLICATE CHECK
# ============================================================

def signal_already_saved(
    signal,
    existing
):

    if existing.empty:
        return False

    required_columns = [
        "Symbol",
        "Signal Date",
        "Signal"
    ]

    for column in required_columns:

        if column not in existing.columns:

            return False

    matches = existing[

        (existing["Symbol"] == signal["Symbol"])

        &

        (
            existing["Signal Date"]
            ==
            signal["Signal Date"]
        )

        &

        (
            existing["Signal"]
            ==
            signal["Signal"]
        )

    ]

    return not matches.empty


# ============================================================
# SAVE SIGNALS
# ============================================================

def save_signals(
    signals
):

    if not signals:

        print()
        print(
            "No new paper signals to save."
        )

        return

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    existing = load_existing_signals()

    new_signals = []

    for signal in signals:

        if signal_already_saved(
            signal,
            existing
        ):

            print(
                f"⚠️ Duplicate ignored: "
                f"{signal['Symbol']} "
                f"{signal['Signal']} "
                f"{signal['Signal Date']}"
            )

            continue

        new_signals.append(
            signal
        )

    if not new_signals:

        print()
        print(
            "No new signals were added."
        )

        return

    new_df = pd.DataFrame(
        new_signals
    )

    if existing.empty:

        final_df = new_df

    else:

        final_df = pd.concat(

            [
                existing,
                new_df
            ],

            ignore_index=True

        )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        f"✅ Saved {len(new_signals)} "
        f"new paper signal(s)."
    )

    print(
        f"📁 File: {OUTPUT_FILE}"
    )


# ============================================================
# PRINT SIGNAL
# ============================================================

def print_signal(
    signal
):

    print()
    print("=" * 90)

    print(
        f"{signal['Stock']} - "
        f"{signal['Signal']} PAPER SIGNAL"
    )

    print("=" * 90)

    print(
        f"Signal Date          : "
        f"{signal['Signal Date']}"
    )

    print(
        f"Signal               : "
        f"{signal['Signal']}"
    )

    print(
        f"Close Price          : "
        f"Rs. {signal['Close Price']:.2f}"
    )

    print(
        f"Entry Price          : "
        f"Rs. {signal['Entry Price']:.2f}"
    )

    print(
        f"Stop Loss            : "
        f"Rs. {signal['Stop Loss']:.2f}"
    )

    print(
        f"Take Profit          : "
        f"Rs. {signal['Take Profit']:.2f}"
    )

    print(
        f"Shares               : "
        f"{signal['Shares']}"
    )

    print(
        f"Position Value       : "
        f"Rs. {signal['Position Value']:.2f}"
    )

    print(
        f"Position %           : "
        f"{signal['Position %']:.2f}%"
    )

    print(
        f"Risk Amount          : "
        f"Rs. {signal['Risk Amount']:.2f}"
    )

    print(
        f"Potential Profit     : "
        f"Rs. {signal['Potential Profit']:.2f}"
    )

    print(
        f"Potential Loss       : "
        f"Rs. {signal['Potential Loss']:.2f}"
    )

    print(
        f"SMMA20               : "
        f"{signal['SMMA20']:.2f}"
    )

    print(
        f"SMMA120              : "
        f"{signal['SMMA120']:.2f}"
    )

    print(
        f"Transaction Cost     : "
        f"Rs. {signal['Transaction Cost']:.4f}"
    )

    print(
        f"Entry Slippage       : "
        f"Rs. {signal['Entry Slippage']:.4f}"
    )

    print(
        f"Estimated Total Cost : "
        f"Rs. {signal['Estimated Total Cost']:.4f}"
    )

    print(
        f"Status               : "
        f"{signal['Status']}"
    )

    print("=" * 90)


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_stock(
    symbol
):

    print()
    print(
        "-" * 80
    )

    print(
        f"Checking {symbol}..."
    )

    print(
        "-" * 80
    )

    # --------------------------------------------------------
    # Historical data
    # --------------------------------------------------------

    try:

        df = get_historical_data(

            symbol,

            days=250

        )

    except Exception as error:

        print(
            f"❌ Historical data error "
            f"for {symbol}: {error}"
        )

        return None

    if df is None or df.empty:

        print(
            f"❌ No data for {symbol}"
        )

        return None

    print(
        f"Historical rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    df = normalize_data(
        df
    )

    if df is None:

        return None

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    df = prepare_indicators(
        df
    )

    # --------------------------------------------------------
    # Need enough rows
    # --------------------------------------------------------

    usable = df.dropna(
        subset=[
            "SMMA20",
            "SMMA120"
        ]
    )

    if usable.empty:

        print(
            f"❌ No usable SMMA data "
            f"for {symbol}"
        )

        return None

    # --------------------------------------------------------
    # Latest candle
    # --------------------------------------------------------

    latest_date = df[
        "date"
    ].iloc[-1]

    latest_close = float(
        df[
            "close"
        ].iloc[-1]
    )

    latest_smma20 = df[
        "SMMA20"
    ].iloc[-1]

    latest_smma120 = df[
        "SMMA120"
    ].iloc[-1]

    print(
        f"Latest date : "
        f"{latest_date.strftime('%Y-%m-%d')}"
    )

    print(
        f"Close       : "
        f"Rs. {latest_close:.2f}"
    )

    print(
        f"SMMA20      : "
        f"{latest_smma20:.2f}"
    )

    print(
        f"SMMA120     : "
        f"{latest_smma120:.2f}"
    )

    # --------------------------------------------------------
    # Latest crossover
    # --------------------------------------------------------

    crossover = find_latest_crossover(
        df
    )

    if crossover is None:

        print(
            "No crossover found."
        )

        return None

    print(
        f"Latest crossover : "
        f"{crossover['signal']}"
    )

    print(
        f"Crossover date   : "
        f"{crossover['date'].strftime('%Y-%m-%d')}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Only generate a signal when crossover happened
    # on the latest available candle.
    # --------------------------------------------------------

    if not signal_is_current(

        crossover["date"],

        latest_date

    ):

        print(
            "No NEW signal today."
        )

        return None

    # --------------------------------------------------------
    # Build signal
    # --------------------------------------------------------

    signal = build_paper_signal(

        symbol,

        df,

        crossover

    )

    if signal is None:

        print(
            "❌ Could not calculate "
            "valid position size."
        )

        return None

    print_signal(
        signal
    )

    return signal


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print("              STAGE 10 - PAPER TRADING ENGINE")
    print("=" * 100)

    print()
    print("⚠️ PAPER TRADING ONLY")
    print("⚠️ NO REAL ORDERS WILL BE PLACED")
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
        f"Risk Per Trade       : "
        f"{RISK_PER_TRADE_PERCENT:.2f}%"
    )

    print(
        f"Stop Loss            : "
        f"{STOP_LOSS_PERCENT:.2f}%"
    )

    print(
        f"Take Profit          : "
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

    print()
    print(
        "=" * 100
    )

    signals = []

    # ========================================================
    # PROCESS ALL STOCKS
    # ========================================================

    for symbol in STOCKS:

        signal = process_stock(
            symbol
        )

        if signal is not None:

            signals.append(
                signal
            )

    # ========================================================
    # SAVE
    # ========================================================

    save_signals(
        signals
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print("STAGE 10 - FINAL SUMMARY")
    print("=" * 100)

    print(
        f"Stocks checked       : "
        f"{len(STOCKS)}"
    )

    print(
        f"New paper signals    : "
        f"{len(signals)}"
    )

    buy_count = sum(

        1

        for signal in signals

        if signal["Signal"] == "BUY"

    )

    sell_count = sum(

        1

        for signal in signals

        if signal["Signal"] == "SELL"

    )

    print(
        f"BUY signals          : "
        f"{buy_count}"
    )

    print(
        f"SELL signals         : "
        f"{sell_count}"
    )

    print()
    print(
        f"Output file          : "
        f"{OUTPUT_FILE}"
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("=" * 100)

    if signals:

        print(
            "RESULT: PAPER SIGNALS GENERATED"
        )

        print()
        print(
            "Review the signals above."
        )

        print(
            "No real order has been placed."
        )

    else:

        print(
            "RESULT: NO NEW PAPER SIGNALS"
        )

        print()
        print(
            "No SMMA 20/120 crossover occurred "
            "on the latest available candle."
        )

    print()
    print(
        "=" * 100
    )

    print(
        "STAGE 10 COMPLETE"
    )

    print(
        "=" * 100
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()