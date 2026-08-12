"""
============================================================
STAGE 6
REALISTIC COST + SLIPPAGE BACKTEST
============================================================

Stage 5 selected parameters:

    Fast SMMA       = 20
    Slow SMMA       = 120
    Holding Period  = 60 trading days

Stage 6 adds:

    - Brokerage
    - STT
    - Exchange transaction charges
    - SEBI charges
    - Stamp duty
    - GST
    - Entry slippage
    - Exit slippage

IMPORTANT:
This is a research/backtest model.
SELL crossover trades are evaluated directionally as
short trades. Actual Indian cash-equity short-selling
execution has additional restrictions.

============================================================
"""

import pandas as pd

from api.historical_data import get_historical_data
from indicators.smma import calculate_smma


# ============================================================
# STAGE 6 CONFIGURATION
# ============================================================

FAST_SMMA = 20
SLOW_SMMA = 120

HOLDING_PERIOD = 60

HISTORICAL_DAYS = 500


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
# COST CONFIGURATION
# ============================================================

# Brokerage
# 0.0 means zero brokerage.
# Change only if your broker charges brokerage.
BROKERAGE_RATE = 0.0


# STT for equity delivery
STT_BUY_RATE = 0.0010
STT_SELL_RATE = 0.0010


# Exchange transaction charge
EXCHANGE_TRANSACTION_RATE = 0.0000307


# SEBI turnover fee
SEBI_RATE = 0.000001


# Stamp duty on buy side
STAMP_DUTY_RATE = 0.00015


# GST
GST_RATE = 0.18


# Slippage
ENTRY_SLIPPAGE_RATE = 0.0010
EXIT_SLIPPAGE_RATE = 0.0010


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_stage6_data(symbol):
    """
    Load historical data and calculate SMMA20/SMMA120.

    Automatically handles different column names such as:

        Date
        date
        DATE
        Close
        close
        CLOSE

    Also handles common alternatives such as:

        datetime
        timestamp
        ltp
    """

    print()
    print("=" * 70)
    print(
        f"Getting historical data for {symbol}..."
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Get historical data
    # --------------------------------------------------------

    df = get_historical_data(
        symbol,
        days=HISTORICAL_DAYS
    )

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
    # Make copy
    # --------------------------------------------------------

    df = df.copy()

    # --------------------------------------------------------
    # Display actual columns
    # --------------------------------------------------------

    print(
        f"{symbol}: Available columns: "
        f"{list(df.columns)}"
    )

    # ========================================================
    # FIND DATE COLUMN
    # ========================================================

    date_column = None

    for column in df.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
        )

        if normalized in [
            "date",
            "datetime",
            "timestamp",
            "time"
        ]:

            date_column = column
            break

    # ========================================================
    # FIND CLOSE COLUMN
    # ========================================================

    close_column = None

    for column in df.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
        )

        if normalized in [
            "close",
            "closing_price",
            "close_price",
            "ltp"
        ]:

            close_column = column
            break

    # ========================================================
    # VALIDATE DATE
    # ========================================================

    if date_column is None:

        print()
        print(
            f"{symbol}: ERROR - Date column not found."
        )

        print(
            f"{symbol}: Available columns:"
        )

        for column in df.columns:

            print(
                f"    {column}"
            )

        return None

    # ========================================================
    # VALIDATE CLOSE
    # ========================================================

    if close_column is None:

        print()
        print(
            f"{symbol}: ERROR - Close column not found."
        )

        print(
            f"{symbol}: Available columns:"
        )

        for column in df.columns:

            print(
                f"    {column}"
            )

        return None

    print()
    print(
        f"{symbol}: Date column  = "
        f"{date_column}"
    )

    print(
        f"{symbol}: Close column = "
        f"{close_column}"
    )

    # ========================================================
    # STANDARDIZE COLUMNS
    # ========================================================

    df["Date"] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df["Close"] = pd.to_numeric(
        df[close_column],
        errors="coerce"
    )

    # ========================================================
    # REMOVE INVALID DATA
    # ========================================================

    df = df.dropna(
        subset=[
            "Date",
            "Close"
        ]
    ).copy()

    # ========================================================
    # SORT DATA
    # ========================================================

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print(
        f"{symbol}: Clean rows: "
        f"{len(df)}"
    )

    # ========================================================
    # CHECK MINIMUM DATA
    # ========================================================

    if len(df) <= SLOW_SMMA:

        print(
            f"{symbol}: Not enough data for "
            f"SMMA{SLOW_SMMA}"
        )

        return None

    # ========================================================
    # CALCULATE SMMA20
    # ========================================================

    df["SMMA20"] = calculate_smma(
        df["Close"],
        FAST_SMMA
    )

    # ========================================================
    # CALCULATE SMMA120
    # ========================================================

    df["SMMA120"] = calculate_smma(
        df["Close"],
        SLOW_SMMA
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

    # ========================================================
    # SELL CROSSOVER
    # ========================================================

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

    # ========================================================
    # REMOVE SMMA NaN ROWS
    # ========================================================

    df = df.dropna(
        subset=[
            "SMMA20",
            "SMMA120",
            "Previous_SMMA20",
            "Previous_SMMA120"
        ]
    ).reset_index(drop=True)

    # ========================================================
    # CROSSOVER COUNTS
    # ========================================================

    buy_count = int(
        df["BUY_CROSSOVER"].sum()
    )

    sell_count = int(
        df["SELL_CROSSOVER"].sum()
    )

    print()
    print(
        f"{symbol}: SMMA calculation completed."
    )

    print(
        f"{symbol}: BUY crossovers  = "
        f"{buy_count}"
    )

    print(
        f"{symbol}: SELL crossovers = "
        f"{sell_count}"
    )

    print(
        f"{symbol}: Final usable rows = "
        f"{len(df)}"
    )

    return df


# ============================================================
# APPLY SLIPPAGE
# ============================================================

def apply_slippage(
    entry_price,
    exit_price,
    signal
):
    """
    Apply adverse execution slippage.

    BUY:
        Entry price increases.
        Exit price decreases.

    SELL:
        Entry price decreases.
        Exit price increases.
    """

    if signal == "BUY":

        actual_entry = (
            entry_price
            * (
                1
                + ENTRY_SLIPPAGE_RATE
            )
        )

        actual_exit = (
            exit_price
            * (
                1
                - EXIT_SLIPPAGE_RATE
            )
        )

    elif signal == "SELL":

        actual_entry = (
            entry_price
            * (
                1
                - ENTRY_SLIPPAGE_RATE
            )
        )

        actual_exit = (
            exit_price
            * (
                1
                + EXIT_SLIPPAGE_RATE
            )
        )

    else:

        return None, None

    return (
        actual_entry,
        actual_exit
    )


# ============================================================
# CALCULATE TRANSACTION COSTS
# ============================================================

def calculate_transaction_costs(
    entry_price,
    exit_price,
    shares
):
    """
    Calculate estimated transaction costs.
    """

    entry_value = (
        entry_price
        * shares
    )

    exit_value = (
        exit_price
        * shares
    )

    turnover = (
        entry_value
        + exit_value
    )

    # --------------------------------------------------------
    # Brokerage
    # --------------------------------------------------------

    brokerage = (
        turnover
        * BROKERAGE_RATE
    )

    # --------------------------------------------------------
    # STT
    # --------------------------------------------------------

    stt_buy = (
        entry_value
        * STT_BUY_RATE
    )

    stt_sell = (
        exit_value
        * STT_SELL_RATE
    )

    stt = (
        stt_buy
        + stt_sell
    )

    # --------------------------------------------------------
    # Exchange charges
    # --------------------------------------------------------

    exchange_charges = (
        turnover
        * EXCHANGE_TRANSACTION_RATE
    )

    # --------------------------------------------------------
    # SEBI
    # --------------------------------------------------------

    sebi_charges = (
        turnover
        * SEBI_RATE
    )

    # --------------------------------------------------------
    # Stamp duty
    # --------------------------------------------------------

    stamp_duty = (
        entry_value
        * STAMP_DUTY_RATE
    )

    # --------------------------------------------------------
    # GST
    # --------------------------------------------------------

    gst = (
        brokerage
        + exchange_charges
    ) * GST_RATE

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    total_cost = (
        brokerage
        + stt
        + exchange_charges
        + sebi_charges
        + stamp_duty
        + gst
    )

    return {
        "Brokerage": brokerage,
        "STT": stt,
        "Exchange Charges": exchange_charges,
        "SEBI Charges": sebi_charges,
        "Stamp Duty": stamp_duty,
        "GST": gst,
        "Total Costs": total_cost
    }


# ============================================================
# CALCULATE ONE TRADE
# ============================================================

def calculate_stage6_trade(
    df,
    crossover_index,
    signal
):
    """
    Calculate one crossover trade.

    Holding period:
        60 trading bars

    Includes:
        - Gross return
        - Slippage
        - Transaction costs
        - Net return
    """

    # --------------------------------------------------------
    # Future exit index
    # --------------------------------------------------------

    exit_index = (
        crossover_index
        + HOLDING_PERIOD
    )

    # --------------------------------------------------------
    # Not enough future data
    # --------------------------------------------------------

    if exit_index >= len(df):

        return None

    # --------------------------------------------------------
    # Raw entry
    # --------------------------------------------------------

    raw_entry_price = float(
        df["Close"].iloc[
            crossover_index
        ]
    )

    # --------------------------------------------------------
    # Raw exit
    # --------------------------------------------------------

    raw_exit_price = float(
        df["Close"].iloc[
            exit_index
        ]
    )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    entry_date = df["Date"].iloc[
        crossover_index
    ]

    exit_date = df["Date"].iloc[
        exit_index
    ]

    # ========================================================
    # GROSS RETURN
    # ========================================================

    if signal == "BUY":

        gross_profit = (
            raw_exit_price
            - raw_entry_price
        )

    elif signal == "SELL":

        gross_profit = (
            raw_entry_price
            - raw_exit_price
        )

    else:

        return None

    gross_return_percent = (
        gross_profit
        / raw_entry_price
    ) * 100

    # ========================================================
    # APPLY SLIPPAGE
    # ========================================================

    actual_entry_price, actual_exit_price = (
        apply_slippage(
            raw_entry_price,
            raw_exit_price,
            signal
        )
    )

    if actual_entry_price is None:

        return None

    # ========================================================
    # PROFIT AFTER SLIPPAGE
    # ========================================================

    if signal == "BUY":

        slippage_adjusted_profit = (
            actual_exit_price
            - actual_entry_price
        )

    else:

        slippage_adjusted_profit = (
            actual_entry_price
            - actual_exit_price
        )

    # --------------------------------------------------------
    # Slippage cost
    # --------------------------------------------------------

    slippage_cost = abs(
        slippage_adjusted_profit
        - gross_profit
    )

    # ========================================================
    # TRANSACTION COSTS
    # ========================================================

    shares = 1

    costs = calculate_transaction_costs(
        entry_price=actual_entry_price,
        exit_price=actual_exit_price,
        shares=shares
    )

    total_costs = costs[
        "Total Costs"
    ]

    # ========================================================
    # NET PROFIT
    # ========================================================

    net_profit = (
        slippage_adjusted_profit
        - total_costs
    )

    net_return_percent = (
        net_profit
        / raw_entry_price
    ) * 100

    # ========================================================
    # RESULT
    # ========================================================

    if net_return_percent > 0:

        result = "PROFITABLE"

    elif net_return_percent < 0:

        result = "FAILED"

    else:

        result = "BREAKEVEN"

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "Entry Date": entry_date,
        "Exit Date": exit_date,
        "Signal": signal,

        "Raw Entry": round(
            raw_entry_price,
            2
        ),

        "Raw Exit": round(
            raw_exit_price,
            2
        ),

        "Actual Entry": round(
            actual_entry_price,
            2
        ),

        "Actual Exit": round(
            actual_exit_price,
            2
        ),

        "Gross Return %": round(
            gross_return_percent,
            2
        ),

        "Slippage Cost": round(
            slippage_cost,
            4
        ),

        "Brokerage": round(
            costs["Brokerage"],
            4
        ),

        "STT": round(
            costs["STT"],
            4
        ),

        "Exchange Charges": round(
            costs["Exchange Charges"],
            4
        ),

        "SEBI Charges": round(
            costs["SEBI Charges"],
            4
        ),

        "Stamp Duty": round(
            costs["Stamp Duty"],
            4
        ),

        "GST": round(
            costs["GST"],
            4
        ),

        "Total Costs": round(
            total_costs,
            4
        ),

        "Net Return %": round(
            net_return_percent,
            2
        ),

        "Result": result,

        "Holding Days": HOLDING_PERIOD
    }


# ============================================================
# ANALYZE ONE STOCK
# ============================================================

def analyze_stock(symbol):

    df = prepare_stage6_data(
        symbol
    )

    if df is None or df.empty:

        summary = {
            "Stock": symbol.replace(
                "-EQ",
                ""
            ),
            "Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "Win Rate": 0.0,
            "Gross Return": 0.0,
            "Net Return": 0.0,
            "Costs": 0.0,
            "Slippage": 0.0
        }

        return summary, []

    trades = []

    # ========================================================
    # FIND CROSSOVERS
    # ========================================================

    for i in range(
        len(df)
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

        # ----------------------------------------------------
        # Calculate trade
        # ----------------------------------------------------

        trade = calculate_stage6_trade(
            df=df,
            crossover_index=i,
            signal=signal
        )

        if trade is not None:

            trades.append(
                trade
            )

    # ========================================================
    # STATISTICS
    # ========================================================

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
        if trade["Net Return %"] <= 0
    )

    if total_trades > 0:

        win_rate = (
            wins
            / total_trades
        ) * 100

    else:

        win_rate = 0.0

    gross_return = sum(
        trade["Gross Return %"]
        for trade in trades
    )

    net_return = sum(
        trade["Net Return %"]
        for trade in trades
    )

    total_costs = sum(
        trade["Total Costs"]
        for trade in trades
    )

    total_slippage = sum(
        trade["Slippage Cost"]
        for trade in trades
    )

    summary = {
        "Stock": symbol.replace(
            "-EQ",
            ""
        ),

        "Trades": total_trades,

        "Wins": wins,

        "Losses": losses,

        "Win Rate": round(
            win_rate,
            2
        ),

        "Gross Return": round(
            gross_return,
            2
        ),

        "Net Return": round(
            net_return,
            2
        ),

        "Costs": round(
            total_costs,
            4
        ),

        "Slippage": round(
            total_slippage,
            4
        )
    }

    return (
        summary,
        trades
    )


# ============================================================
# PRINT TRADE DETAILS
# ============================================================

def print_trade_details(
    symbol,
    trades
):

    print()
    print(
        "=" * 120
    )

    print(
        f"{symbol} - STAGE 6 TRADE DETAILS"
    )

    print(
        "=" * 120
    )

    if not trades:

        print(
            "No usable crossover trades."
        )

        print(
            "=" * 120
        )

        return

    print(
        f"{'Signal':<8}"
        f"{'Entry':>12}"
        f"{'Exit':>12}"
        f"{'Gross %':>12}"
        f"{'Slip.':>12}"
        f"{'Costs':>12}"
        f"{'Net %':>12}"
        f"{'Result':>15}"
    )

    print(
        "-" * 120
    )

    for trade in trades:

        print(
            f"{trade['Signal']:<8}"
            f"₹{trade['Raw Entry']:>10.2f}"
            f"₹{trade['Raw Exit']:>10.2f}"
            f"{trade['Gross Return %']:>10.2f}%"
            f"₹{trade['Slippage Cost']:>10.4f}"
            f"₹{trade['Total Costs']:>10.4f}"
            f"{trade['Net Return %']:>10.2f}%"
            f"{trade['Result']:>15}"
        )

    print(
        "-" * 120
    )

    print(
        "Trade dates:"
    )

    for trade in trades:

        print(
            f"  {trade['Signal']} | "
            f"{trade['Entry Date']} -> "
            f"{trade['Exit Date']}"
        )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

def print_final_summary(
    results
):

    print()
    print(
        "=" * 110
    )

    print(
        "              STAGE 6 - COST + SLIPPAGE RESULTS"
    )

    print(
        "=" * 110
    )

    print()
    print(
        f"SMMA PARAMETERS : "
        f"{FAST_SMMA}/{SLOW_SMMA}"
    )

    print(
        f"HOLDING PERIOD  : "
        f"{HOLDING_PERIOD} trading days"
    )

    print(
        f"ENTRY SLIPPAGE  : "
        f"{ENTRY_SLIPPAGE_RATE * 100:.2f}%"
    )

    print(
        f"EXIT SLIPPAGE   : "
        f"{EXIT_SLIPPAGE_RATE * 100:.2f}%"
    )

    print()

    print(
        f"{'Stock':<15}"
        f"{'Trades':>9}"
        f"{'Wins':>9}"
        f"{'Losses':>9}"
        f"{'Win Rate':>12}"
        f"{'Gross %':>12}"
        f"{'Costs':>14}"
        f"{'Net %':>12}"
    )

    print(
        "-" * 110
    )

    for result in results:

        print(
            f"{result['Stock']:<15}"
            f"{result['Trades']:>9}"
            f"{result['Wins']:>9}"
            f"{result['Losses']:>9}"
            f"{result['Win Rate']:>11.2f}%"
            f"{result['Gross Return']:>11.2f}%"
            f"₹{result['Costs']:>12.4f}"
            f"{result['Net Return']:>11.2f}%"
        )

    print(
        "-" * 110
    )


# ============================================================
# OVERALL STATISTICS
# ============================================================

def print_overall_statistics(
    results
):

    stocks_tested = len(
        results
    )

    total_trades = sum(
        result["Trades"]
        for result in results
    )

    total_wins = sum(
        result["Wins"]
        for result in results
    )

    total_losses = sum(
        result["Losses"]
        for result in results
    )

    total_gross_return = sum(
        result["Gross Return"]
        for result in results
    )

    total_net_return = sum(
        result["Net Return"]
        for result in results
    )

    total_costs = sum(
        result["Costs"]
        for result in results
    )

    total_slippage = sum(
        result["Slippage"]
        for result in results
    )

    if total_trades > 0:

        overall_win_rate = (
            total_wins
            / total_trades
        ) * 100

    else:

        overall_win_rate = 0.0

    cost_impact = (
        total_gross_return
        - total_net_return
    )

    print()
    print(
        "=" * 75
    )

    print(
        "                    OVERALL STATISTICS"
    )

    print(
        "=" * 75
    )

    print(
        f"Stocks Tested          : "
        f"{stocks_tested}"
    )

    print(
        f"Total Trades           : "
        f"{total_trades}"
    )

    print(
        f"Total Wins             : "
        f"{total_wins}"
    )

    print(
        f"Total Losses           : "
        f"{total_losses}"
    )

    print(
        f"Overall Win Rate       : "
        f"{overall_win_rate:.2f}%"
    )

    print(
        f"Gross Return           : "
        f"{total_gross_return:.2f}%"
    )

    print(
        f"Total Transaction Cost : "
        f"₹{total_costs:.4f}"
    )

    print(
        f"Total Slippage Cost    : "
        f"₹{total_slippage:.4f}"
    )

    print(
        f"Net Return             : "
        f"{total_net_return:.2f}%"
    )

    print(
        f"Cost + Slippage Impact : "
        f"{cost_impact:.2f}%"
    )

    print(
        "=" * 75
    )


# ============================================================
# BEST STOCK
# ============================================================

def print_best_stock(
    results
):

    valid_results = [
        result
        for result in results
        if result["Trades"] > 0
    ]

    if not valid_results:

        print()
        print(
            "No valid trades were found."
        )

        return

    best = max(
        valid_results,
        key=lambda x: x["Net Return"]
    )

    print()
    print(
        "=" * 65
    )

    print(
        "             BEST STOCK AFTER COSTS"
    )

    print(
        "=" * 65
    )

    print(
        f"Stock        : "
        f"{best['Stock']}"
    )

    print(
        f"Trades       : "
        f"{best['Trades']}"
    )

    print(
        f"Wins         : "
        f"{best['Wins']}"
    )

    print(
        f"Losses       : "
        f"{best['Losses']}"
    )

    print(
        f"Win Rate     : "
        f"{best['Win Rate']:.2f}%"
    )

    print(
        f"Gross Return : "
        f"{best['Gross Return']:.2f}%"
    )

    print(
        f"Costs        : "
        f"₹{best['Costs']:.4f}"
    )

    print(
        f"Net Return   : "
        f"{best['Net Return']:.2f}%"
    )

    print(
        "=" * 65
    )


# ============================================================
# RUN STAGE 6
# ============================================================

def run_stage6():

    print()
    print(
        "=" * 110
    )

    print(
        "                         STAGE 6"
    )

    print(
        "              REALISTIC COST + SLIPPAGE TEST"
    )

    print(
        "=" * 110
    )

    print()
    print(
        "SELECTED PARAMETERS"
    )

    print(
        "-" * 50
    )

    print(
        f"Fast SMMA       : {FAST_SMMA}"
    )

    print(
        f"Slow SMMA       : {SLOW_SMMA}"
    )

    print(
        f"Holding Period  : "
        f"{HOLDING_PERIOD} trading days"
    )

    print()
    print(
        "COST ASSUMPTIONS"
    )

    print(
        "-" * 50
    )

    print(
        f"Brokerage       : "
        f"{BROKERAGE_RATE * 100:.4f}%"
    )

    print(
        f"STT Buy         : "
        f"{STT_BUY_RATE * 100:.3f}%"
    )

    print(
        f"STT Sell        : "
        f"{STT_SELL_RATE * 100:.3f}%"
    )

    print(
        f"Exchange Charge : "
        f"{EXCHANGE_TRANSACTION_RATE * 100:.5f}%"
    )

    print(
        f"SEBI Charge     : "
        f"{SEBI_RATE * 100:.5f}%"
    )

    print(
        f"Stamp Duty      : "
        f"{STAMP_DUTY_RATE * 100:.3f}%"
    )

    print(
        f"GST             : "
        f"{GST_RATE * 100:.0f}%"
    )

    print()
    print(
        "SLIPPAGE ASSUMPTIONS"
    )

    print(
        "-" * 50
    )

    print(
        f"Entry Slippage  : "
        f"{ENTRY_SLIPPAGE_RATE * 100:.2f}%"
    )

    print(
        f"Exit Slippage   : "
        f"{EXIT_SLIPPAGE_RATE * 100:.2f}%"
    )

    results = []

    # ========================================================
    # PROCESS ALL STOCKS
    # ========================================================

    for symbol in STOCKS:

        try:

            summary, trades = analyze_stock(
                symbol
            )

            results.append(
                summary
            )

            print_trade_details(
                symbol,
                trades
            )

        except Exception as error:

            print()
            print(
                "=" * 70
            )

            print(
                f"{symbol}: ERROR"
            )

            print(
                str(error)
            )

            print(
                "=" * 70
            )

            results.append({
                "Stock": symbol.replace(
                    "-EQ",
                    ""
                ),
                "Trades": 0,
                "Wins": 0,
                "Losses": 0,
                "Win Rate": 0.0,
                "Gross Return": 0.0,
                "Net Return": 0.0,
                "Costs": 0.0,
                "Slippage": 0.0
            })

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print_final_summary(
        results
    )

    print_overall_statistics(
        results
    )

    print_best_stock(
        results
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print(
        "=" * 110
    )

    print(
        "                    STAGE 6 COMPLETE"
    )

    print(
        "=" * 110
    )

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_stage6()