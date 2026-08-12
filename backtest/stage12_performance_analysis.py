"""
============================================================
STAGE 12 - PERFORMANCE & RISK ANALYSIS
============================================================

PURPOSE
-------
Analyze the results produced by Stage 11.

INPUT
-----
data/paper_trade_history.csv
data/paper_portfolio.csv

OUTPUT
------
data/stage12_performance_report.csv

IMPORTANT
---------
This stage is ANALYSIS ONLY.
No real orders are placed.
No strategy parameters are changed.
"""


# ============================================================
# IMPORTS
# ============================================================

import os
import math
import numpy as np
import pandas as pd


# ============================================================
# FILES
# ============================================================

DATA_DIRECTORY = "data"

TRADE_HISTORY_FILE = os.path.join(
    DATA_DIRECTORY,
    "paper_trade_history.csv"
)

PORTFOLIO_FILE = os.path.join(
    DATA_DIRECTORY,
    "paper_portfolio.csv"
)

REPORT_FILE = os.path.join(
    DATA_DIRECTORY,
    "stage12_performance_report.csv"
)


# ============================================================
# LOCKED CAPITAL
# ============================================================

INITIAL_CAPITAL = 100000.0


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value, default=0.0):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


# ============================================================
# LOAD TRADE HISTORY
# ============================================================

def load_trade_history():

    if not os.path.exists(
        TRADE_HISTORY_FILE
    ):

        print()
        print(
            "⚠️ Trade history file not found:"
        )

        print(
            f"   {TRADE_HISTORY_FILE}"
        )

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            TRADE_HISTORY_FILE
        )

    except Exception as error:

        print(
            f"❌ Could not read trade history: "
            f"{error}"
        )

        return pd.DataFrame()

    if df.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Normalize columns
    # --------------------------------------------------------

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    for column in [
        "Signal Date",
        "Entry Date",
        "Exit Date"
    ]:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Convert numeric fields
    # --------------------------------------------------------

    numeric_columns = [
        "Entry Price",
        "Exit Price",
        "Shares",
        "Bars Held",
        "Gross PnL",
        "Transaction Costs",
        "Net PnL",
        "Return %"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# LOAD PORTFOLIO
# ============================================================

def load_portfolio():

    if not os.path.exists(
        PORTFOLIO_FILE
    ):

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            PORTFOLIO_FILE
        )

        return df

    except Exception as error:

        print(
            f"⚠️ Could not read portfolio: "
            f"{error}"
        )

        return pd.DataFrame()


# ============================================================
# FILTER CLOSED TRADES
# ============================================================

def get_closed_trades(
    history
):

    if history.empty:

        return pd.DataFrame()

    if "Status" not in history.columns:

        return pd.DataFrame()

    closed = history[
        history["Status"]
        .astype(str)
        .str.upper()
        .eq("CLOSED")
    ].copy()

    if closed.empty:

        return closed

    if "Net PnL" in closed.columns:

        closed["Net PnL"] = pd.to_numeric(
            closed["Net PnL"],
            errors="coerce"
        ).fillna(0.0)

    if "Return %" in closed.columns:

        closed["Return %"] = pd.to_numeric(
            closed["Return %"],
            errors="coerce"
        ).fillna(0.0)

    return closed.reset_index(
        drop=True
    )


# ============================================================
# CALCULATE BASIC STATISTICS
# ============================================================

def calculate_trade_statistics(
    closed
):

    if closed.empty:

        return {

            "Total Trades": 0,
            "Winning Trades": 0,
            "Losing Trades": 0,
            "Break Even Trades": 0,
            "Win Rate %": 0.0,
            "Loss Rate %": 0.0,
            "Average Trade PnL": 0.0,
            "Average Winning Trade": 0.0,
            "Average Losing Trade": 0.0,
            "Largest Win": 0.0,
            "Largest Loss": 0.0,
            "Profit Factor": 0.0

        }

    pnl = closed[
        "Net PnL"
    ].astype(float)

    wins = pnl[
        pnl > 0
    ]

    losses = pnl[
        pnl < 0
    ]

    break_even = pnl[
        pnl == 0
    ]

    total_trades = len(pnl)

    winning_trades = len(wins)

    losing_trades = len(losses)

    break_even_trades = len(
        break_even
    )

    if total_trades > 0:

        win_rate = (
            winning_trades
            /
            total_trades
            *
            100
        )

        loss_rate = (
            losing_trades
            /
            total_trades
            *
            100
        )

    else:

        win_rate = 0.0
        loss_rate = 0.0

    average_trade = pnl.mean()

    average_win = (
        wins.mean()
        if not wins.empty
        else 0.0
    )

    average_loss = (
        losses.mean()
        if not losses.empty
        else 0.0
    )

    largest_win = (
        wins.max()
        if not wins.empty
        else 0.0
    )

    largest_loss = (
        losses.min()
        if not losses.empty
        else 0.0
    )

    gross_profit = (
        wins.sum()
        if not wins.empty
        else 0.0
    )

    gross_loss = abs(
        losses.sum()
    ) if not losses.empty else 0.0

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            /
            gross_loss
        )

    else:

        profit_factor = 0.0

    return {

        "Total Trades":
            total_trades,

        "Winning Trades":
            winning_trades,

        "Losing Trades":
            losing_trades,

        "Break Even Trades":
            break_even_trades,

        "Win Rate %":
            win_rate,

        "Loss Rate %":
            loss_rate,

        "Average Trade PnL":
            average_trade,

        "Average Winning Trade":
            average_win,

        "Average Losing Trade":
            average_loss,

        "Largest Win":
            largest_win,

        "Largest Loss":
            largest_loss,

        "Profit Factor":
            profit_factor

    }


# ============================================================
# CALCULATE NET RETURN
# ============================================================

def calculate_return(
    closed
):

    if closed.empty:

        return {

            "Gross PnL": 0.0,
            "Transaction Costs": 0.0,
            "Net PnL": 0.0,
            "Net Return %": 0.0

        }

    gross_pnl = safe_float(
        closed["Gross PnL"].sum()
    )

    transaction_costs = safe_float(
        closed["Transaction Costs"].sum()
    )

    net_pnl = safe_float(
        closed["Net PnL"].sum()
    )

    net_return = (
        net_pnl
        /
        INITIAL_CAPITAL
        *
        100
    )

    return {

        "Gross PnL":
            gross_pnl,

        "Transaction Costs":
            transaction_costs,

        "Net PnL":
            net_pnl,

        "Net Return %":
            net_return

    }


# ============================================================
# EQUITY CURVE
# ============================================================

def build_equity_curve(
    closed
):

    if closed.empty:

        return pd.DataFrame({

            "Trade Number": [0],

            "Net PnL": [0.0],

            "Equity": [
                INITIAL_CAPITAL
            ]

        })

    working = closed.copy()

    # --------------------------------------------------------
    # Sort by exit date
    # --------------------------------------------------------

    if "Exit Date" in working.columns:

        working = working.sort_values(
            "Exit Date"
        )

    working = working.reset_index(
        drop=True
    )

    working["Trade Number"] = (
        np.arange(
            1,
            len(working) + 1
        )
    )

    working["Cumulative PnL"] = (
        working["Net PnL"]
        .cumsum()
    )

    working["Equity"] = (
        INITIAL_CAPITAL
        +
        working["Cumulative PnL"]
    )

    return working[
        [
            "Trade Number",
            "Net PnL",
            "Equity"
        ]
    ]


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================

def calculate_drawdown(
    equity_curve
):

    if equity_curve.empty:

        return {

            "Peak Equity": INITIAL_CAPITAL,
            "Maximum Drawdown": 0.0,
            "Maximum Drawdown %": 0.0

        }

    equity = equity_curve[
        "Equity"
    ].astype(float)

    running_peak = equity.cummax()

    drawdown = (
        equity
        -
        running_peak
    )

    drawdown_percent = (
        drawdown
        /
        running_peak
        *
        100
    )

    maximum_drawdown = drawdown.min()

    maximum_drawdown_percent = (
        drawdown_percent.min()
    )

    peak_equity = running_peak.max()

    return {

        "Peak Equity":
            float(peak_equity),

        "Maximum Drawdown":
            float(maximum_drawdown),

        "Maximum Drawdown %":
            float(maximum_drawdown_percent)

    }


# ============================================================
# CALCULATE SHARPE-STYLE METRIC
# ============================================================

def calculate_sharpe(
    closed
):

    if closed.empty:

        return 0.0

    returns = (
        closed["Return %"]
        .astype(float)
        /
        100.0
    )

    if len(returns) < 2:

        return 0.0

    standard_deviation = returns.std(
        ddof=1
    )

    if standard_deviation == 0:

        return 0.0

    # Trade-level Sharpe-style metric.
    # This is NOT an annualized Sharpe ratio.

    sharpe = (
        returns.mean()
        /
        standard_deviation
        *
        math.sqrt(
            len(returns)
        )
    )

    return float(sharpe)


# ============================================================
# CALCULATE AVERAGE HOLDING PERIOD
# ============================================================

def calculate_holding_period(
    closed
):

    if closed.empty:

        return 0.0

    if "Bars Held" not in closed.columns:

        return 0.0

    bars = pd.to_numeric(
        closed["Bars Held"],
        errors="coerce"
    ).dropna()

    if bars.empty:

        return 0.0

    return float(
        bars.mean()
    )


# ============================================================
# STOCK LEVEL ANALYSIS
# ============================================================

def calculate_stock_statistics(
    closed
):

    if closed.empty:

        return pd.DataFrame()

    if "Stock" not in closed.columns:

        return pd.DataFrame()

    rows = []

    for stock, group in closed.groupby(
        "Stock"
    ):

        pnl = pd.to_numeric(
            group["Net PnL"],
            errors="coerce"
        ).fillna(0.0)

        wins = (
            pnl > 0
        ).sum()

        losses = (
            pnl < 0
        ).sum()

        total = len(
            pnl
        )

        if total > 0:

            win_rate = (
                wins
                /
                total
                *
                100
            )

        else:

            win_rate = 0.0

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
                win_rate,

            "Net PnL":
                pnl.sum(),

            "Average PnL":
                pnl.mean()

        })

    result = pd.DataFrame(
        rows
    )

    if not result.empty:

        result = result.sort_values(
            "Net PnL",
            ascending=False
        )

    return result


# ============================================================
# EXIT REASON ANALYSIS
# ============================================================

def calculate_exit_statistics(
    closed
):

    if closed.empty:

        return pd.DataFrame()

    if "Exit Reason" not in closed.columns:

        return pd.DataFrame()

    rows = []

    for reason, group in closed.groupby(
        "Exit Reason"
    ):

        pnl = pd.to_numeric(
            group["Net PnL"],
            errors="coerce"
        ).fillna(0.0)

        rows.append({

            "Exit Reason":
                reason,

            "Trades":
                len(group),

            "Net PnL":
                pnl.sum(),

            "Average PnL":
                pnl.mean()

        })

    return pd.DataFrame(
        rows
    ).sort_values(
        "Net PnL",
        ascending=False
    )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    metrics
):

    rows = []

    for metric, value in metrics.items():

        rows.append({

            "Metric":
                metric,

            "Value":
                value

        })

    report = pd.DataFrame(
        rows
    )

    report.to_csv(
        REPORT_FILE,
        index=False
    )


# ============================================================
# PRINT TRADE STATISTICS
# ============================================================

def print_trade_statistics(
    statistics
):

    print()
    print(
        "=" * 100
    )

    print(
        "TRADE PERFORMANCE"
    )

    print(
        "=" * 100
    )

    print(
        f"Total Trades          : "
        f"{statistics['Total Trades']}"
    )

    print(
        f"Winning Trades        : "
        f"{statistics['Winning Trades']}"
    )

    print(
        f"Losing Trades         : "
        f"{statistics['Losing Trades']}"
    )

    print(
        f"Break-even Trades     : "
        f"{statistics['Break Even Trades']}"
    )

    print(
        f"Win Rate              : "
        f"{statistics['Win Rate %']:.2f}%"
    )

    print(
        f"Loss Rate             : "
        f"{statistics['Loss Rate %']:.2f}%"
    )

    print(
        f"Average Trade PnL     : "
        f"Rs. {statistics['Average Trade PnL']:,.2f}"
    )

    print(
        f"Average Winning Trade : "
        f"Rs. {statistics['Average Winning Trade']:,.2f}"
    )

    print(
        f"Average Losing Trade  : "
        f"Rs. {statistics['Average Losing Trade']:,.2f}"
    )

    print(
        f"Largest Win           : "
        f"Rs. {statistics['Largest Win']:,.2f}"
    )

    print(
        f"Largest Loss          : "
        f"Rs. {statistics['Largest Loss']:,.2f}"
    )

    print(
        f"Profit Factor         : "
        f"{statistics['Profit Factor']:.2f}"
    )


# ============================================================
# PRINT RISK STATISTICS
# ============================================================

def print_risk_statistics(
    returns,
    drawdown,
    sharpe,
    average_holding
):

    print()
    print(
        "=" * 100
    )

    print(
        "RISK & RETURN ANALYSIS"
    )

    print(
        "=" * 100
    )

    print(
        f"Gross PnL             : "
        f"Rs. {returns['Gross PnL']:,.2f}"
    )

    print(
        f"Transaction Costs     : "
        f"Rs. {returns['Transaction Costs']:,.2f}"
    )

    print(
        f"Net PnL               : "
        f"Rs. {returns['Net PnL']:,.2f}"
    )

    print(
        f"Net Return            : "
        f"{returns['Net Return %']:.2f}%"
    )

    print(
        f"Peak Equity           : "
        f"Rs. {drawdown['Peak Equity']:,.2f}"
    )

    print(
        f"Maximum Drawdown      : "
        f"Rs. {drawdown['Maximum Drawdown']:,.2f}"
    )

    print(
        f"Maximum Drawdown %    : "
        f"{drawdown['Maximum Drawdown %']:.2f}%"
    )

    print(
        f"Sharpe-style Metric   : "
        f"{sharpe:.3f}"
    )

    print(
        f"Average Holding Bars  : "
        f"{average_holding:.2f}"
    )


# ============================================================
# PRINT STOCK ANALYSIS
# ============================================================

def print_stock_analysis(
    stock_statistics
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

    if stock_statistics.empty:

        print(
            "No closed trades available."
        )

        return

    print(
        stock_statistics.to_string(
            index=False
        )
    )


# ============================================================
# PRINT EXIT ANALYSIS
# ============================================================

def print_exit_analysis(
    exit_statistics
):

    print()
    print(
        "=" * 100
    )

    print(
        "EXIT REASON ANALYSIS"
    )

    print(
        "=" * 100
    )

    if exit_statistics.empty:

        print(
            "No closed trades available."
        )

        return

    print(
        exit_statistics.to_string(
            index=False
        )
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
        "          STAGE 12 - PERFORMANCE & RISK ANALYSIS"
    )

    print(
        "=" * 100
    )

    print()
    print(
        "PAPER TRADING ANALYSIS ONLY"
    )

    print(
        "NO REAL ORDERS WILL BE PLACED"
    )

    print()
    print(
        f"Initial Capital : "
        f"Rs. {INITIAL_CAPITAL:,.2f}"
    )

    print(
        "=" * 100
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    history = load_trade_history()

    portfolio = load_portfolio()

    print()

    print(
        f"Trade history records : "
        f"{len(history)}"
    )

    print(
        f"Portfolio records     : "
        f"{len(portfolio)}"
    )

    # ========================================================
    # CLOSED TRADES
    # ========================================================

    closed = get_closed_trades(
        history
    )

    print(
        f"Closed trades         : "
        f"{len(closed)}"
    )

    # ========================================================
    # CALCULATE
    # ========================================================

    trade_statistics = (
        calculate_trade_statistics(
            closed
        )
    )

    return_statistics = (
        calculate_return(
            closed
        )
    )

    equity_curve = (
        build_equity_curve(
            closed
        )
    )

    drawdown_statistics = (
        calculate_drawdown(
            equity_curve
        )
    )

    sharpe = calculate_sharpe(
        closed
    )

    average_holding = (
        calculate_holding_period(
            closed
        )
    )

    stock_statistics = (
        calculate_stock_statistics(
            closed
        )
    )

    exit_statistics = (
        calculate_exit_statistics(
            closed
        )
    )

    # ========================================================
    # PRINT
    # ========================================================

    print_trade_statistics(
        trade_statistics
    )

    print_risk_statistics(
        return_statistics,
        drawdown_statistics,
        sharpe,
        average_holding
    )

    print_stock_analysis(
        stock_statistics
    )

    print_exit_analysis(
        exit_statistics
    )

    # ========================================================
    # BUILD FINAL METRICS
    # ========================================================

    metrics = {

        "Initial Capital":
            INITIAL_CAPITAL,

        "Total Trades":
            trade_statistics[
                "Total Trades"
            ],

        "Winning Trades":
            trade_statistics[
                "Winning Trades"
            ],

        "Losing Trades":
            trade_statistics[
                "Losing Trades"
            ],

        "Break Even Trades":
            trade_statistics[
                "Break Even Trades"
            ],

        "Win Rate %":
            trade_statistics[
                "Win Rate %"
            ],

        "Average Trade PnL":
            trade_statistics[
                "Average Trade PnL"
            ],

        "Average Winning Trade":
            trade_statistics[
                "Average Winning Trade"
            ],

        "Average Losing Trade":
            trade_statistics[
                "Average Losing Trade"
            ],

        "Largest Win":
            trade_statistics[
                "Largest Win"
            ],

        "Largest Loss":
            trade_statistics[
                "Largest Loss"
            ],

        "Profit Factor":
            trade_statistics[
                "Profit Factor"
            ],

        "Gross PnL":
            return_statistics[
                "Gross PnL"
            ],

        "Transaction Costs":
            return_statistics[
                "Transaction Costs"
            ],

        "Net PnL":
            return_statistics[
                "Net PnL"
            ],

        "Net Return %":
            return_statistics[
                "Net Return %"
            ],

        "Peak Equity":
            drawdown_statistics[
                "Peak Equity"
            ],

        "Maximum Drawdown":
            drawdown_statistics[
                "Maximum Drawdown"
            ],

        "Maximum Drawdown %":
            drawdown_statistics[
                "Maximum Drawdown %"
            ],

        "Sharpe Style Metric":
            sharpe,

        "Average Holding Bars":
            average_holding,

        "Final Portfolio Value":
            (
                INITIAL_CAPITAL
                +
                return_statistics[
                    "Net PnL"
                ]
            )

    }

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        DATA_DIRECTORY,
        exist_ok=True
    )

    save_report(
        metrics
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        "STAGE 12 - FINAL VALIDATION"
    )

    print(
        "=" * 100
    )

    if closed.empty:

        print()
        print(
            "RESULT: NO CLOSED TRADES"
        )

        print()
        print(
            "Stage 11 currently contains no closed "
            "paper trades."
        )

        print(
            "Performance metrics are therefore "
            "mostly zero."
        )

        print(
            "This is NOT an error."
        )

    else:

        net_pnl = return_statistics[
            "Net PnL"
        ]

        if net_pnl > 0:

            print()
            print(
                "RESULT: POSITIVE PERFORMANCE"
            )

            print(
                "The current closed-trade sample "
                "has positive net PnL."
            )

        elif net_pnl < 0:

            print()
            print(
                "RESULT: NEGATIVE PERFORMANCE"
            )

            print(
                "The current closed-trade sample "
                "has negative net PnL."
            )

        else:

            print()
            print(
                "RESULT: BREAK-EVEN PERFORMANCE"
            )

    print()
    print(
        f"Report file: "
        f"{REPORT_FILE}"
    )

    print()
    print(
        "No strategy parameters were changed."
    )

    print(
        "No real order has been placed."
    )

    print()
    print(
        "=" * 100
    )

    print(
        "STAGE 12 COMPLETE"
    )

    print(
        "=" * 100
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()