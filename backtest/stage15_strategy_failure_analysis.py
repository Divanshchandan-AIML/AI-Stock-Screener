"""
============================================================
STAGE 15 - STRATEGY FAILURE ANALYSIS
============================================================

PURPOSE
-------
Stage 15 analyzes the OUT-OF-SAMPLE results produced by
Stage 14.

IMPORTANT
---------
This stage is DIAGNOSTIC ONLY.

It does NOT:
- change SMMA parameters
- optimize the strategy
- modify holding period
- modify stop loss
- modify take profit
- place real orders

LOCKED STRATEGY
---------------
SMMA FAST       = 20
SMMA SLOW       = 120
HOLDING PERIOD  = 60 trading days
STOP LOSS       = 5%
TAKE PROFIT     = 10%

INPUT
-----
Stage 14 output:

data/stage14/walk_forward_trades.csv

OUTPUT
------
data/stage15/
    failure_analysis_summary.csv
    trade_failure_analysis.csv
    stock_failure_analysis.csv
    signal_failure_analysis.csv
    exit_reason_analysis.csv
    holding_period_analysis.csv
    stage15_report.csv

ANALYSIS
--------
1. Overall OOS performance
2. Winning vs losing trades
3. Stock-level failures
4. BUY vs SELL performance
5. Exit reason performance
6. Holding-period behavior
7. Loss concentration
8. Largest losses
9. Largest winners
10. Strategy failure conclusion
"""


# ============================================================
# IMPORTS
# ============================================================

import os
import pandas as pd


# ============================================================
# LOCKED STRATEGY
# ============================================================

FAST_PERIOD = 20
SLOW_PERIOD = 120

HOLDING_PERIOD = 60

STOP_LOSS_PERCENT = 5.0
TAKE_PROFIT_PERCENT = 10.0

INITIAL_CAPITAL = 100000.0


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = os.path.join(
    "data",
    "stage14",
    "walk_forward_trades.csv"
)

OUTPUT_DIRECTORY = os.path.join(
    "data",
    "stage15"
)

FAILURE_SUMMARY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "failure_analysis_summary.csv"
)

TRADE_ANALYSIS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "trade_failure_analysis.csv"
)

STOCK_ANALYSIS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "stock_failure_analysis.csv"
)

SIGNAL_ANALYSIS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "signal_failure_analysis.csv"
)

EXIT_ANALYSIS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "exit_reason_analysis.csv"
)

HOLDING_ANALYSIS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "holding_period_analysis.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "stage15_report.csv"
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
# SAFE INT
# ============================================================

def safe_int(
    value,
    default=0
):

    try:

        if pd.isna(value):
            return default

        return int(float(value))

    except Exception:

        return default


# ============================================================
# LOAD STAGE 14 TRADES
# ============================================================

def load_stage14_trades():

    print()
    print("=" * 100)
    print("LOADING STAGE 14 OUT-OF-SAMPLE TRADES")
    print("=" * 100)

    if not os.path.exists(
        INPUT_FILE
    ):

        print()
        print(
            "❌ Stage 14 trade file not found."
        )

        print(
            f"Expected: {INPUT_FILE}"
        )

        print()
        print(
            "Run Stage 14 first."
        )

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            INPUT_FILE
        )

    except Exception as error:

        print()
        print(
            f"❌ Could not read Stage 14 file: "
            f"{error}"
        )

        return pd.DataFrame()

    if df.empty:

        print(
            "⚠️ Stage 14 trade file is empty."
        )

        return pd.DataFrame()

    print(
        f"Raw Stage 14 records: {len(df)}"
    )

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
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
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print()
        print(
            "❌ Missing required columns:"
        )

        for column in missing:

            print(
                f"   - {column}"
            )

        print()
        print(
            "Stage 15 cannot continue."
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Date conversion
    # --------------------------------------------------------

    for column in [
        "Entry Date",
        "Exit Date"
    ]:

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "Entry Price",
        "Shares",
        "Exit Price",
        "Bars Held",
        "Net PnL",
        "Return %",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Clean strings
    # --------------------------------------------------------

    df["Stock"] = (
        df["Stock"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["Signal"] = (
        df["Signal"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["Exit Reason"] = (
        df["Exit Reason"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "Stock",
            "Signal",
            "Entry Date",
            "Entry Price",
            "Shares",
            "Bars Held",
            "Net PnL",
            "Return %",
        ]
    ).copy()

    # --------------------------------------------------------
    # Only closed OOS trades
    # --------------------------------------------------------

    if "Status" in df.columns:

        status = (
            df["Status"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        closed_mask = (
            status == "CLOSED"
        )

        df = df.loc[
            closed_mask
        ].copy()

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            "Entry Date"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Usable closed OOS trades: {len(df)}"
    )

    return df


# ============================================================
# CLASSIFY TRADE
# ============================================================

def classify_trade(
    pnl
):

    pnl = safe_float(
        pnl
    )

    if pnl > 0:

        return "WIN"

    if pnl < 0:

        return "LOSS"

    return "BREAKEVEN"


# ============================================================
# ADD TRADE ANALYSIS COLUMNS
# ============================================================

def add_trade_analysis(
    df
):

    if df.empty:

        return df

    data = df.copy()

    # --------------------------------------------------------
    # Trade result
    # --------------------------------------------------------

    data["Result"] = (
        data["Net PnL"]
        .apply(
            classify_trade
        )
    )

    # --------------------------------------------------------
    # Absolute loss
    # --------------------------------------------------------

    data["Loss Amount"] = (
        data["Net PnL"]
        .apply(
            lambda x:
            abs(x)
            if safe_float(x) < 0
            else 0.0
        )
    )

    # --------------------------------------------------------
    # Absolute return
    # --------------------------------------------------------

    data["Absolute Return %"] = (
        data["Return %"]
        .abs()
    )

    # --------------------------------------------------------
    # Holding category
    # --------------------------------------------------------

    def holding_category(
        bars
    ):

        bars = safe_int(
            bars
        )

        if bars <= 5:

            return "VERY_SHORT"

        if bars <= 20:

            return "SHORT"

        if bars <= 40:

            return "MEDIUM"

        if bars < HOLDING_PERIOD:

            return "LONG"

        return "MAX_HOLDING"

    data["Holding Category"] = (
        data["Bars Held"]
        .apply(
            holding_category
        )
    )

    # --------------------------------------------------------
    # Exit classification
    # --------------------------------------------------------

    def exit_category(
        reason
    ):

        reason = str(
            reason
        ).upper()

        if "STOP" in reason:

            return "STOP_LOSS"

        if "TAKE" in reason:

            return "TAKE_PROFIT"

        if "TIME" in reason:

            return "TIME_EXIT"

        return "OTHER"

    data["Exit Category"] = (
        data["Exit Reason"]
        .apply(
            exit_category
        )
    )

    # --------------------------------------------------------
    # Direction label
    # --------------------------------------------------------

    data["Direction"] = (
        data["Signal"]
        .map({
            "BUY": "LONG",
            "SELL": "SHORT"
        })
        .fillna("UNKNOWN")
    )

    return data


# ============================================================
# OVERALL ANALYSIS
# ============================================================

def calculate_overall_analysis(
    df
):

    if df.empty:

        return {
            "Metric": "NO DATA",
            "Value": 0
        }

    total_trades = len(
        df
    )

    wins = int(
        (
            df["Result"]
            == "WIN"
        ).sum()
    )

    losses = int(
        (
            df["Result"]
            == "LOSS"
        ).sum()
    )

    breakeven = int(
        (
            df["Result"]
            == "BREAKEVEN"
        ).sum()
    )

    net_pnl = (
        df["Net PnL"]
        .sum()
    )

    gross_positive = (
        df.loc[
            df["Net PnL"] > 0,
            "Net PnL"
        ]
        .sum()
    )

    gross_negative = (
        abs(
            df.loc[
                df["Net PnL"] < 0,
                "Net PnL"
            ]
            .sum()
        )
    )

    if gross_negative > 0:

        profit_factor = (
            gross_positive
            /
            gross_negative
        )

    else:

        profit_factor = 0.0

    if total_trades > 0:

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

        average_pnl = (
            net_pnl
            /
            total_trades
        )

        average_return = (
            df["Return %"]
            .mean()
        )

        average_bars = (
            df["Bars Held"]
            .mean()
        )

    else:

        win_rate = 0.0
        loss_rate = 0.0
        average_pnl = 0.0
        average_return = 0.0
        average_bars = 0.0

    winning_trades = df[
        df["Net PnL"] > 0
    ]

    losing_trades = df[
        df["Net PnL"] < 0
    ]

    average_win = (
        winning_trades["Net PnL"].mean()
        if not winning_trades.empty
        else 0.0
    )

    average_loss = (
        losing_trades["Net PnL"].mean()
        if not losing_trades.empty
        else 0.0
    )

    largest_win = (
        df["Net PnL"].max()
        if not df.empty
        else 0.0
    )

    largest_loss = (
        df["Net PnL"].min()
        if not df.empty
        else 0.0
    )

    # --------------------------------------------------------
    # Loss concentration
    # --------------------------------------------------------

    total_loss = (
        abs(
            df.loc[
                df["Net PnL"] < 0,
                "Net PnL"
            ].sum()
        )
    )

    largest_loss_share = 0.0

    if total_loss > 0:

        largest_loss_share = (
            abs(largest_loss)
            /
            total_loss
            *
            100
        )

    # --------------------------------------------------------
    # Stop-loss rate
    # --------------------------------------------------------

    stop_loss_count = int(
        (
            df["Exit Category"]
            == "STOP_LOSS"
        ).sum()
    )

    stop_loss_rate = (
        stop_loss_count
        /
        total_trades
        *
        100
        if total_trades > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Time exits
    # --------------------------------------------------------

    time_exit_count = int(
        (
            df["Exit Category"]
            == "TIME_EXIT"
        ).sum()
    )

    # --------------------------------------------------------
    # Take-profit count
    # --------------------------------------------------------

    take_profit_count = int(
        (
            df["Exit Category"]
            == "TAKE_PROFIT"
        ).sum()
    )

    # --------------------------------------------------------
    # Determine conclusion
    # --------------------------------------------------------

    if net_pnl < 0:

        conclusion = (
            "NEGATIVE_OOS_PERFORMANCE"
        )

    elif net_pnl > 0:

        conclusion = (
            "POSITIVE_OOS_PERFORMANCE"
        )

    else:

        conclusion = (
            "BREAK_EVEN_OOS_PERFORMANCE"
        )

    return {

        "Initial Capital":
            INITIAL_CAPITAL,

        "Fast SMMA":
            FAST_PERIOD,

        "Slow SMMA":
            SLOW_PERIOD,

        "Holding Period":
            HOLDING_PERIOD,

        "Stop Loss %":
            STOP_LOSS_PERCENT,

        "Take Profit %":
            TAKE_PROFIT_PERCENT,

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

        "Average Trade PnL":
            average_pnl,

        "Average Return %":
            average_return,

        "Average Winning PnL":
            average_win,

        "Average Losing PnL":
            average_loss,

        "Largest Win":
            largest_win,

        "Largest Loss":
            largest_loss,

        "Gross Winning PnL":
            gross_positive,

        "Gross Losing PnL":
            -gross_negative,

        "Net PnL":
            net_pnl,

        "Profit Factor":
            profit_factor,

        "Stop Loss Trades":
            stop_loss_count,

        "Stop Loss Rate %":
            stop_loss_rate,

        "Take Profit Trades":
            take_profit_count,

        "Time Exit Trades":
            time_exit_count,

        "Largest Loss Share %":
            largest_loss_share,

        "Average Holding Bars":
            average_bars,

        "Conclusion":
            conclusion
    }


# ============================================================
# STOCK ANALYSIS
# ============================================================

def analyze_by_stock(
    df
):

    if df.empty:

        return pd.DataFrame()

    rows = []

    for stock, group in df.groupby(
        "Stock"
    ):

        trades = len(
            group
        )

        wins = int(
            (
                group["Result"]
                == "WIN"
            ).sum()
        )

        losses = int(
            (
                group["Result"]
                == "LOSS"
            ).sum()
        )

        net_pnl = (
            group["Net PnL"]
            .sum()
        )

        average_pnl = (
            group["Net PnL"]
            .mean()
        )

        win_rate = (
            wins
            /
            trades
            *
            100
            if trades > 0
            else 0.0
        )

        winning = group[
            group["Net PnL"] > 0
        ]

        losing = group[
            group["Net PnL"] < 0
        ]

        largest_win = (
            winning["Net PnL"].max()
            if not winning.empty
            else 0.0
        )

        largest_loss = (
            losing["Net PnL"].min()
            if not losing.empty
            else 0.0
        )

        stop_losses = int(
            (
                group["Exit Category"]
                == "STOP_LOSS"
            ).sum()
        )

        take_profits = int(
            (
                group["Exit Category"]
                == "TAKE_PROFIT"
            ).sum()
        )

        time_exits = int(
            (
                group["Exit Category"]
                == "TIME_EXIT"
            ).sum()
        )

        rows.append({

            "Stock":
                stock,

            "Trades":
                trades,

            "Wins":
                wins,

            "Losses":
                losses,

            "Win Rate %":
                round(
                    win_rate,
                    2
                ),

            "Net PnL":
                round(
                    net_pnl,
                    2
                ),

            "Average PnL":
                round(
                    average_pnl,
                    2
                ),

            "Largest Win":
                round(
                    largest_win,
                    2
                ),

            "Largest Loss":
                round(
                    largest_loss,
                    2
                ),

            "Stop Loss Trades":
                stop_losses,

            "Take Profit Trades":
                take_profits,

            "Time Exit Trades":
                time_exits,

            "Average Holding Bars":
                round(
                    group["Bars Held"].mean(),
                    2
                )
        })

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        "Net PnL",
        ascending=False
    ).reset_index(
        drop=True
    )


# ============================================================
# BUY / SELL ANALYSIS
# ============================================================

def analyze_by_signal(
    df
):

    if df.empty:

        return pd.DataFrame()

    rows = []

    for signal, group in df.groupby(
        "Signal"
    ):

        trades = len(
            group
        )

        wins = int(
            (
                group["Result"]
                == "WIN"
            ).sum()
        )

        losses = int(
            (
                group["Result"]
                == "LOSS"
            ).sum()
        )

        net_pnl = (
            group["Net PnL"]
            .sum()
        )

        win_rate = (
            wins
            /
            trades
            *
            100
            if trades > 0
            else 0.0
        )

        positive_pnl = (
            group.loc[
                group["Net PnL"] > 0,
                "Net PnL"
            ].sum()
        )

        negative_pnl = abs(
            group.loc[
                group["Net PnL"] < 0,
                "Net PnL"
            ].sum()
        )

        profit_factor = (
            positive_pnl
            /
            negative_pnl
            if negative_pnl > 0
            else 0.0
        )

        rows.append({

            "Signal":
                signal,

            "Trades":
                trades,

            "Wins":
                wins,

            "Losses":
                losses,

            "Win Rate %":
                round(
                    win_rate,
                    2
                ),

            "Net PnL":
                round(
                    net_pnl,
                    2
                ),

            "Average PnL":
                round(
                    group["Net PnL"].mean(),
                    2
                ),

            "Average Return %":
                round(
                    group["Return %"].mean(),
                    2
                ),

            "Largest Win":
                round(
                    group["Net PnL"].max(),
                    2
                ),

            "Largest Loss":
                round(
                    group["Net PnL"].min(),
                    2
                ),

            "Profit Factor":
                round(
                    profit_factor,
                    2
                )
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# EXIT REASON ANALYSIS
# ============================================================

def analyze_exit_reasons(
    df
):

    if df.empty:

        return pd.DataFrame()

    rows = []

    for reason, group in df.groupby(
        "Exit Category"
    ):

        trades = len(
            group
        )

        wins = int(
            (
                group["Result"]
                == "WIN"
            ).sum()
        )

        losses = int(
            (
                group["Result"]
                == "LOSS"
            ).sum()
        )

        net_pnl = (
            group["Net PnL"]
            .sum()
        )

        win_rate = (
            wins
            /
            trades
            *
            100
            if trades > 0
            else 0.0
        )

        rows.append({

            "Exit Reason":
                reason,

            "Trades":
                trades,

            "Wins":
                wins,

            "Losses":
                losses,

            "Win Rate %":
                round(
                    win_rate,
                    2
                ),

            "Net PnL":
                round(
                    net_pnl,
                    2
                ),

            "Average PnL":
                round(
                    group["Net PnL"].mean(),
                    2
                ),

            "Average Return %":
                round(
                    group["Return %"].mean(),
                    2
                ),

            "Average Holding Bars":
                round(
                    group["Bars Held"].mean(),
                    2
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
# HOLDING PERIOD ANALYSIS
# ============================================================

def analyze_holding_period(
    df
):

    if df.empty:

        return pd.DataFrame()

    categories = [
        "VERY_SHORT",
        "SHORT",
        "MEDIUM",
        "LONG",
        "MAX_HOLDING"
    ]

    rows = []

    for category in categories:

        group = df[
            df["Holding Category"]
            == category
        ]

        if group.empty:

            continue

        trades = len(
            group
        )

        wins = int(
            (
                group["Result"]
                == "WIN"
            ).sum()
        )

        losses = int(
            (
                group["Result"]
                == "LOSS"
            ).sum()
        )

        net_pnl = (
            group["Net PnL"]
            .sum()
        )

        win_rate = (
            wins
            /
            trades
            *
            100
            if trades > 0
            else 0.0
        )

        rows.append({

            "Holding Category":
                category,

            "Trades":
                trades,

            "Wins":
                wins,

            "Losses":
                losses,

            "Win Rate %":
                round(
                    win_rate,
                    2
                ),

            "Net PnL":
                round(
                    net_pnl,
                    2
                ),

            "Average PnL":
                round(
                    group["Net PnL"].mean(),
                    2
                ),

            "Average Return %":
                round(
                    group["Return %"].mean(),
                    2
                ),

            "Average Bars":
                round(
                    group["Bars Held"].mean(),
                    2
                )
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# LOSS ANALYSIS
# ============================================================

def analyze_losses(
    df
):

    if df.empty:

        return pd.DataFrame()

    losses = df[
        df["Net PnL"] < 0
    ].copy()

    if losses.empty:

        return pd.DataFrame()

    losses = losses.sort_values(
        "Net PnL",
        ascending=True
    ).reset_index(
        drop=True
    )

    losses["Loss Rank"] = (
        losses.index + 1
    )

    columns = [
        "Loss Rank",
        "Stock",
        "Signal",
        "Entry Date",
        "Entry Price",
        "Exit Date",
        "Exit Price",
        "Bars Held",
        "Exit Reason",
        "Net PnL",
        "Return %",
        "Result"
    ]

    available = [
        column
        for column in columns
        if column in losses.columns
    ]

    return losses[
        available
    ]


# ============================================================
# WIN ANALYSIS
# ============================================================

def analyze_winners(
    df
):

    if df.empty:

        return pd.DataFrame()

    winners = df[
        df["Net PnL"] > 0
    ].copy()

    if winners.empty:

        return pd.DataFrame()

    winners = winners.sort_values(
        "Net PnL",
        ascending=False
    ).reset_index(
        drop=True
    )

    winners["Win Rank"] = (
        winners.index + 1
    )

    columns = [
        "Win Rank",
        "Stock",
        "Signal",
        "Entry Date",
        "Entry Price",
        "Exit Date",
        "Exit Price",
        "Bars Held",
        "Exit Reason",
        "Net PnL",
        "Return %",
        "Result"
    ]

    available = [
        column
        for column in columns
        if column in winners.columns
    ]

    return winners[
        available
    ]


# ============================================================
# GENERATE DIAGNOSTIC CONCLUSIONS
# ============================================================

def generate_conclusions(
    overall,
    stock_df,
    signal_df,
    exit_df,
    holding_df
):

    conclusions = []

    net_pnl = safe_float(
        overall.get(
            "Net PnL",
            0
        )
    )

    win_rate = safe_float(
        overall.get(
            "Win Rate %",
            0
        )
    )

    profit_factor = safe_float(
        overall.get(
            "Profit Factor",
            0
        )
    )

    stop_loss_rate = safe_float(
        overall.get(
            "Stop Loss Rate %",
            0
        )
    )

    # --------------------------------------------------------
    # Overall result
    # --------------------------------------------------------

    if net_pnl < 0:

        conclusions.append(
            "OOS strategy performance was negative."
        )

    elif net_pnl > 0:

        conclusions.append(
            "OOS strategy performance was positive."
        )

    else:

        conclusions.append(
            "OOS strategy performance was approximately break-even."
        )

    # --------------------------------------------------------
    # Win rate
    # --------------------------------------------------------

    if win_rate < 40:

        conclusions.append(
            "Win rate is low and indicates weak trade selection."
        )

    elif win_rate < 50:

        conclusions.append(
            "Win rate is below 50%, requiring positive payoff asymmetry."
        )

    else:

        conclusions.append(
            "Win rate is at or above 50%."
        )

    # --------------------------------------------------------
    # Profit factor
    # --------------------------------------------------------

    if profit_factor < 1:

        conclusions.append(
            "Profit factor is below 1.0, meaning losing PnL exceeds winning PnL."
        )

    elif profit_factor < 1.2:

        conclusions.append(
            "Profit factor is positive but relatively weak."
        )

    else:

        conclusions.append(
            "Profit factor indicates positive gross payoff asymmetry."
        )

    # --------------------------------------------------------
    # Stop loss behavior
    # --------------------------------------------------------

    if stop_loss_rate >= 50:

        conclusions.append(
            "A high proportion of trades ended through stop loss."
        )

    # --------------------------------------------------------
    # Worst stock
    # --------------------------------------------------------

    if (
        stock_df is not None
        and
        not stock_df.empty
    ):

        worst_stock = stock_df.iloc[
            stock_df["Net PnL"].argmin()
        ]

        if safe_float(
            worst_stock["Net PnL"]
        ) < 0:

            conclusions.append(
                f"{worst_stock['Stock']} was the weakest stock by net PnL."
            )

    # --------------------------------------------------------
    # BUY / SELL
    # --------------------------------------------------------

    if (
        signal_df is not None
        and
        not signal_df.empty
    ):

        for _, row in signal_df.iterrows():

            if safe_float(
                row["Net PnL"]
            ) < 0:

                conclusions.append(
                    f"{row['Signal']} signals produced negative net PnL."
                )

    # --------------------------------------------------------
    # Exit reason
    # --------------------------------------------------------

    if (
        exit_df is not None
        and
        not exit_df.empty
    ):

        worst_exit = exit_df.iloc[
            exit_df["Net PnL"].argmin()
        ]

        if safe_float(
            worst_exit["Net PnL"]
        ) < 0:

            conclusions.append(
                f"{worst_exit['Exit Reason']} exits were a major source of negative PnL."
            )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------

    conclusions.append(
        "No strategy parameters were changed by Stage 15."
    )

    conclusions.append(
        "No real orders were placed."
    )

    return conclusions


# ============================================================
# SAVE ANALYSIS
# ============================================================

def save_dataframe(
    df,
    filename
):

    if df is None:

        return

    if df.empty:

        print(
            f"⚠️ No data to save: {filename}"
        )

        return

    try:

        df.to_csv(
            filename,
            index=False
        )

        print(
            f"✅ Saved: {filename}"
        )

    except Exception as error:

        print(
            f"❌ Could not save {filename}: "
            f"{error}"
        )


# ============================================================
# PRINT OVERALL SUMMARY
# ============================================================

def print_overall_summary(
    overall
):

    print()
    print("=" * 100)
    print("STAGE 15 - OVERALL FAILURE ANALYSIS")
    print("=" * 100)

    for key, value in overall.items():

        if isinstance(
            value,
            float
        ):

            if "PnL" in key or "Capital" in key:

                print(
                    f"{key:<28}: "
                    f"Rs. {value:,.2f}"
                )

            elif "%" in key:

                print(
                    f"{key:<28}: "
                    f"{value:.2f}%"
                )

            else:

                print(
                    f"{key:<28}: "
                    f"{value:.2f}"
                )

        else:

            print(
                f"{key:<28}: "
                f"{value}"
            )


# ============================================================
# PRINT STOCK ANALYSIS
# ============================================================

def print_stock_analysis(
    stock_df
):

    print()
    print("=" * 110)
    print("STOCK FAILURE ANALYSIS")
    print("=" * 110)

    if stock_df.empty:

        print(
            "No stock analysis available."
        )

        return

    columns = [
        "Stock",
        "Trades",
        "Wins",
        "Losses",
        "Win Rate %",
        "Net PnL",
        "Average PnL",
        "Largest Win",
        "Largest Loss"
    ]

    available = [
        column
        for column in columns
        if column in stock_df.columns
    ]

    print(
        stock_df[
            available
        ].to_string(
            index=False
        )
    )


# ============================================================
# PRINT SIGNAL ANALYSIS
# ============================================================

def print_signal_analysis(
    signal_df
):

    print()
    print("=" * 100)
    print("BUY / SELL FAILURE ANALYSIS")
    print("=" * 100)

    if signal_df.empty:

        print(
            "No signal analysis available."
        )

        return

    print(
        signal_df.to_string(
            index=False
        )
    )


# ============================================================
# PRINT EXIT ANALYSIS
# ============================================================

def print_exit_analysis(
    exit_df
):

    print()
    print("=" * 100)
    print("EXIT REASON ANALYSIS")
    print("=" * 100)

    if exit_df.empty:

        print(
            "No exit analysis available."
        )

        return

    print(
        exit_df.to_string(
            index=False
        )
    )


# ============================================================
# PRINT HOLDING ANALYSIS
# ============================================================

def print_holding_analysis(
    holding_df
):

    print()
    print("=" * 100)
    print("HOLDING PERIOD ANALYSIS")
    print("=" * 100)

    if holding_df.empty:

        print(
            "No holding-period analysis available."
        )

        return

    print(
        holding_df.to_string(
            index=False
        )
    )


# ============================================================
# PRINT TOP LOSSES
# ============================================================

def print_top_losses(
    df
):

    print()
    print("=" * 100)
    print("TOP LOSING TRADES")
    print("=" * 100)

    losses = analyze_losses(
        df
    )

    if losses.empty:

        print(
            "No losing trades."
        )

        return

    print(
        losses.head(10).to_string(
            index=False
        )
    )


# ============================================================
# PRINT TOP WINNERS
# ============================================================

def print_top_winners(
    df
):

    print()
    print("=" * 100)
    print("TOP WINNING TRADES")
    print("=" * 100)

    winners = analyze_winners(
        df
    )

    if winners.empty:

        print(
            "No winning trades."
        )

        return

    print(
        winners.head(10).to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print("          STAGE 15 - STRATEGY FAILURE ANALYSIS")
    print("=" * 100)

    print()
    print(
        "PAPER / HISTORICAL ANALYSIS ONLY"
    )

    print(
        "NO REAL ORDERS WILL BE PLACED"
    )

    print()
    print(
        "LOCKED STRATEGY"
    )

    print(
        f"SMMA                 : "
        f"{FAST_PERIOD}/{SLOW_PERIOD}"
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

    print(
        f"Initial Capital      : "
        f"Rs. {INITIAL_CAPITAL:,.2f}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Stage 15 is diagnostic only."
    )

    print(
        "No parameters will be optimized."
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_stage14_trades()

    if df.empty:

        print()
        print("=" * 100)
        print("STAGE 15 STOPPED")
        print("=" * 100)

        print(
            "No Stage 14 closed OOS trades are available."
        )

        print()
        print(
            "Run Stage 14 first:"
        )

        print(
            "python backtest/stage14_walk_forward.py"
        )

        return

    # ========================================================
    # ADD ANALYSIS
    # ========================================================

    df = add_trade_analysis(
        df
    )

    # ========================================================
    # OVERALL
    # ========================================================

    overall = calculate_overall_analysis(
        df
    )

    # ========================================================
    # STOCK
    # ========================================================

    stock_df = analyze_by_stock(
        df
    )

    # ========================================================
    # BUY / SELL
    # ========================================================

    signal_df = analyze_by_signal(
        df
    )

    # ========================================================
    # EXIT
    # ========================================================

    exit_df = analyze_exit_reasons(
        df
    )

    # ========================================================
    # HOLDING PERIOD
    # ========================================================

    holding_df = analyze_holding_period(
        df
    )

    # ========================================================
    # LOSS / WIN DETAILS
    # ========================================================

    losses_df = analyze_losses(
        df
    )

    winners_df = analyze_winners(
        df
    )

    # ========================================================
    # CONCLUSIONS
    # ========================================================

    conclusions = generate_conclusions(
        overall,
        stock_df,
        signal_df,
        exit_df,
        holding_df
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print_overall_summary(
        overall
    )

    print_stock_analysis(
        stock_df
    )

    print_signal_analysis(
        signal_df
    )

    print_exit_analysis(
        exit_df
    )

    print_holding_analysis(
        holding_df
    )

    print_top_losses(
        df
    )

    print_top_winners(
        df
    )

    # ========================================================
    # DIAGNOSTIC CONCLUSIONS
    # ========================================================

    print()
    print("=" * 100)
    print("STAGE 15 DIAGNOSTIC CONCLUSIONS")
    print("=" * 100)

    for number, conclusion in enumerate(
        conclusions,
        start=1
    ):

        print(
            f"{number}. {conclusion}"
        )

    # ========================================================
    # SAVE FILES
    # ========================================================

    print()
    print("=" * 100)
    print("STAGE 15 OUTPUT FILES")
    print("=" * 100)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        [
            overall
        ]
    )

    save_dataframe(
        summary_df,
        FAILURE_SUMMARY_FILE
    )

    # --------------------------------------------------------
    # Trade analysis
    # --------------------------------------------------------

    save_dataframe(
        df,
        TRADE_ANALYSIS_FILE
    )

    # --------------------------------------------------------
    # Stock analysis
    # --------------------------------------------------------

    save_dataframe(
        stock_df,
        STOCK_ANALYSIS_FILE
    )

    # --------------------------------------------------------
    # Signal analysis
    # --------------------------------------------------------

    save_dataframe(
        signal_df,
        SIGNAL_ANALYSIS_FILE
    )

    # --------------------------------------------------------
    # Exit analysis
    # --------------------------------------------------------

    save_dataframe(
        exit_df,
        EXIT_ANALYSIS_FILE
    )

    # --------------------------------------------------------
    # Holding analysis
    # --------------------------------------------------------

    save_dataframe(
        holding_df,
        HOLDING_ANALYSIS_FILE
    )

    # ========================================================
    # COMPLETE REPORT
    # ========================================================

    report_rows = []

    for conclusion in conclusions:

        report_rows.append({

            "Section":
                "CONCLUSION",

            "Metric":
                "Diagnostic",

            "Value":
                conclusion
        })

    # Overall metrics

    for key, value in overall.items():

        report_rows.append({

            "Section":
                "OVERALL",

            "Metric":
                key,

            "Value":
                value
        })

    # Stock metrics

    if not stock_df.empty:

        for _, row in stock_df.iterrows():

            report_rows.append({

                "Section":
                    "STOCK",

                "Metric":
                    row["Stock"],

                "Value":
                    f"Net PnL={row['Net PnL']:.2f}, "
                    f"Win Rate={row['Win Rate %']:.2f}%"
            })

    # Signal metrics

    if not signal_df.empty:

        for _, row in signal_df.iterrows():

            report_rows.append({

                "Section":
                    "SIGNAL",

                "Metric":
                    row["Signal"],

                "Value":
                    f"Net PnL={row['Net PnL']:.2f}, "
                    f"Win Rate={row['Win Rate %']:.2f}%"
            })

    report_df = pd.DataFrame(
        report_rows
    )

    save_dataframe(
        report_df,
        REPORT_FILE
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    net_pnl = safe_float(
        overall["Net PnL"]
    )

    profit_factor = safe_float(
        overall["Profit Factor"]
    )

    print()
    print("=" * 100)
    print("STAGE 15 VALIDATION DECISION")
    print("=" * 100)

    if net_pnl < 0:

        print(
            "RESULT: STRATEGY FAILURE EVIDENCE"
        )

        print()
        print(
            "The Stage 14 out-of-sample period "
            "was not profitable."
        )

        if profit_factor < 1:

            print()
            print(
                "Profit factor is below 1.0."
            )

            print(
                "Losses exceeded winning PnL."
            )

        print()
        print(
            "Do NOT change parameters inside Stage 15."
        )

        print(
            "Further investigation should be performed "
            "before considering any strategy modification."
        )

    elif net_pnl > 0:

        print(
            "RESULT: POSITIVE OOS EVIDENCE"
        )

        print()
        print(
            "The Stage 14 out-of-sample period "
            "was profitable."
        )

        print(
            "Stage 15 confirms the performance "
            "without changing strategy parameters."
        )

    else:

        print(
            "RESULT: BREAK-EVEN OOS EVIDENCE"
        )

        print()
        print(
            "The strategy produced approximately "
            "zero net PnL."
        )

    # ========================================================
    # FILE LIST
    # ========================================================

    print()
    print("=" * 100)
    print("OUTPUT FILES")
    print("=" * 100)

    print(
        f"Summary             : "
        f"{FAILURE_SUMMARY_FILE}"
    )

    print(
        f"Trade Analysis      : "
        f"{TRADE_ANALYSIS_FILE}"
    )

    print(
        f"Stock Analysis      : "
        f"{STOCK_ANALYSIS_FILE}"
    )

    print(
        f"BUY/SELL Analysis   : "
        f"{SIGNAL_ANALYSIS_FILE}"
    )

    print(
        f"Exit Analysis       : "
        f"{EXIT_ANALYSIS_FILE}"
    )

    print(
        f"Holding Analysis    : "
        f"{HOLDING_ANALYSIS_FILE}"
    )

    print(
        f"Full Report         : "
        f"{REPORT_FILE}"
    )

    # ========================================================
    # SAFETY
    # ========================================================

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
        "STAGE 15 COMPLETE"
    )

    print(
        "=" * 100
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()