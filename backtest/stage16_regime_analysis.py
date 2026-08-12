# ============================================================
# STAGE 16 - REGIME ANALYSIS
# ============================================================
#
# PURPOSE:
# Analyze Stage 14 out-of-sample trades under different
# market/price regimes.
#
# IMPORTANT:
# - PAPER / HISTORICAL ANALYSIS ONLY
# - NO REAL ORDERS
# - NO PARAMETER OPTIMIZATION
# - NO STRATEGY PARAMETERS ARE CHANGED
#
# LOCKED STRATEGY:
# SMMA FAST       = 20
# SMMA SLOW       = 120
# HOLDING PERIOD  = 60 trading days
# STOP LOSS       = 5%
# TAKE PROFIT     = 10%
#
# FIXES INCLUDED:
# 1. Supports both Stage 14 trade-file names.
# 2. Timezone-aware and timezone-naive datetime values
#    are normalized before date comparison.
# 3. Historical data is cached per stock.
# 4. Diagnostic analysis only.
#
# ============================================================

import os
import warnings

import pandas as pd
import numpy as np

from api.historical_data import get_historical_data

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INITIAL_CAPITAL = 100000.00

FAST_SMMA = 20
SLOW_SMMA = 120

HOLDING_PERIOD = 60

STOP_LOSS_PERCENT = 5.00
TAKE_PROFIT_PERCENT = 10.00

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
# STAGE 14 TRADE FILES
# ============================================================
#
# Stage 14 may have produced either:
#
# data/stage14/walk_forward_trades.csv
#
# OR:
#
# data/stage14/stage14_walk_forward_trades.csv
#
# Stage 16 will automatically use whichever exists.
#
# ============================================================

STAGE14_TRADE_FILES = [
    "data/stage14/walk_forward_trades.csv",
    "data/stage14/stage14_walk_forward_trades.csv",
]


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "data/stage16"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DATETIME NORMALIZATION
# ============================================================

def normalize_datetime_value(value):
    """
    Convert one datetime value into a timezone-naive
    pandas Timestamp.

    This prevents errors such as:

    TypeError:
    Cannot compare tz-naive and tz-aware datetime-like objects
    """

    if value is None:
        return pd.NaT

    try:

        value = pd.Timestamp(value)

        if pd.isna(value):
            return pd.NaT

        # Remove timezone information.

        if value.tzinfo is not None:
            value = value.tz_localize(None)

        return value

    except Exception:

        return pd.NaT


def normalize_datetime_series(series):
    """
    Convert a pandas datetime series into timezone-naive
    timestamps.
    """

    if series is None:
        return series

    try:

        # utc=True handles mixed timezone values safely.

        result = pd.to_datetime(
            series,
            errors="coerce",
            utc=True
        )

        # Convert UTC timezone-aware timestamps into
        # timezone-naive timestamps.

        result = result.dt.tz_localize(None)

        return result

    except Exception:

        # Fallback for unusual datetime values.

        return series.apply(
            normalize_datetime_value
        )


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

def normalize_columns(df):

    if df is None or df.empty:
        return None

    df = df.copy()

    # Remove accidental spaces.

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # Lowercase lookup.

    column_map = {
        str(column).lower().strip(): column
        for column in df.columns
    }

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    if "date" not in df.columns:

        if "datetime" in column_map:

            df["date"] = df[
                column_map["datetime"]
            ]

        elif "entry date" in column_map:

            df["date"] = df[
                column_map["entry date"]
            ]

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    if "close" not in df.columns:

        if "Close" in df.columns:

            df["close"] = df["Close"]

        elif "CLOSE" in df.columns:

            df["close"] = df["CLOSE"]

        elif "closing price" in column_map:

            df["close"] = df[
                column_map["closing price"]
            ]

    return df


# ============================================================
# FIND STAGE 14 FILE
# ============================================================

def find_stage14_trade_file():

    for file_path in STAGE14_TRADE_FILES:

        if os.path.exists(file_path):

            return file_path

    return None


# ============================================================
# LOAD STAGE 14 TRADES
# ============================================================

def load_stage14_trades():

    print("\n")
    print("=" * 80)
    print("LOADING STAGE 14 OUT-OF-SAMPLE TRADES")
    print("=" * 80)

    # --------------------------------------------------------
    # Find available Stage 14 file
    # --------------------------------------------------------

    trade_file = find_stage14_trade_file()

    if trade_file is None:

        print(
            "\nERROR: Stage 14 trade file not found."
        )

        print(
            "\nChecked:"
        )

        for file_path in STAGE14_TRADE_FILES:

            print(
                f"  - {file_path}"
            )

        print(
            "\nPlease run Stage 14 first."
        )

        return None

    print(
        "\nUsing Stage 14 file:"
    )

    print(
        f"  {trade_file}"
    )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            trade_file
        )

    except Exception as e:

        print(
            f"\nERROR reading Stage 14 file: {e}"
        )

        return None

    print(
        f"\nRaw Stage 14 records: {len(df)}"
    )

    if df.empty:

        print(
            "Stage 14 file contains no records."
        )

        return None

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Normalize dates
    # --------------------------------------------------------

    if "Entry Date" in df.columns:

        df["Entry Date"] = normalize_datetime_series(
            df["Entry Date"]
        )

    if "Exit Date" in df.columns:

        df["Exit Date"] = normalize_datetime_series(
            df["Exit Date"]
        )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "Entry Price",
        "Exit Price",
        "Shares",
        "Bars Held",
        "Net PnL",
        "Return %",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Remove invalid entry dates
    # --------------------------------------------------------

    if "Entry Date" in df.columns:

        df = df.dropna(
            subset=["Entry Date"]
        )

    # --------------------------------------------------------
    # Remove invalid exit dates
    # --------------------------------------------------------

    if "Exit Date" in df.columns:

        df = df.dropna(
            subset=["Exit Date"]
        )

    # --------------------------------------------------------
    # Only closed trades
    # --------------------------------------------------------

    if "Exit Date" in df.columns:

        df = df[
            df["Exit Date"].notna()
        ]

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    df = df.reset_index(
        drop=True
    )

    print(
        f"Usable closed OOS trades: {len(df)}"
    )

    return df


# ============================================================
# CALCULATE SMMA
# ============================================================

def calculate_smma(
    series,
    period
):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    if len(series) < period:

        return pd.Series(
            np.nan,
            index=series.index
        )

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    # First SMMA value = SMA.

    first_value = series.iloc[
        :period
    ].mean()

    result.iloc[
        period - 1
    ] = first_value

    # Wilder-style SMMA.

    for i in range(
        period,
        len(series)
    ):

        previous = result.iloc[
            i - 1
        ]

        current = series.iloc[
            i
        ]

        if pd.isna(current):

            result.iloc[i] = previous

        elif pd.isna(previous):

            result.iloc[i] = current

        else:

            result.iloc[i] = (
                (
                    previous
                    * (period - 1)
                )
                + current
            ) / period

    return result


# ============================================================
# PREPARE HISTORICAL DATA
# ============================================================

def prepare_historical_data(
    symbol
):

    print(
        f"\nGetting historical data for {symbol}..."
    )

    try:

        df = get_historical_data(
            symbol,
            days=HISTORICAL_DAYS
        )

    except Exception as e:

        print(
            f"{symbol}: Historical data error: {e}"
        )

        return None

    if df is None or df.empty:

        print(
            f"{symbol}: No historical data."
        )

        return None

    print(
        f"{symbol}: Historical rows loaded: {len(df)}"
    )

    # --------------------------------------------------------
    # Normalize columns
    # --------------------------------------------------------

    df = normalize_columns(
        df
    )

    if df is None or df.empty:

        print(
            f"{symbol}: Unable to normalize data."
        )

        return None

    # --------------------------------------------------------
    # Check date column
    # --------------------------------------------------------

    if "date" not in df.columns:

        print(
            f"{symbol}: Date column unavailable."
        )

        return None

    # --------------------------------------------------------
    # Check close column
    # --------------------------------------------------------

    if "close" not in df.columns:

        print(
            f"{symbol}: Close column unavailable."
        )

        return None

    # --------------------------------------------------------
    # Normalize datetime
    # --------------------------------------------------------

    df["date"] = normalize_datetime_series(
        df["date"]
    )

    # --------------------------------------------------------
    # Normalize close
    # --------------------------------------------------------

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "date",
            "close"
        ]
    ).copy()

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Remove duplicate dates
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["date"],
        keep="last"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Calculate SMMA20
    # --------------------------------------------------------

    df["SMMA20"] = calculate_smma(
        df["close"],
        FAST_SMMA
    )

    # --------------------------------------------------------
    # Calculate SMMA120
    # --------------------------------------------------------

    df["SMMA120"] = calculate_smma(
        df["close"],
        SLOW_SMMA
    )

    # --------------------------------------------------------
    # SMMA difference
    # --------------------------------------------------------

    df["SMMA_difference"] = (
        df["SMMA20"]
        - df["SMMA120"]
    )

    # --------------------------------------------------------
    # SMMA percentage difference
    # --------------------------------------------------------

    df["SMMA_difference_pct"] = np.where(
        df["SMMA120"] != 0,
        (
            df["SMMA_difference"]
            / df["SMMA120"]
        ) * 100,
        np.nan
    )

    # --------------------------------------------------------
    # Daily return
    # --------------------------------------------------------

    df["daily_return"] = (
        df["close"]
        .pct_change()
    )

    # --------------------------------------------------------
    # 20-day volatility
    # --------------------------------------------------------

    df["volatility_20"] = (
        df["daily_return"]
        .rolling(20)
        .std()
        * 100
    )

    # --------------------------------------------------------
    # Price / SMMA20 ratio
    # --------------------------------------------------------

    df["price_smma_ratio"] = np.where(
        df["SMMA20"] != 0,
        (
            df["close"]
            / df["SMMA20"]
        ),
        np.nan
    )

    return df


# ============================================================
# GET ENTRY CONTEXT
# ============================================================

def get_entry_context(
    df,
    target_date
):

    if df is None or df.empty:

        return None

    # --------------------------------------------------------
    # Normalize target date
    # --------------------------------------------------------

    target_date = normalize_datetime_value(
        target_date
    )

    # --------------------------------------------------------
    # Normalize dataframe dates
    # --------------------------------------------------------

    df = df.copy()

    df["date"] = normalize_datetime_series(
        df["date"]
    )

    # --------------------------------------------------------
    # Invalid target
    # --------------------------------------------------------

    if pd.isna(target_date):

        return None

    # --------------------------------------------------------
    # Find historical row ON OR BEFORE entry date
    # --------------------------------------------------------

    eligible = df[
        df["date"] <= target_date
    ]

    if eligible.empty:

        return None

    row = eligible.iloc[-1]

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    close = float(
        row["close"]
    )

    # --------------------------------------------------------
    # SMMA values
    # --------------------------------------------------------

    smma20 = row[
        "SMMA20"
    ]

    smma120 = row[
        "SMMA120"
    ]

    difference_pct = row[
        "SMMA_difference_pct"
    ]

    volatility = row[
        "volatility_20"
    ]

    # --------------------------------------------------------
    # Trend regime
    # --------------------------------------------------------

    if (
        pd.notna(smma20)
        and
        pd.notna(smma120)
    ):

        if smma20 > smma120:

            trend_regime = "BULL"

        elif smma20 < smma120:

            trend_regime = "BEAR"

        else:

            trend_regime = "NEUTRAL"

    else:

        trend_regime = "UNKNOWN"

    # --------------------------------------------------------
    # Volatility regime
    #
    # LOW    < 2%
    # MEDIUM 2%-4%
    # HIGH   >= 4%
    # --------------------------------------------------------

    if pd.isna(volatility):

        volatility_regime = "UNKNOWN"

    elif volatility < 2:

        volatility_regime = "LOW"

    elif volatility < 4:

        volatility_regime = "MEDIUM"

    else:

        volatility_regime = "HIGH"

    # --------------------------------------------------------
    # Price position
    # --------------------------------------------------------

    if (
        pd.notna(smma20)
        and
        close > smma20
    ):

        price_position = "ABOVE_SMMA20"

    elif (
        pd.notna(smma20)
        and
        close < smma20
    ):

        price_position = "BELOW_SMMA20"

    else:

        price_position = "UNKNOWN"

    return {

        "Context Date": row["date"],

        "Context Close": close,

        "SMMA20": smma20,

        "SMMA120": smma120,

        "SMMA Difference %": difference_pct,

        "Volatility 20D %": volatility,

        "Trend Regime": trend_regime,

        "Volatility Regime": volatility_regime,

        "Price Position": price_position,

    }


# ============================================================
# ENRICH STAGE 14 TRADES
# ============================================================

def enrich_trades(
    trades
):

    if trades is None or trades.empty:

        return pd.DataFrame()

    all_records = []

    # --------------------------------------------------------
    # Historical data cache
    # --------------------------------------------------------

    historical_cache = {}

    for index, trade in trades.iterrows():

        stock = str(
            trade.get(
                "Stock",
                ""
            )
        ).strip()

        if not stock:

            continue

        # ----------------------------------------------------
        # Normalize symbol
        # ----------------------------------------------------

        symbol = stock

        if not symbol.endswith(
            "-EQ"
        ):

            symbol = (
                symbol
                + "-EQ"
            )

        # ----------------------------------------------------
        # Load once per stock
        # ----------------------------------------------------

        if symbol not in historical_cache:

            historical_cache[
                symbol
            ] = prepare_historical_data(
                symbol
            )

        historical_df = historical_cache[
            symbol
        ]

        if (
            historical_df is None
            or
            historical_df.empty
        ):

            continue

        # ----------------------------------------------------
        # Entry date
        # ----------------------------------------------------

        entry_date = trade.get(
            "Entry Date"
        )

        entry_date = normalize_datetime_value(
            entry_date
        )

        if pd.isna(entry_date):

            continue

        # ----------------------------------------------------
        # Exit date
        # ----------------------------------------------------

        exit_date = trade.get(
            "Exit Date"
        )

        exit_date = normalize_datetime_value(
            exit_date
        )

        # ----------------------------------------------------
        # Get entry context
        # ----------------------------------------------------

        context = get_entry_context(
            historical_df,
            entry_date
        )

        if context is None:

            continue

        # ----------------------------------------------------
        # Build enriched record
        # ----------------------------------------------------

        record = trade.to_dict()

        record.update(
            context
        )

        record[
            "Stock Normalized"
        ] = stock

        all_records.append(
            record
        )

    if not all_records:

        return pd.DataFrame()

    return pd.DataFrame(
        all_records
    )


# ============================================================
# SAFE NUMERIC
# ============================================================

def safe_numeric(
    series
):

    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0.0)


# ============================================================
# ANALYZE GROUP
# ============================================================

def analyze_group(
    df,
    group_column
):

    if (
        df is None
        or
        df.empty
        or
        group_column not in df.columns
    ):

        return pd.DataFrame()

    records = []

    for group_name, group in df.groupby(
        group_column,
        dropna=False
    ):

        group = group.copy()

        pnl = safe_numeric(
            group["Net PnL"]
        )

        returns = safe_numeric(
            group["Return %"]
        )

        trades_count = len(
            group
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

        total_pnl = float(
            pnl.sum()
        )

        average_pnl = float(
            pnl.mean()
        )

        average_return = float(
            returns.mean()
        )

        if trades_count > 0:

            win_rate = (
                wins
                / trades_count
            ) * 100

        else:

            win_rate = 0.0

        records.append({

            group_column: str(
                group_name
            ),

            "Trades": trades_count,

            "Wins": wins,

            "Losses": losses,

            "Break-even": breakeven,

            "Win Rate %": round(
                win_rate,
                2
            ),

            "Net PnL": round(
                total_pnl,
                2
            ),

            "Average PnL": round(
                average_pnl,
                2
            ),

            "Average Return %": round(
                average_return,
                2
            ),

            "Average Holding Bars": round(
                safe_numeric(
                    group["Bars Held"]
                ).mean(),
                2
            )
        })

    return pd.DataFrame(
        records
    )


# ============================================================
# EXIT REASON ANALYSIS
# ============================================================

def analyze_exit_reasons(
    df
):

    if (
        df is None
        or
        df.empty
        or
        "Exit Reason" not in df.columns
    ):

        return pd.DataFrame()

    return analyze_group(
        df,
        "Exit Reason"
    )


# ============================================================
# SIGNAL ANALYSIS
# ============================================================

def analyze_signals(
    df
):

    if (
        df is None
        or
        df.empty
        or
        "Signal" not in df.columns
    ):

        return pd.DataFrame()

    return analyze_group(
        df,
        "Signal"
    )


# ============================================================
# STOCK ANALYSIS
# ============================================================

def analyze_stocks(
    df
):

    if (
        df is None
        or
        df.empty
        or
        "Stock" not in df.columns
    ):

        return pd.DataFrame()

    return analyze_group(
        df,
        "Stock"
    )


# ============================================================
# TREND + VOLATILITY ANALYSIS
# ============================================================

def analyze_regimes(
    df
):

    if (
        df is None
        or
        df.empty
    ):

        return pd.DataFrame()

    required = [
        "Trend Regime",
        "Volatility Regime"
    ]

    if not all(
        column in df.columns
        for column in required
    ):

        return pd.DataFrame()

    records = []

    grouped = df.groupby(
        [
            "Trend Regime",
            "Volatility Regime"
        ],
        dropna=False
    )

    for (
        trend,
        volatility
    ), group in grouped:

        pnl = safe_numeric(
            group["Net PnL"]
        )

        returns = safe_numeric(
            group["Return %"]
        )

        trades_count = len(
            group
        )

        wins = int(
            (pnl > 0).sum()
        )

        losses = int(
            (pnl < 0).sum()
        )

        if trades_count > 0:

            win_rate = (
                wins
                / trades_count
            ) * 100

        else:

            win_rate = 0.0

        records.append({

            "Trend Regime": str(
                trend
            ),

            "Volatility Regime": str(
                volatility
            ),

            "Trades": trades_count,

            "Wins": wins,

            "Losses": losses,

            "Win Rate %": round(
                win_rate,
                2
            ),

            "Net PnL": round(
                pnl.sum(),
                2
            ),

            "Average PnL": round(
                pnl.mean(),
                2
            ),

            "Average Return %": round(
                returns.mean(),
                2
            ),

            "Average Holding Bars": round(
                safe_numeric(
                    group["Bars Held"]
                ).mean(),
                2
            )
        })

    return pd.DataFrame(
        records
    )


# ============================================================
# OVERALL SUMMARY
# ============================================================

def calculate_overall_summary(
    df
):

    if df is None or df.empty:

        return {}

    pnl = safe_numeric(
        df["Net PnL"]
    )

    returns = safe_numeric(
        df["Return %"]
    )

    trades = len(
        df
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

    winning_pnl = pnl[
        pnl > 0
    ]

    losing_pnl = pnl[
        pnl < 0
    ]

    gross_profit = float(
        winning_pnl.sum()
    )

    gross_loss = float(
        losing_pnl.sum()
    )

    if abs(gross_loss) > 0:

        profit_factor = (
            gross_profit
            / abs(gross_loss)
        )

    else:

        profit_factor = 0.0

    if trades > 0:

        win_rate = (
            wins
            / trades
        ) * 100

    else:

        win_rate = 0.0

    return {

        "Initial Capital": INITIAL_CAPITAL,

        "Fast SMMA": FAST_SMMA,

        "Slow SMMA": SLOW_SMMA,

        "Holding Period": HOLDING_PERIOD,

        "Stop Loss %": STOP_LOSS_PERCENT,

        "Take Profit %": TAKE_PROFIT_PERCENT,

        "Total Trades": trades,

        "Winning Trades": wins,

        "Losing Trades": losses,

        "Break-even Trades": breakeven,

        "Win Rate %": round(
            win_rate,
            2
        ),

        "Loss Rate %": round(
            (
                losses
                / trades
                * 100
            )
            if trades > 0
            else 0,
            2
        ),

        "Gross PnL": round(
            pnl.sum(),
            2
        ),

        "Average Trade PnL": round(
            pnl.mean(),
            2
        ),

        "Average Return %": round(
            returns.mean(),
            2
        ),

        "Average Winning PnL": round(
            winning_pnl.mean()
            if not winning_pnl.empty
            else 0.0,
            2
        ),

        "Average Losing PnL": round(
            losing_pnl.mean()
            if not losing_pnl.empty
            else 0.0,
            2
        ),

        "Largest Win": round(
            winning_pnl.max()
            if not winning_pnl.empty
            else 0.0,
            2
        ),

        "Largest Loss": round(
            losing_pnl.min()
            if not losing_pnl.empty
            else 0.0,
            2
        ),

        "Profit Factor": round(
            profit_factor,
            2
        )
    }


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    df,
    filename
):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if df is None:

        df = pd.DataFrame()

    df.to_csv(
        path,
        index=False
    )

    print(
        f"Saved: {path}"
    )

    return path


# ============================================================
# PRINT TABLE
# ============================================================

def print_table(
    title,
    df
):

    print("\n")
    print("=" * 100)
    print(title)
    print("=" * 100)

    if df is None or df.empty:

        print(
            "No data available."
        )

        return

    print(
        df.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 100)
    print(
        "                 STAGE 16 - REGIME ANALYSIS"
    )
    print("=" * 100)

    print(
        "\nPAPER / HISTORICAL ANALYSIS ONLY"
    )

    print(
        "NO REAL ORDERS WILL BE PLACED"
    )

    # --------------------------------------------------------
    # LOCKED STRATEGY
    # --------------------------------------------------------

    print("\n")
    print("LOCKED STRATEGY")
    print("-" * 50)

    print(
        f"SMMA             : "
        f"{FAST_SMMA}/{SLOW_SMMA}"
    )

    print(
        f"Holding Period   : "
        f"{HOLDING_PERIOD} trading days"
    )

    print(
        f"Stop Loss        : "
        f"{STOP_LOSS_PERCENT:.2f}%"
    )

    print(
        f"Take Profit      : "
        f"{TAKE_PROFIT_PERCENT:.2f}%"
    )

    print(
        f"Initial Capital  : "
        f"Rs. {INITIAL_CAPITAL:,.2f}"
    )

    print("\n")
    print("IMPORTANT:")

    print(
        "Stage 16 is diagnostic only."
    )

    print(
        "No parameters will be optimized."
    )

    print(
        "No parameters will be changed."
    )

    # --------------------------------------------------------
    # LOAD STAGE 14
    # --------------------------------------------------------

    trades = load_stage14_trades()

    if trades is None or trades.empty:

        print("\n")
        print(
            "STAGE 16 STOPPED"
        )

        print(
            "No usable Stage 14 OOS trades were found."
        )

        return

    # --------------------------------------------------------
    # ENRICH
    # --------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("BUILDING REGIME CONTEXT")
    print("=" * 100)

    enriched = enrich_trades(
        trades
    )

    if enriched.empty:

        print(
            "\nERROR:"
        )

        print(
            "Unable to build regime context."
        )

        print(
            "Check Stage 14 trade dates and historical data."
        )

        return

    print(
        f"Regime-enriched trades: "
        f"{len(enriched)}"
    )

    # ========================================================
    # OVERALL ANALYSIS
    # ========================================================

    summary = calculate_overall_summary(
        enriched
    )

    print("\n")
    print("=" * 100)
    print(
        "STAGE 16 - OVERALL OOS PERFORMANCE"
    )
    print("=" * 100)

    for key, value in summary.items():

        if isinstance(
            value,
            float
        ):

            print(
                f"{key:<25}: {value:.2f}"
            )

        else:

            print(
                f"{key:<25}: {value}"
            )

    # ========================================================
    # TREND REGIME
    # ========================================================

    trend_analysis = analyze_group(
        enriched,
        "Trend Regime"
    )

    print_table(
        "TREND REGIME PERFORMANCE - OUT OF SAMPLE",
        trend_analysis
    )

    # ========================================================
    # VOLATILITY REGIME
    # ========================================================

    volatility_analysis = analyze_group(
        enriched,
        "Volatility Regime"
    )

    print_table(
        "VOLATILITY REGIME PERFORMANCE - OUT OF SAMPLE",
        volatility_analysis
    )

    # ========================================================
    # COMBINED REGIME
    # ========================================================

    regime_analysis = analyze_regimes(
        enriched
    )

    print_table(
        "COMBINED REGIME PERFORMANCE - OUT OF SAMPLE",
        regime_analysis
    )

    # ========================================================
    # STOCK PERFORMANCE
    # ========================================================

    stock_analysis = analyze_stocks(
        enriched
    )

    print_table(
        "STOCK PERFORMANCE - OUT OF SAMPLE",
        stock_analysis
    )

    # ========================================================
    # BUY / SELL PERFORMANCE
    # ========================================================

    signal_analysis = analyze_signals(
        enriched
    )

    print_table(
        "BUY / SELL PERFORMANCE - OUT OF SAMPLE",
        signal_analysis
    )

    # ========================================================
    # EXIT REASON PERFORMANCE
    # ========================================================

    exit_analysis = analyze_exit_reasons(
        enriched
    )

    print_table(
        "EXIT REASON PERFORMANCE - OUT OF SAMPLE",
        exit_analysis
    )

    # ========================================================
    # TRADE DETAILS
    # ========================================================

    print_table(
        "STAGE 16 ENRICHED TRADE RESULTS",
        enriched
    )

    # ========================================================
    # SAVE OUTPUT FILES
    # ========================================================

    print("\n")
    print("=" * 100)
    print("STAGE 16 OUTPUT FILES")
    print("=" * 100)

    save_csv(
        pd.DataFrame(
            [summary]
        ),
        "stage16_summary.csv"
    )

    save_csv(
        enriched,
        "stage16_regime_trades.csv"
    )

    save_csv(
        trend_analysis,
        "stage16_trend_analysis.csv"
    )

    save_csv(
        volatility_analysis,
        "stage16_volatility_analysis.csv"
    )

    save_csv(
        regime_analysis,
        "stage16_regime_performance.csv"
    )

    save_csv(
        stock_analysis,
        "stage16_stock_performance.csv"
    )

    save_csv(
        signal_analysis,
        "stage16_signal_performance.csv"
    )

    save_csv(
        exit_analysis,
        "stage16_exit_analysis.csv"
    )

    # ========================================================
    # DIAGNOSTIC DECISION
    # ========================================================

    print("\n")
    print("=" * 100)
    print("STAGE 16 DIAGNOSTIC CONCLUSION")
    print("=" * 100)

    net_pnl = summary.get(
        "Gross PnL",
        0.0
    )

    profit_factor = summary.get(
        "Profit Factor",
        0.0
    )

    win_rate = summary.get(
        "Win Rate %",
        0.0
    )

    if (
        net_pnl > 0
        and
        profit_factor >= 1.0
    ):

        print(
            "RESULT: POSITIVE REGIME EVIDENCE"
        )

        print(
            "\nThe Stage 14 OOS trades show positive "
            "performance across the analyzed sample."
        )

    elif (
        net_pnl < 0
        and
        profit_factor < 1.0
    ):

        print(
            "RESULT: NEGATIVE REGIME EVIDENCE"
        )

        print(
            "\nThe Stage 14 OOS trades remain "
            "unprofitable in the analyzed regimes."
        )

    else:

        print(
            "RESULT: MIXED REGIME EVIDENCE"
        )

        print(
            "\nPerformance differs across regimes."
        )

    print(
        f"\nOOS Trades       : "
        f"{summary.get('Total Trades', 0)}"
    )

    print(
        f"OOS Win Rate     : "
        f"{win_rate:.2f}%"
    )

    print(
        f"OOS Net/Gross PnL: "
        f"Rs. {net_pnl:,.2f}"
    )

    print(
        f"Profit Factor    : "
        f"{profit_factor:.2f}"
    )

    print("\n")

    print(
        "No strategy parameters were changed."
    )

    print(
        "No real order has been placed."
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 100)
    print("STAGE 16 COMPLETE")
    print("=" * 100)

    print(
        f"Stocks analyzed: {len(STOCKS)}"
    )

    print(
        f"Trades analyzed: "
        f"{len(enriched)}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()