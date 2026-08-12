# ============================================================
# STAGE 19 - SIGNAL ENGINE
# stage19/signal_engine.py
# ============================================================

import os
from datetime import datetime

import numpy as np
import pandas as pd

from api.historical_data import get_historical_data


# ============================================================
# CONFIGURATION
# ============================================================

STAGE17_FILE = "data/stage17/stage17_normalized.csv"

OUTPUT_DIR = "data/stage19"

RSI_PERIOD = 14
SMMA_FAST = 20
SMMA_SLOW = 120

# Minimum historical candles required
MIN_CANDLES = 130


# ============================================================
# DIRECTORY
# ============================================================

def ensure_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD STAGE 17
# ============================================================

def load_stage17():

    print()
    print("=" * 80)
    print("LOADING STAGE 17 DATA")
    print("=" * 80)

    if not os.path.exists(STAGE17_FILE):

        print(
            f"❌ Stage 17 file not found: "
            f"{STAGE17_FILE}"
        )

        return None

    try:

        df = pd.read_csv(STAGE17_FILE)

        print(
            f"Loaded Stage 17 file: "
            f"{STAGE17_FILE}"
        )

        print(
            f"Rows: {len(df)}"
        )

        print(
            f"Columns: {list(df.columns)}"
        )

        return df

    except Exception as e:

        print(
            f"❌ Error loading Stage 17 file: {e}"
        )

        return None


# ============================================================
# FIND SYMBOL COLUMN
# ============================================================

def get_symbol_column(df):

    possible_columns = [
        "Symbol",
        "symbol",
        "STOCK",
        "Stock",
        "SYMBOL",
    ]

    for column in possible_columns:

        if column in df.columns:

            return column

    return None


# ============================================================
# SMMA
# ============================================================

def calculate_smma(series, period):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    valid = series.dropna()

    if len(valid) < period:

        return result

    first_index = valid.index[period - 1]

    first_value = (
        valid.iloc[:period].mean()
    )

    result.loc[first_index] = first_value

    previous = first_value

    for index in valid.index[period:]:

        current = float(
            valid.loc[index]
        )

        previous = (
            (previous * (period - 1))
            + current
        ) / period

        result.loc[index] = previous

    return result


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    close,
    period=14
):

    close = pd.to_numeric(
        close,
        errors="coerce"
    )

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    # When average loss is zero,
    # RSI is effectively 100.
    rsi = rsi.where(
        avg_loss != 0,
        100
    )

    return rsi


# ============================================================
# PREPARE HISTORICAL DATA
# ============================================================

def prepare_historical_data(df):

    if df is None:

        return None

    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        print(
            f"❌ Missing historical columns: "
            f"{missing}"
        )

        return None

    result = df.copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce"
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    result = result.dropna(
        subset=[
            "date",
            "close",
        ]
    )

    result = result.sort_values(
        "date"
    )

    result = result.drop_duplicates(
        subset=["date"],
        keep="last"
    )

    result = result.reset_index(
        drop=True
    )

    return result


# ============================================================
# CALCULATE INDICATORS
# ============================================================

def calculate_indicators(df):

    df = prepare_historical_data(df)

    if df is None:

        return None

    if len(df) < MIN_CANDLES:

        print(
            f"⚠️ Only {len(df)} candles available. "
            f"Required: {MIN_CANDLES}"
        )

        return None

    df["SMMA20"] = calculate_smma(
        df["close"],
        SMMA_FAST
    )

    df["SMMA120"] = calculate_smma(
        df["close"],
        SMMA_SLOW
    )

    df["RSI"] = calculate_rsi(
        df["close"],
        RSI_PERIOD
    )

    return df


# ============================================================
# GENERATE SIGNAL
# ============================================================

def generate_signal(df):

    if df is None or df.empty:

        return {
            "Signal": "HOLD",
            "Confidence": 0.0,
            "Reason": "No historical data",
        }

    latest = df.iloc[-1]

    close = float(
        latest["close"]
    )

    smma20 = latest["SMMA20"]
    smma120 = latest["SMMA120"]
    rsi = latest["RSI"]

    # --------------------------------------------------------
    # Indicator validation
    # --------------------------------------------------------

    if (
        pd.isna(smma20)
        or pd.isna(smma120)
        or pd.isna(rsi)
    ):

        return {
            "Signal": "HOLD",
            "Confidence": 0.0,
            "Reason": "Indicators unavailable",
        }

    smma20 = float(smma20)
    smma120 = float(smma120)
    rsi = float(rsi)

    # --------------------------------------------------------
    # Bullish conditions
    # --------------------------------------------------------

    bullish_conditions = 0

    bullish_reasons = []

    if close > smma20:

        bullish_conditions += 1

        bullish_reasons.append(
            "Close above SMMA20"
        )

    if smma20 > smma120:

        bullish_conditions += 1

        bullish_reasons.append(
            "SMMA20 above SMMA120"
        )

    if 50 <= rsi <= 70:

        bullish_conditions += 1

        bullish_reasons.append(
            "RSI bullish range"
        )

    # --------------------------------------------------------
    # Bearish conditions
    # --------------------------------------------------------

    bearish_conditions = 0

    bearish_reasons = []

    if close < smma20:

        bearish_conditions += 1

        bearish_reasons.append(
            "Close below SMMA20"
        )

    if smma20 < smma120:

        bearish_conditions += 1

        bearish_reasons.append(
            "SMMA20 below SMMA120"
        )

    if 30 <= rsi < 50:

        bearish_conditions += 1

        bearish_reasons.append(
            "RSI bearish range"
        )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if bullish_conditions >= 3:

        signal = "BUY"

        confidence = (
            bullish_conditions / 3
        ) * 100

        reason = "; ".join(
            bullish_reasons
        )

    elif bearish_conditions >= 3:

        signal = "SELL"

        confidence = (
            bearish_conditions / 3
        ) * 100

        reason = "; ".join(
            bearish_reasons
        )

    else:

        signal = "HOLD"

        confidence = (
            max(
                bullish_conditions,
                bearish_conditions
            ) / 3
        ) * 100

        if bullish_conditions > bearish_conditions:

            reason = (
                "Partial bullish conditions"
            )

        elif bearish_conditions > bullish_conditions:

            reason = (
                "Partial bearish conditions"
            )

        else:

            reason = (
                "No complete signal"
            )

    return {
        "Signal": signal,
        "Confidence": round(
            confidence,
            2
        ),
        "Reason": reason,

        "Close": round(
            close,
            2
        ),

        "SMMA20": round(
            smma20,
            2
        ),

        "SMMA120": round(
            smma120,
            2
        ),

        "RSI": round(
            rsi,
            2
        ),
    }


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_stock(symbol):

    print()
    print("-" * 80)
    print(f"Processing: {symbol}")
    print("-" * 80)

    try:

        print(
            f"📡 Getting historical OHLC data "
            f"for {symbol}..."
        )

        historical = get_historical_data(
            symbol,
            days=250
        )

        if historical is None:

            print(
                f"❌ Historical data unavailable "
                f"for {symbol}"
            )

            return None

        print(
            f"Historical candles: "
            f"{len(historical)}"
        )

        indicators = calculate_indicators(
            historical
        )

        if indicators is None:

            print(
                f"⚠️ Indicators unavailable "
                f"for {symbol}"
            )

            return None

        signal = generate_signal(
            indicators
        )

        result = {
            "Symbol": symbol,

            "LTP": signal.get(
                "Close"
            ),

            "SMMA20": signal.get(
                "SMMA20"
            ),

            "SMMA120": signal.get(
                "SMMA120"
            ),

            "RSI": signal.get(
                "RSI"
            ),

            "Signal": signal.get(
                "Signal"
            ),

            "Confidence": signal.get(
                "Confidence"
            ),

            "Reason": signal.get(
                "Reason"
            ),

            "Historical Rows": len(
                indicators
            ),

            "Scan Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

        print()
        print(
            f"Signal     : {result['Signal']}"
        )

        print(
            f"Confidence : "
            f"{result['Confidence']}%"
        )

        print(
            f"Close      : {result['LTP']}"
        )

        print(
            f"SMMA20     : {result['SMMA20']}"
        )

        print(
            f"SMMA120    : {result['SMMA120']}"
        )

        print(
            f"RSI        : {result['RSI']}"
        )

        print(
            f"Reason     : {result['Reason']}"
        )

        return result

    except Exception as e:

        print(
            f"❌ Failed processing "
            f"{symbol}: {e}"
        )

        return None


# ============================================================
# RUN STAGE 19
# ============================================================

def run_stage19():

    print()
    print("=" * 80)
    print("STAGE 19 - SIGNAL ENGINE")
    print("=" * 80)

    ensure_output_directory()

    # --------------------------------------------------------
    # Load Stage 17
    # --------------------------------------------------------

    stage17 = load_stage17()

    if stage17 is None:

        return None

    symbol_column = get_symbol_column(
        stage17
    )

    if symbol_column is None:

        print(
            "❌ Could not find Symbol column "
            "in Stage 17 CSV."
        )

        return None

    # --------------------------------------------------------
    # Extract symbols
    # --------------------------------------------------------

    symbols = (
        stage17[symbol_column]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    symbols = [
        symbol
        for symbol in symbols
        if symbol
    ]

    print()
    print(
        f"Unique stocks to process: "
        f"{len(symbols)}"
    )

    if not symbols:

        print(
            "❌ No symbols found."
        )

        return None

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    results = []

    failed = 0

    for number, symbol in enumerate(
        symbols,
        start=1
    ):

        print()
        print(
            f"[{number}/{len(symbols)}] "
            f"{symbol}"
        )

        result = process_stock(
            symbol
        )

        if result is not None:

            results.append(
                result
            )

        else:

            failed += 1

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    if results:

        result_df = pd.DataFrame(
            results
        )

    else:

        result_df = pd.DataFrame(
            columns=[
                "Symbol",
                "LTP",
                "SMMA20",
                "SMMA120",
                "RSI",
                "Signal",
                "Confidence",
                "Reason",
                "Historical Rows",
                "Scan Time",
            ]
        )

    # --------------------------------------------------------
    # Save complete results
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    all_file = os.path.join(
        OUTPUT_DIR,
        f"stage19_{timestamp}.csv"
    )

    result_df.to_csv(
        all_file,
        index=False
    )

    # --------------------------------------------------------
    # Save qualified signals
    # --------------------------------------------------------

    qualified_df = result_df[
        result_df["Signal"].isin(
            ["BUY", "SELL"]
        )
    ].copy()

    qualified_file = os.path.join(
        OUTPUT_DIR,
        "stage19_signals.csv"
    )

    qualified_df.to_csv(
        qualified_file,
        index=False
    )

    # --------------------------------------------------------
    # Save latest result
    # --------------------------------------------------------

    latest_file = os.path.join(
        OUTPUT_DIR,
        "stage19_latest.csv"
    )

    result_df.to_csv(
        latest_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    buy_count = len(
        result_df[
            result_df["Signal"] == "BUY"
        ]
    )

    sell_count = len(
        result_df[
            result_df["Signal"] == "SELL"
        ]
    )

    hold_count = len(
        result_df[
            result_df["Signal"] == "HOLD"
        ]
    )

    print()
    print("=" * 80)
    print("STAGE 19 RESULTS")
    print("=" * 80)

    print(
        f"Stocks processed : "
        f"{len(symbols)}"
    )

    print(
        f"Successful       : "
        f"{len(result_df)}"
    )

    print(
        f"Failed/skipped   : "
        f"{failed}"
    )

    print(
        f"BUY signals      : "
        f"{buy_count}"
    )

    print(
        f"SELL signals     : "
        f"{sell_count}"
    )

    print(
        f"HOLD signals     : "
        f"{hold_count}"
    )

    print()
    print(
        f"Saved: {all_file}"
    )

    print(
        f"Saved: {qualified_file}"
    )

    print(
        f"Saved: {latest_file}"
    )

    # --------------------------------------------------------
    # Display qualified signals
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("QUALIFIED STAGE 19 SIGNALS")
    print("=" * 80)

    if qualified_df.empty:

        print(
            "No BUY/SELL signals generated."
        )

    else:

        display_columns = [
            "Symbol",
            "LTP",
            "SMMA20",
            "SMMA120",
            "RSI",
            "Signal",
            "Confidence",
            "Reason",
        ]

        print(
            qualified_df[
                display_columns
            ].to_string(
                index=False
            )
        )

    print()
    print("=" * 80)
    print("STAGE 19 COMPLETE")
    print("=" * 80)

    return result_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_stage19()