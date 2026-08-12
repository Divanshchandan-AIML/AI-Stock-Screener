# ============================================================
# STAGE 20 - FEATURE BUILDER
# stage20/feature_builder.py
# ============================================================

import os
import time

import pandas as pd
import numpy as np

from api.historical_data import get_historical_data
from utils.token_map import TOKENS


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "data/stage20"

TRAINING_FILE = os.path.join(
    OUTPUT_DIR,
    "stage20_training_data.csv"
)

# Future return horizon
FUTURE_HORIZON = 5

# Minimum daily candles required
MIN_CANDLES = 150


# ============================================================
# DIRECTORY
# ============================================================

def ensure_directories():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# ============================================================
# SMA
# ============================================================

def calculate_sma(
    series,
    period
):

    return (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .rolling(
            window=period,
            min_periods=period
        )
        .mean()
    )


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

    average_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = (
        average_gain
        /
        average_loss.replace(
            0,
            np.nan
        )
    )

    rsi = (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )

    # If there are gains but no losses,
    # RSI should be 100.
    rsi = rsi.where(
        average_loss != 0,
        100
    )

    return rsi


# ============================================================
# PREPARE OHLC DATA
# ============================================================

def prepare_data(df):

    if df is None:

        return None

    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print(
            f"❌ Missing columns: {missing}"
        )

        return None

    data = df.copy()

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove invalid OHLC rows
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    data = data.sort_values(
        "date"
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    data = data.drop_duplicates(
        subset=["date"],
        keep="last"
    )

    data = data.reset_index(
        drop=True
    )

    return data


# ============================================================
# HISTORICAL ETQ PROXY
# ============================================================
#
# IMPORTANT:
#
# We do NOT use today's live intraday values for historical
# rows because that would create data leakage.
#
# For historical training, ETQ is represented using daily
# volume information:
#
#   etq_5m  = estimated 5-minute quantity
#   etq_20m = estimated 20-minute quantity
#   etq_60m = estimated 60-minute quantity
#
# NSE regular session is approximately 375 minutes.
#
# Later, Stage 20.3 will use REAL 1-minute data for live
# prediction.
#
# ============================================================

def calculate_historical_etq(
    volume
):

    volume = pd.to_numeric(
        volume,
        errors="coerce"
    )

    # NSE session:
    # 09:15 - 15:30 = 375 minutes

    session_minutes = 375

    estimated_per_minute = (
        volume
        /
        session_minutes
    )

    etq_5m = (
        estimated_per_minute
        * 5
    )

    etq_20m = (
        estimated_per_minute
        * 20
    )

    etq_60m = (
        estimated_per_minute
        * 60
    )

    return (
        etq_5m,
        etq_20m,
        etq_60m
    )


# ============================================================
# BUILD FEATURES FOR ONE STOCK
# ============================================================

def build_features(
    df,
    symbol
):

    data = prepare_data(
        df
    )

    if data is None:

        return None

    if len(data) < MIN_CANDLES:

        print(
            f"⚠️ {symbol}: "
            f"only {len(data)} candles. "
            f"Required {MIN_CANDLES}."
        )

        return None

    # ========================================================
    # RETURNS
    # ========================================================

    data["return_1d"] = (
        data["close"]
        .pct_change()
    )

    data["return_5d"] = (
        data["close"]
        .pct_change(5)
    )

    data["return_10d"] = (
        data["close"]
        .pct_change(10)
    )

    # ========================================================
    # SMA 20
    # ========================================================

    data["sma20"] = calculate_sma(
        data["close"],
        20
    )

    # ========================================================
    # SMA 120
    # ========================================================

    data["sma120"] = calculate_sma(
        data["close"],
        120
    )

    # ========================================================
    # SMA RELATIONSHIP
    # ========================================================

    data["sma20_sma120_ratio"] = (
        data["sma20"]
        /
        data["sma120"]
    )

    # ========================================================
    # RSI
    # ========================================================

    data["rsi"] = calculate_rsi(
        data["close"],
        14
    )

    # ========================================================
    # ETQ FEATURES
    # ========================================================

    (
        data["etq_5m"],
        data["etq_20m"],
        data["etq_60m"]
    ) = calculate_historical_etq(
        data["volume"]
    )

    # ========================================================
    # PRICE RANGE
    # ========================================================

    data["high_low_range"] = (
        (
            data["high"]
            -
            data["low"]
        )
        /
        data["close"]
    )

    data["open_close_range"] = (
        (
            data["close"]
            -
            data["open"]
        )
        /
        data["open"]
    )

    # ========================================================
    # VOLATILITY
    # ========================================================

    data["volatility_10d"] = (
        data["return_1d"]
        .rolling(
            10
        )
        .std()
    )

    data["volatility_20d"] = (
        data["return_1d"]
        .rolling(
            20
        )
        .std()
    )

    # ========================================================
    # VOLUME
    # ========================================================

    data["volume_change"] = (
        data["volume"]
        .pct_change()
    )

    data["volume_sma20"] = (
        data["volume"]
        .rolling(
            20
        )
        .mean()
    )

    data["volume_ratio"] = (
        data["volume"]
        /
        data["volume_sma20"]
    )

    # ========================================================
    # MOMENTUM
    # ========================================================

    data["momentum_5d"] = (
        data["close"]
        /
        data["close"].shift(5)
        - 1
    )

    data["momentum_20d"] = (
        data["close"]
        /
        data["close"].shift(20)
        - 1
    )

    # ========================================================
    # FUTURE CLOSE
    # ========================================================

    data["future_close"] = (
        data["close"]
        .shift(
            -FUTURE_HORIZON
        )
    )

    # ========================================================
    # FUTURE RETURN
    # ========================================================

    data["future_return"] = (
        data["future_close"]
        /
        data["close"]
        - 1
    )

    # ========================================================
    # TARGET
    # ========================================================
    #
    # 1 = price increases after 5 candles
    # 0 = price does not increase
    #
    # ========================================================

    data["target"] = np.where(

        data["future_return"].notna(),

        (
            data["future_return"]
            > 0
        ).astype(int),

        np.nan
    )

    # ========================================================
    # SYMBOL
    # ========================================================

    data["Symbol"] = symbol

    # ========================================================
    # EXACT ML FEATURE LIST
    # ========================================================

    feature_columns = [

        "close",

        "volume",

        "return_1d",

        "return_5d",

        "return_10d",

        "sma20",

        "sma120",

        "sma20_sma120_ratio",

        "rsi",

        "etq_5m",

        "etq_20m",

        "etq_60m",

        "high_low_range",

        "open_close_range",

        "volatility_10d",

        "volatility_20d",

        "volume_change",

        "volume_ratio",

        "momentum_5d",

        "momentum_20d",
    ]

    # ========================================================
    # FINAL OUTPUT COLUMNS
    # ========================================================

    final_columns = [

        "date",

        "Symbol",

        *feature_columns,

        "future_return",

        "target",
    ]

    data = data[
        final_columns
    ]

    # ========================================================
    # REMOVE INDICATOR WARM-UP ROWS
    # ========================================================

    data = data.dropna(
        subset=feature_columns
    )

    # ========================================================
    # REMOVE FUTURE TARGET ROWS
    # ========================================================

    data = data.dropna(
        subset=[
            "future_return",
            "target",
        ]
    )

    # ========================================================
    # TARGET INTEGER
    # ========================================================

    data["target"] = (
        data["target"]
        .astype(int)
    )

    # ========================================================
    # RESET INDEX
    # ========================================================

    data = data.reset_index(
        drop=True
    )

    return data


# ============================================================
# GET SYMBOLS
# ============================================================

def get_symbols():

    symbols = list(
        TOKENS.keys()
    )

    symbols = list(
        dict.fromkeys(
            symbols
        )
    )

    return symbols


# ============================================================
# BUILD TRAINING DATASET
# ============================================================

def build_training_dataset(
    symbols=None,
    days=500,
    limit=None
):

    ensure_directories()

    print()
    print("=" * 80)
    print("STAGE 20 - BUILDING ML TRAINING DATA")
    print("=" * 80)

    # --------------------------------------------------------
    # Symbols
    # --------------------------------------------------------

    if symbols is None:

        symbols = get_symbols()

    if limit is not None:

        symbols = symbols[
            :limit
        ]

    print()
    print(
        f"Stocks selected: "
        f"{len(symbols)}"
    )

    all_features = []

    failed = 0

    # ========================================================
    # PROCESS STOCKS
    # ========================================================

    for number, symbol in enumerate(
        symbols,
        start=1
    ):

        print()
        print(
            f"[{number}/{len(symbols)}] "
            f"{symbol}"
        )

        try:

            # ------------------------------------------------
            # Historical data
            # ------------------------------------------------

            print(
                f"Getting historical data "
                f"for {symbol}..."
            )

            historical = (
                get_historical_data(
                    symbol,
                    days=days
                )
            )

            if historical is None:

                print(
                    f"⚠️ No historical data "
                    f"for {symbol}"
                )

                failed += 1

                continue

            print(
                f"Historical candles: "
                f"{len(historical)}"
            )

            # ------------------------------------------------
            # Build features
            # ------------------------------------------------

            features = build_features(
                historical,
                symbol
            )

            if features is None:

                failed += 1

                continue

            print(
                f"Training rows created: "
                f"{len(features)}"
            )

            all_features.append(
                features
            )

            # ------------------------------------------------
            # Small delay
            # ------------------------------------------------

            time.sleep(
                0.5
            )

        except Exception as e:

            print(
                f"❌ Failed {symbol}: "
                f"{type(e).__name__}: {e}"
            )

            failed += 1

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if not all_features:

        print()
        print(
            "❌ No training data generated."
        )

        return None

    # ========================================================
    # COMBINE
    # ========================================================

    training_df = pd.concat(
        all_features,
        ignore_index=True
    )

    # ========================================================
    # DATE
    # ========================================================

    training_df["date"] = (
        pd.to_datetime(
            training_df["date"],
            errors="coerce"
        )
    )

    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    training_df = (
        training_df
        .sort_values(
            [
                "date",
                "Symbol"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    training_df.to_csv(
        TRAINING_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("STAGE 20 TRAINING DATA COMPLETE")
    print("=" * 80)

    print(
        f"Stocks requested : "
        f"{len(symbols)}"
    )

    print(
        f"Stocks failed    : "
        f"{failed}"
    )

    print(
        f"Training rows    : "
        f"{len(training_df)}"
    )

    up_count = int(
        (
            training_df["target"]
            == 1
        ).sum()
    )

    down_count = int(
        (
            training_df["target"]
            == 0
        ).sum()
    )

    print(
        f"UP target        : "
        f"{up_count}"
    )

    print(
        f"DOWN target      : "
        f"{down_count}"
    )

    print()
    print(
        f"Saved: {TRAINING_FILE}"
    )

    # ========================================================
    # PRINT FEATURES
    # ========================================================

    print()
    print(
        "ML feature columns:"
    )

    feature_columns = [

        "close",
        "volume",
        "return_1d",
        "return_5d",
        "return_10d",
        "sma20",
        "sma120",
        "sma20_sma120_ratio",
        "rsi",
        "etq_5m",
        "etq_20m",
        "etq_60m",
        "high_low_range",
        "open_close_range",
        "volatility_10d",
        "volatility_20d",
        "volume_change",
        "volume_ratio",
        "momentum_5d",
        "momentum_20d",
    ]

    for column in feature_columns:

        print(
            f"  - {column}"
        )

    print()
    print(
        "=" * 80
    )

    return training_df


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("STAGE 20 FEATURE BUILDER TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # TEST ONLY 5 STOCKS
    # --------------------------------------------------------

    test_symbols = [

        "SBIN-EQ",

        "RELIANCE-EQ",

        "ITC-EQ",

        "SUZLON-EQ",

        "TATAMOTORS-EQ",
    ]

    result = build_training_dataset(

        symbols=test_symbols,

        days=500
    )

    if result is not None:

        print()
        print(
            "Sample training data:"
        )

        print(
            result.tail(
                10
            ).to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # Verify exact columns
        # ----------------------------------------------------

        expected_features = [

            "close",
            "volume",
            "return_1d",
            "return_5d",
            "return_10d",
            "sma20",
            "sma120",
            "sma20_sma120_ratio",
            "rsi",
            "etq_5m",
            "etq_20m",
            "etq_60m",
            "high_low_range",
            "open_close_range",
            "volatility_10d",
            "volatility_20d",
            "volume_change",
            "volume_ratio",
            "momentum_5d",
            "momentum_20d",
        ]

        print()
        print(
            "=" * 80
        )

        print(
            "FEATURE COLUMN VERIFICATION"
        )

        print(
            "=" * 80
        )

        missing = [
            column
            for column in expected_features
            if column not in result.columns
        ]

        if not missing:

            print(
                "✅ ALL 20 ML FEATURES ARE PRESENT"
            )

        else:

            print(
                "❌ Missing features:"
            )

            for column in missing:

                print(
                    f"   - {column}"
                )

    else:

        print(
            "❌ Feature builder test failed."
        )

    print()
    print("=" * 80)
    print(
        "STAGE 20 FEATURE BUILDER TEST COMPLETE"
    )
    print("=" * 80)