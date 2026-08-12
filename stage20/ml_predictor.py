# ============================================================
# STAGE 20.3 - ML PREDICTOR
# stage20/ml_predictor.py
# ============================================================

import os
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from api.historical_data import get_historical_data
from utils.token_map import TOKENS


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "stock_classifier.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "feature_columns.json"
)

METRICS_FILE = os.path.join(
    MODEL_DIR,
    "training_metrics.json"
)

OUTPUT_DIR = "data/stage20"

PREDICTION_FILE = os.path.join(
    OUTPUT_DIR,
    "stage20_predictions.csv"
)

# Number of historical calendar days.
HISTORICAL_DAYS = 500

# Minimum candles required.
MIN_CANDLES = 150

# Delay between API requests.
REQUEST_DELAY = 0.5


warnings.filterwarnings(
    "ignore"
)


# ============================================================
# EXPECTED FEATURE SCHEMA
# ============================================================

EXPECTED_FEATURES = [
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


# ============================================================
# DIRECTORY
# ============================================================

def ensure_directories():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print("=" * 80)
    print("LOADING ML MODEL")
    print("=" * 80)

    if not os.path.exists(
        MODEL_FILE
    ):

        raise FileNotFoundError(
            f"Model file not found: "
            f"{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        f"Model loaded: {MODEL_FILE}"
    )

    print(
        f"Model type: "
        f"{type(model).__name__}"
    )

    return model


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

def load_feature_columns():

    print()
    print("=" * 80)
    print("LOADING FEATURE COLUMNS")
    print("=" * 80)

    if not os.path.exists(
        FEATURE_FILE
    ):

        raise FileNotFoundError(
            f"Feature file not found: "
            f"{FEATURE_FILE}"
        )

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(
            file
        )

    # --------------------------------------------------------
    # Support both:
    #
    # ["close", "volume", ...]
    #
    # and:
    #
    # {"features": ["close", "volume", ...]}
    # --------------------------------------------------------

    if isinstance(
        data,
        list
    ):

        feature_columns = data

    elif isinstance(
        data,
        dict
    ):

        if "features" in data:

            feature_columns = (
                data["features"]
            )

        elif "feature_columns" in data:

            feature_columns = (
                data["feature_columns"]
            )

        else:

            raise ValueError(
                "feature_columns.json does "
                "not contain a recognized "
                "feature list."
            )

    else:

        raise ValueError(
            "Invalid feature_columns.json format."
        )

    feature_columns = list(
        feature_columns
    )

    print(
        f"Features loaded: "
        f"{len(feature_columns)}"
    )

    for number, feature in enumerate(
        feature_columns,
        start=1
    ):

        print(
            f"  {number:02d}. {feature}"
        )

    return feature_columns


# ============================================================
# VALIDATE FEATURE SCHEMA
# ============================================================

def validate_feature_schema(
    feature_columns
):

    print()
    print("=" * 80)
    print("VALIDATING FEATURE SCHEMA")
    print("=" * 80)

    missing = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in feature_columns
    ]

    extra = [
        feature
        for feature in feature_columns
        if feature not in EXPECTED_FEATURES
    ]

    if missing:

        print()
        print(
            "Missing expected features:"
        )

        for feature in missing:

            print(
                f"  - {feature}"
            )

        raise ValueError(
            "Trained model feature schema "
            "does not match Stage 20.3."
        )

    if extra:

        print()
        print(
            "Additional model features:"
        )

        for feature in extra:

            print(
                f"  - {feature}"
            )

    print()
    print(
        "Feature schema validation: OK"
    )

    return True


# ============================================================
# SMOOTH MOVING AVERAGE
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
            period
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

    rsi = rsi.where(
        average_loss != 0,
        100
    )

    return rsi


# ============================================================
# PREPARE HISTORICAL DATA
# ============================================================

def prepare_data(
    df
):

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
            f"Missing OHLC columns: "
            f"{missing}"
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
    # Numeric
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
    # Remove invalid rows
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            "date",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    data = data.sort_values(
        "date"
    )

    # --------------------------------------------------------
    # Remove duplicate dates
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
# ETQ FEATURES
# ============================================================
#
# Stage 20 training uses daily-volume historical proxies.
#
# NSE session ≈ 375 minutes.
#
# ETQ:
#
# 5m  = daily volume * 5 / 375
# 20m = daily volume * 20 / 375
# 60m = daily volume * 60 / 375
#
# These are historical proxies and do not use future data.
#
# ============================================================

def calculate_etq_features(
    data
):

    session_minutes = 375.0

    data["etq_5m"] = (
        data["volume"]
        *
        5.0
        /
        session_minutes
    )

    data["etq_20m"] = (
        data["volume"]
        *
        20.0
        /
        session_minutes
    )

    data["etq_60m"] = (
        data["volume"]
        *
        60.0
        /
        session_minutes
    )

    return data


# ============================================================
# BUILD PREDICTION FEATURES
# ============================================================

def build_prediction_features(
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
            f"{symbol}: only "
            f"{len(data)} candles. "
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
    # SMA
    # ========================================================

    data["sma20"] = calculate_sma(
        data["close"],
        20
    )

    data["sma120"] = calculate_sma(
        data["close"],
        120
    )

    # ========================================================
    # SMA RATIO
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
    # ETQ
    # ========================================================

    data = calculate_etq_features(
        data
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
        .rolling(10)
        .std()
    )

    data["volatility_20d"] = (
        data["return_1d"]
        .rolling(20)
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
        .rolling(20)
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
    # SYMBOL
    # ========================================================

    data["Symbol"] = symbol

    # ========================================================
    # REQUIRED FEATURES
    # ========================================================

    missing_features = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in data.columns
    ]

    if missing_features:

        print(
            f"{symbol}: missing prediction "
            f"features: {missing_features}"
        )

        return None

    # ========================================================
    # REMOVE INVALID FEATURE ROWS
    # ========================================================

    data = data.dropna(
        subset=EXPECTED_FEATURES
    )

    if data.empty:

        print(
            f"{symbol}: no valid feature rows."
        )

        return None

    data = data.reset_index(
        drop=True
    )

    return data


# ============================================================
# GET LATEST FEATURE ROW
# ============================================================

def get_latest_feature_row(
    historical,
    symbol,
    feature_columns
):

    features = build_prediction_features(
        historical,
        symbol
    )

    if features is None:

        return None

    latest = features.iloc[
        -1
    ].copy()

    # --------------------------------------------------------
    # Verify every model feature exists
    # --------------------------------------------------------

    missing = [
        feature
        for feature in feature_columns
        if feature not in latest.index
    ]

    if missing:

        print()
        print(
            f"{symbol}: model features missing:"
        )

        for feature in missing:

            print(
                f"  - {feature}"
            )

        return None

    # --------------------------------------------------------
    # Check numeric values
    # --------------------------------------------------------

    for feature in feature_columns:

        value = latest[
            feature
        ]

        if not np.isfinite(
            float(value)
        ):

            print(
                f"{symbol}: invalid value "
                f"for {feature}: {value}"
            )

            return None

    return latest


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_stock(
    model,
    feature_columns,
    historical,
    symbol
):

    latest = get_latest_feature_row(
        historical,
        symbol,
        feature_columns
    )

    if latest is None:

        return None

    # ========================================================
    # CREATE MODEL INPUT
    # ========================================================

    X = pd.DataFrame(
        [
            [
                latest[feature]
                for feature in feature_columns
            ]
        ],
        columns=feature_columns
    )

    # ========================================================
    # PREDICT CLASS
    # ========================================================

    prediction = model.predict(
        X
    )

    prediction_value = int(
        prediction[0]
    )

    # ========================================================
    # PREDICT PROBABILITY
    # ========================================================

    probability_up = np.nan
    probability_down = np.nan

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(X)[0]
        )

        classes = list(
            model.classes_
        )

        if 1 in classes:

            index_up = classes.index(
                1
            )

            probability_up = float(
                probabilities[index_up]
            )

        if 0 in classes:

            index_down = classes.index(
                0
            )

            probability_down = float(
                probabilities[index_down]
            )

    # ========================================================
    # SIGNAL
    # ========================================================

    if prediction_value == 1:

        signal = "BUY"

    else:

        signal = "SELL"

    # ========================================================
    # CONFIDENCE
    # ========================================================

    probabilities_available = (
        np.isfinite(probability_up)
        and
        np.isfinite(probability_down)
    )

    if probabilities_available:

        confidence = max(
            probability_up,
            probability_down
        ) * 100.0

    else:

        confidence = np.nan

    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "date": latest["date"],

        "Symbol": symbol,

        "close": float(
            latest["close"]
        ),

        "volume": float(
            latest["volume"]
        ),

        "sma20": float(
            latest["sma20"]
        ),

        "sma120": float(
            latest["sma120"]
        ),

        "rsi": float(
            latest["rsi"]
        ),

        "etq_5m": float(
            latest["etq_5m"]
        ),

        "etq_20m": float(
            latest["etq_20m"]
        ),

        "etq_60m": float(
            latest["etq_60m"]
        ),

        "momentum_5d": float(
            latest["momentum_5d"]
        ),

        "momentum_20d": float(
            latest["momentum_20d"]
        ),

        "prediction": prediction_value,

        "signal": signal,

        "probability_up": probability_up,

        "probability_down": probability_down,

        "confidence": confidence,
    }

    return result


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
# PREDICT MULTIPLE STOCKS
# ============================================================

def predict_stocks(
    symbols=None,
    days=HISTORICAL_DAYS,
    limit=None
):

    ensure_directories()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # LOAD FEATURES
    # ========================================================

    feature_columns = (
        load_feature_columns()
    )

    validate_feature_schema(
        feature_columns
    )

    # ========================================================
    # SYMBOLS
    # ========================================================

    if symbols is None:

        symbols = get_symbols()

    if limit is not None:

        symbols = symbols[
            :limit
        ]

    print()
    print("=" * 80)
    print("STAGE 20.3 - ML STOCK PREDICTION")
    print("=" * 80)

    print()
    print(
        f"Stocks selected: {len(symbols)}"
    )

    results = []

    failed = 0

    # ========================================================
    # LOOP
    # ========================================================

    for number, symbol in enumerate(
        symbols,
        start=1
    ):

        print()
        print(
            "-" * 80
        )

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
                    f"No historical data "
                    f"for {symbol}"
                )

                failed += 1

                continue

            print(
                f"Historical candles: "
                f"{len(historical)}"
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            result = predict_stock(
                model=model,
                feature_columns=feature_columns,
                historical=historical,
                symbol=symbol
            )

            if result is None:

                print(
                    f"Prediction failed "
                    f"for {symbol}"
                )

                failed += 1

                continue

            results.append(
                result
            )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            print()
            print(
                f"Close       : "
                f"{result['close']:.2f}"
            )

            print(
                f"RSI         : "
                f"{result['rsi']:.2f}"
            )

            print(
                f"SMA20       : "
                f"{result['sma20']:.2f}"
            )

            print(
                f"SMA120      : "
                f"{result['sma120']:.2f}"
            )

            print(
                f"Momentum 5D : "
                f"{result['momentum_5d']:.4f}"
            )

            print(
                f"Momentum 20D: "
                f"{result['momentum_20d']:.4f}"
            )

            print(
                f"Prediction  : "
                f"{result['prediction']}"
            )

            print(
                f"Signal      : "
                f"{result['signal']}"
            )

            if np.isfinite(
                result["probability_up"]
            ):

                print(
                    f"UP probability: "
                    f"{result['probability_up'] * 100:.2f}%"
                )

            if np.isfinite(
                result["probability_down"]
            ):

                print(
                    f"DOWN probability: "
                    f"{result['probability_down'] * 100:.2f}%"
                )

            if np.isfinite(
                result["confidence"]
            ):

                print(
                    f"Confidence  : "
                    f"{result['confidence']:.2f}%"
                )

            time.sleep(
                REQUEST_DELAY
            )

        except Exception as e:

            print()
            print(
                f"❌ Failed {symbol}: "
                f"{type(e).__name__}: {e}"
            )

            failed += 1

    # ========================================================
    # NO RESULTS
    # ========================================================

    if not results:

        print()
        print("=" * 80)
        print(
            "❌ NO PREDICTIONS GENERATED"
        )
        print("=" * 80)

        print(
            f"Stocks attempted: {len(symbols)}"
        )

        print(
            f"Failed/skipped: {failed}"
        )

        return None

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    # ========================================================
    # SORT
    # ========================================================

    if "confidence" in result_df.columns:

        result_df = (
            result_df
            .sort_values(
                "confidence",
                ascending=False,
                na_position="last"
            )
            .reset_index(
                drop=True
            )
        )

    # ========================================================
    # SAVE
    # ========================================================

    result_df.to_csv(
        PREDICTION_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    buy_count = int(
        (
            result_df["signal"]
            == "BUY"
        ).sum()
    )

    sell_count = int(
        (
            result_df["signal"]
            == "SELL"
        ).sum()
    )

    print()
    print("=" * 80)
    print("STAGE 20.3 ML PREDICTION COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Stocks requested : {len(symbols)}"
    )

    print(
        f"Predictions      : {len(result_df)}"
    )

    print(
        f"Failed/skipped   : {failed}"
    )

    print(
        f"BUY predictions  : {buy_count}"
    )

    print(
        f"SELL predictions : {sell_count}"
    )

    print()
    print(
        f"Saved: {PREDICTION_FILE}"
    )

    # ========================================================
    # DISPLAY TOP RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("ML PREDICTION RESULTS")
    print("=" * 80)

    display_columns = [
        "Symbol",
        "close",
        "rsi",
        "sma20",
        "sma120",
        "prediction",
        "signal",
        "probability_up",
        "probability_down",
        "confidence",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in result_df.columns
    ]

    print()

    print(
        result_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 80)

    return result_df


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print()
    print("=" * 80)
    print("STAGE 20.3 ML PREDICTOR TEST")
    print("=" * 80)

    try:

        # ----------------------------------------------------
        # FIRST TEST ONLY
        # ----------------------------------------------------
        #
        # Use the same 5 stocks used in Stage 20.1.
        #
        # Once this works, we can remove the limit and
        # scan the complete stock universe.
        #
        # ----------------------------------------------------

        test_symbols = [
            "SBIN-EQ",
            "RELIANCE-EQ",
            "ITC-EQ",
            "SUZLON-EQ",
            "TATAMOTORS-EQ",
        ]

        result = predict_stocks(
            symbols=test_symbols,
            days=HISTORICAL_DAYS
        )

        if result is None:

            print()
            print(
                "❌ Stage 20.3 prediction test failed."
            )

            return

        print()
        print(
            "Top prediction results:"
        )

        print(
            result.head(
                10
            ).to_string(
                index=False
            )
        )

    except Exception as e:

        print()
        print("=" * 80)
        print("STAGE 20.3 PREDICTION FAILED")
        print("=" * 80)

        print()
        print(
            f"Error type: "
            f"{type(e).__name__}"
        )

        print(
            f"Error: {e}"
        )

    print()
    print("=" * 80)
    print("STAGE 20.3 TEST COMPLETE")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()