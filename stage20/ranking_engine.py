# ============================================================
# STAGE 20.4 - ML + STAGE 19 RANKING ENGINE
# stage20/ranking_engine.py
# ============================================================

import os
import json
import glob
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

STAGE20_DIR = "data/stage20"
STAGE19_DIR = "data/stage19"

ML_FILE = os.path.join(
    STAGE20_DIR,
    "stage20_predictions.csv"
)

STAGE19_FILE = os.path.join(
    STAGE19_DIR,
    "stage19_latest.csv"
)

OUTPUT_FILE = os.path.join(
    STAGE20_DIR,
    "stage20_ranked_results.csv"
)

METRICS_FILE = os.path.join(
    STAGE20_DIR,
    "stage20_ranking_metrics.json"
)


# ============================================================
# WEIGHTS
# ============================================================

ML_WEIGHT = 0.60
STAGE19_WEIGHT = 0.25
CONFIDENCE_WEIGHT = 0.15


# ============================================================
# DIRECTORY
# ============================================================

def ensure_directories():

    os.makedirs(
        STAGE20_DIR,
        exist_ok=True
    )


# ============================================================
# NORMALIZE SYMBOL
# ============================================================

def normalize_symbol(value):

    if pd.isna(value):

        return ""

    return (
        str(value)
        .strip()
        .upper()
    )


# ============================================================
# LOAD ML PREDICTIONS
# ============================================================

def load_ml_predictions():

    print()
    print("=" * 80)
    print("LOADING STAGE 20.3 ML PREDICTIONS")
    print("=" * 80)

    if not os.path.exists(
        ML_FILE
    ):

        raise FileNotFoundError(
            f"ML prediction file not found:\n"
            f"{ML_FILE}"
        )

    df = pd.read_csv(
        ML_FILE
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    required = [
        "Symbol",
        "prediction",
        "signal",
        "probability_up",
        "probability_down",
        "confidence",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing ML columns: "
            + str(missing)
        )

    # --------------------------------------------------------
    # Symbol
    # --------------------------------------------------------

    df["Symbol"] = (
        df["Symbol"]
        .apply(
            normalize_symbol
        )
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    df["prediction"] = pd.to_numeric(
        df["prediction"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Signal
    # --------------------------------------------------------

    df["signal"] = (
        df["signal"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    df["probability_up"] = pd.to_numeric(
        df["probability_up"],
        errors="coerce"
    )

    df["probability_down"] = pd.to_numeric(
        df["probability_down"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Some ML output files store probabilities as:
    #
    # 0.3564
    #
    # instead of:
    #
    # 35.64
    #
    # Convert automatically.
    # --------------------------------------------------------

    if (
        df["probability_up"]
        .dropna()
        .max()
        <= 1.0
    ):

        df["probability_up"] = (
            df["probability_up"]
            * 100
        )

    if (
        df["probability_down"]
        .dropna()
        .max()
        <= 1.0
    ):

        df["probability_down"] = (
            df["probability_down"]
            * 100
        )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    df["confidence"] = pd.to_numeric(
        df["confidence"],
        errors="coerce"
    )

    if (
        df["confidence"]
        .dropna()
        .max()
        <= 1.0
    ):

        df["confidence"] = (
            df["confidence"]
            * 100
        )

    # --------------------------------------------------------
    # Clip
    # --------------------------------------------------------

    df["probability_up"] = (
        df["probability_up"]
        .clip(0, 100)
    )

    df["probability_down"] = (
        df["probability_down"]
        .clip(0, 100)
    )

    df["confidence"] = (
        df["confidence"]
        .clip(0, 100)
    )

    # --------------------------------------------------------
    # Remove invalid symbols
    # --------------------------------------------------------

    df = df[
        df["Symbol"] != ""
    ]

    # --------------------------------------------------------
    # One row per stock
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset=["Symbol"],
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "ML predictions loaded successfully."
    )

    print(
        f"Valid ML stocks: {len(df)}"
    )

    return df


# ============================================================
# LOAD STAGE 19
# ============================================================

def load_stage19():

    print()
    print("=" * 80)
    print("LOADING STAGE 19 SIGNALS")
    print("=" * 80)

    # --------------------------------------------------------
    # Prefer exact latest file
    # --------------------------------------------------------

    if os.path.exists(
        STAGE19_FILE
    ):

        file_path = STAGE19_FILE

    else:

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        files = glob.glob(
            os.path.join(
                STAGE19_DIR,
                "*.csv"
            )
        )

        files = [
            file
            for file in files
            if os.path.getsize(file) > 0
        ]

        if not files:

            print(
                "No Stage 19 CSV found."
            )

            return None

        file_path = max(
            files,
            key=os.path.getmtime
        )

    print()
    print(
        f"Stage 19 file selected:"
    )

    print(
        file_path
    )

    df = pd.read_csv(
        file_path
    )

    print(
        f"Stage 19 rows loaded: "
        f"{len(df)}"
    )

    print()
    print(
        "Stage 19 columns:"
    )

    for column in df.columns:

        print(
            f"  - {column}"
        )

    # --------------------------------------------------------
    # EXACT FORMAT FROM YOUR FILE
    # --------------------------------------------------------

    required = [
        "Symbol",
        "Signal",
        "Confidence",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Stage 19 file is missing: "
            + str(missing)
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    df["Symbol"] = (
        df["Symbol"]
        .apply(
            normalize_symbol
        )
    )

    df["Signal"] = (
        df["Signal"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["Confidence"] = pd.to_numeric(
        df["Confidence"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Rename Stage 19 columns
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "Signal":
            "stage19_signal",

            "Confidence":
            "stage19_confidence",

            "LTP":
            "stage19_ltp",

            "SMMA20":
            "stage19_smma20",

            "SMMA120":
            "stage19_smma120",

            "RSI":
            "stage19_rsi",

            "Reason":
            "stage19_reason",

            "Historical Rows":
            "stage19_historical_rows",

            "Scan Time":
            "stage19_scan_time",
        }
    )

    # --------------------------------------------------------
    # Keep useful columns
    # --------------------------------------------------------

    keep_columns = [

        "Symbol",

        "stage19_signal",

        "stage19_confidence",

        "stage19_ltp",

        "stage19_smma20",

        "stage19_smma120",

        "stage19_rsi",

        "stage19_reason",

        "stage19_historical_rows",

        "stage19_scan_time",

    ]

    keep_columns = [
        column
        for column in keep_columns
        if column in df.columns
    ]

    df = df[
        keep_columns
    ]

    # --------------------------------------------------------
    # Remove invalid symbols
    # --------------------------------------------------------

    df = df[
        df["Symbol"] != ""
    ]

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset=["Symbol"],
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "Stage 19 signals loaded successfully."
    )

    return df


# ============================================================
# CALCULATE ML SCORE
# ============================================================

def calculate_ml_score(row):

    if int(
        row["prediction"]
    ) == 1:

        return float(
            row["probability_up"]
        )

    return float(
        row["probability_down"]
    )


# ============================================================
# STAGE 19 SCORE
# ============================================================

def calculate_stage19_score(
    signal
):

    signal = str(
        signal
    ).upper().strip()

    if signal in [
        "BUY",
        "SELL",
    ]:

        return 100.0

    if signal == "HOLD":

        return 50.0

    return 50.0


# ============================================================
# SIGNAL AGREEMENT
# ============================================================

def calculate_agreement(
    ml_signal,
    stage19_signal
):

    ml_signal = str(
        ml_signal
    ).upper().strip()

    stage19_signal = str(
        stage19_signal
    ).upper().strip()

    if stage19_signal in [
        "BUY",
        "SELL",
    ]:

        if (
            ml_signal
            ==
            stage19_signal
        ):

            return 100.0

        return 0.0

    return 50.0


# ============================================================
# BUILD RANKING
# ============================================================

def build_ranking(
    ml_df,
    stage19_df
):

    print()
    print("=" * 80)
    print("BUILDING STAGE 20.4 RANKING")
    print("=" * 80)

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    df = ml_df.merge(
        stage19_df,
        on="Symbol",
        how="left"
    )

    # --------------------------------------------------------
    # Missing Stage 19 values
    # --------------------------------------------------------

    df["stage19_signal"] = (
        df["stage19_signal"]
        .fillna(
            "UNAVAILABLE"
        )
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # ML score
    # --------------------------------------------------------

    df["ml_score"] = (
        df.apply(
            calculate_ml_score,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Stage 19 score
    # --------------------------------------------------------

    df["stage19_score"] = (
        df["stage19_signal"]
        .apply(
            calculate_stage19_score
        )
    )

    # --------------------------------------------------------
    # Agreement
    # --------------------------------------------------------

    df["signal_agreement"] = (
        df.apply(
            lambda row:
            calculate_agreement(
                row["signal"],
                row["stage19_signal"]
            ),
            axis=1
        )
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    df["final_score"] = (

        (
            df["ml_score"]
            * ML_WEIGHT
        )

        +

        (
            df["signal_agreement"]
            * STAGE19_WEIGHT
        )

        +

        (
            df["confidence"]
            * CONFIDENCE_WEIGHT
        )

    )

    df["final_score"] = (
        df["final_score"]
        .clip(0, 100)
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    df["direction"] = np.where(

        df["prediction"] == 1,

        "BUY",

        "SELL"

    )

    # --------------------------------------------------------
    # Agreement text
    # --------------------------------------------------------

    def agreement_text(row):

        if (
            row["stage19_signal"]
            == "UNAVAILABLE"
        ):

            return "NO_STAGE19"

        if (
            row["signal"]
            ==
            row["stage19_signal"]
        ):

            return "AGREE"

        return "CONFLICT"

    df["agreement"] = (
        df.apply(
            agreement_text,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Ranking category
    # --------------------------------------------------------

    def category(row):

        score = float(
            row["final_score"]
        )

        confidence = float(
            row["confidence"]
        )

        agreement = (
            row["agreement"]
        )

        # Strong:
        # high score + high confidence
        # + Stage 19 confirmation

        if (
            score >= 75
            and confidence >= 65
            and agreement == "AGREE"
        ):

            return "STRONG"

        # Candidate

        if (
            score >= 60
            and confidence >= 55
        ):

            return "CANDIDATE"

        # Watch

        if score >= 50:

            return "WATCH"

        return "WEAK"

    df["rank_category"] = (
        df.apply(
            category,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            [
                "final_score",
                "confidence",
                "ml_score",
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    df["rank"] = (
        np.arange(
            1,
            len(df) + 1
        )
    )

    # --------------------------------------------------------
    # Column order
    # --------------------------------------------------------

    preferred = [

        "rank",

        "Symbol",

        "direction",

        "signal",

        "stage19_signal",

        "agreement",

        "probability_up",

        "probability_down",

        "confidence",

        "stage19_confidence",

        "ml_score",

        "signal_agreement",

        "final_score",

        "rank_category",

        "stage19_ltp",

        "stage19_smma20",

        "stage19_smma120",

        "stage19_rsi",

        "stage19_reason",

        "stage19_historical_rows",

        "stage19_scan_time",

    ]

    existing = [
        column
        for column in preferred
        if column in df.columns
    ]

    remaining = [
        column
        for column in df.columns
        if column not in existing
    ]

    df = df[
        existing
        +
        remaining
    ]

    return df


# ============================================================
# SAVE
# ============================================================

def save_results(
    df
):

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    metrics = {

        "generated_at":
        datetime.now().isoformat(),

        "stocks_ranked":
        int(len(df)),

        "buy_predictions":
        int(
            (
                df["direction"]
                == "BUY"
            ).sum()
        ),

        "sell_predictions":
        int(
            (
                df["direction"]
                == "SELL"
            ).sum()
        ),

        "stage19_agreement":
        int(
            (
                df["agreement"]
                == "AGREE"
            ).sum()
        ),

        "stage19_conflict":
        int(
            (
                df["agreement"]
                == "CONFLICT"
            ).sum()
        ),

        "strong":
        int(
            (
                df["rank_category"]
                == "STRONG"
            ).sum()
        ),

        "candidate":
        int(
            (
                df["rank_category"]
                == "CANDIDATE"
            ).sum()
        ),

        "watch":
        int(
            (
                df["rank_category"]
                == "WATCH"
            ).sum()
        ),

        "weak":
        int(
            (
                df["rank_category"]
                == "WEAK"
            ).sum()
        ),

        "ml_weight":
        ML_WEIGHT,

        "stage19_weight":
        STAGE19_WEIGHT,

        "confidence_weight":
        CONFIDENCE_WEIGHT,

    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    return metrics


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    df,
    metrics
):

    print()
    print("=" * 110)
    print("STAGE 20.4 - FINAL RANKED RESULTS")
    print("=" * 110)

    display_columns = [

        "rank",
        "Symbol",
        "direction",
        "signal",
        "stage19_signal",
        "agreement",
        "probability_up",
        "probability_down",
        "confidence",
        "final_score",
        "rank_category",

    ]

    display_columns = [
        column
        for column in display_columns
        if column in df.columns
    ]

    display = df[
        display_columns
    ].copy()

    # --------------------------------------------------------
    # Round
    # --------------------------------------------------------

    numeric_columns = [

        "probability_up",
        "probability_down",
        "confidence",
        "final_score",

    ]

    for column in numeric_columns:

        if column in display.columns:

            display[column] = (
                pd.to_numeric(
                    display[column],
                    errors="coerce"
                )
                .round(2)
            )

    print()

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("STAGE 20.4 SUMMARY")
    print("=" * 80)

    print()

    print(
        f"Stocks ranked        : "
        f"{metrics['stocks_ranked']}"
    )

    print(
        f"BUY predictions      : "
        f"{metrics['buy_predictions']}"
    )

    print(
        f"SELL predictions     : "
        f"{metrics['sell_predictions']}"
    )

    print(
        f"Stage 19 agreement   : "
        f"{metrics['stage19_agreement']}"
    )

    print(
        f"Stage 19 conflict    : "
        f"{metrics['stage19_conflict']}"
    )

    print(
        f"STRONG               : "
        f"{metrics['strong']}"
    )

    print(
        f"CANDIDATE            : "
        f"{metrics['candidate']}"
    )

    print(
        f"WATCH                : "
        f"{metrics['watch']}"
    )

    print(
        f"WEAK                 : "
        f"{metrics['weak']}"
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Metrics: {METRICS_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print("STAGE 20.4 - ML + STAGE 19 RANKING ENGINE")
    print("=" * 100)

    try:

        ensure_directories()

        # ----------------------------------------------------
        # ML
        # ----------------------------------------------------

        ml_df = (
            load_ml_predictions()
        )

        # ----------------------------------------------------
        # Stage 19
        # ----------------------------------------------------

        stage19_df = (
            load_stage19()
        )

        if stage19_df is None:

            raise ValueError(
                "Stage 19 data is required "
                "for Stage 20.4."
            )

        # ----------------------------------------------------
        # Ranking
        # ----------------------------------------------------

        ranked_df = build_ranking(
            ml_df,
            stage19_df
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        metrics = save_results(
            ranked_df
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print_results(
            ranked_df,
            metrics
        )

        print()
        print("=" * 100)
        print("STAGE 20.4 RANKING COMPLETE")
        print("=" * 100)

        return ranked_df

    except Exception as e:

        print()
        print("=" * 100)
        print("STAGE 20.4 RANKING FAILED")
        print("=" * 100)

        print()
        print(
            f"Error type: "
            f"{type(e).__name__}"
        )

        print(
            f"Error: {e}"
        )

        return None


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
    