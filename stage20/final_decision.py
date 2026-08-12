# ============================================================
# STAGE 20.5 - FINAL DECISION ENGINE
# stage20/final_decision.py
# ============================================================

import os
import json
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "data/stage20"
OUTPUT_DIR = "data/stage20"

INPUT_FILE = os.path.join(
    INPUT_DIR,
    "stage20_ranked_results.csv"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "stage20_final_candidates.csv"
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "stage20_final_metrics.json"
)


# ============================================================
# DECISION THRESHOLDS
# ============================================================

# Minimum ML probability for a trade candidate
MIN_PROBABILITY = 0.55

# Minimum final score
MIN_SCORE = 55.0

# Strong candidate threshold
STRONG_SCORE = 70.0

# Candidate threshold
CANDIDATE_SCORE = 55.0


# ============================================================
# DIRECTORY
# ============================================================

def ensure_directories():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# ============================================================
# LOAD STAGE 20.4
# ============================================================

def load_ranked_results():

    print()
    print("=" * 80)
    print("LOADING STAGE 20.4 RESULTS")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Stage 20.4 file not found: "
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        f"File: {INPUT_FILE}"
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    print()
    print("Columns:")

    for column in df.columns:

        print(
            f"  - {column}"
        )

    return df


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

def normalize_columns(df):

    df = df.copy()

    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        for column in df.columns
    ]

    return df


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    df,
    possible_names
):

    for name in possible_names:

        if name in df.columns:

            return name

    return None


# ============================================================
# NUMERIC CONVERSION
# ============================================================

def convert_numeric(
    df,
    columns
):

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# CALCULATE FINAL SCORE
# ============================================================

def calculate_final_score(df):

    print()
    print("=" * 80)
    print("CALCULATING FINAL SCORE")
    print("=" * 80)

    df = df.copy()

    # --------------------------------------------------------
    # ML probability
    # --------------------------------------------------------

    probability_column = find_column(
        df,
        [
            "probability_up",
            "ml_probability",
            "probability",
            "prob_up",
        ]
    )

    if probability_column is None:

        raise ValueError(
            "ML probability column not found."
        )

    df["ml_probability"] = pd.to_numeric(
        df[probability_column],
        errors="coerce"
    )

    # Convert percentage values such as 63.5 -> 0.635
    df["ml_probability"] = np.where(
        df["ml_probability"] > 1,
        df["ml_probability"] / 100,
        df["ml_probability"]
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    direction_column = find_column(
        df,
        [
            "direction",
            "signal",
            "prediction_signal",
        ]
    )

    if direction_column is None:

        raise ValueError(
            "Direction column not found."
        )

    df["direction"] = (
        df[direction_column]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # Stage 19 agreement
    # --------------------------------------------------------

    agreement_column = find_column(
        df,
        [
            "agreement",
            "stage19_agreement",
        ]
    )

    if agreement_column is None:

        df["stage19_agreement"] = (
            "UNKNOWN"
        )

    else:

        df["stage19_agreement"] = (
            df[agreement_column]
            .astype(str)
            .str.upper()
            .str.strip()
        )

    # --------------------------------------------------------
    # Stage 19 signal
    # --------------------------------------------------------

    stage19_column = find_column(
        df,
        [
            "stage19_signal",
            "stage19_direction",
            "stage_19_signal",
        ]
    )

    if stage19_column is None:

        df["stage19_signal"] = (
            "UNAVAILABLE"
        )

    else:

        df["stage19_signal"] = (
            df[stage19_column]
            .astype(str)
            .str.upper()
            .str.strip()
        )

    # ========================================================
    # BASE SCORE
    # ========================================================

    # Probability contribution:
    #
    # 50% probability = 50 points
    # 70% probability = 70 points
    #
    # This keeps the score easy to understand.
    # ========================================================

    df["probability_score"] = (
        df["ml_probability"]
        * 100
    )

    # --------------------------------------------------------
    # Agreement bonus
    # --------------------------------------------------------

    df["agreement_score"] = 0.0

    df.loc[
        df["stage19_agreement"] == "AGREE",
        "agreement_score"
    ] = 15.0

    df.loc[
        df["stage19_agreement"] == "CONFLICT",
        "agreement_score"
    ] = -15.0

    # --------------------------------------------------------
    # Stage 19 unavailable
    # --------------------------------------------------------

    df.loc[
        df["stage19_signal"] == "UNAVAILABLE",
        "agreement_score"
    ] = -5.0

    # ========================================================
    # DIRECTION CONFIRMATION
    # ========================================================

    df["direction_score"] = 0.0

    agree_mask = (
        (
            df["direction"] == "BUY"
        )
        &
        (
            df["stage19_signal"] == "BUY"
        )
    ) | (
        (
            df["direction"] == "SELL"
        )
        &
        (
            df["stage19_signal"] == "SELL"
        )
    )

    conflict_mask = (
        (
            df["direction"] == "BUY"
        )
        &
        (
            df["stage19_signal"] == "SELL"
        )
    ) | (
        (
            df["direction"] == "SELL"
        )
        &
        (
            df["stage19_signal"] == "BUY"
        )
    )

    df.loc[
        agree_mask,
        "direction_score"
    ] = 10.0

    df.loc[
        conflict_mask,
        "direction_score"
    ] = -10.0

    # ========================================================
    # FINAL SCORE
    # ========================================================

    df["final_score"] = (
        df["probability_score"]
        + df["agreement_score"]
        + df["direction_score"]
    )

    # Keep score in 0-100 range
    df["final_score"] = (
        df["final_score"]
        .clip(
            lower=0,
            upper=100
        )
    )

    return df


# ============================================================
# FINAL DECISION
# ============================================================

def assign_decision(row):

    probability = float(
        row["ml_probability"]
    )

    score = float(
        row["final_score"]
    )

    direction = str(
        row["direction"]
    ).upper()

    agreement = str(
        row["stage19_agreement"]
    ).upper()

    stage19_signal = str(
        row["stage19_signal"]
    ).upper()

    # --------------------------------------------------------
    # Invalid direction
    # --------------------------------------------------------

    if direction not in [
        "BUY",
        "SELL"
    ]:

        return "REJECT"

    # --------------------------------------------------------
    # Low ML probability
    # --------------------------------------------------------

    if probability < MIN_PROBABILITY:

        return "WATCH"

    # --------------------------------------------------------
    # Stage 19 conflict
    # --------------------------------------------------------

    if agreement == "CONFLICT":

        return "WATCH"

    if (
        stage19_signal != "UNAVAILABLE"
        and stage19_signal != direction
    ):

        return "WATCH"

    # --------------------------------------------------------
    # Strong candidate
    # --------------------------------------------------------

    if score >= STRONG_SCORE:

        return "STRONG"

    # --------------------------------------------------------
    # Candidate
    # --------------------------------------------------------

    if score >= CANDIDATE_SCORE:

        return "CANDIDATE"

    # --------------------------------------------------------
    # Weak
    # --------------------------------------------------------

    return "WEAK"


# ============================================================
# APPLY DECISIONS
# ============================================================

def apply_decisions(df):

    print()
    print("=" * 80)
    print("APPLYING FINAL DECISIONS")
    print("=" * 80)

    df = df.copy()

    df["decision"] = df.apply(
        assign_decision,
        axis=1
    )

    # --------------------------------------------------------
    # Reason
    # --------------------------------------------------------

    reasons = []

    for _, row in df.iterrows():

        direction = row["direction"]

        probability = (
            float(row["ml_probability"])
        )

        score = (
            float(row["final_score"])
        )

        agreement = (
            row["stage19_agreement"]
        )

        stage19_signal = (
            row["stage19_signal"]
        )

        decision = row["decision"]

        reason_parts = []

        reason_parts.append(
            f"ML probability "
            f"{probability * 100:.2f}%"
        )

        reason_parts.append(
            f"Final score "
            f"{score:.2f}"
        )

        if agreement == "AGREE":

            reason_parts.append(
                "Stage 19 agrees"
            )

        elif agreement == "CONFLICT":

            reason_parts.append(
                "Stage 19 conflict"
            )

        elif stage19_signal == "UNAVAILABLE":

            reason_parts.append(
                "Stage 19 unavailable"
            )

        reasons.append(
            f"{decision}: "
            + "; ".join(reason_parts)
        )

    df["reason"] = reasons

    return df


# ============================================================
# SORT FINAL RESULTS
# ============================================================

def sort_results(df):

    df = df.copy()

    decision_priority = {

        "STRONG": 1,

        "CANDIDATE": 2,

        "WATCH": 3,

        "WEAK": 4,

        "REJECT": 5,
    }

    df["decision_priority"] = (
        df["decision"]
        .map(
            decision_priority
        )
        .fillna(99)
    )

    df = (
        df
        .sort_values(
            [
                "decision_priority",
                "final_score",
            ],
            ascending=[
                True,
                False,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    df["final_rank"] = (
        range(
            1,
            len(df) + 1
        )
    )

    return df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(df):

    # --------------------------------------------------------
    # Remove internal sorting column
    # --------------------------------------------------------

    output_columns = [
        "final_rank",
        "symbol",
        "direction",
        "ml_probability",
        "probability_score",
        "agreement_score",
        "direction_score",
        "final_score",
        "stage19_signal",
        "stage19_agreement",
        "decision",
        "reason",
    ]

    available_columns = [
        column
        for column in output_columns
        if column in df.columns
    ]

    output_df = df[
        available_columns
    ].copy()

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = {

        "stocks_processed": int(
            len(df)
        ),

        "strong": int(
            (
                df["decision"]
                == "STRONG"
            ).sum()
        ),

        "candidate": int(
            (
                df["decision"]
                == "CANDIDATE"
            ).sum()
        ),

        "watch": int(
            (
                df["decision"]
                == "WATCH"
            ).sum()
        ),

        "weak": int(
            (
                df["decision"]
                == "WEAK"
            ).sum()
        ),

        "reject": int(
            (
                df["decision"]
                == "REJECT"
            ).sum()
        ),

        "buy": int(
            (
                df["direction"]
                == "BUY"
            ).sum()
        ),

        "sell": int(
            (
                df["direction"]
                == "SELL"
            ).sum()
        ),

        "average_score": round(
            float(
                df["final_score"]
                .mean()
            ),
            4
        ),

        "max_score": round(
            float(
                df["final_score"]
                .max()
            ),
            4
        ),

        "min_score": round(
            float(
                df["final_score"]
                .min()
            ),
            4
        ),
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

    return output_df, metrics


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    output_df,
    metrics
):

    print()
    print("=" * 100)
    print("STAGE 20.5 - FINAL DECISION RESULTS")
    print("=" * 100)

    print()

    if output_df.empty:

        print(
            "No final results."
        )

        return

    display_columns = [
        "final_rank",
        "symbol",
        "direction",
        "ml_probability",
        "final_score",
        "stage19_signal",
        "stage19_agreement",
        "decision",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in output_df.columns
    ]

    display_df = (
        output_df[
            display_columns
        ]
        .copy()
    )

    if "ml_probability" in display_df.columns:

        display_df[
            "ml_probability"
        ] = (
            display_df[
                "ml_probability"
            ]
            * 100
        ).round(2)

    if "final_score" in display_df.columns:

        display_df[
            "final_score"
        ] = display_df[
            "final_score"
        ].round(2)

    print(
        display_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("STAGE 20.5 SUMMARY")
    print("=" * 80)

    print(
        f"Stocks processed : "
        f"{metrics['stocks_processed']}"
    )

    print(
        f"BUY              : "
        f"{metrics['buy']}"
    )

    print(
        f"SELL             : "
        f"{metrics['sell']}"
    )

    print(
        f"STRONG           : "
        f"{metrics['strong']}"
    )

    print(
        f"CANDIDATE        : "
        f"{metrics['candidate']}"
    )

    print(
        f"WATCH            : "
        f"{metrics['watch']}"
    )

    print(
        f"WEAK             : "
        f"{metrics['weak']}"
    )

    print(
        f"REJECT           : "
        f"{metrics['reject']}"
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
    print("=" * 80)
    print("STAGE 20.5 - FINAL DECISION ENGINE")
    print("=" * 80)

    try:

        ensure_directories()

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        df = load_ranked_results()

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        df = normalize_columns(
            df
        )

        # ----------------------------------------------------
        # NUMERIC
        # ----------------------------------------------------

        numeric_columns = [
            "probability_up",
            "probability_down",
            "confidence",
            "final_score",
            "rank",
        ]

        df = convert_numeric(
            df,
            numeric_columns
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        df = calculate_final_score(
            df
        )

        # ----------------------------------------------------
        # DECISION
        # ----------------------------------------------------

        df = apply_decisions(
            df
        )

        # ----------------------------------------------------
        # SORT
        # ----------------------------------------------------

        df = sort_results(
            df
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output_df, metrics = (
            save_results(
                df
            )
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        display_results(
            output_df,
            metrics
        )

        print()
        print("=" * 80)
        print("STAGE 20.5 COMPLETE")
        print("=" * 80)

    except Exception as e:

        print()
        print("=" * 80)
        print("STAGE 20.5 FAILED")
        print("=" * 80)

        print(
            f"Error type: "
            f"{type(e).__name__}"
        )

        print(
            f"Error: {e}"
        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()