# ============================================================
# STAGE 20.10 - STRATEGY OPTIMIZER
# stage20/strategy_optimizer.py
# ============================================================

import os
import json
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "data/stage20"

WALK_FORWARD_FILE = os.path.join(
    BASE_DIR,
    "stage20_walk_forward_results.csv"
)

RISK_FILE = os.path.join(
    BASE_DIR,
    "stage20_risk_validation.csv"
)

FINAL_FILE = os.path.join(
    BASE_DIR,
    "stage20_final_candidates.csv"
)

OUTPUT_RESULTS = os.path.join(
    BASE_DIR,
    "stage20_optimization_results.csv"
)

OUTPUT_TRADES = os.path.join(
    BASE_DIR,
    "stage20_optimized_trades.csv"
)

OUTPUT_METRICS = os.path.join(
    BASE_DIR,
    "stage20_optimization_metrics.json"
)


# ============================================================
# OPTIMIZATION GRID
# ============================================================

CONFIDENCE_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]

POSITION_SIZES = [
    0.25,
    0.50,
    0.75,
    1.00,
]

MAX_POSITIONS = [
    3,
    5,
    10,
    20,
]

MIN_TRADES = 30


# ============================================================
# DIRECTORY
# ============================================================

def ensure_directories():

    os.makedirs(
        BASE_DIR,
        exist_ok=True
    )


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(
    df,
    candidates
):

    columns = {
        str(column).strip().lower():
        column
        for column in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in columns:

            return columns[key]

    # Fuzzy matching
    for column in df.columns:

        name = (
            str(column)
            .strip()
            .lower()
        )

        for candidate in candidates:

            candidate = (
                candidate
                .strip()
                .lower()
            )

            if candidate in name:

                return column

    return None


# ============================================================
# PRINT COLUMNS
# ============================================================

def print_columns(df):

    print()
    print("Available columns:")

    for number, column in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"  {number:02d}. {column}"
        )


# ============================================================
# LOAD WALK-FORWARD DATA
# ============================================================

def load_data():

    print()
    print("=" * 80)
    print("LOADING STAGE 20 WALK-FORWARD DATA")
    print("=" * 80)

    # --------------------------------------------------------
    # PRIMARY SOURCE
    # --------------------------------------------------------

    if os.path.exists(
        WALK_FORWARD_FILE
    ):

        print()
        print(
            f"File: {WALK_FORWARD_FILE}"
        )

        df = pd.read_csv(
            WALK_FORWARD_FILE
        )

        print(
            f"Rows loaded: {len(df)}"
        )

        print(
            f"Columns loaded: "
            f"{len(df.columns)}"
        )

        print_columns(
            df
        )

        if len(df) > 0:

            return df, WALK_FORWARD_FILE

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if os.path.exists(
        RISK_FILE
    ):

        print()
        print(
            "⚠️ Walk-forward file unavailable."
        )

        print(
            f"Trying: {RISK_FILE}"
        )

        df = pd.read_csv(
            RISK_FILE
        )

        print(
            f"Rows loaded: {len(df)}"
        )

        print_columns(
            df
        )

        return df, RISK_FILE

    # --------------------------------------------------------
    # SECOND FALLBACK
    # --------------------------------------------------------

    if os.path.exists(
        FINAL_FILE
    ):

        print()
        print(
            f"Trying: {FINAL_FILE}"
        )

        df = pd.read_csv(
            FINAL_FILE
        )

        print(
            f"Rows loaded: {len(df)}"
        )

        print_columns(
            df
        )

        return df, FINAL_FILE

    raise FileNotFoundError(
        "No Stage 20 optimization input file found."
    )


# ============================================================
# NORMALIZE PROBABILITY
# ============================================================

def normalize_probability(
    series
):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    # Convert percentage to probability.
    values = np.where(
        np.abs(values) > 1.0,
        values / 100.0,
        values
    )

    return pd.Series(
        values,
        index=series.index,
        dtype=float
    )


# ============================================================
# PREPARE WALK-FORWARD DATA
# ============================================================

def prepare_data(
    df
):

    print()
    print("=" * 80)
    print("PREPARING WALK-FORWARD OPTIMIZATION DATA")
    print("=" * 80)

    data = df.copy()

    # ========================================================
    # SYMBOL
    # ========================================================

    symbol_column = find_column(
        data,
        [
            "symbol",
            "Symbol",
            "stock",
        ]
    )

    if symbol_column:

        data["Symbol"] = (
            data[symbol_column]
            .astype(str)
            .str.strip()
        )

    else:

        data["Symbol"] = "UNKNOWN"

    # ========================================================
    # DATE
    # ========================================================

    date_column = find_column(
        data,
        [
            "date",
            "datetime",
            "timestamp",
            "scan_time",
        ]
    )

    if date_column:

        data["date"] = pd.to_datetime(
            data[date_column],
            errors="coerce"
        )

    else:

        data["date"] = pd.NaT

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction_column = find_column(
        data,
        [
            "prediction",
            "predicted",
            "prediction_class",
            "ml_prediction",
        ]
    )

    if prediction_column:

        data["prediction"] = pd.to_numeric(
            data[prediction_column],
            errors="coerce"
        )

    else:

        data["prediction"] = np.nan

    # ========================================================
    # TARGET
    # ========================================================

    target_column = find_column(
        data,
        [
            "target",
            "actual",
            "actual_target",
            "y_true",
        ]
    )

    if target_column:

        data["target"] = pd.to_numeric(
            data[target_column],
            errors="coerce"
        )

    else:

        data["target"] = np.nan

    # ========================================================
    # PROBABILITY UP
    # ========================================================

    probability_up_column = find_column(
        data,
        [
            "probability_up",
            "prob_up",
            "up_probability",
            "probability",
            "ml_probability",
        ]
    )

    if probability_up_column:

        data["probability_up"] = (
            normalize_probability(
                data[
                    probability_up_column
                ]
            )
        )

    else:

        data["probability_up"] = np.nan

    # ========================================================
    # PROBABILITY DOWN
    # ========================================================

    probability_down_column = find_column(
        data,
        [
            "probability_down",
            "prob_down",
            "down_probability",
        ]
    )

    if probability_down_column:

        data["probability_down"] = (
            normalize_probability(
                data[
                    probability_down_column
                ]
            )
        )

    else:

        data["probability_down"] = (
            1.0
            -
            data["probability_up"]
        )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence_column = find_column(
        data,
        [
            "confidence",
            "ml_confidence",
        ]
    )

    if confidence_column:

        data["confidence"] = (
            normalize_probability(
                data[
                    confidence_column
                ]
            )
        )

    else:

        data["confidence"] = np.maximum(
            data["probability_up"],
            data["probability_down"]
        )

    # ========================================================
    # DIRECTION
    # ========================================================

    direction_column = find_column(
        data,
        [
            "direction",
            "signal",
            "prediction_signal",
            "stage19_signal",
        ]
    )

    if direction_column:

        data["direction"] = (
            data[direction_column]
            .astype(str)
            .str.upper()
            .str.strip()
        )

    else:

        data["direction"] = np.where(
            data["prediction"] == 1,
            "BUY",
            np.where(
                data["prediction"] == 0,
                "SELL",
                "HOLD"
            )
        )

    # ========================================================
    # FUTURE RETURN
    # ========================================================

    future_return_column = find_column(
        data,
        [
            "future_return",
            "future_return_5d",
            "forward_return",
            "realized_return",
            "trade_return",
            "strategy_return",
        ]
    )

    if future_return_column:

        data["future_return"] = pd.to_numeric(
            data[
                future_return_column
            ],
            errors="coerce"
        )

    else:

        data["future_return"] = np.nan

    # ========================================================
    # NORMALIZE RETURN
    # ========================================================

    return_mask = (
        data["future_return"]
        .abs()
        > 1.0
    )

    data.loc[
        return_mask,
        "future_return"
    ] = (
        data.loc[
            return_mask,
            "future_return"
        ]
        / 100.0
    )

    # ========================================================
    # BUILD TRADE RETURN
    # ========================================================

    # BUY:
    #   profit when future return is positive.
    #
    # SELL:
    #   profit when future return is negative.
    #
    # HOLD:
    #   no trade.

    data["trade_return"] = np.where(
        data["direction"] == "BUY",
        data["future_return"],
        np.where(
            data["direction"] == "SELL",
            -data["future_return"],
            0.0
        )
    )

    # ========================================================
    # IF FUTURE RETURN IS NOT AVAILABLE
    # USE TARGET + PREDICTION AS CLASSIFICATION RESULT
    # ========================================================

    missing_return = (
        data["trade_return"]
        .isna()
    )

    if missing_return.any():

        data.loc[
            missing_return,
            "trade_return"
        ] = np.where(
            (
                data.loc[
                    missing_return,
                    "prediction"
                ]
                ==
                data.loc[
                    missing_return,
                    "target"
                ]
            ),
            0.001,
            -0.001
        )

    # ========================================================
    # REMOVE HOLD
    # ========================================================

    data = data[
        data["direction"].isin(
            [
                "BUY",
                "SELL",
            ]
        )
    ].copy()

    # ========================================================
    # CLEAN
    # ========================================================

    data = data.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan
    )

    data = data.dropna(
        subset=[
            "confidence",
            "trade_return",
        ]
    )

    # ========================================================
    # SORT
    # ========================================================

    if data["date"].notna().any():

        data = (
            data
            .sort_values(
                "date"
            )
        )

    data = (
        data
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        f"Usable trade rows: "
        f"{len(data)}"
    )

    print(
        f"BUY rows: "
        f"{int((data['direction'] == 'BUY').sum())}"
    )

    print(
        f"SELL rows: "
        f"{int((data['direction'] == 'SELL').sum())}"
    )

    if len(data) == 0:

        raise ValueError(
            "Walk-forward file does not contain "
            "usable prediction/return information."
        )

    return data


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    returns
):

    returns = pd.Series(
        returns,
        dtype=float
    )

    returns = returns.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan
    )

    returns = returns.dropna()

    if len(returns) == 0:

        return {
            "trades": 0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "maximum_drawdown": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
        }

    wins = returns[
        returns > 0
    ]

    losses = returns[
        returns < 0
    ]

    win_rate = (
        len(wins)
        /
        len(returns)
    )

    equity = (
        1.0
        *
        (1.0 + returns)
        .cumprod()
    )

    total_return = (
        equity.iloc[-1]
        - 1.0
    )

    running_max = (
        equity
        .cummax()
    )

    drawdown = (
        equity
        /
        running_max
        - 1.0
    )

    maximum_drawdown = (
        drawdown.min()
    )

    gross_profit = (
        wins.sum()
    )

    gross_loss = abs(
        losses.sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            /
            gross_loss
        )

    elif gross_profit > 0:

        profit_factor = 999.0

    else:

        profit_factor = 0.0

    standard_deviation = (
        returns.std(
            ddof=1
        )
    )

    if (
        standard_deviation > 0
        and
        len(returns) > 1
    ):

        sharpe_ratio = (
            returns.mean()
            /
            standard_deviation
        ) * math.sqrt(
            len(returns)
        )

    else:

        sharpe_ratio = 0.0

    return {
        "trades": int(
            len(returns)
        ),

        "win_rate": float(
            win_rate
        ),

        "total_return": float(
            total_return
        ),

        "maximum_drawdown": float(
            maximum_drawdown
        ),

        "profit_factor": float(
            profit_factor
        ),

        "sharpe_ratio": float(
            sharpe_ratio
        ),
    }


# ============================================================
# EVALUATE CONFIGURATION
# ============================================================

def evaluate_configuration(
    data,
    threshold,
    position_size,
    max_positions
):

    selected = data[
        data["confidence"]
        >= threshold
    ].copy()

    if len(selected) == 0:

        return None

    # --------------------------------------------------------
    # Position sizing
    # --------------------------------------------------------

    selected[
        "optimized_return"
    ] = (
        selected["trade_return"]
        *
        position_size
    )

    # --------------------------------------------------------
    # Exposure limiter
    #
    # Keep the implementation deterministic.
    # Every group of max_positions represents one
    # portfolio exposure batch.
    # --------------------------------------------------------

    if max_positions > 0:

        selected[
            "position_number"
        ] = (
            np.arange(
                len(selected)
            )
            %
            max_positions
        )

    metrics = calculate_metrics(
        selected[
            "optimized_return"
        ]
    )

    metrics.update(
        {
            "confidence_threshold":
                threshold,

            "position_size":
                position_size,

            "max_positions":
                max_positions,
        }
    )

    return metrics


# ============================================================
# OPTIMIZATION SCORE
# ============================================================

def calculate_score(
    metrics
):

    if (
        metrics["trades"]
        <
        MIN_TRADES
    ):

        return -999999.0

    total_return = (
        metrics["total_return"]
    )

    drawdown = abs(
        metrics["maximum_drawdown"]
    )

    sharpe = (
        metrics["sharpe_ratio"]
    )

    profit_factor = (
        metrics["profit_factor"]
    )

    win_rate = (
        metrics["win_rate"]
    )

    score = (
        total_return * 100.0
        +
        sharpe * 2.0
        +
        min(
            profit_factor,
            5.0
        )
        +
        win_rate
        -
        drawdown * 120.0
    )

    return float(
        score
    )


# ============================================================
# RUN OPTIMIZATION
# ============================================================

def run_optimization(
    data
):

    print()
    print("=" * 80)
    print("RUNNING STRATEGY OPTIMIZATION")
    print("=" * 80)

    results = []

    total = (
        len(CONFIDENCE_THRESHOLDS)
        *
        len(POSITION_SIZES)
        *
        len(MAX_POSITIONS)
    )

    completed = 0

    for threshold in (
        CONFIDENCE_THRESHOLDS
    ):

        for position_size in (
            POSITION_SIZES
        ):

            for max_positions in (
                MAX_POSITIONS
            ):

                completed += 1

                metrics = evaluate_configuration(
                    data,
                    threshold,
                    position_size,
                    max_positions
                )

                if metrics is None:

                    continue

                metrics[
                    "optimization_score"
                ] = calculate_score(
                    metrics
                )

                results.append(
                    metrics
                )

    if not results:

        raise ValueError(
            "No optimization configurations "
            "could be evaluated."
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "optimization_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    results_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(results_df) + 1
        )
    )

    print()
    print(
        f"Configurations evaluated: "
        f"{completed}"
    )

    return results_df


# ============================================================
# CREATE OPTIMIZED DATASET
# ============================================================

def create_optimized_trades(
    data,
    best
):

    threshold = float(
        best[
            "confidence_threshold"
        ]
    )

    position_size = float(
        best[
            "position_size"
        ]
    )

    optimized = data[
        data["confidence"]
        >= threshold
    ].copy()

    optimized[
        "position_size"
    ] = position_size

    optimized[
        "optimized_return"
    ] = (
        optimized[
            "trade_return"
        ]
        *
        position_size
    )

    optimized["result"] = np.where(
        optimized[
            "optimized_return"
        ] > 0,
        "WIN",
        np.where(
            optimized[
                "optimized_return"
            ] < 0,
            "LOSS",
            "FLAT"
        )
    )

    return optimized


# ============================================================
# SAVE
# ============================================================

def save_results(
    results_df,
    optimized_trades,
    best,
    source_file
):

    print()
    print("=" * 80)
    print("SAVING OPTIMIZATION RESULTS")
    print("=" * 80)

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_RESULTS,
        index=False
    )

    optimized_trades.to_csv(
        OUTPUT_TRADES,
        index=False
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    trades = int(
        best["trades"]
    )

    profit_factor = float(
        best["profit_factor"]
    )

    sharpe = float(
        best["sharpe_ratio"]
    )

    drawdown = float(
        best["maximum_drawdown"]
    )

    total_return = float(
        best["total_return"]
    )

    if (
        trades >= MIN_TRADES
        and
        profit_factor >= 1.10
        and
        sharpe >= 0.50
        and
        drawdown >= -0.30
    ):

        decision = "PASS"

    elif (
        trades >= MIN_TRADES
        and
        profit_factor > 1.00
        and
        total_return > 0
    ):

        decision = "REVIEW"

    else:

        decision = "REJECT"

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {
        "stage": "20.10",

        "source_file":
            source_file,

        "tested_configurations":
            int(len(results_df)),

        "best_rank":
            int(best["rank"]),

        "confidence_threshold":
            float(
                best[
                    "confidence_threshold"
                ]
            ),

        "position_size":
            float(
                best[
                    "position_size"
                ]
            ),

        "max_positions":
            int(
                best[
                    "max_positions"
                ]
            ),

        "trades":
            trades,

        "win_rate":
            float(
                best[
                    "win_rate"
                ]
            ),

        "total_return":
            total_return,

        "maximum_drawdown":
            drawdown,

        "profit_factor":
            profit_factor,

        "sharpe_ratio":
            sharpe,

        "final_decision":
            decision,
    }

    with open(
        OUTPUT_METRICS,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print()
    print(
        f"Results saved:"
    )

    print(
        f"  {OUTPUT_RESULTS}"
    )

    print()
    print(
        f"Optimized trades saved:"
    )

    print(
        f"  {OUTPUT_TRADES}"
    )

    print()
    print(
        f"Metrics saved:"
    )

    print(
        f"  {OUTPUT_METRICS}"
    )

    return metrics


# ============================================================
# DISPLAY TOP RESULTS
# ============================================================

def display_results(
    results_df
):

    print()
    print("=" * 80)
    print("TOP OPTIMIZATION RESULTS")
    print("=" * 80)

    columns = [
        "rank",
        "confidence_threshold",
        "position_size",
        "max_positions",
        "trades",
        "win_rate",
        "total_return",
        "maximum_drawdown",
        "profit_factor",
        "sharpe_ratio",
        "optimization_score",
    ]

    available = [
        column
        for column in columns
        if column in results_df.columns
    ]

    output = (
        results_df[
            available
        ]
        .head(10)
        .copy()
    )

    # Percent display
    for column in [
        "win_rate",
        "total_return",
        "maximum_drawdown",
    ]:

        if column in output.columns:

            output[column] = (
                output[column]
                * 100
            ).round(2)

    if "profit_factor" in output.columns:

        output[
            "profit_factor"
        ] = output[
            "profit_factor"
        ].round(4)

    if "sharpe_ratio" in output.columns:

        output[
            "sharpe_ratio"
        ] = output[
            "sharpe_ratio"
        ].round(4)

    print(
        output.to_string(
            index=False
        )
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    metrics
):

    print()
    print("=" * 80)
    print("STAGE 20.10 STRATEGY OPTIMIZATION SUMMARY")
    print("=" * 80)

    print()

    print(
        f"Confidence threshold : "
        f"{metrics['confidence_threshold']:.2f}"
    )

    print(
        f"Position size        : "
        f"{metrics['position_size']:.2f}"
    )

    print(
        f"Maximum positions    : "
        f"{metrics['max_positions']}"
    )

    print()

    print(
        f"Trades               : "
        f"{metrics['trades']}"
    )

    print(
        f"Win rate             : "
        f"{metrics['win_rate'] * 100:.2f}%"
    )

    print(
        f"Total return         : "
        f"{metrics['total_return'] * 100:.2f}%"
    )

    print(
        f"Maximum drawdown     : "
        f"{metrics['maximum_drawdown'] * 100:.2f}%"
    )

    print(
        f"Profit factor        : "
        f"{metrics['profit_factor']:.4f}"
    )

    print(
        f"Sharpe ratio         : "
        f"{metrics['sharpe_ratio']:.4f}"
    )

    print()

    print(
        f"FINAL DECISION       : "
        f"{metrics['final_decision']}"
    )

    print()
    print("=" * 80)
    print("STAGE 20.10 OPTIMIZATION COMPLETE")
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("STAGE 20.10 - STRATEGY OPTIMIZER")
    print("=" * 80)

    try:

        ensure_directories()

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        raw_data, source_file = (
            load_data()
        )

        # ----------------------------------------------------
        # PREPARE
        # ----------------------------------------------------

        data = prepare_data(
            raw_data
        )

        # ----------------------------------------------------
        # OPTIMIZE
        # ----------------------------------------------------

        results_df = run_optimization(
            data
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        display_results(
            results_df
        )

        # ----------------------------------------------------
        # BEST
        # ----------------------------------------------------

        best = (
            results_df
            .iloc[0]
            .to_dict()
        )

        print()
        print("=" * 80)
        print("BEST CONFIGURATION")
        print("=" * 80)

        print(
            f"Confidence threshold : "
            f"{best['confidence_threshold']:.2f}"
        )

        print(
            f"Position size        : "
            f"{best['position_size']:.2f}"
        )

        print(
            f"Max positions        : "
            f"{int(best['max_positions'])}"
        )

        print(
            f"Trades               : "
            f"{int(best['trades'])}"
        )

        print(
            f"Win rate             : "
            f"{best['win_rate'] * 100:.2f}%"
        )

        print(
            f"Total return         : "
            f"{best['total_return'] * 100:.2f}%"
        )

        print(
            f"Maximum drawdown     : "
            f"{best['maximum_drawdown'] * 100:.2f}%"
        )

        print(
            f"Profit factor        : "
            f"{best['profit_factor']:.4f}"
        )

        print(
            f"Sharpe ratio         : "
            f"{best['sharpe_ratio']:.4f}"
        )

        # ----------------------------------------------------
        # OPTIMIZED TRADES
        # ----------------------------------------------------

        optimized_trades = (
            create_optimized_trades(
                data,
                best
            )
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        metrics = save_results(
            results_df,
            optimized_trades,
            best,
            source_file
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        print_summary(
            metrics
        )

    except Exception as e:

        print()
        print("=" * 80)
        print("STAGE 20.10 STRATEGY OPTIMIZATION FAILED")
        print("=" * 80)

        print()
        print(
            f"Error type: "
            f"{type(e).__name__}"
        )

        print(
            f"Error: "
            f"{e}"
        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()