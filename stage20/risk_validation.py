# ============================================================
# STAGE 20.9 - RISK & ROBUSTNESS VALIDATION
# stage20/risk_validation.py
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_RESULTS = os.path.join(
    "data",
    "stage20",
    "stage20_walk_forward_results.csv"
)

INPUT_METRICS = os.path.join(
    "data",
    "stage20",
    "stage20_walk_forward_metrics.json"
)

OUTPUT_DIR = os.path.join(
    "data",
    "stage20"
)

RISK_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "stage20_risk_validation.csv"
)

RISK_METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "stage20_risk_metrics.json"
)


# ============================================================
# RISK THRESHOLDS
# ============================================================

# These are validation thresholds, NOT guarantees of safety.

MIN_WIN_RATE = 0.50

MIN_ACCURACY = 0.50

MIN_SHARPE = 0.50

MIN_TOTAL_RETURN = 0.00

MAX_ALLOWED_DRAWDOWN = -0.20

MIN_FOLD_ACCURACY = 0.48

MAX_FOLD_ACCURACY_SPREAD = 0.20


# ============================================================
# TRANSACTION COST STRESS TEST
# ============================================================

TRANSACTION_COSTS = [
    0.0000,
    0.0010,
    0.0025,
    0.0050,
    0.0100,
]


# ============================================================
# DIRECTORY
# ============================================================

def ensure_directories():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# ============================================================
# HEADER
# ============================================================

def print_header(
    title
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    print_header(
        "LOADING STAGE 20.8 WALK-FORWARD RESULTS"
    )

    print(
        f"File: {INPUT_RESULTS}"
    )

    if not os.path.exists(
        INPUT_RESULTS
    ):

        raise FileNotFoundError(
            f"Walk-forward results not found:\n"
            f"{INPUT_RESULTS}"
        )

    df = pd.read_csv(
        INPUT_RESULTS
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    print()
    print(
        "Columns:"
    )

    for column in df.columns:

        print(
            f"  - {column}"
        )

    return df


# ============================================================
# LOAD WALK-FORWARD METRICS
# ============================================================

def load_previous_metrics():

    print_header(
        "LOADING STAGE 20.8 METRICS"
    )

    if not os.path.exists(
        INPUT_METRICS
    ):

        print(
            "⚠️ Stage 20.8 metrics file "
            "not found."
        )

        return {}

    try:

        with open(
            INPUT_METRICS,
            "r",
            encoding="utf-8"
        ) as file:

            metrics = json.load(
                file
            )

        print(
            "Stage 20.8 metrics loaded."
        )

        return metrics

    except Exception as e:

        print(
            f"⚠️ Could not read metrics: "
            f"{e}"
        )

        return {}


# ============================================================
# VALIDATE COLUMNS
# ============================================================

def validate_columns(
    df
):

    print_header(
        "VALIDATING RISK DATA"
    )

    required = [
        "date",
        "target",
        "prediction",
        "future_return",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            +
            "\n".join(
                f"- {column}"
                for column in missing
            )
        )

    print(
        "Required columns are available."
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    df
):

    print_header(
        "PREPARING RISK VALIDATION DATA"
    )

    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data["target"] = pd.to_numeric(
        data["target"],
        errors="coerce"
    )

    data["prediction"] = pd.to_numeric(
        data["prediction"],
        errors="coerce"
    )

    data["future_return"] = pd.to_numeric(
        data["future_return"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "date",
            "target",
            "prediction",
            "future_return",
        ]
    )

    data["target"] = (
        data["target"]
        .astype(int)
    )

    data["prediction"] = (
        data["prediction"]
        .astype(int)
    )

    data = data[
        data["prediction"].isin(
            [0, 1]
        )
    ]

    data = data[
        data["target"].isin(
            [0, 1]
        )
    ]

    data = (
        data
        .sort_values(
            [
                "date",
                "Symbol"
            ]
            if "Symbol" in data.columns
            else ["date"]
        )
        .reset_index(
            drop=True
        )
    )

    if len(data) == 0:

        raise ValueError(
            "No valid risk-validation rows."
        )

    print(
        f"Prepared rows: "
        f"{len(data)}"
    )

    print(
        f"Start date: "
        f"{data['date'].min()}"
    )

    print(
        f"End date: "
        f"{data['date'].max()}"
    )

    return data


# ============================================================
# BUILD STRATEGY RETURNS
# ============================================================

def calculate_strategy_returns(
    data
):

    # --------------------------------------------------------
    # Prediction 1:
    # positive/up direction
    #
    # Prediction 0:
    # negative/down direction
    #
    # This is a directional validation model.
    # --------------------------------------------------------

    data = data.copy()

    data["strategy_return"] = np.where(
        data["prediction"] == 1,
        data["future_return"],
        -data["future_return"]
    )

    data["strategy_return"] = pd.to_numeric(
        data["strategy_return"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "strategy_return"
        ]
    )

    return data


# ============================================================
# EQUITY CURVE
# ============================================================

def calculate_equity_curve(
    returns
):

    returns = pd.Series(
        returns,
        dtype=float
    )

    returns = returns.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    returns = returns.dropna()

    if len(returns) == 0:

        return pd.Series(
            dtype=float
        )

    equity = (
        1.0
        +
        returns
    ).cumprod()

    return equity


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================

def calculate_max_drawdown(
    returns
):

    equity = (
        calculate_equity_curve(
            returns
        )
    )

    if equity.empty:

        return 0.0

    running_max = (
        equity
        .cummax()
    )

    drawdown = (
        equity
        /
        running_max
        -
        1.0
    )

    return float(
        drawdown.min()
    )


# ============================================================
# WIN RATE
# ============================================================

def calculate_win_rate(
    returns
):

    returns = pd.Series(
        returns,
        dtype=float
    )

    returns = returns.dropna()

    if len(returns) == 0:

        return 0.0

    return float(
        (
            returns > 0
        ).mean()
    )


# ============================================================
# PROFIT FACTOR
# ============================================================

def calculate_profit_factor(
    returns
):

    returns = pd.Series(
        returns,
        dtype=float
    )

    returns = returns.dropna()

    gross_profit = (
        returns[
            returns > 0
        ].sum()
    )

    gross_loss = -(
        returns[
            returns < 0
        ].sum()
    )

    if gross_loss == 0:

        if gross_profit > 0:

            return float(
                "inf"
            )

        return 0.0

    return float(
        gross_profit
        /
        gross_loss
    )


# ============================================================
# SHARPE RATIO
# ============================================================

def calculate_sharpe(
    returns
):

    returns = pd.Series(
        returns,
        dtype=float
    )

    returns = returns.dropna()

    if len(returns) < 2:

        return 0.0

    std = returns.std()

    if std == 0:

        return 0.0

    return float(
        (
            returns.mean()
            /
            std
        )
        *
        np.sqrt(252)
    )


# ============================================================
# TOTAL RETURN
# ============================================================

def calculate_total_return(
    returns
):

    equity = (
        calculate_equity_curve(
            returns
        )
    )

    if equity.empty:

        return 0.0

    return float(
        equity.iloc[-1]
        -
        1.0
    )


# ============================================================
# BASIC RISK METRICS
# ============================================================

def calculate_basic_metrics(
    data
):

    print_header(
        "CALCULATING CORE RISK METRICS"
    )

    returns = (
        data[
            "strategy_return"
        ]
    )

    total_return = (
        calculate_total_return(
            returns
        )
    )

    max_drawdown = (
        calculate_max_drawdown(
            returns
        )
    )

    win_rate = (
        calculate_win_rate(
            returns
        )
    )

    profit_factor = (
        calculate_profit_factor(
            returns
        )
    )

    sharpe = (
        calculate_sharpe(
            returns
        )
    )

    total_trades = len(
        returns
    )

    buy_trades = int(
        (
            data["prediction"] == 1
        ).sum()
    )

    sell_trades = int(
        (
            data["prediction"] == 0
        ).sum()
    )

    metrics = {

        "total_trades": int(
            total_trades
        ),

        "buy_trades": int(
            buy_trades
        ),

        "sell_trades": int(
            sell_trades
        ),

        "win_rate": float(
            win_rate
        ),

        "total_return": float(
            total_return
        ),

        "maximum_drawdown": float(
            max_drawdown
        ),

        "profit_factor": float(
            profit_factor
        ),

        "sharpe_ratio": float(
            sharpe
        ),
    }

    print()
    print(
        f"Total trades      : "
        f"{total_trades}"
    )

    print(
        f"BUY trades        : "
        f"{buy_trades}"
    )

    print(
        f"SELL trades       : "
        f"{sell_trades}"
    )

    print(
        f"Win rate          : "
        f"{win_rate * 100:.2f}%"
    )

    print(
        f"Total return      : "
        f"{total_return * 100:.2f}%"
    )

    print(
        f"Maximum drawdown  : "
        f"{max_drawdown * 100:.2f}%"
    )

    if np.isinf(
        profit_factor
    ):

        print(
            "Profit factor    : "
            "INF"
        )

    else:

        print(
            f"Profit factor     : "
            f"{profit_factor:.4f}"
        )

    print(
        f"Sharpe ratio      : "
        f"{sharpe:.4f}"
    )

    return metrics


# ============================================================
# ML METRICS
# ============================================================

def calculate_ml_metrics(
    data
):

    print_header(
        "CALCULATING ML ROBUSTNESS"
    )

    target = (
        data["target"]
        .astype(int)
    )

    prediction = (
        data["prediction"]
        .astype(int)
    )

    accuracy = (
        target == prediction
    ).mean()

    true_positive = int(
        (
            (
                target == 1
            )
            &
            (
                prediction == 1
            )
        ).sum()
    )

    predicted_positive = int(
        (
            prediction == 1
        ).sum()
    )

    actual_positive = int(
        (
            target == 1
        ).sum()
    )

    if predicted_positive > 0:

        precision = (
            true_positive
            /
            predicted_positive
        )

    else:

        precision = 0.0

    if actual_positive > 0:

        recall = (
            true_positive
            /
            actual_positive
        )

    else:

        recall = 0.0

    metrics = {

        "ml_accuracy": float(
            accuracy
        ),

        "ml_precision": float(
            precision
        ),

        "ml_recall": float(
            recall
        ),
    }

    print(
        f"ML accuracy      : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"ML precision     : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"ML recall        : "
        f"{recall * 100:.2f}%"
    )

    return metrics


# ============================================================
# FOLD ROBUSTNESS
# ============================================================

def calculate_fold_robustness(
    data
):

    print_header(
        "ANALYZING FOLD ROBUSTNESS"
    )

    if "fold" not in data.columns:

        print(
            "⚠️ Fold column unavailable."
        )

        return {
            "fold_count": 0,
            "fold_accuracy_mean": 0.0,
            "fold_accuracy_min": 0.0,
            "fold_accuracy_max": 0.0,
            "fold_accuracy_spread": 0.0,
        }

    fold_records = []

    for fold_number, group in (
        data.groupby(
            "fold"
        )
    ):

        accuracy = (
            group["target"]
            ==
            group["prediction"]
        ).mean()

        strategy_returns = (
            group[
                "strategy_return"
            ]
        )

        fold_return = (
            calculate_total_return(
                strategy_returns
            )
        )

        fold_drawdown = (
            calculate_max_drawdown(
                strategy_returns
            )
        )

        fold_records.append(
            {
                "fold": int(
                    fold_number
                ),
                "rows": int(
                    len(group)
                ),
                "accuracy": float(
                    accuracy
                ),
                "return": float(
                    fold_return
                ),
                "drawdown": float(
                    fold_drawdown
                ),
            }
        )

        print()
        print(
            f"Fold {fold_number}:"
        )

        print(
            f"  Rows      : "
            f"{len(group)}"
        )

        print(
            f"  Accuracy  : "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"  Return    : "
            f"{fold_return * 100:.2f}%"
        )

        print(
            f"  Drawdown  : "
            f"{fold_drawdown * 100:.2f}%"
        )

    fold_df = pd.DataFrame(
        fold_records
    )

    if fold_df.empty:

        return {
            "fold_count": 0,
            "fold_accuracy_mean": 0.0,
            "fold_accuracy_min": 0.0,
            "fold_accuracy_max": 0.0,
            "fold_accuracy_spread": 0.0,
            "fold_details": [],
        }

    mean_accuracy = (
        fold_df["accuracy"]
        .mean()
    )

    min_accuracy = (
        fold_df["accuracy"]
        .min()
    )

    max_accuracy = (
        fold_df["accuracy"]
        .max()
    )

    spread = (
        max_accuracy
        -
        min_accuracy
    )

    return {

        "fold_count": int(
            len(fold_df)
        ),

        "fold_accuracy_mean": float(
            mean_accuracy
        ),

        "fold_accuracy_min": float(
            min_accuracy
        ),

        "fold_accuracy_max": float(
            max_accuracy
        ),

        "fold_accuracy_spread": float(
            spread
        ),

        "fold_details":
            fold_records,
    }


# ============================================================
# TRANSACTION COST STRESS TEST
# ============================================================

def transaction_cost_stress_test(
    data
):

    print_header(
        "TRANSACTION COST STRESS TEST"
    )

    results = []

    base_returns = (
        data[
            "strategy_return"
        ]
        .astype(float)
    )

    # --------------------------------------------------------
    # A conservative simplified stress test:
    # one cost applied per validated trade.
    #
    # This is intentionally a stress test rather than a
    # broker-specific execution simulation.
    # --------------------------------------------------------

    for cost in TRANSACTION_COSTS:

        stressed_returns = (
            base_returns
            -
            cost
        )

        total_return = (
            calculate_total_return(
                stressed_returns
            )
        )

        max_drawdown = (
            calculate_max_drawdown(
                stressed_returns
            )
        )

        sharpe = (
            calculate_sharpe(
                stressed_returns
            )
        )

        win_rate = (
            calculate_win_rate(
                stressed_returns
            )
        )

        record = {

            "transaction_cost": float(
                cost
            ),

            "transaction_cost_percent":
                float(
                    cost * 100
                ),

            "total_return": float(
                total_return
            ),

            "maximum_drawdown": float(
                max_drawdown
            ),

            "sharpe_ratio": float(
                sharpe
            ),

            "win_rate": float(
                win_rate
            ),
        }

        results.append(
            record
        )

        print()
        print(
            f"Cost: "
            f"{cost * 100:.2f}%"
        )

        print(
            f"  Return   : "
            f"{total_return * 100:.2f}%"
        )

        print(
            f"  Drawdown : "
            f"{max_drawdown * 100:.2f}%"
        )

        print(
            f"  Sharpe   : "
            f"{sharpe:.4f}"
        )

        print(
            f"  Win rate : "
            f"{win_rate * 100:.2f}%"
        )

    return results


# ============================================================
# BUY / SELL BALANCE
# ============================================================

def calculate_signal_balance(
    data
):

    print_header(
        "SIGNAL BALANCE ANALYSIS"
    )

    total = len(
        data
    )

    buy_count = int(
        (
            data["prediction"] == 1
        ).sum()
    )

    sell_count = int(
        (
            data["prediction"] == 0
        ).sum()
    )

    if total > 0:

        buy_ratio = (
            buy_count
            /
            total
        )

        sell_ratio = (
            sell_count
            /
            total
        )

    else:

        buy_ratio = 0.0
        sell_ratio = 0.0

    print(
        f"BUY  : "
        f"{buy_count} "
        f"({buy_ratio * 100:.2f}%)"
    )

    print(
        f"SELL : "
        f"{sell_count} "
        f"({sell_ratio * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Flag extreme imbalance.
    # --------------------------------------------------------

    if (
        buy_ratio > 0.90
        or
        sell_ratio > 0.90
    ):

        balance_status = (
            "WARNING"
        )

    else:

        balance_status = (
            "OK"
        )

    print(
        f"Balance status: "
        f"{balance_status}"
    )

    return {

        "buy_count": buy_count,

        "sell_count": sell_count,

        "buy_ratio": float(
            buy_ratio
        ),

        "sell_ratio": float(
            sell_ratio
        ),

        "balance_status":
            balance_status,
    }


# ============================================================
# DRAWDOWN WARNING
# ============================================================

def evaluate_drawdown(
    maximum_drawdown
):

    print_header(
        "DRAWDOWN ANALYSIS"
    )

    print(
        f"Observed maximum drawdown: "
        f"{maximum_drawdown * 100:.2f}%"
    )

    print(
        f"Allowed validation threshold: "
        f"{MAX_ALLOWED_DRAWDOWN * 100:.2f}%"
    )

    if (
        maximum_drawdown
        <
        MAX_ALLOWED_DRAWDOWN
    ):

        status = "FAIL"

        print(
            "⚠️ Drawdown exceeds "
            "the validation threshold."
        )

    else:

        status = "PASS"

        print(
            "Drawdown is within "
            "the validation threshold."
        )

    return status


# ============================================================
# ROBUSTNESS DECISION
# ============================================================

def calculate_final_decision(
    core,
    ml,
    fold,
    signal_balance,
    drawdown_status,
    cost_results
):

    print_header(
        "CALCULATING STAGE 20.9 DECISION"
    )

    checks = {}

    # --------------------------------------------------------
    # ML accuracy
    # --------------------------------------------------------

    checks[
        "ml_accuracy"
    ] = (
        ml["ml_accuracy"]
        >=
        MIN_ACCURACY
    )

    # --------------------------------------------------------
    # Win rate
    # --------------------------------------------------------

    checks[
        "win_rate"
    ] = (
        core["win_rate"]
        >=
        MIN_WIN_RATE
    )

    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    checks[
        "sharpe"
    ] = (
        core["sharpe_ratio"]
        >=
        MIN_SHARPE
    )

    # --------------------------------------------------------
    # Total return
    # --------------------------------------------------------

    checks[
        "total_return"
    ] = (
        core["total_return"]
        >
        MIN_TOTAL_RETURN
    )

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    checks[
        "drawdown"
    ] = (
        drawdown_status
        ==
        "PASS"
    )

    # --------------------------------------------------------
    # Fold minimum accuracy
    # --------------------------------------------------------

    checks[
        "fold_min_accuracy"
    ] = (
        fold[
            "fold_accuracy_min"
        ]
        >=
        MIN_FOLD_ACCURACY
    )

    # --------------------------------------------------------
    # Fold consistency
    # --------------------------------------------------------

    checks[
        "fold_consistency"
    ] = (
        fold[
            "fold_accuracy_spread"
        ]
        <=
        MAX_FOLD_ACCURACY_SPREAD
    )

    # --------------------------------------------------------
    # Signal balance
    # --------------------------------------------------------

    checks[
        "signal_balance"
    ] = (
        signal_balance[
            "balance_status"
        ]
        ==
        "OK"
    )

    # --------------------------------------------------------
    # Transaction cost robustness
    #
    # We check whether the strategy remains profitable
    # at the highest configured stress cost.
    # --------------------------------------------------------

    highest_cost = (
        cost_results[-1]
    )

    checks[
        "cost_robustness"
    ] = (
        highest_cost[
            "total_return"
        ]
        >
        0
    )

    # --------------------------------------------------------
    # Display checks
    # --------------------------------------------------------

    print()

    for name, passed in checks.items():

        status = (
            "PASS"
            if passed
            else
            "FAIL"
        )

        print(
            f"{name:25s}: "
            f"{status}"
        )

    passed_count = sum(
        checks.values()
    )

    total_checks = len(
        checks
    )

    print()
    print(
        f"Checks passed: "
        f"{passed_count}/{total_checks}"
    )

    # --------------------------------------------------------
    # Final decision
    #
    # Any critical risk failure prevents PASS.
    # --------------------------------------------------------

    critical_checks = [
        "ml_accuracy",
        "win_rate",
        "sharpe",
        "total_return",
        "drawdown",
        "fold_min_accuracy",
        "fold_consistency",
        "cost_robustness",
    ]

    critical_pass = all(
        checks[name]
        for name in critical_checks
    )

    if critical_pass:

        decision = "PASS"

    else:

        decision = "REVIEW"

    print()
    print(
        f"STAGE 20.9 DECISION: "
        f"{decision}"
    )

    return decision, checks


# ============================================================
# BUILD FINAL METRICS
# ============================================================

def build_final_metrics(
    core,
    ml,
    fold,
    signal_balance,
    cost_results,
    decision,
    checks,
    previous_metrics
):

    metrics = {

        "stage": "20.9",

        "stage_name":
            "Risk & Robustness Validation",

        "decision":
            decision,

        "checks":
            checks,

        "core_metrics":
            core,

        "ml_metrics":
            ml,

        "fold_robustness":
            fold,

        "signal_balance":
            signal_balance,

        "transaction_cost_stress":
            cost_results,

        "stage20_8_metrics":
            previous_metrics,
    }

    return metrics


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    data,
    metrics
):

    print_header(
        "SAVING STAGE 20.9 RESULTS"
    )

    # --------------------------------------------------------
    # Save row-level results
    # --------------------------------------------------------

    data.to_csv(
        RISK_RESULTS_FILE,
        index=False
    )

    print(
        f"Risk results saved:"
    )

    print(
        f"  {RISK_RESULTS_FILE}"
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    with open(
        RISK_METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
            allow_nan=True
        )

    print()
    print(
        f"Risk metrics saved:"
    )

    print(
        f"  {RISK_METRICS_FILE}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    core,
    ml,
    fold,
    decision
):

    print_header(
        "STAGE 20.9 RISK & ROBUSTNESS SUMMARY"
    )

    print(
        f"Total trades       : "
        f"{core['total_trades']}"
    )

    print(
        f"BUY trades         : "
        f"{core['buy_trades']}"
    )

    print(
        f"SELL trades        : "
        f"{core['sell_trades']}"
    )

    print()

    print(
        f"ML Accuracy        : "
        f"{ml['ml_accuracy'] * 100:.2f}%"
    )

    print(
        f"ML Precision       : "
        f"{ml['ml_precision'] * 100:.2f}%"
    )

    print(
        f"ML Recall          : "
        f"{ml['ml_recall'] * 100:.2f}%"
    )

    print()

    print(
        f"Win Rate           : "
        f"{core['win_rate'] * 100:.2f}%"
    )

    print(
        f"Total Return       : "
        f"{core['total_return'] * 100:.2f}%"
    )

    print(
        f"Maximum Drawdown   : "
        f"{core['maximum_drawdown'] * 100:.2f}%"
    )

    print(
        f"Profit Factor      : "
        f"{core['profit_factor']:.4f}"
        if not np.isinf(
            core["profit_factor"]
        )
        else
        "Profit Factor      : INF"
    )

    print(
        f"Sharpe Ratio       : "
        f"{core['sharpe_ratio']:.4f}"
    )

    print()

    print(
        f"Fold count         : "
        f"{fold['fold_count']}"
    )

    print(
        f"Fold min accuracy  : "
        f"{fold['fold_accuracy_min'] * 100:.2f}%"
    )

    print(
        f"Fold max accuracy  : "
        f"{fold['fold_accuracy_max'] * 100:.2f}%"
    )

    print(
        f"Fold spread        : "
        f"{fold['fold_accuracy_spread'] * 100:.2f}%"
    )

    print()

    print(
        "=" * 80
    )

    print(
        f"FINAL DECISION     : "
        f"{decision}"
    )

    print(
        "=" * 80
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("STAGE 20.9 - RISK & ROBUSTNESS VALIDATION")
    print("=" * 80)

    try:

        # ----------------------------------------------------
        # Directories
        # ----------------------------------------------------

        ensure_directories()

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        data = (
            load_results()
        )

        previous_metrics = (
            load_previous_metrics()
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        validate_columns(
            data
        )

        # ----------------------------------------------------
        # Prepare
        # ----------------------------------------------------

        data = (
            prepare_data(
                data
            )
        )

        # ----------------------------------------------------
        # Strategy returns
        # ----------------------------------------------------

        data = (
            calculate_strategy_returns(
                data
            )
        )

        # ----------------------------------------------------
        # Core metrics
        # ----------------------------------------------------

        core = (
            calculate_basic_metrics(
                data
            )
        )

        # ----------------------------------------------------
        # ML metrics
        # ----------------------------------------------------

        ml = (
            calculate_ml_metrics(
                data
            )
        )

        # ----------------------------------------------------
        # Fold robustness
        # ----------------------------------------------------

        fold = (
            calculate_fold_robustness(
                data
            )
        )

        # ----------------------------------------------------
        # Signal balance
        # ----------------------------------------------------

        signal_balance = (
            calculate_signal_balance(
                data
            )
        )

        # ----------------------------------------------------
        # Drawdown
        # ----------------------------------------------------

        drawdown_status = (
            evaluate_drawdown(
                core[
                    "maximum_drawdown"
                ]
            )
        )

        # ----------------------------------------------------
        # Transaction costs
        # ----------------------------------------------------

        cost_results = (
            transaction_cost_stress_test(
                data
            )
        )

        # ----------------------------------------------------
        # Final decision
        # ----------------------------------------------------

        decision, checks = (
            calculate_final_decision(
                core,
                ml,
                fold,
                signal_balance,
                drawdown_status,
                cost_results
            )
        )

        # ----------------------------------------------------
        # Metrics object
        # ----------------------------------------------------

        metrics = (
            build_final_metrics(
                core,
                ml,
                fold,
                signal_balance,
                cost_results,
                decision,
                checks,
                previous_metrics
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_outputs(
            data,
            metrics
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print_final_summary(
            core,
            ml,
            fold,
            decision
        )

        print()
        print("=" * 80)
        print("STAGE 20.9 RISK VALIDATION COMPLETE")
        print("=" * 80)

        print()
        print(
            f"Results : "
            f"{RISK_RESULTS_FILE}"
        )

        print(
            f"Metrics : "
            f"{RISK_METRICS_FILE}"
        )

        print(
            f"Decision: "
            f"{decision}"
        )

        return metrics

    except Exception as e:

        print()
        print("=" * 80)
        print("STAGE 20.9 RISK VALIDATION FAILED")
        print("=" * 80)

        print()
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