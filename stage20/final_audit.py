# ============================================================
# STAGE 20.13 - FINAL SYSTEM AUDIT
# stage20/final_audit.py
# ============================================================

import os
import json
import math
from datetime import datetime

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/stage20"

FILES_TO_CHECK = {
    "training_data":
        "stage20_training_data.csv",

    "predictions":
        "stage20_predictions.csv",

    "ranked_results":
        "stage20_ranked_results.csv",

    "final_candidates":
        "stage20_final_candidates.csv",

    "backtest_results":
        "stage20_backtest_results.csv",

    "walk_forward_results":
        "stage20_walk_forward_results.csv",

    "risk_validation":
        "stage20_risk_validation.csv",

    "optimized_trades":
        "stage20_optimized_trades.csv",

    "portfolio":
        "stage20_portfolio.csv",

    "portfolio_validation":
        "stage20_portfolio_validation.csv",
}


METRICS_FILES = {
    "training_metrics":
        "training_metrics.json",

    "backtest_metrics":
        "stage20_backtest_metrics.json",

    "walk_forward_metrics":
        "stage20_walk_forward_metrics.json",

    "risk_metrics":
        "stage20_risk_metrics.json",

    "optimization_metrics":
        "stage20_optimization_metrics.json",

    "portfolio_metrics":
        "stage20_portfolio_metrics.json",

    "portfolio_validation_metrics":
        "stage20_portfolio_validation_metrics.json",
}


OUTPUT_REPORT = os.path.join(
    DATA_DIR,
    "stage20_final_audit.csv"
)

OUTPUT_METRICS = os.path.join(
    DATA_DIR,
    "stage20_final_audit_metrics.json"
)


# ============================================================
# CONFIGURATION LIMITS
# ============================================================

MIN_ML_ACCURACY = 0.50
MIN_WIN_RATE = 0.50

MIN_PROFIT_FACTOR = 1.00

MIN_SHARPE = 0.00

MAX_DRAWDOWN = -0.50

MIN_PORTFOLIO_CHECKS = 1


# ============================================================
# DIRECTORY
# ============================================================

def ensure_directory():

    os.makedirs(
        DATA_DIR,
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

        number = float(value)

        if not math.isfinite(number):

            return default

        return number

    except Exception:

        return default


# ============================================================
# LOAD JSON
# ============================================================

def load_json(
    path
):

    if not os.path.exists(path):

        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception as e:

        print(
            f"⚠️ Could not read {path}: {e}"
        )

        return {}


# ============================================================
# CHECK FILES
# ============================================================

def check_required_files():

    print()
    print("=" * 80)
    print("CHECKING STAGE 20 OUTPUT FILES")
    print("=" * 80)

    results = []

    for name, filename in FILES_TO_CHECK.items():

        path = os.path.join(
            DATA_DIR,
            filename
        )

        exists = os.path.exists(
            path
        )

        size = (
            os.path.getsize(path)
            if exists
            else 0
        )

        results.append({
            "component": name,
            "file": filename,
            "exists": exists,
            "size_bytes": size,
            "status":
                "PASS"
                if exists
                else "FAIL"
        })

        print(
            f"{name:<28} : "
            f"{'PASS' if exists else 'FAIL'}"
        )

    return results


# ============================================================
# CHECK METRICS FILES
# ============================================================

def check_metric_files():

    print()
    print("=" * 80)
    print("CHECKING METRICS FILES")
    print("=" * 80)

    results = []

    for name, filename in METRICS_FILES.items():

        path = os.path.join(
            DATA_DIR,
            filename
        )

        exists = os.path.exists(
            path
        )

        results.append({
            "component": name,
            "file": filename,
            "exists": exists,
            "status":
                "PASS"
                if exists
                else "FAIL"
        })

        print(
            f"{name:<32} : "
            f"{'PASS' if exists else 'FAIL'}"
        )

    return results


# ============================================================
# LOAD DATASET
# ============================================================

def load_csv(
    filename
):

    path = os.path.join(
        DATA_DIR,
        filename
    )

    if not os.path.exists(path):

        return None

    try:

        return pd.read_csv(
            path
        )

    except Exception as e:

        print(
            f"⚠️ Failed reading "
            f"{filename}: {e}"
        )

        return None


# ============================================================
# DATA QUALITY
# ============================================================

def check_data_quality():

    print()
    print("=" * 80)
    print("CHECKING DATA QUALITY")
    print("=" * 80)

    results = []

    for name, filename in FILES_TO_CHECK.items():

        df = load_csv(
            filename
        )

        if df is None:

            results.append({
                "component":
                    f"data_{name}",
                "status":
                    "FAIL",
                "rows": 0,
                "columns": 0,
                "missing_values": 0
            })

            continue

        rows = len(df)

        columns = len(
            df.columns
        )

        missing = int(
            df.isna()
            .sum()
            .sum()
        )

        # A completely empty CSV is invalid.
        passed = (
            rows > 0
            and columns > 0
        )

        results.append({
            "component":
                f"data_{name}",
            "status":
                "PASS"
                if passed
                else "FAIL",
            "rows": rows,
            "columns": columns,
            "missing_values": missing
        })

        print(
            f"{name:<28} : "
            f"rows={rows:<6} "
            f"columns={columns:<3} "
            f"missing={missing}"
        )

    return results


# ============================================================
# NORMALIZE METRIC
# ============================================================

def get_metric(
    metrics,
    names,
    default=0.0
):

    for name in names:

        if name in metrics:

            return safe_float(
                metrics[name],
                default
            )

    return default


# ============================================================
# NORMALIZE PERCENTAGE
# ============================================================

def normalize_percentage(
    value
):

    value = safe_float(
        value
    )

    if abs(value) > 1:

        return value / 100.0

    return value


# ============================================================
# LOAD ALL METRICS
# ============================================================

def load_all_metrics():

    print()
    print("=" * 80)
    print("LOADING STAGE 20 METRICS")
    print("=" * 80)

    metrics = {}

    for name, filename in METRICS_FILES.items():

        path = os.path.join(
            DATA_DIR,
            filename
        )

        data = load_json(
            path
        )

        metrics[name] = data

        print(
            f"{name:<32} : "
            f"{'LOADED' if data else 'NOT AVAILABLE'}"
        )

    return metrics


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_model(
    metrics
):

    print()
    print("=" * 80)
    print("VALIDATING ML PERFORMANCE")
    print("=" * 80)

    checks = []

    backtest = metrics.get(
        "backtest_metrics",
        {}
    )

    walk_forward = metrics.get(
        "walk_forward_metrics",
        {}
    )

    risk = metrics.get(
        "risk_metrics",
        {}
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = get_metric(
        walk_forward,
        [
            "ml_accuracy",
            "accuracy"
        ]
    )

    if accuracy == 0:

        accuracy = get_metric(
            backtest,
            [
                "ml_accuracy",
                "accuracy"
            ]
        )

    accuracy = normalize_percentage(
        accuracy
    )

    accuracy_pass = (
        accuracy
        >= MIN_ML_ACCURACY
    )

    checks.append({
        "check":
            "ml_accuracy",
        "value":
            accuracy,
        "threshold":
            MIN_ML_ACCURACY,
        "passed":
            accuracy_pass
    })

    print(
        f"ML accuracy           : "
        f"{accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Win rate
    # --------------------------------------------------------

    win_rate = get_metric(
        walk_forward,
        [
            "win_rate"
        ]
    )

    if win_rate == 0:

        win_rate = get_metric(
            risk,
            [
                "win_rate"
            ]
        )

    win_rate = normalize_percentage(
        win_rate
    )

    win_pass = (
        win_rate
        >= MIN_WIN_RATE
    )

    checks.append({
        "check":
            "win_rate",
        "value":
            win_rate,
        "threshold":
            MIN_WIN_RATE,
        "passed":
            win_pass
    })

    print(
        f"Win rate             : "
        f"{win_rate * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Profit factor
    # --------------------------------------------------------

    profit_factor = get_metric(
        risk,
        [
            "profit_factor"
        ]
    )

    if profit_factor == 0:

        profit_factor = get_metric(
            backtest,
            [
                "profit_factor"
            ]
        )

    profit_pass = (
        profit_factor
        >= MIN_PROFIT_FACTOR
    )

    checks.append({
        "check":
            "profit_factor",
        "value":
            profit_factor,
        "threshold":
            MIN_PROFIT_FACTOR,
        "passed":
            profit_pass
    })

    print(
        f"Profit factor        : "
        f"{profit_factor:.4f}"
    )

    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    sharpe = get_metric(
        risk,
        [
            "sharpe_ratio",
            "sharpe"
        ]
    )

    if sharpe == 0:

        sharpe = get_metric(
            backtest,
            [
                "sharpe_ratio",
                "sharpe"
            ]
        )

    sharpe_pass = (
        sharpe
        >= MIN_SHARPE
    )

    checks.append({
        "check":
            "sharpe_ratio",
        "value":
            sharpe,
        "threshold":
            MIN_SHARPE,
        "passed":
            sharpe_pass
    })

    print(
        f"Sharpe ratio         : "
        f"{sharpe:.4f}"
    )

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    drawdown = get_metric(
        risk,
        [
            "maximum_drawdown",
            "max_drawdown"
        ]
    )

    if drawdown == 0:

        drawdown = get_metric(
            backtest,
            [
                "maximum_drawdown",
                "max_drawdown"
            ]
        )

    drawdown = normalize_percentage(
        drawdown
    )

    drawdown_pass = (
        drawdown
        >= MAX_DRAWDOWN
    )

    checks.append({
        "check":
            "maximum_drawdown",
        "value":
            drawdown,
        "threshold":
            MAX_DRAWDOWN,
        "passed":
            drawdown_pass
    })

    print(
        f"Maximum drawdown     : "
        f"{drawdown * 100:.2f}%"
    )

    return checks


# ============================================================
# PORTFOLIO VALIDATION
# ============================================================

def validate_portfolio(
    metrics
):

    print()
    print("=" * 80)
    print("VALIDATING FINAL PORTFOLIO")
    print("=" * 80)

    portfolio = metrics.get(
        "portfolio_validation_metrics",
        {}
    )

    checks = []

    validation_status = str(
        portfolio.get(
            "validation_status",
            ""
        )
    ).upper()

    checks_passed = int(
        get_metric(
            portfolio,
            [
                "checks_passed"
            ]
        )
    )

    checks_failed = int(
        get_metric(
            portfolio,
            [
                "checks_failed"
            ]
        )
    )

    positions = int(
        get_metric(
            portfolio,
            [
                "positions"
            ]
        )
    )

    valid_positions = int(
        get_metric(
            portfolio,
            [
                "valid_positions"
            ]
        )
    )

    exposure = get_metric(
        portfolio,
        [
            "total_exposure"
        ]
    )

    largest_position = get_metric(
        portfolio,
        [
            "largest_position"
        ]
    )

    # --------------------------------------------------------
    # Validation status
    # --------------------------------------------------------

    status_pass = (
        validation_status == "PASS"
    )

    checks.append({
        "check":
            "portfolio_validation",
        "value":
            validation_status,
        "threshold":
            "PASS",
        "passed":
            status_pass
    })

    # --------------------------------------------------------
    # Invalid positions
    # --------------------------------------------------------

    positions_pass = (
        positions > 0
        and
        valid_positions == positions
    )

    checks.append({
        "check":
            "valid_positions",
        "value":
            valid_positions,
        "threshold":
            positions,
        "passed":
            positions_pass
    })

    # --------------------------------------------------------
    # Failed checks
    # --------------------------------------------------------

    failed_pass = (
        checks_failed == 0
    )

    checks.append({
        "check":
            "portfolio_checks",
        "value":
            checks_failed,
        "threshold":
            0,
        "passed":
            failed_pass
    })

    print(
        f"Portfolio status      : "
        f"{validation_status}"
    )

    print(
        f"Positions             : "
        f"{positions}"
    )

    print(
        f"Valid positions       : "
        f"{valid_positions}"
    )

    print(
        f"Checks passed         : "
        f"{checks_passed}"
    )

    print(
        f"Checks failed         : "
        f"{checks_failed}"
    )

    print(
        f"Total exposure        : "
        f"{exposure * 100:.2f}%"
    )

    print(
        f"Largest position      : "
        f"{largest_position * 100:.2f}%"
    )

    return checks


# ============================================================
# BUILD AUDIT TABLE
# ============================================================

def build_audit_table(
    file_results,
    metric_results,
    data_results,
    model_checks,
    portfolio_checks
):

    rows = []

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    for item in file_results:

        rows.append({
            "category":
                "FILE",
            "component":
                item["component"],
            "status":
                item["status"],
            "value":
                item.get(
                    "size_bytes",
                    0
                ),
            "threshold":
                "EXISTS"
        })

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    for item in metric_results:

        rows.append({
            "category":
                "METRICS",
            "component":
                item["component"],
            "status":
                item["status"],
            "value":
                "",
            "threshold":
                "EXISTS"
        })

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    for item in data_results:

        rows.append({
            "category":
                "DATA",
            "component":
                item["component"],
            "status":
                item["status"],
            "value":
                item.get(
                    "rows",
                    0
                ),
            "threshold":
                "> 0 ROWS"
        })

    # --------------------------------------------------------
    # Model checks
    # --------------------------------------------------------

    for item in model_checks:

        rows.append({
            "category":
                "MODEL",
            "component":
                item["check"],
            "status":
                "PASS"
                if item["passed"]
                else "FAIL",
            "value":
                item["value"],
            "threshold":
                item["threshold"]
        })

    # --------------------------------------------------------
    # Portfolio checks
    # --------------------------------------------------------

    for item in portfolio_checks:

        rows.append({
            "category":
                "PORTFOLIO",
            "component":
                item["check"],
            "status":
                "PASS"
                if item["passed"]
                else "FAIL",
            "value":
                item["value"],
            "threshold":
                item["threshold"]
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# FINAL DECISION
# ============================================================

def final_decision(
    audit_df
):

    failed = int(
        (
            audit_df[
                "status"
            ]
            == "FAIL"
        ).sum()
    )

    passed = int(
        (
            audit_df[
                "status"
            ]
            == "PASS"
        ).sum()
    )

    total = len(
        audit_df
    )

    if failed == 0:

        decision = "PASS"

    elif failed <= 2:

        decision = "REVIEW"

    else:

        decision = "REJECT"

    return (
        decision,
        passed,
        failed,
        total
    )


# ============================================================
# SAVE
# ============================================================

def save_results(
    audit_df,
    metrics
):

    print()
    print("=" * 80)
    print("SAVING FINAL AUDIT")
    print("=" * 80)

    audit_df.to_csv(
        OUTPUT_REPORT,
        index=False
    )

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
        f"Audit report saved:"
    )

    print(
        f"  {OUTPUT_REPORT}"
    )

    print()
    print(
        f"Audit metrics saved:"
    )

    print(
        f"  {OUTPUT_METRICS}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("STAGE 20.13 - FINAL SYSTEM AUDIT")
    print("=" * 80)

    try:

        ensure_directory()

        # ----------------------------------------------------
        # FILE CHECK
        # ----------------------------------------------------

        file_results = (
            check_required_files()
        )

        # ----------------------------------------------------
        # METRIC CHECK
        # ----------------------------------------------------

        metric_results = (
            check_metric_files()
        )

        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        data_results = (
            check_data_quality()
        )

        # ----------------------------------------------------
        # LOAD METRICS
        # ----------------------------------------------------

        metrics = (
            load_all_metrics()
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model_checks = (
            validate_model(
                metrics
            )
        )

        # ----------------------------------------------------
        # PORTFOLIO
        # ----------------------------------------------------

        portfolio_checks = (
            validate_portfolio(
                metrics
            )
        )

        # ----------------------------------------------------
        # BUILD AUDIT
        # ----------------------------------------------------

        audit_df = (
            build_audit_table(
                file_results,
                metric_results,
                data_results,
                model_checks,
                portfolio_checks
            )
        )

        # ----------------------------------------------------
        # FINAL DECISION
        # ----------------------------------------------------

        decision, passed, failed, total = (
            final_decision(
                audit_df
            )
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        final_metrics = {

            "stage":
                "20.13",

            "stage_name":
                "Final System Audit",

            "timestamp":
                datetime.now().isoformat(),

            "checks_total":
                total,

            "checks_passed":
                passed,

            "checks_failed":
                failed,

            "file_checks":
                len(file_results),

            "metric_file_checks":
                len(metric_results),

            "data_quality_checks":
                len(data_results),

            "model_checks":
                len(model_checks),

            "portfolio_checks":
                len(portfolio_checks),

            "final_decision":
                decision
        }

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_results(
            audit_df,
            final_metrics
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        print()
        print("=" * 80)
        print("STAGE 20.13 FINAL AUDIT SUMMARY")
        print("=" * 80)

        print()
        print(
            f"Total checks          : "
            f"{total}"
        )

        print(
            f"Checks passed         : "
            f"{passed}"
        )

        print(
            f"Checks failed         : "
            f"{failed}"
        )

        print()
        print(
            "=" * 80
        )

        print(
            f"FINAL DECISION        : "
            f"{decision}"
        )

        print(
            "=" * 80
        )

        print()
        print(
            "STAGE 20.13 FINAL SYSTEM AUDIT COMPLETE"
        )

        print(
            "=" * 80
        )

        print()
        print(
            f"Report : {OUTPUT_REPORT}"
        )

        print(
            f"Metrics: {OUTPUT_METRICS}"
        )

        return final_metrics

    except Exception as e:

        print()
        print("=" * 80)
        print("STAGE 20.13 FINAL AUDIT FAILED")
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