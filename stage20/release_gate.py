# ============================================================
# STAGE 20.14 - PRODUCTION RELEASE GATE
# stage20/release_gate.py
# ============================================================

import os
import json
import time
from datetime import datetime

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

STAGE_DIR = "data/stage20"

AUDIT_FILE = os.path.join(
    STAGE_DIR,
    "stage20_final_audit.csv"
)

AUDIT_METRICS_FILE = os.path.join(
    STAGE_DIR,
    "stage20_final_audit_metrics.json"
)

PORTFOLIO_FILE = os.path.join(
    STAGE_DIR,
    "stage20_portfolio.csv"
)

PORTFOLIO_VALIDATION_FILE = os.path.join(
    STAGE_DIR,
    "stage20_portfolio_validation.csv"
)

PORTFOLIO_VALIDATION_METRICS = os.path.join(
    STAGE_DIR,
    "stage20_portfolio_validation_metrics.json"
)

RISK_FILE = os.path.join(
    STAGE_DIR,
    "stage20_risk_validation.csv"
)

RISK_METRICS_FILE = os.path.join(
    STAGE_DIR,
    "stage20_risk_metrics.json"
)

OPTIMIZATION_FILE = os.path.join(
    STAGE_DIR,
    "stage20_optimization_results.csv"
)

OPTIMIZATION_METRICS_FILE = os.path.join(
    STAGE_DIR,
    "stage20_optimization_metrics.json"
)

WALK_FORWARD_FILE = os.path.join(
    STAGE_DIR,
    "stage20_walk_forward_results.csv"
)

WALK_FORWARD_METRICS_FILE = os.path.join(
    STAGE_DIR,
    "stage20_walk_forward_metrics.json"
)

BACKTEST_FILE = os.path.join(
    STAGE_DIR,
    "stage20_backtest_results.csv"
)

BACKTEST_METRICS_FILE = os.path.join(
    STAGE_DIR,
    "stage20_backtest_metrics.json"
)

FINAL_CANDIDATES_FILE = os.path.join(
    STAGE_DIR,
    "stage20_final_candidates.csv"
)

RELEASE_REPORT_FILE = os.path.join(
    STAGE_DIR,
    "stage20_release_report.csv"
)

RELEASE_METRICS_FILE = os.path.join(
    STAGE_DIR,
    "stage20_release_metrics.json"
)

RELEASE_MANIFEST_FILE = os.path.join(
    STAGE_DIR,
    "stage20_release_manifest.json"
)


# ============================================================
# THRESHOLDS
# ============================================================

MIN_AUDIT_PASS_RATE = 0.95

MIN_WALK_FORWARD_ACCURACY = 0.50

MIN_WIN_RATE = 0.50

MIN_PROFIT_FACTOR = 1.00

MAX_DRAWDOWN = -50.00

MIN_SHARPE = 0.00

MAX_FAILED_AUDIT_CHECKS = 0

MIN_PORTFOLIO_POSITIONS = 1

MAX_PORTFOLIO_POSITIONS = 20


# ============================================================
# DIRECTORY
# ============================================================

def ensure_directory():

    os.makedirs(
        STAGE_DIR,
        exist_ok=True
    )


# ============================================================
# PRINT HEADER
# ============================================================

def print_header():

    print()
    print("=" * 80)
    print("STAGE 20.14 - PRODUCTION RELEASE GATE")
    print("=" * 80)


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_float(
    value,
    default=np.nan
):

    try:

        if value is None:

            return default

        if isinstance(value, str):

            value = (
                value
                .replace("%", "")
                .replace(",", "")
                .strip()
            )

        return float(value)

    except Exception:

        return default


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(
    path,
    required=False
):

    if not os.path.exists(path):

        if required:

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        return None

    try:

        df = pd.read_csv(path)

        return df

    except Exception as e:

        if required:

            raise ValueError(
                f"Could not read {path}: {e}"
            )

        print(
            f"WARNING: Could not read {path}: {e}"
        )

        return None


# ============================================================
# LOAD JSON
# ============================================================

def load_json(
    path,
    required=False
):

    if not os.path.exists(path):

        if required:

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, dict):

            return data

        return {}

    except Exception as e:

        if required:

            raise ValueError(
                f"Could not read {path}: {e}"
            )

        print(
            f"WARNING: Could not read {path}: {e}"
        )

        return {}


# ============================================================
# FIND METRIC
# ============================================================

def find_metric(
    metrics,
    names,
    default=np.nan
):

    if not isinstance(
        metrics,
        dict
    ):

        return default

    normalized = {}

    for key, value in metrics.items():

        normalized[
            str(key).strip().lower()
        ] = value

    for name in names:

        key = (
            str(name)
            .strip()
            .lower()
        )

        if key in normalized:

            return safe_float(
                normalized[key],
                default
            )

    # Nested dictionaries
    for value in metrics.values():

        if isinstance(
            value,
            dict
        ):

            result = find_metric(
                value,
                names,
                default
            )

            if not np.isnan(result):

                return result

    return default


# ============================================================
# CHECK FILE
# ============================================================

def check_file(
    path,
    name,
    checks
):

    exists = os.path.exists(path)

    size = (
        os.path.getsize(path)
        if exists
        else 0
    )

    status = (
        "PASS"
        if exists and size > 0
        else "FAIL"
    )

    checks.append(
        {
            "check": name,
            "status": status,
            "value": (
                f"{size} bytes"
                if exists
                else "missing"
            ),
            "reason": (
                "File exists"
                if status == "PASS"
                else "Required file missing or empty"
            ),
        }
    )

    return exists


# ============================================================
# LOAD AUDIT
# ============================================================

def load_audit():

    print()
    print("=" * 80)
    print("LOADING STAGE 20.13 AUDIT")
    print("=" * 80)

    audit = load_csv(
        AUDIT_FILE,
        required=True
    )

    metrics = load_json(
        AUDIT_METRICS_FILE,
        required=False
    )

    print()
    print(
        f"Audit rows loaded: {len(audit)}"
    )

    print(
        f"Audit columns loaded: {len(audit.columns)}"
    )

    return audit, metrics


# ============================================================
# AUDIT STATUS COLUMN
# ============================================================

def find_status_column(df):

    candidates = [
        "status",
        "result",
        "decision",
        "check_status",
        "validation_status",
    ]

    mapping = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        if candidate in mapping:

            return mapping[candidate]

    return None


# ============================================================
# AUDIT CHECK
# ============================================================

def evaluate_audit(
    audit,
    audit_metrics,
    checks
):

    print()
    print("=" * 80)
    print("EVALUATING AUDIT")
    print("=" * 80)

    status_column = find_status_column(
        audit
    )

    failed = 0
    passed = 0
    total = len(audit)

    if status_column is not None:

        statuses = (
            audit[status_column]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        passed = int(
            (
                statuses
                .isin(
                    [
                        "PASS",
                        "PASSED",
                        "OK",
                        "TRUE",
                    ]
                )
            ).sum()
        )

        failed = int(
            (
                statuses
                .isin(
                    [
                        "FAIL",
                        "FAILED",
                        "ERROR",
                        "FALSE",
                    ]
                )
            ).sum()
        )

    else:

        failed = int(
            find_metric(
                audit_metrics,
                [
                    "checks_failed",
                    "failed_checks",
                    "failures",
                ],
                0
            )
        )

        passed = int(
            find_metric(
                audit_metrics,
                [
                    "checks_passed",
                    "passed_checks",
                    "passes",
                ],
                0
            )
        )

        total = max(
            total,
            passed + failed
        )

    if total > 0:

        pass_rate = (
            passed / total
        )

    else:

        pass_rate = 0.0

    audit_pass = (
        failed <= MAX_FAILED_AUDIT_CHECKS
        and pass_rate >= MIN_AUDIT_PASS_RATE
    )

    checks.append(
        {
            "check": "Final audit pass rate",
            "status": (
                "PASS"
                if audit_pass
                else "FAIL"
            ),
            "value": (
                f"{pass_rate * 100:.2f}%"
            ),
            "reason": (
                f"{passed}/{total} checks passed; "
                f"{failed} failed"
            ),
        }
    )

    print(
        f"Audit checks : {total}"
    )

    print(
        f"Passed       : {passed}"
    )

    print(
        f"Failed       : {failed}"
    )

    print(
        f"Pass rate    : {pass_rate * 100:.2f}%"
    )

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "passed_gate": audit_pass,
    }


# ============================================================
# LOAD PERFORMANCE METRICS
# ============================================================

def evaluate_performance(
    checks
):

    print()
    print("=" * 80)
    print("EVALUATING PERFORMANCE")
    print("=" * 80)

    walk_metrics = load_json(
        WALK_FORWARD_METRICS_FILE
    )

    backtest_metrics = load_json(
        BACKTEST_METRICS_FILE
    )

    risk_metrics = load_json(
        RISK_METRICS_FILE
    )

    optimization_metrics = load_json(
        OPTIMIZATION_METRICS_FILE
    )

    # --------------------------------------------------------
    # Walk-forward accuracy
    # --------------------------------------------------------

    wf_accuracy = find_metric(
        walk_metrics,
        [
            "ml_accuracy",
            "accuracy",
            "walk_forward_accuracy",
        ]
    )

    if np.isnan(wf_accuracy):

        wf_accuracy = find_metric(
            backtest_metrics,
            [
                "ml_accuracy",
                "accuracy",
            ]
        )

    if not np.isnan(wf_accuracy):

        if wf_accuracy > 1:

            wf_accuracy /= 100.0

        accuracy_pass = (
            wf_accuracy
            >= MIN_WALK_FORWARD_ACCURACY
        )

        checks.append(
            {
                "check": "ML accuracy",
                "status": (
                    "PASS"
                    if accuracy_pass
                    else "FAIL"
                ),
                "value": (
                    f"{wf_accuracy * 100:.2f}%"
                ),
                "reason": (
                    f"Minimum required: "
                    f"{MIN_WALK_FORWARD_ACCURACY * 100:.2f}%"
                ),
            }
        )

    else:

        checks.append(
            {
                "check": "ML accuracy",
                "status": "FAIL",
                "value": "UNAVAILABLE",
                "reason": "No accuracy metric found",
            }
        )

    # --------------------------------------------------------
    # Win rate
    # --------------------------------------------------------

    win_rate = find_metric(
        risk_metrics,
        [
            "win_rate",
        ]
    )

    if np.isnan(win_rate):

        win_rate = find_metric(
            optimization_metrics,
            [
                "win_rate",
            ]
        )

    if not np.isnan(win_rate):

        if win_rate > 1:

            win_rate /= 100.0

        win_pass = (
            win_rate
            >= MIN_WIN_RATE
        )

        checks.append(
            {
                "check": "Win rate",
                "status": (
                    "PASS"
                    if win_pass
                    else "FAIL"
                ),
                "value": (
                    f"{win_rate * 100:.2f}%"
                ),
                "reason": (
                    f"Minimum required: "
                    f"{MIN_WIN_RATE * 100:.2f}%"
                ),
            }
        )

    else:

        checks.append(
            {
                "check": "Win rate",
                "status": "FAIL",
                "value": "UNAVAILABLE",
                "reason": "No win rate metric found",
            }
        )

    # --------------------------------------------------------
    # Profit factor
    # --------------------------------------------------------

    profit_factor = find_metric(
        risk_metrics,
        [
            "profit_factor",
        ]
    )

    if np.isnan(profit_factor):

        profit_factor = find_metric(
            optimization_metrics,
            [
                "profit_factor",
            ]
        )

    if not np.isnan(profit_factor):

        pf_pass = (
            profit_factor
            >= MIN_PROFIT_FACTOR
        )

        checks.append(
            {
                "check": "Profit factor",
                "status": (
                    "PASS"
                    if pf_pass
                    else "FAIL"
                ),
                "value": (
                    f"{profit_factor:.4f}"
                ),
                "reason": (
                    f"Minimum required: "
                    f"{MIN_PROFIT_FACTOR:.2f}"
                ),
            }
        )

    else:

        checks.append(
            {
                "check": "Profit factor",
                "status": "FAIL",
                "value": "UNAVAILABLE",
                "reason": "No profit factor found",
            }
        )

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    drawdown = find_metric(
        risk_metrics,
        [
            "maximum_drawdown",
            "max_drawdown",
        ]
    )

    if np.isnan(drawdown):

        drawdown = find_metric(
            optimization_metrics,
            [
                "maximum_drawdown",
                "max_drawdown",
            ]
        )

    if not np.isnan(drawdown):

        drawdown_pass = (
            drawdown >= MAX_DRAWDOWN
        )

        checks.append(
            {
                "check": "Maximum drawdown",
                "status": (
                    "PASS"
                    if drawdown_pass
                    else "FAIL"
                ),
                "value": (
                    f"{drawdown:.2f}%"
                ),
                "reason": (
                    f"Maximum allowed: "
                    f"{MAX_DRAWDOWN:.2f}%"
                ),
            }
        )

    else:

        checks.append(
            {
                "check": "Maximum drawdown",
                "status": "FAIL",
                "value": "UNAVAILABLE",
                "reason": "No drawdown metric found",
            }
        )

    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    sharpe = find_metric(
        risk_metrics,
        [
            "sharpe_ratio",
            "sharpe",
        ]
    )

    if np.isnan(sharpe):

        sharpe = find_metric(
            optimization_metrics,
            [
                "sharpe_ratio",
                "sharpe",
            ]
        )

    if not np.isnan(sharpe):

        sharpe_pass = (
            sharpe >= MIN_SHARPE
        )

        checks.append(
            {
                "check": "Sharpe ratio",
                "status": (
                    "PASS"
                    if sharpe_pass
                    else "FAIL"
                ),
                "value": (
                    f"{sharpe:.4f}"
                ),
                "reason": (
                    f"Minimum required: "
                    f"{MIN_SHARPE:.2f}"
                ),
            }
        )

    else:

        checks.append(
            {
                "check": "Sharpe ratio",
                "status": "FAIL",
                "value": "UNAVAILABLE",
                "reason": "No Sharpe metric found",
            }
        )

    return {
        "ml_accuracy": wf_accuracy,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "maximum_drawdown": drawdown,
        "sharpe_ratio": sharpe,
    }


# ============================================================
# VALIDATE PORTFOLIO
# ============================================================

def evaluate_portfolio(
    checks
):

    print()
    print("=" * 80)
    print("EVALUATING PORTFOLIO")
    print("=" * 80)

    portfolio = load_csv(
        PORTFOLIO_FILE
    )

    validation = load_csv(
        PORTFOLIO_VALIDATION_FILE
    )

    validation_metrics = load_json(
        PORTFOLIO_VALIDATION_METRICS
    )

    if portfolio is None:

        checks.append(
            {
                "check": "Portfolio file",
                "status": "FAIL",
                "value": "MISSING",
                "reason": "Portfolio file not found",
            }
        )

        return {
            "positions": 0,
            "valid_positions": 0,
        }

    positions = len(
        portfolio
    )

    valid_positions = positions

    if validation is not None:

        status_column = find_status_column(
            validation
        )

        if status_column:

            statuses = (
                validation[status_column]
                .astype(str)
                .str.upper()
                .str.strip()
            )

            valid_positions = int(
                statuses.isin(
                    [
                        "PASS",
                        "VALID",
                        "OK",
                        "TRUE",
                    ]
                ).sum()
            )

    if validation_metrics:

        metric_valid = find_metric(
            validation_metrics,
            [
                "valid_positions",
                "valid_position_count",
            ],
            np.nan
        )

        if not np.isnan(metric_valid):

            valid_positions = int(
                metric_valid
            )

    position_pass = (
        MIN_PORTFOLIO_POSITIONS
        <= positions
        <= MAX_PORTFOLIO_POSITIONS
    )

    validity_pass = (
        valid_positions == positions
    )

    checks.append(
        {
            "check": "Portfolio position count",
            "status": (
                "PASS"
                if position_pass
                else "FAIL"
            ),
            "value": str(positions),
            "reason": (
                f"Allowed range: "
                f"{MIN_PORTFOLIO_POSITIONS}-"
                f"{MAX_PORTFOLIO_POSITIONS}"
            ),
        }
    )

    checks.append(
        {
            "check": "Portfolio position validity",
            "status": (
                "PASS"
                if validity_pass
                else "FAIL"
            ),
            "value": (
                f"{valid_positions}/{positions}"
            ),
            "reason": (
                "All positions must be valid"
            ),
        }
    )

    return {
        "positions": positions,
        "valid_positions": valid_positions,
    }


# ============================================================
# CHECK REQUIRED ARTIFACTS
# ============================================================

def evaluate_artifacts(
    checks
):

    print()
    print("=" * 80)
    print("CHECKING REQUIRED ARTIFACTS")
    print("=" * 80)

    artifacts = [
        (
            AUDIT_FILE,
            "Final audit report"
        ),
        (
            AUDIT_METRICS_FILE,
            "Final audit metrics"
        ),
        (
            BACKTEST_FILE,
            "Backtest results"
        ),
        (
            BACKTEST_METRICS_FILE,
            "Backtest metrics"
        ),
        (
            WALK_FORWARD_FILE,
            "Walk-forward results"
        ),
        (
            WALK_FORWARD_METRICS_FILE,
            "Walk-forward metrics"
        ),
        (
            RISK_FILE,
            "Risk validation"
        ),
        (
            RISK_METRICS_FILE,
            "Risk metrics"
        ),
        (
            OPTIMIZATION_FILE,
            "Optimization results"
        ),
        (
            OPTIMIZATION_METRICS_FILE,
            "Optimization metrics"
        ),
        (
            PORTFOLIO_FILE,
            "Portfolio"
        ),
        (
            PORTFOLIO_VALIDATION_FILE,
            "Portfolio validation"
        ),
        (
            PORTFOLIO_VALIDATION_METRICS,
            "Portfolio validation metrics"
        ),
        (
            FINAL_CANDIDATES_FILE,
            "Final candidates"
        ),
    ]

    existing = 0

    for path, name in artifacts:

        if check_file(
            path,
            name,
            checks
        ):

            existing += 1

    print()
    print(
        f"Artifacts present: "
        f"{existing}/{len(artifacts)}"
    )

    return {
        "total": len(artifacts),
        "existing": existing,
    }


# ============================================================
# FINAL DECISION
# ============================================================

def calculate_decision(
    checks
):

    print()
    print("=" * 80)
    print("CALCULATING RELEASE DECISION")
    print("=" * 80)

    failed = [
        check
        for check in checks
        if check["status"] == "FAIL"
    ]

    total = len(checks)

    passed = (
        total - len(failed)
    )

    pass_rate = (
        passed / total
        if total > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Strict release gate
    # --------------------------------------------------------

    if len(failed) == 0:

        decision = "READY"

    elif pass_rate >= 0.90:

        decision = "REVIEW"

    else:

        decision = "BLOCKED"

    return {
        "decision": decision,
        "total_checks": total,
        "checks_passed": passed,
        "checks_failed": len(failed),
        "check_pass_rate": pass_rate,
        "failed_checks": failed,
    }


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    checks,
    decision,
    performance,
    portfolio,
    artifacts
):

    print()
    print("=" * 80)
    print("SAVING RELEASE REPORT")
    print("=" * 80)

    report = pd.DataFrame(
        checks
    )

    report.to_csv(
        RELEASE_REPORT_FILE,
        index=False
    )

    metrics = {
        "stage": "20.14",
        "stage_name": "Production Release Gate",
        "generated_at": datetime.now().isoformat(),

        "decision": decision["decision"],

        "total_checks": decision[
            "total_checks"
        ],

        "checks_passed": decision[
            "checks_passed"
        ],

        "checks_failed": decision[
            "checks_failed"
        ],

        "check_pass_rate": decision[
            "check_pass_rate"
        ],

        "performance": {
            key: (
                None
                if pd.isna(value)
                else float(value)
            )
            for key, value
            in performance.items()
        },

        "portfolio": portfolio,

        "artifacts": artifacts,

        "thresholds": {
            "minimum_audit_pass_rate":
                MIN_AUDIT_PASS_RATE,

            "maximum_failed_audit_checks":
                MAX_FAILED_AUDIT_CHECKS,

            "minimum_walk_forward_accuracy":
                MIN_WALK_FORWARD_ACCURACY,

            "minimum_win_rate":
                MIN_WIN_RATE,

            "minimum_profit_factor":
                MIN_PROFIT_FACTOR,

            "maximum_drawdown":
                MAX_DRAWDOWN,

            "minimum_sharpe":
                MIN_SHARPE,
        },
    }

    with open(
        RELEASE_METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
            default=str
        )

    # --------------------------------------------------------
    # Release manifest
    # --------------------------------------------------------

    manifest = {
        "stage": "20.14",
        "name": "Production Release Gate",
        "generated_at": datetime.now().isoformat(),
        "decision": decision["decision"],

        "report": RELEASE_REPORT_FILE,
        "metrics": RELEASE_METRICS_FILE,

        "system_status": (
            "NOT_APPROVED"
            if decision["decision"] != "READY"
            else "APPROVED"
        ),

        "failed_checks": [
            item["check"]
            for item in decision["failed_checks"]
        ],
    }

    with open(
        RELEASE_MANIFEST_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4
        )

    print()
    print(
        "Release report saved:"
    )

    print(
        f"  {RELEASE_REPORT_FILE}"
    )

    print()
    print(
        "Release metrics saved:"
    )

    print(
        f"  {RELEASE_METRICS_FILE}"
    )

    print()
    print(
        "Release manifest saved:"
    )

    print(
        f"  {RELEASE_MANIFEST_FILE}"
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    decision,
    performance,
    portfolio,
    artifacts
):

    print()
    print("=" * 80)
    print("STAGE 20.14 - RELEASE GATE SUMMARY")
    print("=" * 80)

    print()

    print(
        f"Total checks       : "
        f"{decision['total_checks']}"
    )

    print(
        f"Checks passed      : "
        f"{decision['checks_passed']}"
    )

    print(
        f"Checks failed      : "
        f"{decision['checks_failed']}"
    )

    print(
        f"Check pass rate    : "
        f"{decision['check_pass_rate'] * 100:.2f}%"
    )

    print()

    print(
        "ML accuracy        : "
        + (
            f"{performance['ml_accuracy'] * 100:.2f}%"
            if not np.isnan(
                performance["ml_accuracy"]
            )
            else "N/A"
        )
    )

    print(
        "Win rate           : "
        + (
            f"{performance['win_rate'] * 100:.2f}%"
            if not np.isnan(
                performance["win_rate"]
            )
            else "N/A"
        )
    )

    print(
        "Profit factor      : "
        + (
            f"{performance['profit_factor']:.4f}"
            if not np.isnan(
                performance["profit_factor"]
            )
            else "N/A"
        )
    )

    print(
        "Maximum drawdown   : "
        + (
            f"{performance['maximum_drawdown']:.2f}%"
            if not np.isnan(
                performance["maximum_drawdown"]
            )
            else "N/A"
        )
    )

    print(
        "Sharpe ratio       : "
        + (
            f"{performance['sharpe_ratio']:.4f}"
            if not np.isnan(
                performance["sharpe_ratio"]
            )
            else "N/A"
        )
    )

    print()

    print(
        f"Portfolio positions: "
        f"{portfolio['positions']}"
    )

    print(
        f"Valid positions    : "
        f"{portfolio['valid_positions']}"
    )

    print()

    print(
        f"Artifacts present  : "
        f"{artifacts['existing']}/"
        f"{artifacts['total']}"
    )

    print()

    print(
        "FINAL DECISION     : "
        f"{decision['decision']}"
    )

    if decision["failed_checks"]:

        print()
        print(
            "FAILED CHECKS:"
        )

        for item in decision[
            "failed_checks"
        ]:

            print(
                f"  - {item['check']}: "
                f"{item['reason']}"
            )

    print()
    print("=" * 80)
    print(
        "STAGE 20.14 PRODUCTION RELEASE GATE COMPLETE"
    )
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_directory()

    print_header()

    try:

        # ----------------------------------------------------
        # LOAD AUDIT
        # ----------------------------------------------------

        audit, audit_metrics = load_audit()

        checks = []

        # ----------------------------------------------------
        # AUDIT
        # ----------------------------------------------------

        audit_summary = evaluate_audit(
            audit,
            audit_metrics,
            checks
        )

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        performance = evaluate_performance(
            checks
        )

        # ----------------------------------------------------
        # PORTFOLIO
        # ----------------------------------------------------

        portfolio = evaluate_portfolio(
            checks
        )

        # ----------------------------------------------------
        # ARTIFACTS
        # ----------------------------------------------------

        artifacts = evaluate_artifacts(
            checks
        )

        # ----------------------------------------------------
        # DECISION
        # ----------------------------------------------------

        decision = calculate_decision(
            checks
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_report(
            checks,
            decision,
            performance,
            portfolio,
            artifacts
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        display_results(
            decision,
            performance,
            portfolio,
            artifacts
        )

        return decision

    except Exception as e:

        print()
        print("=" * 80)
        print("STAGE 20.14 RELEASE GATE FAILED")
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

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()