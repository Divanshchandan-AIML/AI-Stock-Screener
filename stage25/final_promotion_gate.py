from pathlib import Path
import json
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# STAGE 25 - FINAL RISK-CONTROLLED PROMOTION GATE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = PROJECT_ROOT / "data" / "stage24"
OUTPUT_DIR = PROJECT_ROOT / "data" / "stage25"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REQUIRED STAGE 24 ARTIFACTS
# ============================================================

REQUIRED_FILES = [
    "stage24_health_report.csv",
    "stage24_metrics.json",
    "stage24_risk_controlled_trades.csv",
    "stage24_release_manifest.json",
]


# ============================================================
# FINAL PROMOTION THRESHOLDS
# ============================================================

MIN_WIN_RATE = 0.50
MIN_PROFIT_FACTOR = 1.00
MIN_SHARPE = 0.00
MAX_ALLOWED_DRAWDOWN = -0.30

PAPER_TRADING_ONLY = True


# ============================================================
# HELPERS
# ============================================================

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def safe_float(value, default=0.0):
    try:
        value = float(value)
        if pd.isna(value):
            return default
        return value
    except Exception:
        return default


def add_check(
    checks,
    category,
    name,
    passed,
    value,
    threshold,
    message
):
    checks.append({
        "timestamp": now_iso(),
        "category": category,
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "value": value,
        "threshold": threshold,
        "message": message,
    })


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("STAGE 25 - FINAL RISK-CONTROLLED PROMOTION GATE")
print("=" * 80)

print()
print("Source directory:")
print(SOURCE_DIR)

print()
print("Output directory:")
print(OUTPUT_DIR)


# ============================================================
# CHECK STAGE 24 ARTIFACTS
# ============================================================

checks = []

missing_files = []

for filename in REQUIRED_FILES:

    path = SOURCE_DIR / filename

    exists = path.exists()

    if not exists:
        missing_files.append(filename)

    add_check(
        checks,
        "ARTIFACT",
        f"{filename}_exists",
        exists,
        exists,
        True,
        "Required Stage 24 artifact must exist."
    )


# ============================================================
# STOP EARLY IF ARTIFACTS ARE MISSING
# ============================================================

if missing_files:

    print()
    print("Missing Stage 24 artifacts:")

    for filename in missing_files:
        print(f" - {filename}")

    final_decision = "REVIEW"

else:

    # ========================================================
    # LOAD STAGE 24 METRICS
    # ========================================================

    metrics_path = (
        SOURCE_DIR
        / "stage24_metrics.json"
    )

    with open(
        metrics_path,
        "r",
        encoding="utf-8"
    ) as f:

        stage24_metrics = json.load(f)


    # ========================================================
    # EXTRACT RISK-CONTROLLED METRICS
    # ========================================================

    controlled = stage24_metrics.get(
        "risk_controlled_metrics",
        {}
    )

    checks_summary = stage24_metrics.get(
        "checks",
        {}
    )

    win_rate = safe_float(
        controlled.get("win_rate")
    )

    total_return = safe_float(
        controlled.get("total_return")
    )

    maximum_drawdown = safe_float(
        controlled.get("maximum_drawdown")
    )

    profit_factor = safe_float(
        controlled.get("profit_factor")
    )

    sharpe_ratio = safe_float(
        controlled.get("sharpe_ratio")
    )

    trades = int(
        controlled.get("trades", 0)
    )

    stage24_passed = int(
        checks_summary.get("passed", 0)
    )

    stage24_failed = int(
        checks_summary.get("failed", 0)
    )


    # ========================================================
    # DISPLAY INPUT METRICS
    # ========================================================

    print()
    print("-" * 80)
    print("STAGE 24 INPUT METRICS")
    print("-" * 80)

    print()
    print(f"Trades                 : {trades}")
    print(f"Win rate               : {win_rate:.2%}")
    print(f"Total return           : {total_return:.2%}")
    print(f"Maximum drawdown       : {maximum_drawdown:.2%}")
    print(f"Profit factor          : {profit_factor:.4f}")
    print(f"Sharpe ratio           : {sharpe_ratio:.4f}")

    print()
    print(
        f"Stage 24 checks passed : "
        f"{stage24_passed}"
    )

    print(
        f"Stage 24 checks failed : "
        f"{stage24_failed}"
    )


    # ========================================================
    # FINAL GATE CHECKS
    # ========================================================

    add_check(
        checks,
        "STAGE24",
        "stage24_checks_passed",
        stage24_failed == 0,
        stage24_failed,
        0,
        "Stage 24 must have zero failed checks."
    )


    add_check(
        checks,
        "PERFORMANCE",
        "win_rate",
        win_rate >= MIN_WIN_RATE,
        win_rate,
        MIN_WIN_RATE,
        "Win rate must meet minimum requirement."
    )


    add_check(
        checks,
        "PERFORMANCE",
        "profit_factor",
        profit_factor >= MIN_PROFIT_FACTOR,
        profit_factor,
        MIN_PROFIT_FACTOR,
        "Profit factor must meet minimum requirement."
    )


    add_check(
        checks,
        "PERFORMANCE",
        "sharpe_ratio",
        sharpe_ratio >= MIN_SHARPE,
        sharpe_ratio,
        MIN_SHARPE,
        "Sharpe ratio must meet minimum requirement."
    )


    add_check(
        checks,
        "RISK",
        "maximum_drawdown",
        maximum_drawdown >= MAX_ALLOWED_DRAWDOWN,
        maximum_drawdown,
        MAX_ALLOWED_DRAWDOWN,
        "Maximum drawdown must remain within the risk limit."
    )


    add_check(
        checks,
        "SAFETY",
        "paper_trading_only",
        PAPER_TRADING_ONLY is True,
        PAPER_TRADING_ONLY,
        True,
        "Live trading must remain disabled."
    )


    add_check(
        checks,
        "DATA",
        "valid_trade_count",
        trades > 0,
        trades,
        1,
        "At least one valid trade is required."
    )


    # ========================================================
    # FINAL DECISION
    # ========================================================

    failed_checks = [
        check
        for check in checks
        if check["status"] == "FAIL"
    ]

    if len(failed_checks) == 0:

        final_decision = "PROMOTION_READY"

    else:

        final_decision = "REVIEW"


# ============================================================
# CHECK SUMMARY
# ============================================================

checks_df = pd.DataFrame(checks)

total_checks = len(checks_df)

passed_checks = int(
    (checks_df["status"] == "PASS").sum()
)

failed_checks_count = int(
    (checks_df["status"] == "FAIL").sum()
)

pass_rate = (
    passed_checks / total_checks
    if total_checks > 0
    else 0.0
)


# ============================================================
# SAVE HEALTH REPORT
# ============================================================

health_report_path = (
    OUTPUT_DIR
    / "stage25_health_report.csv"
)

checks_df.to_csv(
    health_report_path,
    index=False
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

final_metrics = {
    "stage": "25",
    "stage_name": "Final Risk-Controlled Promotion Gate",
    "generated_at": now_iso(),

    "source_stage": "24",

    "paper_trading_only": PAPER_TRADING_ONLY,

    "thresholds": {
        "minimum_win_rate": MIN_WIN_RATE,
        "minimum_profit_factor": MIN_PROFIT_FACTOR,
        "minimum_sharpe": MIN_SHARPE,
        "maximum_allowed_drawdown": MAX_ALLOWED_DRAWDOWN,
    },

    "risk_controlled_metrics": {
        "trades": trades,
        "win_rate": win_rate,
        "total_return": total_return,
        "maximum_drawdown": maximum_drawdown,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
    },

    "checks": {
        "total": total_checks,
        "passed": passed_checks,
        "failed": failed_checks_count,
        "pass_rate": pass_rate,
    },

    "final_decision": final_decision,
}


metrics_path = (
    OUTPUT_DIR
    / "stage25_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_metrics,
        f,
        indent=2
    )


# ============================================================
# SAVE RELEASE MANIFEST
# ============================================================

manifest = {
    "stage": "25",
    "stage_name": "Final Risk-Controlled Promotion Gate",
    "generated_at": now_iso(),

    "source_stage": "24",

    "paper_trading_only": True,

    "final_decision": final_decision,

    "checks": {
        "total": total_checks,
        "passed": passed_checks,
        "failed": failed_checks_count,
        "pass_rate": pass_rate,
    },

    "artifacts": {
        "health_report": str(
            health_report_path
        ),
        "metrics": str(
            metrics_path
        ),
    },
}


manifest_path = (
    OUTPUT_DIR
    / "stage25_release_manifest.json"
)

with open(
    manifest_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ============================================================
# FINAL TERMINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("STAGE 25 FINAL PROMOTION SUMMARY")
print("=" * 80)

print()
print(f"Checks total        : {total_checks}")
print(f"Checks passed       : {passed_checks}")
print(f"Checks failed       : {failed_checks_count}")
print(f"Pass rate           : {pass_rate:.2%}")

print()

if not missing_files:

    print(f"Win rate            : {win_rate:.2%}")
    print(f"Profit factor       : {profit_factor:.4f}")
    print(f"Sharpe ratio        : {sharpe_ratio:.4f}")
    print(f"Maximum drawdown    : {maximum_drawdown:.2%}")

print()
print(
    f"PAPER TRADING ONLY : "
    f"{'YES' if PAPER_TRADING_ONLY else 'NO'}"
)

print()
print(f"FINAL DECISION      : {final_decision}")

print()
print("Health report:")
print(health_report_path)

print()
print("Metrics:")
print(metrics_path)

print()
print("Manifest:")
print(manifest_path)

print()
print("=" * 80)
print("STAGE 25 COMPLETE")
print("=" * 80)