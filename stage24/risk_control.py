from pathlib import Path
import json
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# STAGE 24 - RISK CONTROL & DRAWDOWN REDUCTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "stage21"
    / "stage21_paper_trades.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "stage24"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# RISK PARAMETERS
# ------------------------------------------------------------

MAX_POSITION_SIZE = 0.50       # cap each trade at 50% of original size
MAX_DRAWDOWN_LIMIT = -0.30     # target maximum drawdown: -30%
MIN_WIN_RATE = 0.50
MIN_PROFIT_FACTOR = 1.00
MIN_SHARPE = 0.00

PAPER_TRADING_ONLY = True


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

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


def calculate_metrics(returns):
    returns = pd.Series(returns).dropna().astype(float)

    if len(returns) == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "maximum_drawdown": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
        }

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = float((returns > 0).mean())

    total_return = float((1.0 + returns).prod() - 1.0)

    equity = (1.0 + returns).cumprod()

    running_max = equity.cummax()

    drawdown = (equity / running_max) - 1.0

    maximum_drawdown = float(drawdown.min())

    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    std = float(returns.std(ddof=1))

    if std > 0:
        sharpe_ratio = float(
            returns.mean() / std * (252 ** 0.5)
        )
    else:
        sharpe_ratio = 0.0

    return {
        "trades": int(len(returns)),
        "win_rate": win_rate,
        "total_return": total_return,
        "maximum_drawdown": maximum_drawdown,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
    }


# ------------------------------------------------------------
# LOAD STAGE 21 TRADES
# ------------------------------------------------------------

print("=" * 80)
print("STAGE 24 - RISK CONTROL & DRAWDOWN REDUCTION")
print("=" * 80)

print()
print("Input:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Stage 21 paper trades not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print()
print(f"Rows loaded : {len(df)}")

print()
print("Columns:")
print(list(df.columns))


# ------------------------------------------------------------
# FIND RETURN COLUMN
# ------------------------------------------------------------

return_column = None

possible_return_columns = [
    "trade_return",
    "paper_return",
    "future_return",
]

for column in possible_return_columns:
    if column in df.columns:
        return_column = column
        break

if return_column is None:
    raise ValueError(
        "Could not find a usable return column. "
        "Expected one of: trade_return, paper_return, future_return."
    )

print()
print(f"Return column : {return_column}")


# ------------------------------------------------------------
# CLEAN RETURNS
# ------------------------------------------------------------

df[return_column] = pd.to_numeric(
    df[return_column],
    errors="coerce"
)

df = df.dropna(subset=[return_column]).copy()

if len(df) == 0:
    raise ValueError("No valid trade returns found.")


# ------------------------------------------------------------
# ORIGINAL METRICS
# ------------------------------------------------------------

original_returns = df[return_column].astype(float)

original_metrics = calculate_metrics(original_returns)


# ------------------------------------------------------------
# APPLY RISK CONTROL
# ------------------------------------------------------------
#
# The original return is scaled by MAX_POSITION_SIZE.
#
# Example:
# original trade return = -10%
# risk-controlled return = -5%
#
# This does NOT modify the prediction/model.
# It only simulates smaller exposure.
# ------------------------------------------------------------

df["original_trade_return"] = df[return_column]

df["risk_control_factor"] = MAX_POSITION_SIZE

df["risk_controlled_return"] = (
    df["original_trade_return"]
    * df["risk_control_factor"]
)


# ------------------------------------------------------------
# CALCULATE RISK-CONTROLLED METRICS
# ------------------------------------------------------------

controlled_returns = df["risk_controlled_return"]

controlled_metrics = calculate_metrics(
    controlled_returns
)


# ------------------------------------------------------------
# DECISION
# ------------------------------------------------------------

checks = []

def add_check(name, passed, value, threshold, message):
    checks.append({
        "timestamp": now_iso(),
        "category": "RISK_CONTROL",
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "value": value,
        "threshold": threshold,
        "message": message,
    })


# Drawdown
add_check(
    "risk_controlled_max_drawdown",
    controlled_metrics["maximum_drawdown"] >= MAX_DRAWDOWN_LIMIT,
    controlled_metrics["maximum_drawdown"],
    MAX_DRAWDOWN_LIMIT,
    "Risk-controlled drawdown must remain within limit.",
)


# Win rate
add_check(
    "risk_controlled_win_rate",
    controlled_metrics["win_rate"] >= MIN_WIN_RATE,
    controlled_metrics["win_rate"],
    MIN_WIN_RATE,
    "Risk-controlled win rate must meet minimum requirement.",
)


# Profit factor
add_check(
    "risk_controlled_profit_factor",
    controlled_metrics["profit_factor"] >= MIN_PROFIT_FACTOR,
    controlled_metrics["profit_factor"],
    MIN_PROFIT_FACTOR,
    "Risk-controlled profit factor must meet minimum requirement.",
)


# Sharpe
add_check(
    "risk_controlled_sharpe",
    controlled_metrics["sharpe_ratio"] >= MIN_SHARPE,
    controlled_metrics["sharpe_ratio"],
    MIN_SHARPE,
    "Risk-controlled Sharpe ratio must meet minimum requirement.",
)


# Paper-only safety
add_check(
    "paper_trading_only",
    PAPER_TRADING_ONLY is True,
    PAPER_TRADING_ONLY,
    True,
    "Live trading remains disabled.",
)


# ------------------------------------------------------------
# DATA QUALITY CHECKS
# ------------------------------------------------------------

add_check(
    "trade_count_available",
    len(df) > 0,
    len(df),
    1,
    "At least one valid paper trade is required.",
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

checks_df = pd.DataFrame(checks)

total_checks = len(checks_df)
passed_checks = int(
    (checks_df["status"] == "PASS").sum()
)
failed_checks = int(
    (checks_df["status"] == "FAIL").sum()
)

pass_rate = (
    passed_checks / total_checks
    if total_checks > 0
    else 0.0
)


# ------------------------------------------------------------
# PROMOTION DECISION
# ------------------------------------------------------------

if failed_checks == 0:
    final_decision = "READY_FOR_STAGE_25"
else:
    final_decision = "REVIEW"


# ------------------------------------------------------------
# SAVE RISK-CONTROLLED TRADES
# ------------------------------------------------------------

adjusted_trades_path = (
    OUTPUT_DIR
    / "stage24_risk_controlled_trades.csv"
)

df.to_csv(
    adjusted_trades_path,
    index=False
)


# ------------------------------------------------------------
# SAVE HEALTH REPORT
# ------------------------------------------------------------

health_report_path = (
    OUTPUT_DIR
    / "stage24_health_report.csv"
)

checks_df.to_csv(
    health_report_path,
    index=False
)


# ------------------------------------------------------------
# SAVE METRICS
# ------------------------------------------------------------

metrics = {
    "stage": "24",
    "stage_name": "Risk Control & Drawdown Reduction",
    "generated_at": now_iso(),

    "paper_trading_only": PAPER_TRADING_ONLY,

    "input_file": str(INPUT_FILE),

    "risk_parameters": {
        "max_position_size": MAX_POSITION_SIZE,
        "maximum_drawdown_limit": MAX_DRAWDOWN_LIMIT,
        "minimum_win_rate": MIN_WIN_RATE,
        "minimum_profit_factor": MIN_PROFIT_FACTOR,
        "minimum_sharpe": MIN_SHARPE,
    },

    "original_metrics": original_metrics,

    "risk_controlled_metrics": controlled_metrics,

    "checks": {
        "total": total_checks,
        "passed": passed_checks,
        "failed": failed_checks,
        "pass_rate": pass_rate,
    },

    "final_decision": final_decision,
}


metrics_path = (
    OUTPUT_DIR
    / "stage24_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=2,
        default=str
    )


# ------------------------------------------------------------
# SAVE RELEASE MANIFEST
# ------------------------------------------------------------

manifest = {
    "stage": "24",
    "stage_name": "Risk Control & Drawdown Reduction",
    "generated_at": now_iso(),

    "paper_trading_only": True,

    "source_stage": "21",

    "input_directory": str(
        INPUT_FILE.parent
    ),

    "output_directory": str(
        OUTPUT_DIR
    ),

    "final_decision": final_decision,

    "checks": {
        "total": total_checks,
        "passed": passed_checks,
        "failed": failed_checks,
        "pass_rate": pass_rate,
    },

    "artifacts": {
        "health_report": str(
            health_report_path
        ),
        "metrics": str(
            metrics_path
        ),
        "risk_controlled_trades": str(
            adjusted_trades_path
        ),
    },
}


manifest_path = (
    OUTPUT_DIR
    / "stage24_release_manifest.json"
)

with open(
    manifest_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
        default=str
    )


# ------------------------------------------------------------
# TERMINAL SUMMARY
# ------------------------------------------------------------

print()
print("=" * 80)
print("STAGE 24 RISK CONTROL SUMMARY")
print("=" * 80)

print()
print(f"Original trades          : {original_metrics['trades']}")
print(
    f"Original win rate        : "
    f"{original_metrics['win_rate']:.2%}"
)
print(
    f"Original total return    : "
    f"{original_metrics['total_return']:.2%}"
)
print(
    f"Original max drawdown    : "
    f"{original_metrics['maximum_drawdown']:.2%}"
)
print(
    f"Original profit factor   : "
    f"{original_metrics['profit_factor']:.4f}"
)
print(
    f"Original Sharpe          : "
    f"{original_metrics['sharpe_ratio']:.4f}"
)

print()
print("-" * 80)

print(
    f"Risk-control position cap : "
    f"{MAX_POSITION_SIZE:.0%}"
)

print(
    f"Controlled win rate       : "
    f"{controlled_metrics['win_rate']:.2%}"
)

print(
    f"Controlled total return   : "
    f"{controlled_metrics['total_return']:.2%}"
)

print(
    f"Controlled max drawdown   : "
    f"{controlled_metrics['maximum_drawdown']:.2%}"
)

print(
    f"Controlled profit factor  : "
    f"{controlled_metrics['profit_factor']:.4f}"
)

print(
    f"Controlled Sharpe         : "
    f"{controlled_metrics['sharpe_ratio']:.4f}"
)

print()
print("-" * 80)

print(f"Checks total : {total_checks}")
print(f"Checks passed: {passed_checks}")
print(f"Checks failed: {failed_checks}")
print(f"Pass rate    : {pass_rate:.2%}")

print()
print(f"PAPER TRADING ONLY : {'YES' if PAPER_TRADING_ONLY else 'NO'}")

print()
print(f"FINAL DECISION : {final_decision}")

print()
print("Health report:")
print(health_report_path)

print()
print("Metrics:")
print(metrics_path)

print()
print("Risk-controlled trades:")
print(adjusted_trades_path)

print()
print("Manifest:")
print(manifest_path)

print()
print("=" * 80)
print("STAGE 24 COMPLETE")
print("=" * 80)