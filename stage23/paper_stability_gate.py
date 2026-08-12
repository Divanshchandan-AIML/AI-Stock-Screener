"""
STAGE 23 - PAPER TRADING STABILITY & PROMOTION GATE

Purpose
-------
Evaluate the Stage 22 paper-trading results using the actual Stage 21
paper-trade data.

IMPORTANT
---------
This stage is an evaluation gate only.

It does NOT:
    - place real orders
    - enable live trading
    - modify Stage 21
    - modify Stage 22
    - change the original paper-trading results

A risk-capped scenario is calculated separately for analysis.
It must never be confused with the actual strategy performance.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

STAGE21_DIR = ROOT / "data" / "stage21"
STAGE22_DIR = ROOT / "data" / "stage22"
STAGE23_DIR = ROOT / "data" / "stage23"

STAGE21_TRADES = (
    STAGE21_DIR / "stage21_paper_trades.csv"
)

STAGE21_METRICS = (
    STAGE21_DIR / "stage21_metrics.json"
)

STAGE22_METRICS = (
    STAGE22_DIR / "stage22_metrics.json"
)

STAGE22_REPORT = (
    STAGE22_DIR / "stage22_evaluation_report.csv"
)

STAGE22_MANIFEST = (
    STAGE22_DIR / "stage22_release_manifest.json"
)

OUTPUT_HEALTH = (
    STAGE23_DIR / "stage23_health_report.csv"
)

OUTPUT_METRICS = (
    STAGE23_DIR / "stage23_metrics.json"
)

OUTPUT_MANIFEST = (
    STAGE23_DIR / "stage23_release_manifest.json"
)

STAGE23_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAFETY / PROMOTION THRESHOLDS
# ============================================================

MIN_WIN_RATE = 0.50

MIN_PROFIT_FACTOR = 1.20

MIN_SHARPE = 0.80

MIN_PAPER_TRADES = 100

# Actual strategy drawdown must be no worse than -30%.
MAX_ALLOWED_DRAWDOWN = -0.30

# Separate hypothetical risk-control scenario.
# This DOES NOT modify the original strategy.
RISK_CAP = 0.50

# Live trading is always disabled in this stage.
LIVE_TRADING_ENABLED = False

PAPER_TRADING_ONLY = True


# ============================================================
# HELPERS
# ============================================================

def now_utc() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_float(
    value: Any,
    default: float = 0.0
) -> float:

    try:

        if value is None:
            return default

        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (
        TypeError,
        ValueError
    ):

        return default


def safe_int(
    value: Any,
    default: int = 0
) -> int:

    try:
        return int(value)

    except (
        TypeError,
        ValueError
    ):

        return default


def load_json(
    path: Path
) -> dict:

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if not isinstance(
        data,
        dict
    ):

        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return data


def save_json(
    path: Path,
    data: dict
) -> None:

    with path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            default=str
        )


# ============================================================
# CHECK COLLECTOR
# ============================================================

class CheckCollector:

    def __init__(self):

        self.rows = []

    def add(
        self,
        category: str,
        check: str,
        passed: bool,
        value: Any,
        threshold: Any,
        message: str = ""
    ):

        self.rows.append(
            {
                "timestamp": now_utc(),
                "category": category,
                "check": check,
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
                "value": value,
                "threshold": threshold,
                "message": message,
            }
        )

    @property
    def total(self) -> int:

        return len(
            self.rows
        )

    @property
    def passed(self) -> int:

        return sum(
            1
            for row in self.rows
            if row["status"] == "PASS"
        )

    @property
    def failed(self) -> int:

        return sum(
            1
            for row in self.rows
            if row["status"] == "FAIL"
        )

    @property
    def pass_rate(self) -> float:

        if self.total == 0:
            return 0.0

        return (
            self.passed
            /
            self.total
        )


# ============================================================
# REQUIRED ARTIFACT CHECK
# ============================================================

def check_required_artifacts(
    checks: CheckCollector
):

    artifacts = {

        "stage21_paper_trades":
            STAGE21_TRADES,

        "stage21_metrics":
            STAGE21_METRICS,

        "stage22_metrics":
            STAGE22_METRICS,

        "stage22_evaluation_report":
            STAGE22_REPORT,

        "stage22_release_manifest":
            STAGE22_MANIFEST,
    }

    for name, path in artifacts.items():

        exists = (
            path.exists()
            and path.is_file()
        )

        checks.add(
            "ARTIFACT",
            f"{name}_exists",
            exists,
            str(path),
            "EXISTS",
            str(path)
        )

    return artifacts


# ============================================================
# LOAD STAGE 22 METRICS
# ============================================================

def load_stage22_metrics(
    checks: CheckCollector
) -> dict:

    if not STAGE22_METRICS.exists():

        checks.add(
            "INPUT",
            "stage22_metrics_readable",
            False,
            "MISSING",
            "READABLE"
        )

        return {}

    try:

        metrics = load_json(
            STAGE22_METRICS
        )

        checks.add(
            "INPUT",
            "stage22_metrics_readable",
            True,
            "READABLE",
            "READABLE"
        )

        return metrics

    except Exception as exc:

        checks.add(
            "INPUT",
            "stage22_metrics_readable",
            False,
            str(exc),
            "READABLE"
        )

        return {}


# ============================================================
# LOAD STAGE 21 PAPER TRADES
# ============================================================

def load_paper_trades(
    checks: CheckCollector
) -> pd.DataFrame:

    if not STAGE21_TRADES.exists():

        checks.add(
            "TRADES",
            "stage21_paper_trades_readable",
            False,
            "MISSING",
            "READABLE"
        )

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            STAGE21_TRADES
        )

        checks.add(
            "TRADES",
            "stage21_paper_trades_readable",
            True,
            len(df),
            ">0"
        )

        return df

    except Exception as exc:

        checks.add(
            "TRADES",
            "stage21_paper_trades_readable",
            False,
            str(exc),
            "READABLE"
        )

        return pd.DataFrame()


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    df: pd.DataFrame,
    names: list[str]
) -> Optional[str]:

    lookup = {
        str(column).strip().lower():
            column
        for column in df.columns
    }

    for name in names:

        key = (
            name
            .strip()
            .lower()
        )

        if key in lookup:

            return lookup[key]

    return None


# ============================================================
# CALCULATE DRAWdown
# ============================================================

def calculate_drawdown(
    returns: pd.Series
) -> tuple[float, float]:

    returns = pd.to_numeric(
        returns,
        errors="coerce"
    ).fillna(0.0)

    capital = (
        1.0
        *
        (
            1.0 + returns
        ).cumprod()
    )

    running_max = (
        capital.cummax()
    )

    drawdown = (
        capital
        /
        running_max
        - 1.0
    )

    maximum_drawdown = float(
        drawdown.min()
    )

    total_return = float(
        capital.iloc[-1]
        - 1.0
    )

    return (
        total_return,
        maximum_drawdown
    )


# ============================================================
# RISK-CAPPED HYPOTHETICAL SCENARIO
# ============================================================

def calculate_risk_capped_scenario(
    trades: pd.DataFrame
) -> dict:

    if trades.empty:

        return {
            "available": False
        }

    return_column = find_column(
        trades,
        [
            "trade_return",
            "paper_return"
        ]
    )

    position_column = find_column(
        trades,
        [
            "position_size"
        ]
    )

    if return_column is None:

        return {
            "available": False,
            "reason":
                "trade_return/paper_return missing"
        }

    returns = pd.to_numeric(
        trades[return_column],
        errors="coerce"
    ).fillna(0.0)

    # --------------------------------------------------------
    # If position size exists, cap it at 50%.
    # --------------------------------------------------------

    if position_column is not None:

        positions = pd.to_numeric(
            trades[position_column],
            errors="coerce"
        ).fillna(0.0)

        position_factor = (
            positions.abs()
            .clip(
                upper=RISK_CAP
            )
        )

        original_abs = (
            positions.abs()
        )

        scaling = pd.Series(
            1.0,
            index=trades.index
        )

        valid = (
            original_abs > 0
        )

        scaling.loc[valid] = (
            position_factor.loc[valid]
            /
            original_abs.loc[valid]
        )

        capped_returns = (
            returns
            *
            scaling
        )

    else:

        # Conservative fallback.
        capped_returns = (
            returns
            * RISK_CAP
        )

    total_return, max_drawdown = (
        calculate_drawdown(
            capped_returns
        )
    )

    return {
        "available": True,

        "risk_cap":
            RISK_CAP,

        "total_return":
            total_return,

        "maximum_drawdown":
            max_drawdown,

        "trade_count":
            len(capped_returns),
    }


# ============================================================
# PERFORMANCE CHECKS
# ============================================================

def evaluate_performance(
    checks: CheckCollector,
    metrics: dict
):

    win_rate = safe_float(
        metrics.get(
            "paper_win_rate"
        )
    )

    profit_factor = safe_float(
        metrics.get(
            "paper_profit_factor"
        )
    )

    sharpe = safe_float(
        metrics.get(
            "paper_sharpe_ratio"
        )
    )

    paper_trades = safe_int(
        metrics.get(
            "paper_trades"
        )
    )

    max_drawdown = safe_float(
        metrics.get(
            "paper_maximum_drawdown"
        )
    )

    checks.add(
        "PERFORMANCE",
        "paper_win_rate",
        win_rate >= MIN_WIN_RATE,
        win_rate,
        MIN_WIN_RATE,
        "Actual Stage 22 win rate"
    )

    checks.add(
        "PERFORMANCE",
        "paper_profit_factor",
        profit_factor >= MIN_PROFIT_FACTOR,
        profit_factor,
        MIN_PROFIT_FACTOR,
        "Actual Stage 22 profit factor"
    )

    checks.add(
        "PERFORMANCE",
        "paper_sharpe_ratio",
        sharpe >= MIN_SHARPE,
        sharpe,
        MIN_SHARPE,
        "Actual Stage 22 Sharpe ratio"
    )

    checks.add(
        "PERFORMANCE",
        "paper_trade_count",
        paper_trades >= MIN_PAPER_TRADES,
        paper_trades,
        MIN_PAPER_TRADES,
        "Minimum paper-trading sample size"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # This is an actual-performance check.
    # We do NOT replace it with the risk-capped scenario.
    # --------------------------------------------------------

    checks.add(
        "RISK",
        "actual_paper_max_drawdown",
        max_drawdown >= MAX_ALLOWED_DRAWDOWN,
        max_drawdown,
        MAX_ALLOWED_DRAWDOWN,
        "Actual strategy drawdown must be within limit"
    )

    return {
        "win_rate":
            win_rate,

        "profit_factor":
            profit_factor,

        "sharpe_ratio":
            sharpe,

        "paper_trades":
            paper_trades,

        "maximum_drawdown":
            max_drawdown,
    }


# ============================================================
# SAFETY CHECKS
# ============================================================

def evaluate_safety(
    checks: CheckCollector,
    metrics: dict
):

    paper_only = bool(
        metrics.get(
            "paper_trading_only",
            False
        )
    )

    live_enabled = bool(
        metrics.get(
            "live_trading_enabled",
            False
        )
    )

    checks.add(
        "SAFETY",
        "paper_trading_only",
        paper_only,
        paper_only,
        True,
        "Paper trading must remain enabled"
    )

    checks.add(
        "SAFETY",
        "live_trading_disabled",
        not live_enabled,
        live_enabled,
        False,
        "Live trading must remain disabled"
    )

    return {
        "paper_trading_only":
            paper_only,

        "live_trading_enabled":
            live_enabled,
    }


# ============================================================
# DECISION
# ============================================================

def make_decision(
    checks: CheckCollector
) -> str:

    if checks.failed == 0:

        return "CONTINUE_PAPER_TRADING"

    return "REVIEW"


# ============================================================
# SAVE HEALTH REPORT
# ============================================================

def save_health_report(
    checks: CheckCollector
):

    with OUTPUT_HEALTH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "timestamp",
            "category",
            "check",
            "status",
            "value",
            "threshold",
            "message",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            checks.rows
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "STAGE 23 - PAPER TRADING "
        "STABILITY & PROMOTION GATE"
    )
    print("=" * 80)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This stage does NOT enable live trading."
    )

    print(
        "Original paper-trading performance "
        "is never modified."
    )

    checks = CheckCollector()

    # --------------------------------------------------------
    # Artifacts
    # --------------------------------------------------------

    check_required_artifacts(
        checks
    )

    # --------------------------------------------------------
    # Stage 22 metrics
    # --------------------------------------------------------

    metrics = load_stage22_metrics(
        checks
    )

    # --------------------------------------------------------
    # Stage 21 trades
    # --------------------------------------------------------

    trades = load_paper_trades(
        checks
    )

    # --------------------------------------------------------
    # Stage 22 performance
    # --------------------------------------------------------

    performance = evaluate_performance(
        checks,
        metrics
    )

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    safety = evaluate_safety(
        checks,
        metrics
    )

    # --------------------------------------------------------
    # Risk-capped hypothetical scenario
    # --------------------------------------------------------

    capped = calculate_risk_capped_scenario(
        trades
    )

    if capped.get("available"):

        capped_drawdown = safe_float(
            capped.get(
                "maximum_drawdown"
            )
        )

        capped_return = safe_float(
            capped.get(
                "total_return"
            )
        )

        print()
        print(
            "RISK-CAPPED HYPOTHETICAL SCENARIO"
        )

        print(
            f"Position cap       : "
            f"{RISK_CAP:.0%}"
        )

        print(
            f"Hypothetical return : "
            f"{capped_return:.2%}"
        )

        print(
            f"Hypothetical DD     : "
            f"{capped_drawdown:.2%}"
        )

        print()
        print(
            "NOTE:"
        )

        print(
            "This scenario is analytical only."
        )

        print(
            "It does NOT change the actual "
            "Stage 22 performance."
        )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    decision = make_decision(
        checks
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    output_metrics = {

        "stage":
            23,

        "stage_name":
            "Paper Trading Stability & Promotion Gate",

        "generated_at":
            now_utc(),

        "source_stage":
            22,

        "source_trade_stage":
            21,

        "paper_trading_only":
            PAPER_TRADING_ONLY,

        "live_trading_enabled":
            LIVE_TRADING_ENABLED,

        "actual_performance":
            performance,

        "safety":
            safety,

        "risk_capped_scenario":
            capped,

        "thresholds":
            {
                "minimum_win_rate":
                    MIN_WIN_RATE,

                "minimum_profit_factor":
                    MIN_PROFIT_FACTOR,

                "minimum_sharpe":
                    MIN_SHARPE,

                "minimum_paper_trades":
                    MIN_PAPER_TRADES,

                "maximum_allowed_drawdown":
                    MAX_ALLOWED_DRAWDOWN,

                "risk_cap":
                    RISK_CAP,
            },

        "checks":
            {
                "total":
                    checks.total,

                "passed":
                    checks.passed,

                "failed":
                    checks.failed,

                "pass_rate":
                    checks.pass_rate,
            },

        "final_decision":
            decision,
    }

    save_json(
        OUTPUT_METRICS,
        output_metrics
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {

        "stage":
            "23",

        "stage_name":
            "Paper Trading Stability & Promotion Gate",

        "source_stage":
            "22",

        "source_trade_stage":
            "21",

        "generated_at":
            now_utc(),

        "paper_trading_only":
            True,

        "live_trading_enabled":
            False,

        "final_decision":
            decision,

        "checks":
            {
                "total":
                    checks.total,

                "passed":
                    checks.passed,

                "failed":
                    checks.failed,

                "pass_rate":
                    checks.pass_rate,
            },

        "artifacts":
            {
                "health_report":
                    str(OUTPUT_HEALTH),

                "metrics":
                    str(OUTPUT_METRICS),

                "manifest":
                    str(OUTPUT_MANIFEST),
            },
    }

    save_json(
        OUTPUT_MANIFEST,
        manifest
    )

    save_health_report(
        checks
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "STAGE 23 STABILITY GATE SUMMARY"
    )
    print("=" * 80)

    print(
        f"Checks total       : "
        f"{checks.total}"
    )

    print(
        f"Checks passed      : "
        f"{checks.passed}"
    )

    print(
        f"Checks failed      : "
        f"{checks.failed}"
    )

    print(
        f"Pass rate          : "
        f"{checks.pass_rate:.2%}"
    )

    print()

    print(
        f"Actual win rate    : "
        f"{performance['win_rate']:.2%}"
    )

    print(
        f"Actual profit fac. : "
        f"{performance['profit_factor']:.4f}"
    )

    print(
        f"Actual Sharpe      : "
        f"{performance['sharpe_ratio']:.4f}"
    )

    print(
        f"Actual trades      : "
        f"{performance['paper_trades']}"
    )

    print(
        f"Actual max DD      : "
        f"{performance['maximum_drawdown']:.2%}"
    )

    print()

    print(
        f"Paper trading only : "
        f"{'YES' if PAPER_TRADING_ONLY else 'NO'}"
    )

    print(
        f"Live trading       : "
        f"{'ENABLED' if LIVE_TRADING_ENABLED else 'DISABLED'}"
    )

    print()

    print(
        f"FINAL DECISION     : "
        f"{decision}"
    )

    print()

    print(
        f"Health report      : "
        f"{OUTPUT_HEALTH}"
    )

    print(
        f"Metrics            : "
        f"{OUTPUT_METRICS}"
    )

    print(
        f"Manifest           : "
        f"{OUTPUT_MANIFEST}"
    )

    print("=" * 80)
    print(
        "STAGE 23 COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()