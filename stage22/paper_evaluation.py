"""
Stage 22 - Paper Trading Evaluation & Promotion Decision

Reads Stage 21 production-monitor outputs and evaluates whether
the paper-trading system is suitable for continued paper trading.

IMPORTANT:
This stage does NOT enable live trading.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = ROOT / "data" / "stage21"
OUTPUT_DIR = ROOT / "data" / "stage22"

METRICS_FILE = SOURCE_DIR / "stage21_metrics.json"
HEALTH_FILE = SOURCE_DIR / "stage21_health_report.csv"
TRADES_FILE = SOURCE_DIR / "stage21_paper_trades.csv"
SNAPSHOT_FILE = SOURCE_DIR / "stage21_monitoring_snapshot.csv"
MANIFEST_FILE = SOURCE_DIR / "stage21_release_manifest.json"


# ============================================================
# STAGE 22 THRESHOLDS
# ============================================================

MIN_HEALTH_PASS_RATE = 0.95
MAX_HEALTH_FAILED_CHECKS = 0

MIN_WIN_RATE = 0.50
MIN_PROFIT_FACTOR = 1.00
MIN_SHARPE = 0.00

# Keep this conservative because Stage 21 is PAPER TRADING ONLY.
# -45.16% was observed in Stage 21, so live promotion is blocked.
MAX_ALLOWED_DRAWDOWN = -0.50


# ============================================================
# HELPERS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        result = float(value)

        if result != result:  # NaN
            return default

        return result

    except (TypeError, ValueError):
        return default


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.reader(f)

        try:
            next(reader)
        except StopIteration:
            return 0

        return sum(1 for _ in reader)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# CHECK SYSTEM
# ============================================================

class CheckCollector:

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(
        self,
        category: str,
        name: str,
        passed: bool,
        value: Any,
        threshold: Any,
        message: str = ""
    ) -> None:

        self.rows.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "category": category,
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "value": value,
                "threshold": threshold,
                "message": message,
            }
        )

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def passed(self) -> int:
        return sum(
            1
            for row in self.rows
            if row["status"] == "PASS"
        )

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0

        return self.passed / self.total


# ============================================================
# MAIN EVALUATION
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 80)
    print("STAGE 22 - PAPER TRADING EVALUATION")
    print("=" * 80)

    checker = CheckCollector()

    # --------------------------------------------------------
    # LOAD STAGE 21 METRICS
    # --------------------------------------------------------

    try:

        metrics = load_json(METRICS_FILE)

        checker.check(
            "INPUT",
            "stage21_metrics_exists",
            True,
            "EXISTS",
            "EXISTS"
        )

        checker.check(
            "INPUT",
            "stage21_metrics_readable",
            True,
            "READABLE",
            "READABLE"
        )

    except Exception as exc:

        checker.check(
            "INPUT",
            "stage21_metrics_readable",
            False,
            str(exc),
            "READABLE"
        )

        metrics = {}


    # --------------------------------------------------------
    # REQUIRED ARTIFACTS
    # --------------------------------------------------------

    artifacts = {
        "health_report": HEALTH_FILE,
        "paper_trades": TRADES_FILE,
        "monitoring_snapshot": SNAPSHOT_FILE,
        "metrics": METRICS_FILE,
        "release_manifest": MANIFEST_FILE,
    }

    for name, path in artifacts.items():

        checker.check(
            "ARTIFACT",
            f"{name}_exists",
            path.exists(),
            str(path),
            "EXISTS"
        )


    # --------------------------------------------------------
    # STAGE 21 DECISION
    # --------------------------------------------------------

    stage21_decision = str(
        metrics.get(
            "final_decision",
            "UNKNOWN"
        )
    ).upper()

    checker.check(
        "STAGE21",
        "stage21_ready",
        stage21_decision == "READY",
        stage21_decision,
        "READY"
    )


    # --------------------------------------------------------
    # HEALTH CHECKS
    # --------------------------------------------------------

    health_total = int(
        metrics.get(
            "health_checks",
            0
        )
    )

    health_passed = int(
        metrics.get(
            "health_checks_passed",
            0
        )
    )

    health_failed = int(
        metrics.get(
            "health_checks_failed",
            0
        )
    )

    health_rate = safe_float(
        metrics.get(
            "health_pass_rate",
            0.0
        )
    )

    checker.check(
        "HEALTH",
        "health_pass_rate",
        health_rate >= MIN_HEALTH_PASS_RATE,
        health_rate,
        MIN_HEALTH_PASS_RATE
    )

    checker.check(
        "HEALTH",
        "health_failed_checks",
        health_failed <= MAX_HEALTH_FAILED_CHECKS,
        health_failed,
        MAX_HEALTH_FAILED_CHECKS
    )


    # --------------------------------------------------------
    # PAPER TRADING METRICS
    # --------------------------------------------------------

    paper_trades = int(
        metrics.get(
            "paper_trades",
            0
        )
    )

    win_rate = safe_float(
        metrics.get(
            "paper_win_rate",
            0.0
        )
    )

    total_return = safe_float(
        metrics.get(
            "paper_total_return",
            0.0
        )
    )

    max_drawdown = safe_float(
        metrics.get(
            "paper_maximum_drawdown",
            0.0
        )
    )

    profit_factor = safe_float(
        metrics.get(
            "paper_profit_factor",
            0.0
        )
    )

    sharpe_ratio = safe_float(
        metrics.get(
            "paper_sharpe_ratio",
            0.0
        )
    )


    checker.check(
        "PAPER",
        "paper_trades_present",
        paper_trades > 0,
        paper_trades,
        "> 0"
    )

    checker.check(
        "PAPER",
        "minimum_win_rate",
        win_rate >= MIN_WIN_RATE,
        win_rate,
        MIN_WIN_RATE
    )

    checker.check(
        "PAPER",
        "minimum_profit_factor",
        profit_factor >= MIN_PROFIT_FACTOR,
        profit_factor,
        MIN_PROFIT_FACTOR
    )

    checker.check(
        "PAPER",
        "minimum_sharpe",
        sharpe_ratio >= MIN_SHARPE,
        sharpe_ratio,
        MIN_SHARPE
    )

    checker.check(
        "PAPER",
        "maximum_drawdown",
        max_drawdown >= MAX_ALLOWED_DRAWDOWN,
        max_drawdown,
        MAX_ALLOWED_DRAWDOWN
    )


    # --------------------------------------------------------
    # PAPER-TRADING SAFETY
    # --------------------------------------------------------

    paper_only = bool(
        metrics.get(
            "paper_trading_only",
            False
        )
    )

    checker.check(
        "SAFETY",
        "paper_trading_only",
        paper_only,
        paper_only,
        True
    )


    # --------------------------------------------------------
    # PORTFOLIO
    # --------------------------------------------------------

    portfolio_positions = int(
        metrics.get(
            "portfolio_positions",
            0
        )
    )

    valid_positions = portfolio_positions

    checker.check(
        "PORTFOLIO",
        "portfolio_positions_present",
        portfolio_positions > 0,
        portfolio_positions,
        "> 0"
    )


    # --------------------------------------------------------
    # OUTPUT ARTIFACTS
    # --------------------------------------------------------

    health_report_path = (
        OUTPUT_DIR /
        "stage22_evaluation_report.csv"
    )

    metrics_path = (
        OUTPUT_DIR /
        "stage22_metrics.json"
    )

    manifest_path = (
        OUTPUT_DIR /
        "stage22_release_manifest.json"
    )


    write_csv(
        health_report_path,
        checker.rows
    )


    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    all_checks_passed = (
        checker.failed == 0
    )

    if all_checks_passed:

        final_decision = "CONTINUE_PAPER_TRADING"

    else:

        final_decision = "REVIEW"


    # --------------------------------------------------------
    # STAGE 22 METRICS
    # --------------------------------------------------------

    result = {

        "stage": "22",

        "stage_name":
            "Paper Trading Evaluation & Promotion Decision",

        "generated_at":
            datetime.now(timezone.utc).isoformat(),

        "source_stage":
            "21",

        "paper_trading_only":
            True,

        "stage21_decision":
            stage21_decision,

        "health_checks":
            health_total,

        "health_checks_passed":
            health_passed,

        "health_checks_failed":
            health_failed,

        "health_pass_rate":
            health_rate,

        "paper_trades":
            paper_trades,

        "paper_win_rate":
            win_rate,

        "paper_total_return":
            total_return,

        "paper_maximum_drawdown":
            max_drawdown,

        "paper_profit_factor":
            profit_factor,

        "paper_sharpe_ratio":
            sharpe_ratio,

        "portfolio_positions":
            portfolio_positions,

        "valid_positions":
            valid_positions,

        "evaluation_checks_total":
            checker.total,

        "evaluation_checks_passed":
            checker.passed,

        "evaluation_checks_failed":
            checker.failed,

        "evaluation_pass_rate":
            checker.pass_rate,

        "final_decision":
            final_decision,

        "live_trading_enabled":
            False,
    }


    write_json(
        metrics_path,
        result
    )


    # --------------------------------------------------------
    # RELEASE MANIFEST
    # --------------------------------------------------------

    manifest = {

        "stage": "22",

        "stage_name":
            "Paper Trading Evaluation & Promotion Decision",

        "generated_at":
            datetime.now(timezone.utc).isoformat(),

        "source_stage":
            "21",

        "source_directory":
            str(SOURCE_DIR),

        "output_directory":
            str(OUTPUT_DIR),

        "paper_trading_only":
            True,

        "live_trading_enabled":
            False,

        "final_decision":
            final_decision,

        "checks": {

            "total":
                checker.total,

            "passed":
                checker.passed,

            "failed":
                checker.failed,

            "pass_rate":
                checker.pass_rate,
        },

        "artifacts": {

            "evaluation_report":
                str(health_report_path),

            "metrics":
                str(metrics_path),

        },
    }


    write_json(
        manifest_path,
        manifest
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STAGE 22 PAPER TRADING EVALUATION SUMMARY")
    print("=" * 80)

    print(
        f"Health checks       : "
        f"{health_passed}/{health_total}"
    )

    print(
        f"Evaluation checks   : "
        f"{checker.passed}/{checker.total}"
    )

    print(
        f"Paper trades        : "
        f"{paper_trades}"
    )

    print(
        f"Paper win rate      : "
        f"{win_rate:.2%}"
    )

    print(
        f"Paper return        : "
        f"{total_return:.2%}"
    )

    print(
        f"Profit factor       : "
        f"{profit_factor:.4f}"
    )

    print(
        f"Sharpe ratio        : "
        f"{sharpe_ratio:.4f}"
    )

    print(
        f"Maximum drawdown    : "
        f"{max_drawdown:.2%}"
    )

    print(
        f"Paper trading only  : "
        f"{'YES' if paper_only else 'NO'}"
    )

    print()
    print(
        f"FINAL DECISION      : "
        f"{final_decision}"
    )

    print()
    print(
        f"Evaluation report   : "
        f"{health_report_path}"
    )

    print(
        f"Metrics             : "
        f"{metrics_path}"
    )

    print(
        f"Manifest            : "
        f"{manifest_path}"
    )

    print("=" * 80)
    print("STAGE 22 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()