"""
STAGE 21 - PRODUCTION READINESS & PAPER-TRADING MONITOR

Purpose
-------
Stage 20 establishes that the AI stock screening system has passed its
release gate.

Stage 21 performs a controlled post-release readiness check and paper
trading simulation using the artifacts produced by Stage 20.

IMPORTANT:
    This module NEVER places real trades.
    It is a monitoring / validation / paper-trading stage only.

Expected project structure
--------------------------
AI_Stock_Screener/
    data/
        stage20/
            stage20_release_report.csv
            stage20_release_metrics.json
            stage20_release_manifest.json
            stage20_portfolio.csv
            stage20_portfolio_metrics.json
            stage20_portfolio_validation.csv
            stage20_portfolio_validation_metrics.json
            stage20_optimized_trades.csv
            stage20_optimization_metrics.json
            stage20_risk_validation.csv
            stage20_risk_metrics.json
            stage20_walk_forward_results.csv
            stage20_walk_forward_metrics.json
            stage20_predictions.csv
            stage20_validation_metrics.json
            stage20_training_data.csv
            ...

Outputs
-------
data/stage21/
    stage21_health_report.csv
    stage21_paper_trades.csv
    stage21_monitoring_snapshot.csv
    stage21_metrics.json
    stage21_release_manifest.json
"""

from __future__ import annotations

import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGE20_DIR = PROJECT_ROOT / "data" / "stage20"
STAGE21_DIR = PROJECT_ROOT / "data" / "stage21"

STAGE21_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100000.0

MAX_ALLOWED_MISSING_RATIO = 0.10
MAX_ALLOWED_DUPLICATE_RATIO = 0.05

MIN_PORTFOLIO_POSITIONS = 1
MAX_PORTFOLIO_POSITIONS = 20

MIN_EXPECTED_PROFIT_FACTOR = 1.0
MIN_ML_ACCURACY = 0.50

# Paper trading does not use real money.
PAPER_TRADING_ONLY = True


# ============================================================
# DISPLAY HELPERS
# ============================================================

def line(char: str = "=", width: int = 78) -> None:
    print(char * width)


def title(text: str) -> None:
    line()
    print(text)
    line()


def section(text: str) -> None:
    print()
    line()
    print(text)
    line()


# ============================================================
# GENERAL HELPERS
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip().replace("%", "")

        result = float(value)

        if math.isnan(result) or math.isinf(result):
            return default

        return result

    except Exception:
        return default


def percentage(value: float) -> float:
    return safe_float(value) * 100.0


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:

    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    # More tolerant matching.
    for column in df.columns:
        normalized_column = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        for candidate in candidates:
            normalized_candidate = (
                candidate.strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            if normalized_column == normalized_candidate:
                return column

    return None


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        return data

    return {"value": data}


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, default=str)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ============================================================
# STAGE 20 ARTIFACT DISCOVERY
# ============================================================

def artifact_paths() -> Dict[str, Path]:
    return {
        "release_report": STAGE20_DIR / "stage20_release_report.csv",
        "release_metrics": STAGE20_DIR / "stage20_release_metrics.json",
        "release_manifest": STAGE20_DIR / "stage20_release_manifest.json",

        "portfolio": STAGE20_DIR / "stage20_portfolio.csv",
        "portfolio_metrics": STAGE20_DIR / "stage20_portfolio_metrics.json",

        "portfolio_validation": (
            STAGE20_DIR / "stage20_portfolio_validation.csv"
        ),

        "portfolio_validation_metrics": (
            STAGE20_DIR / "stage20_portfolio_validation_metrics.json"
        ),

        "optimized_trades": (
            STAGE20_DIR / "stage20_optimized_trades.csv"
        ),

        "optimization_metrics": (
            STAGE20_DIR / "stage20_optimization_metrics.json"
        ),

        "risk_validation": (
            STAGE20_DIR / "stage20_risk_validation.csv"
        ),

        "risk_metrics": (
            STAGE20_DIR / "stage20_risk_metrics.json"
        ),

        "walk_forward_results": (
            STAGE20_DIR / "stage20_walk_forward_results.csv"
        ),

        "walk_forward_metrics": (
            STAGE20_DIR / "stage20_walk_forward_metrics.json"
        ),

        "predictions": STAGE20_DIR / "stage20_predictions.csv",

        "validation_metrics": (
            STAGE20_DIR / "stage20_validation_metrics.json"
        ),

        "training_data": (
            STAGE20_DIR / "stage20_training_data.csv"
        ),
    }


# ============================================================
# CHECK FRAMEWORK
# ============================================================

class HealthChecker:

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def check(
        self,
        category: str,
        name: str,
        passed: bool,
        value: Any = None,
        threshold: Any = None,
        message: str = "",
    ) -> None:

        self.rows.append(
            {
                "timestamp": utc_now(),
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
            1 for row in self.rows
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

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# ============================================================
# RELEASE MANIFEST VALIDATION
# ============================================================

def validate_release_manifest(
    checker: HealthChecker,
    paths: Dict[str, Path],
) -> Dict[str, Any]:

    section("LOADING STAGE 20 RELEASE")

    manifest_path = paths["release_manifest"]

    if not manifest_path.exists():

        checker.check(
            "RELEASE",
            "release_manifest_exists",
            False,
            value="MISSING",
            threshold="EXISTS",
            message="Stage 20 release manifest is missing.",
        )

        return {}

    checker.check(
        "RELEASE",
        "release_manifest_exists",
        True,
        value="EXISTS",
        threshold="EXISTS",
    )

    try:
        manifest = load_json(manifest_path)
    except Exception as exc:

        checker.check(
            "RELEASE",
            "release_manifest_readable",
            False,
            value=str(exc),
            threshold="READABLE",
        )

        return {}

    checker.check(
        "RELEASE",
        "release_manifest_readable",
        True,
        value="READABLE",
        threshold="READABLE",
    )

    # Search recursively for a READY value.
    text = json.dumps(manifest).upper()

    release_ready = "READY" in text

    checker.check(
        "RELEASE",
        "stage20_release_ready",
        release_ready,
        value="READY" if release_ready else "NOT_READY",
        threshold="READY",
        message=(
            "Stage 20 release manifest contains READY."
            if release_ready
            else
            "Stage 20 release manifest does not contain READY."
        ),
    )

    return manifest


# ============================================================
# ARTIFACT VALIDATION
# ============================================================

def validate_artifacts(
    checker: HealthChecker,
    paths: Dict[str, Path],
) -> Dict[str, bool]:

    section("VALIDATING STAGE 20 ARTIFACTS")

    results: Dict[str, bool] = {}

    for name, path in paths.items():

        exists = path.exists() and path.is_file()

        results[name] = exists

        checker.check(
            "ARTIFACT",
            name,
            exists,
            value="EXISTS" if exists else "MISSING",
            threshold="EXISTS",
            message=str(path),
        )

        status = "PASS" if exists else "FAIL"

        print(
            f"{name:<35} {status:<6} "
            f"{'EXISTS' if exists else 'MISSING'}"
        )

    return results


# ============================================================
# PORTFOLIO VALIDATION
# ============================================================

def load_portfolio(
    paths: Dict[str, Path],
    checker: HealthChecker,
) -> pd.DataFrame:

    path = paths["portfolio"]

    if not path.exists():

        checker.check(
            "PORTFOLIO",
            "portfolio_file",
            False,
            value="MISSING",
            threshold="EXISTS",
        )

        return pd.DataFrame()

    try:
        df = load_csv(path)

        checker.check(
            "PORTFOLIO",
            "portfolio_file_readable",
            True,
            value=len(df),
            threshold=">0",
        )

        return df

    except Exception as exc:

        checker.check(
            "PORTFOLIO",
            "portfolio_file_readable",
            False,
            value=str(exc),
            threshold="READABLE",
        )

        return pd.DataFrame()


def validate_portfolio(
    portfolio: pd.DataFrame,
    checker: HealthChecker,
) -> Dict[str, Any]:

    if portfolio.empty:

        checker.check(
            "PORTFOLIO",
            "portfolio_has_rows",
            False,
            value=0,
            threshold=">0",
        )

        return {
            "positions": 0,
            "buy_positions": 0,
            "sell_positions": 0,
            "exposure": 0.0,
            "largest_position": 0.0,
        }

    checker.check(
        "PORTFOLIO",
        "portfolio_has_rows",
        True,
        value=len(portfolio),
        threshold=">0",
    )

    symbol_col = find_column(
        portfolio,
        [
            "symbol",
            "ticker",
            "stock",
            "security",
            "name",
        ],
    )

    side_col = find_column(
        portfolio,
        [
            "side",
            "signal",
            "action",
            "position",
            "direction",
        ],
    )

    weight_col = find_column(
        portfolio,
        [
            "weight",
            "allocation",
            "allocation_pct",
            "portfolio_weight",
            "position_weight",
            "percentage",
        ],
    )

    positions = len(portfolio)

    checker.check(
        "PORTFOLIO",
        "position_count_valid",
        (
            MIN_PORTFOLIO_POSITIONS
            <= positions
            <= MAX_PORTFOLIO_POSITIONS
        ),
        value=positions,
        threshold=f"{MIN_PORTFOLIO_POSITIONS}-{MAX_PORTFOLIO_POSITIONS}",
    )

    buy_positions = 0
    sell_positions = 0

    if side_col is not None:

        sides = (
            portfolio[side_col]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        buy_positions = int(
            sides.isin(
                ["BUY", "LONG", "B", "1"]
            ).sum()
        )

        sell_positions = int(
            sides.isin(
                ["SELL", "SHORT", "S", "-1"]
            ).sum()
        )

    if weight_col is not None:

        weights = pd.to_numeric(
            portfolio[weight_col],
            errors="coerce",
        ).fillna(0.0)

        weights_abs = weights.abs()

        weight_sum = float(weights_abs.sum())

        # Detect percentage-formatted values.
        if weight_sum > 2.0:
            exposure = weight_sum
        else:
            exposure = weight_sum * 100.0

        if exposure <= 0:
            exposure = 0.0

        largest_position = float(weights_abs.max())

        if largest_position <= 1.0:
            largest_position *= 100.0

        checker.check(
            "PORTFOLIO",
            "portfolio_exposure_reasonable",
            exposure <= 100.0001,
            value=round(exposure, 4),
            threshold="<=100%",
        )

    else:

        exposure = 0.0
        largest_position = 0.0

        checker.check(
            "PORTFOLIO",
            "portfolio_weight_column",
            False,
            value="MISSING",
            threshold="AVAILABLE",
            message="No portfolio weight column detected.",
        )

    if symbol_col is not None:

        duplicate_ratio = (
            portfolio[symbol_col].duplicated().mean()
        )

        checker.check(
            "PORTFOLIO",
            "portfolio_duplicates",
            duplicate_ratio <= MAX_ALLOWED_DUPLICATE_RATIO,
            value=round(float(duplicate_ratio), 6),
            threshold=f"<={MAX_ALLOWED_DUPLICATE_RATIO}",
        )

    return {
        "positions": positions,
        "buy_positions": buy_positions,
        "sell_positions": sell_positions,
        "exposure": exposure,
        "largest_position": largest_position,
    }


# ============================================================
# GENERIC DATA QUALITY CHECKS
# ============================================================

def validate_dataframe_quality(
    checker: HealthChecker,
    name: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:

    if df.empty:

        checker.check(
            "DATA",
            f"{name}_has_rows",
            False,
            value=0,
            threshold=">0",
        )

        return {
            "rows": 0,
            "columns": 0,
            "missing_ratio": 1.0,
            "duplicate_ratio": 1.0,
        }

    rows, columns = df.shape

    missing_ratio = float(
        df.isna().mean().mean()
    )

    duplicate_ratio = float(
        df.duplicated().mean()
    )

    checker.check(
        "DATA",
        f"{name}_has_rows",
        rows > 0,
        value=rows,
        threshold=">0",
    )

    checker.check(
        "DATA",
        f"{name}_has_columns",
        columns > 0,
        value=columns,
        threshold=">0",
    )

    checker.check(
        "DATA",
        f"{name}_missing_values",
        missing_ratio <= MAX_ALLOWED_MISSING_RATIO,
        value=round(missing_ratio, 6),
        threshold=f"<={MAX_ALLOWED_MISSING_RATIO}",
    )

    checker.check(
        "DATA",
        f"{name}_duplicate_rows",
        duplicate_ratio <= MAX_ALLOWED_DUPLICATE_RATIO,
        value=round(duplicate_ratio, 6),
        threshold=f"<={MAX_ALLOWED_DUPLICATE_RATIO}",
    )

    return {
        "rows": rows,
        "columns": columns,
        "missing_ratio": missing_ratio,
        "duplicate_ratio": duplicate_ratio,
    }


# ============================================================
# OPTIMIZED TRADES
# ============================================================

def load_optimized_trades(
    paths: Dict[str, Path],
    checker: HealthChecker,
) -> pd.DataFrame:

    path = paths["optimized_trades"]

    if not path.exists():

        checker.check(
            "TRADES",
            "optimized_trades_exists",
            False,
            value="MISSING",
            threshold="EXISTS",
        )

        return pd.DataFrame()

    try:

        df = load_csv(path)

        checker.check(
            "TRADES",
            "optimized_trades_readable",
            True,
            value=len(df),
            threshold=">0",
        )

        return df

    except Exception as exc:

        checker.check(
            "TRADES",
            "optimized_trades_readable",
            False,
            value=str(exc),
            threshold="READABLE",
        )

        return pd.DataFrame()


def detect_return_column(
    df: pd.DataFrame,
) -> Optional[str]:

    return find_column(
        df,
        [
            "return",
            "returns",
            "trade_return",
            "strategy_return",
            "pnl",
            "profit",
            "profit_loss",
            "pnl_pct",
            "return_pct",
        ],
    )


def detect_side_column(
    df: pd.DataFrame,
) -> Optional[str]:

    return find_column(
        df,
        [
            "side",
            "signal",
            "action",
            "decision",
            "trade",
            "position",
        ],
    )


# ============================================================
# PAPER TRADING ENGINE
# ============================================================

def normalize_return_series(
    values: pd.Series,
) -> pd.Series:

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(0.0)

    # If values appear to be percentages, convert to decimals.
    if numeric.abs().median() > 1.0:
        numeric = numeric / 100.0

    return numeric


def simulate_paper_trading(
    trades: pd.DataFrame,
    checker: HealthChecker,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    section("RUNNING PAPER-TRADING SIMULATION")

    if trades.empty:

        checker.check(
            "PAPER",
            "paper_trade_rows",
            False,
            value=0,
            threshold=">0",
        )

        return pd.DataFrame(), {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "maximum_drawdown": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
        }

    return_col = detect_return_column(trades)

    if return_col is None:

        checker.check(
            "PAPER",
            "trade_return_column",
            False,
            value="MISSING",
            threshold="AVAILABLE",
        )

        return pd.DataFrame(), {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "maximum_drawdown": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
        }

    returns = normalize_return_series(
        trades[return_col]
    )

    paper = trades.copy()

    paper["paper_return"] = returns

    paper["paper_capital_before"] = (
        INITIAL_CAPITAL
        * (1.0 + returns).cumprod().shift(1).fillna(1.0)
    )

    paper["paper_pnl"] = (
        paper["paper_capital_before"]
        * paper["paper_return"]
    )

    paper["paper_capital"] = (
        INITIAL_CAPITAL
        * (1.0 + paper["paper_return"]).cumprod()
    )

    running_max = (
        paper["paper_capital"]
        .cummax()
    )

    paper["paper_drawdown"] = (
        paper["paper_capital"]
        / running_max
        - 1.0
    )

    wins = int((returns > 0).sum())
    losses = int((returns < 0).sum())

    total_trades = len(returns)

    win_rate = (
        wins / total_trades
        if total_trades
        else 0.0
    )

    total_return = (
        paper["paper_capital"].iloc[-1]
        / INITIAL_CAPITAL
        - 1.0
    )

    maximum_drawdown = float(
        paper["paper_drawdown"].min()
    )

    positive = float(
        returns[returns > 0].sum()
    )

    negative = float(
        abs(returns[returns < 0].sum())
    )

    if negative > 0:
        profit_factor = positive / negative
    elif positive > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    mean_return = float(returns.mean())
    std_return = float(returns.std(ddof=1))

    if std_return > 0:
        sharpe_ratio = (
            mean_return
            / std_return
            * math.sqrt(max(total_trades, 1))
        )
    else:
        sharpe_ratio = 0.0

    checker.check(
        "PAPER",
        "paper_trade_rows",
        total_trades > 0,
        value=total_trades,
        threshold=">0",
    )

    checker.check(
        "PAPER",
        "paper_capital_finite",
        bool(
            np.isfinite(
                paper["paper_capital"]
            ).all()
        ),
        value="FINITE",
        threshold="FINITE",
    )

    checker.check(
        "PAPER",
        "paper_drawdown_finite",
        bool(
            np.isfinite(
                paper["paper_drawdown"]
            ).all()
        ),
        value="FINITE",
        threshold="FINITE",
    )

    metrics = {
        "trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_return": float(total_return),
        "maximum_drawdown": maximum_drawdown,
        "profit_factor": (
            None
            if math.isinf(profit_factor)
            else float(profit_factor)
        ),
        "sharpe_ratio": float(sharpe_ratio),
        "ending_capital": float(
            paper["paper_capital"].iloc[-1]
        ),
    }

    print(f"Paper trades       : {total_trades}")
    print(f"Wins               : {wins}")
    print(f"Losses             : {losses}")
    print(f"Win rate           : {win_rate * 100:.2f}%")
    print(f"Total return       : {total_return * 100:.2f}%")
    print(
        f"Maximum drawdown   : "
        f"{maximum_drawdown * 100:.2f}%"
    )
    print(
        f"Profit factor      : "
        f"{profit_factor:.4f}"
        if math.isfinite(profit_factor)
        else "Profit factor      : INF"
    )
    print(f"Sharpe ratio       : {sharpe_ratio:.4f}")
    print(
        f"Ending capital     : "
        f"{metrics['ending_capital']:.2f}"
    )

    return paper, metrics


# ============================================================
# ML METRICS
# ============================================================

def load_ml_metrics(
    paths: Dict[str, Path],
    checker: HealthChecker,
) -> Dict[str, Any]:

    path = Path("models/training_metrics.json")

    if not path.exists():

        checker.check(
            "MODEL",
            "training_metrics_exists",
            False,
            value="MISSING",
            threshold="EXISTS",
        )

        return {}

    try:

        metrics = load_json(path)

        checker.check(
            "MODEL",
            "validation_metrics_readable",
            True,
            value="READABLE",
            threshold="READABLE",
        )

        accuracy = safe_float(
            metrics.get("accuracy"),
            default=0.0,
        )

        checker.check(
            "MODEL",
            "ml_accuracy",
            accuracy >= MIN_ML_ACCURACY,
            value=accuracy,
            threshold=MIN_ML_ACCURACY,
        )

        return metrics
 
    except Exception as exc:

        checker.check(
            "MODEL",
            "validation_metrics_readable",
            False,
            value=str(exc),
            threshold="READABLE",
        )

        return {}


# ============================================================
# RISK METRICS
# ============================================================

def load_risk_metrics(
    paths: Dict[str, Path],
    checker: HealthChecker,
) -> Dict[str, Any]:

    path = paths["risk_metrics"]

    if not path.exists():

        checker.check(
            "RISK",
            "risk_metrics_exists",
            False,
            value="MISSING",
            threshold="EXISTS",
        )

        return {}

    try:

        metrics = load_json(path)

        checker.check(
            "RISK",
            "risk_metrics_readable",
            True,
            value="READABLE",
            threshold="READABLE",
        )

        return metrics

    except Exception as exc:

        checker.check(
            "RISK",
            "risk_metrics_readable",
            False,
            value=str(exc),
            threshold="READABLE",
        )

        return {}


# ============================================================
# MONITORING SNAPSHOT
# ============================================================

def build_monitoring_snapshot(
    artifact_results: Dict[str, bool],
    portfolio_metrics: Dict[str, Any],
    paper_metrics: Dict[str, Any],
    ml_metrics: Dict[str, Any],
    checker: HealthChecker,
) -> Dict[str, Any]:

    artifacts_present = sum(
        1 for value in artifact_results.values()
        if value
    )

    artifact_total = len(artifact_results)

    accuracy = safe_float(
        ml_metrics.get("accuracy"),
        default=0.0,
    )

    profit_factor = safe_float(
        paper_metrics.get("profit_factor"),
        default=0.0,
    )

    if profit_factor == 0.0:

        # Try Stage 20 risk metrics.
        profit_factor = safe_float(
            ml_metrics.get("profit_factor"),
            default=0.0,
        )

    portfolio_positions = int(
        portfolio_metrics.get("positions", 0)
    )

    all_artifacts_present = (
        artifacts_present == artifact_total
    )

    portfolio_valid = (
        MIN_PORTFOLIO_POSITIONS
        <= portfolio_positions
        <= MAX_PORTFOLIO_POSITIONS
    )

    model_valid = accuracy >= MIN_ML_ACCURACY

    checker.check(
        "MONITOR",
        "all_stage20_artifacts_present",
        all_artifacts_present,
        value=f"{artifacts_present}/{artifact_total}",
        threshold=f"{artifact_total}/{artifact_total}",
    )

    checker.check(
        "MONITOR",
        "portfolio_positions_valid",
        portfolio_valid,
        value=portfolio_positions,
        threshold=(
            f"{MIN_PORTFOLIO_POSITIONS}-"
            f"{MAX_PORTFOLIO_POSITIONS}"
        ),
    )

    checker.check(
        "MONITOR",
        "model_accuracy_valid",
        model_valid,
        value=accuracy,
        threshold=MIN_ML_ACCURACY,
    )

    return {
        "timestamp": utc_now(),
        "stage": "21",
        "paper_trading_only": PAPER_TRADING_ONLY,
        "stage20_artifacts_present": artifacts_present,
        "stage20_artifacts_total": artifact_total,
        "portfolio_positions": portfolio_positions,
        "buy_positions": portfolio_metrics.get(
            "buy_positions", 0
        ),
        "sell_positions": portfolio_metrics.get(
            "sell_positions", 0
        ),
        "portfolio_exposure": portfolio_metrics.get(
            "exposure", 0.0
        ),
        "largest_position": portfolio_metrics.get(
            "largest_position", 0.0
        ),
        "ml_accuracy": accuracy,
        "paper_trades": paper_metrics.get(
            "trades", 0
        ),
        "paper_win_rate": paper_metrics.get(
            "win_rate", 0.0
        ),
        "paper_total_return": paper_metrics.get(
            "total_return", 0.0
        ),
        "paper_maximum_drawdown": paper_metrics.get(
            "maximum_drawdown", 0.0
        ),
        "paper_profit_factor": profit_factor,
        "paper_sharpe_ratio": paper_metrics.get(
            "sharpe_ratio", 0.0
        ),
        "paper_ending_capital": paper_metrics.get(
            "ending_capital",
            INITIAL_CAPITAL,
        ),
    }


# ============================================================
# DECISION ENGINE
# ============================================================

def final_decision(
    checker: HealthChecker,
    snapshot: Dict[str, Any],
) -> str:

    # Hard blockers.
    if checker.failed > 0:
        return "REVIEW"

    if not snapshot.get("stage20_artifacts_present"):
        return "BLOCKED"

    if not snapshot.get("paper_trading_only"):
        return "BLOCKED"

    if snapshot.get("portfolio_positions", 0) <= 0:
        return "BLOCKED"

    return "READY"


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    checker: HealthChecker,
    paper_trades: pd.DataFrame,
    snapshot: Dict[str, Any],
    decision: str,
) -> None:

    section("SAVING STAGE 21 RESULTS")

    health_path = (
        STAGE21_DIR
        / "stage21_health_report.csv"
    )

    paper_path = (
        STAGE21_DIR
        / "stage21_paper_trades.csv"
    )

    snapshot_path = (
        STAGE21_DIR
        / "stage21_monitoring_snapshot.csv"
    )

    metrics_path = (
        STAGE21_DIR
        / "stage21_metrics.json"
    )

    manifest_path = (
        STAGE21_DIR
        / "stage21_release_manifest.json"
    )

    # Health report.
    checker.dataframe().to_csv(
        health_path,
        index=False,
    )

    # Paper trades.
    if paper_trades.empty:

        pd.DataFrame(
            columns=[
                "paper_return",
                "paper_pnl",
                "paper_capital",
                "paper_drawdown",
            ]
        ).to_csv(
            paper_path,
            index=False,
        )

    else:

        paper_trades.to_csv(
            paper_path,
            index=False,
        )

    # Snapshot.
    pd.DataFrame(
        [snapshot]
    ).to_csv(
        snapshot_path,
        index=False,
    )

    metrics = {
        **snapshot,
        "health_checks": checker.total,
        "health_checks_passed": checker.passed,
        "health_checks_failed": checker.failed,
        "health_pass_rate": checker.pass_rate,
        "final_decision": decision,
        "generated_at": utc_now(),
    }

    save_json(
        metrics_path,
        metrics,
    )

    manifest = {
        "stage": "21",
        "stage_name": (
            "Production Readiness "
            "& Paper Trading Monitor"
        ),
        "generated_at": utc_now(),
        "paper_trading_only": True,
        "source_stage": "20",
        "source_directory": str(STAGE20_DIR),
        "output_directory": str(STAGE21_DIR),
        "final_decision": decision,
        "checks": {
            "total": checker.total,
            "passed": checker.passed,
            "failed": checker.failed,
            "pass_rate": checker.pass_rate,
        },
        "artifacts": {
            "health_report": str(health_path),
            "paper_trades": str(paper_path),
            "monitoring_snapshot": str(snapshot_path),
            "metrics": str(metrics_path),
        },
    }

    save_json(
        manifest_path,
        manifest,
    )

    print()
    print("Health report saved:")
    print(f"  {health_path}")

    print()
    print("Paper trades saved:")
    print(f"  {paper_path}")

    print()
    print("Monitoring snapshot saved:")
    print(f"  {snapshot_path}")

    print()
    print("Metrics saved:")
    print(f"  {metrics_path}")

    print()
    print("Release manifest saved:")
    print(f"  {manifest_path}")


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    checker: HealthChecker,
    snapshot: Dict[str, Any],
    decision: str,
) -> None:

    section("STAGE 21 PRODUCTION MONITOR SUMMARY")

    print(
        f"Health checks       : "
        f"{checker.total}"
    )

    print(
        f"Checks passed       : "
        f"{checker.passed}"
    )

    print(
        f"Checks failed       : "
        f"{checker.failed}"
    )

    print(
        f"Pass rate           : "
        f"{checker.pass_rate * 100:.2f}%"
    )

    print()

    print(
        f"Stage 20 artifacts  : "
        f"{snapshot['stage20_artifacts_present']}/"
        f"{snapshot['stage20_artifacts_total']}"
    )

    print(
        f"Portfolio positions : "
        f"{snapshot['portfolio_positions']}"
    )

    print(
        f"BUY positions       : "
        f"{snapshot['buy_positions']}"
    )

    print(
        f"SELL positions      : "
        f"{snapshot['sell_positions']}"
    )

    print(
        f"Portfolio exposure  : "
        f"{snapshot['portfolio_exposure']:.2f}%"
    )

    print(
        f"Largest position    : "
        f"{snapshot['largest_position']:.2f}%"
    )

    print()

    print(
        f"ML accuracy         : "
        f"{snapshot['ml_accuracy'] * 100:.2f}%"
    )

    print(
        f"Paper trades        : "
        f"{snapshot['paper_trades']}"
    )

    print(
        f"Paper win rate      : "
        f"{snapshot['paper_win_rate'] * 100:.2f}%"
    )

    print(
        f"Paper total return  : "
        f"{snapshot['paper_total_return'] * 100:.2f}%"
    )

    print(
        f"Paper max drawdown  : "
        f"{snapshot['paper_maximum_drawdown'] * 100:.2f}%"
    )

    print(
        f"Paper profit factor : "
        f"{snapshot['paper_profit_factor']:.4f}"
    )

    print(
        f"Paper Sharpe        : "
        f"{snapshot['paper_sharpe_ratio']:.4f}"
    )

    print()

    print(
        f"PAPER TRADING ONLY  : "
        f"{'YES' if PAPER_TRADING_ONLY else 'NO'}"
    )

    print()

    print(
        f"FINAL DECISION      : "
        f"{decision}"
    )

    line()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    title(
        "STAGE 21 - PRODUCTION READINESS "
        "& PAPER-TRADING MONITOR"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Stage 20    : {STAGE20_DIR}"
    )

    print(
        f"Stage 21    : {STAGE21_DIR}"
    )

    print()
    print(
        "IMPORTANT: PAPER TRADING ONLY - "
        "NO REAL ORDERS WILL BE PLACED."
    )

    checker = HealthChecker()

    try:

        paths = artifact_paths()

        # ----------------------------------------------------
        # Release validation
        # ----------------------------------------------------

        manifest = validate_release_manifest(
            checker,
            paths,
        )

        # ----------------------------------------------------
        # Artifact validation
        # ----------------------------------------------------

        artifact_results = validate_artifacts(
            checker,
            paths,
        )

        # ----------------------------------------------------
        # Portfolio
        # ----------------------------------------------------

        portfolio = load_portfolio(
            paths,
            checker,
        )

        portfolio_quality = (
            validate_dataframe_quality(
                checker,
                "portfolio",
                portfolio,
            )
            if not portfolio.empty
            else {}
        )

        portfolio_metrics = validate_portfolio(
            portfolio,
            checker,
        )

        # ----------------------------------------------------
        # Optimized trades
        # ----------------------------------------------------

        optimized_trades = load_optimized_trades(
            paths,
            checker,
        )

        if not optimized_trades.empty:

            validate_dataframe_quality(
                checker,
                "optimized_trades",
                optimized_trades,
            )

        # ----------------------------------------------------
        # Paper trading
        # ----------------------------------------------------

        paper_trades, paper_metrics = (
            simulate_paper_trading(
                optimized_trades,
                checker,
            )
        )

        # ----------------------------------------------------
        # ML metrics
        # ----------------------------------------------------

        ml_metrics = load_ml_metrics(
            paths,
            checker,
        )

        # ----------------------------------------------------
        # Risk metrics
        # ----------------------------------------------------

        risk_metrics = load_risk_metrics(
            paths,
            checker,
        )

        # Keep these variables available for future extension.
        _ = manifest
        _ = portfolio_quality
        _ = risk_metrics

        # ----------------------------------------------------
        # Monitoring snapshot
        # ----------------------------------------------------

        snapshot = build_monitoring_snapshot(
            artifact_results,
            portfolio_metrics,
            paper_metrics,
            ml_metrics,
            checker,
        )

        # ----------------------------------------------------
        # Final decision
        # ----------------------------------------------------

        decision = final_decision(
            checker,
            snapshot,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_outputs(
            checker,
            paper_trades,
            snapshot,
            decision,
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print_summary(
            checker,
            snapshot,
            decision,
        )

        section(
            "STAGE 21 PRODUCTION MONITOR COMPLETE"
        )

        print(
            f"Decision : {decision}"
        )

        print(
            f"Results  : {STAGE21_DIR}"
        )

    except Exception as exc:

        section(
            "STAGE 21 PRODUCTION MONITOR FAILED"
        )

        print(
            f"Error type: {type(exc).__name__}"
        )

        print(
            f"Error: {exc}"
        )

        traceback.print_exc()

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()