# ============================================================
# STAGE 20.12 - PORTFOLIO VALIDATOR
# stage20/portfolio_validator.py
# ============================================================

import os
import json
import math
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/stage20"

PORTFOLIO_FILE = os.path.join(
    DATA_DIR,
    "stage20_portfolio.csv"
)

PORTFOLIO_METRICS_FILE = os.path.join(
    DATA_DIR,
    "stage20_portfolio_metrics.json"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "stage20_portfolio_validation.csv"
)

METRICS_FILE = os.path.join(
    DATA_DIR,
    "stage20_portfolio_validation_metrics.json"
)


# ============================================================
# VALIDATION LIMITS
# ============================================================

INITIAL_CAPITAL = 100000.0

MIN_POSITIONS = 1
MAX_POSITIONS = 20

MAX_SINGLE_POSITION = 0.25

MAX_TOTAL_EXPOSURE = 1.00

MAX_RISK_CAPITAL = 0.02

MIN_EXPECTED_RETURN = 0.0

MIN_WIN_RATE = 0.50

MIN_RISK_REWARD = 1.0


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
# LOAD PORTFOLIO
# ============================================================

def load_portfolio():

    print()
    print("=" * 80)
    print("LOADING STAGE 20.11 PORTFOLIO")
    print("=" * 80)

    print()
    print(
        f"File: {PORTFOLIO_FILE}"
    )

    if not os.path.exists(
        PORTFOLIO_FILE
    ):

        raise FileNotFoundError(
            f"Portfolio file not found: "
            f"{PORTFOLIO_FILE}"
        )

    df = pd.read_csv(
        PORTFOLIO_FILE
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    print()
    print("Columns:")

    for column in df.columns:

        print(
            f"  - {column}"
        )

    return df


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    df,
    candidates,
    required=True
):

    lower_map = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:

            return lower_map[
                candidate.lower()
            ]

    if required:

        raise ValueError(
            "Required column not found. "
            f"Expected one of: {candidates}"
        )

    return None


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    df
):

    print()
    print("=" * 80)
    print("PREPARING VALIDATION DATA")
    print("=" * 80)

    data = df.copy()

    # --------------------------------------------------------
    # Symbol
    # --------------------------------------------------------

    symbol_column = find_column(
        data,
        [
            "Symbol",
            "symbol",
            "ticker",
            "Ticker"
        ]
    )

    data["validation_symbol"] = (
        data[symbol_column]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    direction_column = find_column(
        data,
        [
            "direction",
            "Direction",
            "signal",
            "Signal"
        ]
    )

    data["validation_direction"] = (
        data[direction_column]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # Allocation / weight
    # --------------------------------------------------------

    weight_column = find_column(
        data,
        [
            "weight",
            "allocation",
            "allocation_pct",
            "position_weight",
            "portfolio_weight",
            "position_size"
        ]
    )

    data["validation_weight_raw"] = pd.to_numeric(
        data[weight_column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Normalize percentage weights
    #
    # Example:
    # 20.0  -> 0.20
    # 0.20  -> 0.20
    # --------------------------------------------------------

    def normalize_weight(
        value
    ):

        if pd.isna(value):

            return np.nan

        value = float(value)

        if abs(value) > 1.0:

            return value / 100.0

        return value

    data["validation_weight"] = (
        data["validation_weight_raw"]
        .apply(
            normalize_weight
        )
    )

    # --------------------------------------------------------
    # Expected return
    # --------------------------------------------------------

    expected_return_column = find_column(
        data,
        [
            "expected_return",
            "expected_return_pct",
            "return",
            "expected"
        ],
        required=False
    )

    if expected_return_column is not None:

        data["validation_expected_return"] = (
            pd.to_numeric(
                data[
                    expected_return_column
                ],
                errors="coerce"
            )
        )

    else:

        data["validation_expected_return"] = 0.0

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability_column = find_column(
        data,
        [
            "ml_probability",
            "probability",
            "probability_up",
            "confidence",
            "ml_confidence"
        ],
        required=False
    )

    if probability_column is not None:

        data["validation_probability"] = (
            pd.to_numeric(
                data[
                    probability_column
                ],
                errors="coerce"
            )
        )

    else:

        data["validation_probability"] = np.nan

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk_column = find_column(
        data,
        [
            "risk",
            "risk_pct",
            "risk_percent",
            "risk_capital"
        ],
        required=False
    )

    if risk_column is not None:

        data["validation_risk"] = (
            pd.to_numeric(
                data[
                    risk_column
                ],
                errors="coerce"
            )
        )

    else:

        data["validation_risk"] = np.nan

    return data


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_basic_structure(
    data
):

    print()
    print("=" * 80)
    print("VALIDATING PORTFOLIO STRUCTURE")
    print("=" * 80)

    checks = []

    position_count = len(data)

    # --------------------------------------------------------
    # Position count
    # --------------------------------------------------------

    position_ok = (
        MIN_POSITIONS
        <= position_count
        <= MAX_POSITIONS
    )

    checks.append({
        "check": "position_count",
        "value": position_count,
        "limit": f"{MIN_POSITIONS}-{MAX_POSITIONS}",
        "passed": position_ok
    })

    print(
        f"Positions              : "
        f"{position_count}"
    )

    print(
        f"Position count check   : "
        f"{'PASS' if position_ok else 'FAIL'}"
    )

    # --------------------------------------------------------
    # Empty symbols
    # --------------------------------------------------------

    empty_symbols = (
        data["validation_symbol"]
        .eq("")
        .sum()
    )

    symbol_ok = (
        empty_symbols == 0
    )

    checks.append({
        "check": "valid_symbols",
        "value": int(empty_symbols),
        "limit": 0,
        "passed": symbol_ok
    })

    print(
        f"Empty symbols          : "
        f"{empty_symbols}"
    )

    print(
        f"Symbol check           : "
        f"{'PASS' if symbol_ok else 'FAIL'}"
    )

    # --------------------------------------------------------
    # Duplicate symbols
    # --------------------------------------------------------

    duplicate_count = (
        data["validation_symbol"]
        .duplicated()
        .sum()
    )

    duplicates_ok = (
        duplicate_count == 0
    )

    checks.append({
        "check": "duplicate_symbols",
        "value": int(duplicate_count),
        "limit": 0,
        "passed": duplicates_ok
    })

    print(
        f"Duplicate symbols      : "
        f"{duplicate_count}"
    )

    print(
        f"Duplicate check        : "
        f"{'PASS' if duplicates_ok else 'FAIL'}"
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    valid_directions = {
        "BUY",
        "SELL",
        "HOLD"
    }

    invalid_directions = (
        ~data[
            "validation_direction"
        ].isin(
            valid_directions
        )
    ).sum()

    direction_ok = (
        invalid_directions == 0
    )

    checks.append({
        "check": "valid_direction",
        "value": int(invalid_directions),
        "limit": 0,
        "passed": direction_ok
    })

    print(
        f"Invalid directions     : "
        f"{invalid_directions}"
    )

    print(
        f"Direction check        : "
        f"{'PASS' if direction_ok else 'FAIL'}"
    )

    return checks


# ============================================================
# ALLOCATION VALIDATION
# ============================================================

def validate_allocation(
    data
):

    print()
    print("=" * 80)
    print("VALIDATING PORTFOLIO ALLOCATION")
    print("=" * 80)

    checks = []

    weights = (
        data["validation_weight"]
        .dropna()
    )

    # --------------------------------------------------------
    # Missing weights
    # --------------------------------------------------------

    missing_weights = (
        data["validation_weight"]
        .isna()
        .sum()
    )

    weights_present = (
        missing_weights == 0
    )

    checks.append({
        "check": "missing_weights",
        "value": int(missing_weights),
        "limit": 0,
        "passed": weights_present
    })

    print(
        f"Missing weights        : "
        f"{missing_weights}"
    )

    # --------------------------------------------------------
    # Negative weights
    # --------------------------------------------------------

    negative_weights = (
        weights < 0
    ).sum()

    negative_ok = (
        negative_weights == 0
    )

    checks.append({
        "check": "negative_weights",
        "value": int(negative_weights),
        "limit": 0,
        "passed": negative_ok
    })

    print(
        f"Negative weights       : "
        f"{negative_weights}"
    )

    # --------------------------------------------------------
    # Single position limit
    # --------------------------------------------------------

    largest_position = (
        weights.max()
        if len(weights)
        else 0.0
    )

    largest_ok = (
        largest_position
        <= MAX_SINGLE_POSITION
        + 1e-9
    )

    checks.append({
        "check": "largest_position",
        "value": largest_position,
        "limit": MAX_SINGLE_POSITION,
        "passed": largest_ok
    })

    print(
        f"Largest position      : "
        f"{largest_position * 100:.2f}%"
    )

    print(
        f"Position limit check   : "
        f"{'PASS' if largest_ok else 'FAIL'}"
    )

    # --------------------------------------------------------
    # Total exposure
    # --------------------------------------------------------

    total_exposure = (
        weights.abs().sum()
    )

    exposure_ok = (
        total_exposure
        <= MAX_TOTAL_EXPOSURE
        + 1e-9
    )

    checks.append({
        "check": "total_exposure",
        "value": total_exposure,
        "limit": MAX_TOTAL_EXPOSURE,
        "passed": exposure_ok
    })

    print(
        f"Total exposure         : "
        f"{total_exposure * 100:.2f}%"
    )

    print(
        f"Exposure limit check   : "
        f"{'PASS' if exposure_ok else 'FAIL'}"
    )

    # --------------------------------------------------------
    # Fully invested check
    # --------------------------------------------------------

    fully_invested = (
        abs(
            total_exposure - 1.0
        )
        <= 0.02
    )

    checks.append({
        "check": "fully_invested",
        "value": total_exposure,
        "limit": 1.0,
        "passed": fully_invested
    })

    print(
        f"Fully invested         : "
        f"{'YES' if fully_invested else 'NO'}"
    )

    return checks, {
        "total_exposure": float(
            total_exposure
        ),
        "largest_position": float(
            largest_position
        )
    }


# ============================================================
# BUY / SELL VALIDATION
# ============================================================

def validate_directions(
    data
):

    print()
    print("=" * 80)
    print("VALIDATING BUY / SELL POSITIONS")
    print("=" * 80)

    buy_count = (
        data[
            "validation_direction"
        ]
        .eq("BUY")
        .sum()
    )

    sell_count = (
        data[
            "validation_direction"
        ]
        .eq("SELL")
        .sum()
    )

    hold_count = (
        data[
            "validation_direction"
        ]
        .eq("HOLD")
        .sum()
    )

    print(
        f"BUY positions          : "
        f"{buy_count}"
    )

    print(
        f"SELL positions         : "
        f"{sell_count}"
    )

    print(
        f"HOLD positions         : "
        f"{hold_count}"
    )

    return {
        "buy_positions": int(
            buy_count
        ),
        "sell_positions": int(
            sell_count
        ),
        "hold_positions": int(
            hold_count
        )
    }


# ============================================================
# RISK VALIDATION
# ============================================================

def validate_risk(
    data,
    portfolio_metrics
):

    print()
    print("=" * 80)
    print("VALIDATING PORTFOLIO RISK")
    print("=" * 80)

    checks = []

    # --------------------------------------------------------
    # Risk from portfolio metrics
    # --------------------------------------------------------

    maximum_risk_capital = safe_float(
        portfolio_metrics.get(
            "maximum_risk_capital",
            portfolio_metrics.get(
                "max_risk_capital",
                0.0
            )
        )
    )

    risk_limit = (
        INITIAL_CAPITAL
        * MAX_RISK_CAPITAL
    )

    risk_ok = (
        maximum_risk_capital
        <= risk_limit
        + 1e-9
    )

    checks.append({
        "check": "maximum_risk_capital",
        "value": maximum_risk_capital,
        "limit": risk_limit,
        "passed": risk_ok
    })

    print(
        f"Maximum risk capital  : "
        f"₹{maximum_risk_capital:,.2f}"
    )

    print(
        f"Risk capital limit     : "
        f"₹{risk_limit:,.2f}"
    )

    print(
        f"Risk check             : "
        f"{'PASS' if risk_ok else 'FAIL'}"
    )

    return checks


# ============================================================
# PERFORMANCE VALIDATION
# ============================================================

def validate_performance(
    portfolio_metrics
):

    print()
    print("=" * 80)
    print("VALIDATING PORTFOLIO PERFORMANCE")
    print("=" * 80)

    checks = []

    expected_return = safe_float(
        portfolio_metrics.get(
            "expected_return",
            portfolio_metrics.get(
                "expected_return_pct",
                0.0
            )
        )
    )

    expected_profit = safe_float(
        portfolio_metrics.get(
            "expected_profit",
            0.0
        )
    )

    win_rate = safe_float(
        portfolio_metrics.get(
            "win_rate",
            0.0
        )
    )

    risk_reward = safe_float(
        portfolio_metrics.get(
            "risk_reward",
            portfolio_metrics.get(
                "risk_reward_ratio",
                0.0
            )
        )
    )

    # --------------------------------------------------------
    # Normalize percentages
    # --------------------------------------------------------

    if win_rate > 1:

        win_rate = (
            win_rate / 100
        )

    if expected_return > 1:

        expected_return = (
            expected_return / 100
        )

    # --------------------------------------------------------
    # Expected return
    # --------------------------------------------------------

    return_ok = (
        expected_return
        >= MIN_EXPECTED_RETURN
    )

    checks.append({
        "check": "expected_return",
        "value": expected_return,
        "limit": MIN_EXPECTED_RETURN,
        "passed": return_ok
    })

    # --------------------------------------------------------
    # Win rate
    # --------------------------------------------------------

    win_rate_ok = (
        win_rate
        >= MIN_WIN_RATE
    )

    checks.append({
        "check": "win_rate",
        "value": win_rate,
        "limit": MIN_WIN_RATE,
        "passed": win_rate_ok
    })

    # --------------------------------------------------------
    # Risk reward
    # --------------------------------------------------------

    risk_reward_ok = (
        risk_reward
        >= MIN_RISK_REWARD
    )

    checks.append({
        "check": "risk_reward",
        "value": risk_reward,
        "limit": MIN_RISK_REWARD,
        "passed": risk_reward_ok
    })

    print(
        f"Expected return        : "
        f"{expected_return * 100:.2f}%"
    )

    print(
        f"Expected profit        : "
        f"₹{expected_profit:,.2f}"
    )

    print(
        f"Win rate               : "
        f"{win_rate * 100:.2f}%"
    )

    print(
        f"Risk / reward          : "
        f"{risk_reward:.2f}"
    )

    print(
        f"Return check           : "
        f"{'PASS' if return_ok else 'FAIL'}"
    )

    print(
        f"Win rate check         : "
        f"{'PASS' if win_rate_ok else 'FAIL'}"
    )

    print(
        f"Risk/reward check      : "
        f"{'PASS' if risk_reward_ok else 'FAIL'}"
    )

    return checks, {
        "expected_return": expected_return,
        "expected_profit": expected_profit,
        "win_rate": win_rate,
        "risk_reward": risk_reward
    }


# ============================================================
# POSITION LEVEL VALIDATION
# ============================================================

def build_position_validation(
    data
):

    print()
    print("=" * 80)
    print("BUILDING POSITION VALIDATION")
    print("=" * 80)

    result = data.copy()

    # --------------------------------------------------------
    # Individual checks
    # --------------------------------------------------------

    result["check_symbol"] = (
        result[
            "validation_symbol"
        ].ne("")
    )

    result["check_direction"] = (
        result[
            "validation_direction"
        ].isin(
            [
                "BUY",
                "SELL",
                "HOLD"
            ]
        )
    )

    result["check_weight"] = (
        result[
            "validation_weight"
        ].notna()
        &
        (
            result[
                "validation_weight"
            ] >= 0
        )
        &
        (
            result[
                "validation_weight"
            ]
            <= MAX_SINGLE_POSITION
        )
    )

    # --------------------------------------------------------
    # Position status
    # --------------------------------------------------------

    result["position_status"] = np.where(
        (
            result["check_symbol"]
            &
            result["check_direction"]
            &
            result["check_weight"]
        ),
        "VALID",
        "INVALID"
    )

    # --------------------------------------------------------
    # Reason
    # --------------------------------------------------------

    def get_reason(
        row
    ):

        reasons = []

        if not row[
            "check_symbol"
        ]:

            reasons.append(
                "INVALID_SYMBOL"
            )

        if not row[
            "check_direction"
        ]:

            reasons.append(
                "INVALID_DIRECTION"
            )

        if not row[
            "check_weight"
        ]:

            reasons.append(
                "INVALID_WEIGHT"
            )

        if not reasons:

            return "OK"

        return ";".join(
            reasons
        )

    result["validation_reason"] = (
        result.apply(
            get_reason,
            axis=1
        )
    )

    return result


# ============================================================
# LOAD METRICS
# ============================================================

def load_portfolio_metrics():

    if not os.path.exists(
        PORTFOLIO_METRICS_FILE
    ):

        print()
        print(
            "⚠️ Portfolio metrics file "
            "not found."
        )

        return {}

    try:

        with open(
            PORTFOLIO_METRICS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            metrics = json.load(
                file
            )

        return metrics

    except Exception as e:

        print(
            f"⚠️ Could not load portfolio "
            f"metrics: {e}"
        )

        return {}


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    data,
    metrics
):

    print()
    print("=" * 80)
    print("SAVING PORTFOLIO VALIDATION")
    print("=" * 80)

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

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

    print()
    print(
        "Validation results saved:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print()
    print(
        "Validation metrics saved:"
    )

    print(
        f"  {METRICS_FILE}"
    )


# ============================================================
# FINAL DECISION
# ============================================================

def calculate_final_decision(
    checks
):

    passed = 0
    failed = 0

    for check in checks:

        if check.get(
            "passed",
            False
        ):

            passed += 1

        else:

            failed += 1

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if failed == 0:

        decision = "PASS"

    elif failed <= 2:

        decision = "REVIEW"

    else:

        decision = "REJECT"

    return decision, passed, failed


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("STAGE 20.12 - PORTFOLIO VALIDATOR")
    print("=" * 80)

    try:

        ensure_directory()

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        portfolio = load_portfolio()

        portfolio_metrics = (
            load_portfolio_metrics()
        )

        # ----------------------------------------------------
        # Prepare
        # ----------------------------------------------------

        data = prepare_data(
            portfolio
        )

        # ----------------------------------------------------
        # Structure validation
        # ----------------------------------------------------

        structure_checks = (
            validate_basic_structure(
                data
            )
        )

        # ----------------------------------------------------
        # Allocation validation
        # ----------------------------------------------------

        allocation_checks, allocation_stats = (
            validate_allocation(
                data
            )
        )

        # ----------------------------------------------------
        # Direction validation
        # ----------------------------------------------------

        direction_stats = (
            validate_directions(
                data
            )
        )

        # ----------------------------------------------------
        # Risk validation
        # ----------------------------------------------------

        risk_checks = (
            validate_risk(
                data,
                portfolio_metrics
            )
        )

        # ----------------------------------------------------
        # Performance validation
        # ----------------------------------------------------

        performance_checks, performance_stats = (
            validate_performance(
                portfolio_metrics
            )
        )

        # ----------------------------------------------------
        # Position validation
        # ----------------------------------------------------

        validated_positions = (
            build_position_validation(
                data
            )
        )

        valid_positions = int(
            (
                validated_positions[
                    "position_status"
                ]
                == "VALID"
            ).sum()
        )

        invalid_positions = int(
            (
                validated_positions[
                    "position_status"
                ]
                == "INVALID"
            ).sum()
        )

        position_check = {
            "check": "all_positions_valid",
            "value": valid_positions,
            "limit": len(
                validated_positions
            ),
            "passed": (
                invalid_positions == 0
            )
        }

        # ----------------------------------------------------
        # Combine checks
        # ----------------------------------------------------

        all_checks = (
            structure_checks
            +
            allocation_checks
            +
            risk_checks
            +
            performance_checks
            +
            [
                position_check
            ]
        )

        # ----------------------------------------------------
        # Final decision
        # ----------------------------------------------------

        decision, passed, failed = (
            calculate_final_decision(
                all_checks
            )
        )

        # ----------------------------------------------------
        # Add validation columns
        # ----------------------------------------------------

        validated_positions[
            "portfolio_validation"
        ] = decision

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = {

            "stage": "20.12",

            "stage_name":
                "Portfolio Validator",

            "validation_status":
                decision,

            "checks_total":
                len(all_checks),

            "checks_passed":
                passed,

            "checks_failed":
                failed,

            "positions":
                len(data),

            "valid_positions":
                valid_positions,

            "invalid_positions":
                invalid_positions,

            "buy_positions":
                direction_stats[
                    "buy_positions"
                ],

            "sell_positions":
                direction_stats[
                    "sell_positions"
                ],

            "hold_positions":
                direction_stats[
                    "hold_positions"
                ],

            "total_exposure":
                allocation_stats[
                    "total_exposure"
                ],

            "largest_position":
                allocation_stats[
                    "largest_position"
                ],

            "expected_return":
                performance_stats[
                    "expected_return"
                ],

            "expected_profit":
                performance_stats[
                    "expected_profit"
                ],

            "win_rate":
                performance_stats[
                    "win_rate"
                ],

            "risk_reward":
                performance_stats[
                    "risk_reward"
                ],

            "initial_capital":
                INITIAL_CAPITAL,

            "maximum_single_position":
                MAX_SINGLE_POSITION,

            "maximum_total_exposure":
                MAX_TOTAL_EXPOSURE,

            "maximum_risk_capital":
                INITIAL_CAPITAL
                * MAX_RISK_CAPITAL
        }

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_results(
            validated_positions,
            metrics
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print()
        print("=" * 80)
        print("STAGE 20.12 PORTFOLIO VALIDATION SUMMARY")
        print("=" * 80)

        print()
        print(
            f"Positions             : "
            f"{len(data)}"
        )

        print(
            f"Valid positions       : "
            f"{valid_positions}"
        )

        print(
            f"Invalid positions     : "
            f"{invalid_positions}"
        )

        print(
            f"BUY positions         : "
            f"{direction_stats['buy_positions']}"
        )

        print(
            f"SELL positions        : "
            f"{direction_stats['sell_positions']}"
        )

        print(
            f"HOLD positions        : "
            f"{direction_stats['hold_positions']}"
        )

        print(
            f"Total exposure        : "
            f"{allocation_stats['total_exposure'] * 100:.2f}%"
        )

        print(
            f"Largest position      : "
            f"{allocation_stats['largest_position'] * 100:.2f}%"
        )

        print(
            f"Expected return       : "
            f"{performance_stats['expected_return'] * 100:.2f}%"
        )

        print(
            f"Expected profit       : "
            f"₹{performance_stats['expected_profit']:,.2f}"
        )

        print(
            f"Win rate              : "
            f"{performance_stats['win_rate'] * 100:.2f}%"
        )

        print(
            f"Risk / reward         : "
            f"{performance_stats['risk_reward']:.2f}"
        )

        print()
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
            "STAGE 20.12 PORTFOLIO VALIDATION COMPLETE"
        )

        print(
            "=" * 80
        )

        print()
        print(
            f"Results : {OUTPUT_FILE}"
        )

        print(
            f"Metrics : {METRICS_FILE}"
        )

        return metrics

    except Exception as e:

        print()
        print("=" * 80)
        print("STAGE 20.12 PORTFOLIO VALIDATION FAILED")
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

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()