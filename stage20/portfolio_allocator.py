# ============================================================
# STAGE 20.11 - PORTFOLIO RISK & ALLOCATION
# stage20/portfolio_allocator.py
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

INPUT_FILE = os.path.join(
    BASE_DIR,
    "stage20_optimized_trades.csv"
)

OPTIMIZATION_METRICS_FILE = os.path.join(
    BASE_DIR,
    "stage20_optimization_metrics.json"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "stage20_portfolio.csv"
)

OUTPUT_METRICS = os.path.join(
    BASE_DIR,
    "stage20_portfolio_metrics.json"
)


# ============================================================
# PORTFOLIO SETTINGS
# ============================================================

INITIAL_CAPITAL = 100000.0

MAX_PORTFOLIO_EXPOSURE = 1.00

MAX_SINGLE_POSITION = 0.20

MIN_SINGLE_POSITION = 0.05

MAX_POSITIONS = 10

RISK_PER_POSITION = 0.02

STOP_LOSS = 0.05

TAKE_PROFIT = 0.10

MIN_CONFIDENCE = 0.60


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

        key = (
            str(candidate)
            .strip()
            .lower()
        )

        if key in columns:

            return columns[key]

    for column in df.columns:

        name = (
            str(column)
            .strip()
            .lower()
        )

        for candidate in candidates:

            candidate = (
                str(candidate)
                .strip()
                .lower()
            )

            if candidate in name:

                return column

    return None


# ============================================================
# LOAD OPTIMIZED TRADES
# ============================================================

def load_data():

    print()
    print("=" * 80)
    print("STAGE 20.11 - LOADING OPTIMIZED TRADES")
    print("=" * 80)

    if not os.path.exists(
        INPUT_FILE
    ):

        raise FileNotFoundError(
            f"Optimized trade file not found: "
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        f"File: {INPUT_FILE}"
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        f"Columns loaded: "
        f"{len(df.columns)}"
    )

    print()
    print("Columns:")

    for column in df.columns:

        print(
            f"  - {column}"
        )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    df
):

    print()
    print("=" * 80)
    print("PREPARING PORTFOLIO DATA")
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
            data[
                symbol_column
            ]
            .astype(str)
            .str.strip()
        )

    else:

        data["Symbol"] = (
            "UNKNOWN"
        )

    # ========================================================
    # DATE
    # ========================================================

    date_column = find_column(
        data,
        [
            "date",
            "datetime",
            "timestamp",
        ]
    )

    if date_column:

        data["date"] = pd.to_datetime(
            data[
                date_column
            ],
            errors="coerce"
        )

    else:

        data["date"] = pd.NaT

    # ========================================================
    # DIRECTION
    # ========================================================

    direction_column = find_column(
        data,
        [
            "direction",
            "signal",
            "prediction_signal",
        ]
    )

    if direction_column:

        data["direction"] = (
            data[
                direction_column
            ]
            .astype(str)
            .str.upper()
            .str.strip()
        )

    else:

        data["direction"] = "HOLD"

    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence_column = find_column(
        data,
        [
            "confidence",
            "ml_confidence",
            "probability_up",
        ]
    )

    if confidence_column:

        data["confidence"] = pd.to_numeric(
            data[
                confidence_column
            ],
            errors="coerce"
        )

    else:

        data["confidence"] = np.nan

    # Convert percentages to decimal.
    percentage_mask = (
        data["confidence"]
        > 1.0
    )

    data.loc[
        percentage_mask,
        "confidence"
    ] = (
        data.loc[
            percentage_mask,
            "confidence"
        ]
        / 100.0
    )

    # ========================================================
    # RETURN
    # ========================================================

    return_column = find_column(
        data,
        [
            "optimized_return",
            "trade_return",
            "future_return",
            "strategy_return",
        ]
    )

    if return_column:

        data["trade_return"] = pd.to_numeric(
            data[
                return_column
            ],
            errors="coerce"
        )

    else:

        data["trade_return"] = np.nan

    # Convert percentage returns.
    return_mask = (
        data["trade_return"]
        .abs()
        > 1.0
    )

    data.loc[
        return_mask,
        "trade_return"
    ] = (
        data.loc[
            return_mask,
            "trade_return"
        ]
        / 100.0
    )

    # ========================================================
    # CLOSE
    # ========================================================

    close_column = find_column(
        data,
        [
            "close",
            "price",
            "ltp",
        ]
    )

    if close_column:

        data["close"] = pd.to_numeric(
            data[
                close_column
            ],
            errors="coerce"
        )

    else:

        data["close"] = np.nan

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
            "Symbol",
            "confidence",
        ]
    )

    # Only actionable signals.
    data = data[
        data["direction"].isin(
            [
                "BUY",
                "SELL",
            ]
        )
    ].copy()

    # Confidence filter.
    data = data[
        data["confidence"]
        >= MIN_CONFIDENCE
    ].copy()

    data = (
        data
        .sort_values(
            [
                "date",
                "confidence",
            ],
            ascending=[
                True,
                False,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        f"Usable portfolio rows: "
        f"{len(data)}"
    )

    print(
        f"BUY signals: "
        f"{int((data['direction'] == 'BUY').sum())}"
    )

    print(
        f"SELL signals: "
        f"{int((data['direction'] == 'SELL').sum())}"
    )

    if len(data) == 0:

        raise ValueError(
            "No usable portfolio candidates "
            "were found."
        )

    return data


# ============================================================
# REMOVE DUPLICATE SYMBOLS
# ============================================================

def select_latest_candidates(
    data
):

    print()
    print("=" * 80)
    print("SELECTING PORTFOLIO CANDIDATES")
    print("=" * 80)

    # Prefer latest signal for each stock.
    if data["date"].notna().any():

        data = (
            data
            .sort_values(
                [
                    "Symbol",
                    "date",
                    "confidence",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ]
            )
        )

    else:

        data = (
            data
            .sort_values(
                [
                    "Symbol",
                    "confidence",
                ],
                ascending=[
                    True,
                    False,
                ]
            )
        )

    data = (
        data
        .drop_duplicates(
            subset=[
                "Symbol"
            ],
            keep="first"
        )
        .copy()
    )

    # Highest confidence first.
    data = (
        data
        .sort_values(
            "confidence",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # Maximum number of positions.
    data = data.head(
        MAX_POSITIONS
    ).copy()

    print()
    print(
        f"Unique candidates: "
        f"{len(data)}"
    )

    return data


# ============================================================
# CALCULATE POSITION SCORE
# ============================================================

def calculate_position_score(
    row
):

    confidence = float(
        row["confidence"]
    )

    trade_return = row[
        "trade_return"
    ]

    if pd.isna(
        trade_return
    ):

        return_score = 0.0

    else:

        return_score = min(
            max(
                float(
                    trade_return
                ),
                -0.10
            ),
            0.10
        )

        return_score = (
            return_score
            / 0.10
        )

    # Confidence receives most weight.
    score = (
        confidence * 0.70
        +
        max(
            return_score,
            0.0
        ) * 0.30
    )

    return float(
        score
    )


# ============================================================
# ALLOCATE POSITIONS
# ============================================================

def allocate_positions(
    data
):

    print()
    print("=" * 80)
    print("CALCULATING PORTFOLIO ALLOCATION")
    print("=" * 80)

    data = data.copy()

    data[
        "position_score"
    ] = data.apply(
        calculate_position_score,
        axis=1
    )

    data = (
        data
        .sort_values(
            [
                "position_score",
                "confidence",
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # SCORE SUM
    # ========================================================

    score_sum = (
        data[
            "position_score"
        ]
        .sum()
    )

    if score_sum <= 0:

        data[
            "raw_weight"
        ] = (
            1.0
            /
            len(data)
        )

    else:

        data[
            "raw_weight"
        ] = (
            data[
                "position_score"
            ]
            /
            score_sum
        )

    # ========================================================
    # POSITION LIMIT
    # ========================================================

    data[
        "weight"
    ] = data[
        "raw_weight"
    ].clip(
        lower=MIN_SINGLE_POSITION,
        upper=MAX_SINGLE_POSITION
    )

    # ========================================================
    # NORMALIZE
    # ========================================================

    total_weight = (
        data["weight"]
        .sum()
    )

    if total_weight > 0:

        data[
            "weight"
        ] = (
            data["weight"]
            /
            total_weight
        )

    # ========================================================
    # PORTFOLIO EXPOSURE
    # ========================================================

    data[
        "portfolio_weight"
    ] = (
        data["weight"]
        *
        MAX_PORTFOLIO_EXPOSURE
    )

    # ========================================================
    # CAPITAL
    # ========================================================

    data[
        "allocated_capital"
    ] = (
        data[
            "portfolio_weight"
        ]
        *
        INITIAL_CAPITAL
    )

    # ========================================================
    # RISK CAPITAL
    # ========================================================

    data[
        "risk_capital"
    ] = (
        data[
            "allocated_capital"
        ]
        *
        RISK_PER_POSITION
    )

    # ========================================================
    # STOP LOSS
    # ========================================================

    data[
        "stop_loss_pct"
    ] = STOP_LOSS

    data[
        "take_profit_pct"
    ] = TAKE_PROFIT

    # ========================================================
    # RISK / REWARD
    # ========================================================

    data[
        "risk_reward"
    ] = (
        TAKE_PROFIT
        /
        STOP_LOSS
    )

    # ========================================================
    # POSITION CLASS
    # ========================================================

    data[
        "position_class"
    ] = np.where(
        data[
            "confidence"
        ] >= 0.70,
        "STRONG",
        np.where(
            data[
                "confidence"
            ] >= 0.60,
            "CANDIDATE",
            "WATCH"
        )
    )

    # ========================================================
    # FINAL RANK
    # ========================================================

    data.insert(
        0,
        "portfolio_rank",
        np.arange(
            1,
            len(data) + 1
        )
    )

    return data


# ============================================================
# PORTFOLIO METRICS
# ============================================================

def calculate_portfolio_metrics(
    portfolio
):

    print()
    print("=" * 80)
    print("CALCULATING PORTFOLIO RISK METRICS")
    print("=" * 80)

    weights = (
        portfolio[
            "portfolio_weight"
        ]
        .astype(float)
    )

    returns = (
        portfolio[
            "trade_return"
        ]
        .fillna(0.0)
        .astype(float)
    )

    # ========================================================
    # EXPECTED RETURN
    # ========================================================

    weighted_returns = (
        weights
        *
        returns
    )

    expected_return = (
        weighted_returns
        .sum()
    )

    # ========================================================
    # WIN RATE
    # ========================================================

    valid_returns = (
        returns[
            returns != 0
        ]
    )

    if len(
        valid_returns
    ) > 0:

        win_rate = (
            valid_returns.gt(0)
            .mean()
        )

    else:

        win_rate = 0.0

    # ========================================================
    # BUY / SELL
    # ========================================================

    buy_count = int(
        (
            portfolio[
                "direction"
            ]
            ==
            "BUY"
        ).sum()
    )

    sell_count = int(
        (
            portfolio[
                "direction"
            ]
            ==
            "SELL"
        ).sum()
    )

    # ========================================================
    # EXPOSURE
    # ========================================================

    total_exposure = (
        weights.sum()
    )

    largest_position = (
        weights.max()
        if len(weights)
        else 0.0
    )

    # ========================================================
    # CONCENTRATION
    # ========================================================

    herfindahl = (
        weights.pow(2)
        .sum()
    )

    # ========================================================
    # CAPITAL
    # ========================================================

    invested_capital = (
        INITIAL_CAPITAL
        *
        total_exposure
    )

    cash_remaining = (
        INITIAL_CAPITAL
        -
        invested_capital
    )

    # ========================================================
    # MAX LOSS
    # ========================================================

    maximum_risk = (
        portfolio[
            "risk_capital"
        ]
        .sum()
    )

    # ========================================================
    # EXPECTED PROFIT
    # ========================================================

    expected_profit = (
        expected_return
        *
        INITIAL_CAPITAL
    )

    # ========================================================
    # DECISION
    # ========================================================

    if (
        len(portfolio) >= 3
        and
        largest_position <= 0.20
        and
        total_exposure <= 1.00
        and
        expected_return > 0
        and
        win_rate >= 0.50
    ):

        decision = "PASS"

    elif (
        len(portfolio) >= 2
        and
        expected_return > 0
    ):

        decision = "REVIEW"

    else:

        decision = "REJECT"

    metrics = {

        "stage":
            "20.11",

        "initial_capital":
            INITIAL_CAPITAL,

        "positions":
            int(
                len(portfolio)
            ),

        "buy_positions":
            buy_count,

        "sell_positions":
            sell_count,

        "total_exposure":
            float(
                total_exposure
            ),

        "largest_position":
            float(
                largest_position
            ),

        "cash_remaining":
            float(
                cash_remaining
            ),

        "invested_capital":
            float(
                invested_capital
            ),

        "maximum_risk_capital":
            float(
                maximum_risk
            ),

        "expected_return":
            float(
                expected_return
            ),

        "expected_profit":
            float(
                expected_profit
            ),

        "win_rate":
            float(
                win_rate
            ),

        "herfindahl_index":
            float(
                herfindahl
            ),

        "stop_loss":
            STOP_LOSS,

        "take_profit":
            TAKE_PROFIT,

        "risk_reward":
            TAKE_PROFIT
            /
            STOP_LOSS,

        "final_decision":
            decision,
    }

    return metrics


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    portfolio,
    metrics
):

    print()
    print("=" * 80)
    print("SAVING PORTFOLIO RESULTS")
    print("=" * 80)

    portfolio.to_csv(
        OUTPUT_FILE,
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
        "Portfolio saved:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print()
    print(
        "Risk metrics saved:"
    )

    print(
        f"  {OUTPUT_METRICS}"
    )


# ============================================================
# DISPLAY PORTFOLIO
# ============================================================

def display_portfolio(
    portfolio
):

    print()
    print("=" * 80)
    print("STAGE 20.11 - FINAL PORTFOLIO ALLOCATION")
    print("=" * 80)

    display_columns = [
        "portfolio_rank",
        "Symbol",
        "direction",
        "confidence",
        "position_score",
        "portfolio_weight",
        "allocated_capital",
        "risk_capital",
        "stop_loss_pct",
        "take_profit_pct",
        "position_class",
    ]

    available = [
        column
        for column in display_columns
        if column in portfolio.columns
    ]

    output = (
        portfolio[
            available
        ]
        .copy()
    )

    if "confidence" in output.columns:

        output[
            "confidence"
        ] = (
            output[
                "confidence"
            ]
            * 100
        ).round(2)

    if "portfolio_weight" in output.columns:

        output[
            "portfolio_weight"
        ] = (
            output[
                "portfolio_weight"
            ]
            * 100
        ).round(2)

    if "stop_loss_pct" in output.columns:

        output[
            "stop_loss_pct"
        ] = (
            output[
                "stop_loss_pct"
            ]
            * 100
        ).round(2)

    if "take_profit_pct" in output.columns:

        output[
            "take_profit_pct"
        ] = (
            output[
                "take_profit_pct"
            ]
            * 100
        ).round(2)

    if "allocated_capital" in output.columns:

        output[
            "allocated_capital"
        ] = output[
            "allocated_capital"
        ].round(2)

    if "risk_capital" in output.columns:

        output[
            "risk_capital"
        ] = output[
            "risk_capital"
        ].round(2)

    print()

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
    print("STAGE 20.11 PORTFOLIO SUMMARY")
    print("=" * 80)

    print()

    print(
        f"Initial capital       : "
        f"{metrics['initial_capital']:.2f}"
    )

    print(
        f"Positions             : "
        f"{metrics['positions']}"
    )

    print(
        f"BUY positions         : "
        f"{metrics['buy_positions']}"
    )

    print(
        f"SELL positions        : "
        f"{metrics['sell_positions']}"
    )

    print()

    print(
        f"Total exposure        : "
        f"{metrics['total_exposure'] * 100:.2f}%"
    )

    print(
        f"Largest position      : "
        f"{metrics['largest_position'] * 100:.2f}%"
    )

    print(
        f"Invested capital      : "
        f"{metrics['invested_capital']:.2f}"
    )

    print(
        f"Cash remaining        : "
        f"{metrics['cash_remaining']:.2f}"
    )

    print()

    print(
        f"Maximum risk capital  : "
        f"{metrics['maximum_risk_capital']:.2f}"
    )

    print(
        f"Expected return       : "
        f"{metrics['expected_return'] * 100:.2f}%"
    )

    print(
        f"Expected profit       : "
        f"{metrics['expected_profit']:.2f}"
    )

    print(
        f"Win rate              : "
        f"{metrics['win_rate'] * 100:.2f}%"
    )

    print(
        f"Risk / reward         : "
        f"{metrics['risk_reward']:.2f}"
    )

    print()

    print(
        f"FINAL DECISION        : "
        f"{metrics['final_decision']}"
    )

    print()
    print("=" * 80)
    print("STAGE 20.11 PORTFOLIO ALLOCATION COMPLETE")
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("STAGE 20.11 - PORTFOLIO RISK & ALLOCATION")
    print("=" * 80)

    try:

        # ----------------------------------------------------
        # DIRECTORY
        # ----------------------------------------------------

        ensure_directories()

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        raw_data = load_data()

        # ----------------------------------------------------
        # PREPARE
        # ----------------------------------------------------

        data = prepare_data(
            raw_data
        )

        # ----------------------------------------------------
        # SELECT
        # ----------------------------------------------------

        candidates = (
            select_latest_candidates(
                data
            )
        )

        # ----------------------------------------------------
        # ALLOCATE
        # ----------------------------------------------------

        portfolio = allocate_positions(
            candidates
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metrics = (
            calculate_portfolio_metrics(
                portfolio
            )
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        display_portfolio(
            portfolio
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_results(
            portfolio,
            metrics
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
        print("STAGE 20.11 PORTFOLIO ALLOCATION FAILED")
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