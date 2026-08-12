# ============================================================
# STAGE 20.7 - BACKTEST VALIDATION
# stage20/validation.py
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

DATA_DIR = "data/stage20"

BACKTEST_FILE = os.path.join(
    DATA_DIR,
    "stage20_backtest_results.csv"
)

BACKTEST_METRICS_FILE = os.path.join(
    DATA_DIR,
    "stage20_backtest_metrics.json"
)

VALIDATION_FILE = os.path.join(
    DATA_DIR,
    "stage20_validation_report.csv"
)

VALIDATION_METRICS_FILE = os.path.join(
    DATA_DIR,
    "stage20_validation_metrics.json"
)


# ============================================================
# VALIDATION THRESHOLDS
# ============================================================

MIN_ACCURACY = 50.0

MIN_WIN_RATE = 50.0

MIN_PROFIT_FACTOR = 1.0

MAX_DRAWDOWN = -20.0

MIN_TOTAL_TRADES = 20

MIN_SHARPE = 0.0


# ============================================================
# LOAD BACKTEST RESULTS
# ============================================================

def load_backtest_results():

    print()
    print("=" * 80)
    print("LOADING STAGE 20.6 BACKTEST RESULTS")
    print("=" * 80)

    if not os.path.exists(BACKTEST_FILE):

        raise FileNotFoundError(
            f"Backtest results not found:\n"
            f"{BACKTEST_FILE}"
        )

    df = pd.read_csv(
        BACKTEST_FILE
    )

    print()
    print(
        f"File: {BACKTEST_FILE}"
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    return df


# ============================================================
# LOAD BACKTEST METRICS
# ============================================================

def load_backtest_metrics():

    if not os.path.exists(
        BACKTEST_METRICS_FILE
    ):

        print(
            "⚠️ Backtest metrics file not found."
        )

        return {}

    with open(
        BACKTEST_METRICS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print()
    print("=" * 80)
    print("PREPARING VALIDATION DATA")
    print("=" * 80)

    df = df.copy()

    # --------------------------------------------------------
    # Clean names
    # --------------------------------------------------------

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        "date",
        "Symbol",
        "target",
        "prediction",
        "trade",
        "strategy_return",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns:\n"
            + "\n".join(
                f" - {column}"
                for column in missing
            )
        )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    numeric_columns = [
        "target",
        "prediction",
        "strategy_return",
    ]

    optional_numeric = [
        "probability_up",
        "probability_down",
        "confidence",
        "future_return",
        "equity",
    ]

    for column in (
        numeric_columns
        + optional_numeric
    ):

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.dropna(
        subset=[
            "date",
            "Symbol",
            "target",
            "prediction",
            "strategy_return",
        ]
    )

    df = (
        df
        .sort_values(
            [
                "date",
                "Symbol",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        f"Valid rows: {len(df)}"
    )

    print(
        f"Symbols: {df['Symbol'].nunique()}"
    )

    print(
        f"Dates: {df['date'].nunique()}"
    )

    print(
        f"Start: {df['date'].min()}"
    )

    print(
        f"End: {df['date'].max()}"
    )

    return df


# ============================================================
# CLASSIFICATION VALIDATION
# ============================================================

def calculate_classification_metrics(df):

    print()
    print("=" * 80)
    print("CLASSIFICATION VALIDATION")
    print("=" * 80)

    actual = df["target"].astype(int)

    predicted = (
        df["prediction"]
        .astype(int)
    )

    accuracy = (
        actual
        == predicted
    ).mean() * 100

    # --------------------------------------------------------
    # BUY precision
    # --------------------------------------------------------

    buy_predictions = (
        predicted == 1
    )

    if buy_predictions.sum() > 0:

        buy_accuracy = (
            actual[
                buy_predictions
            ] == 1
        ).mean() * 100

    else:

        buy_accuracy = 0.0

    # --------------------------------------------------------
    # SELL accuracy
    # --------------------------------------------------------

    sell_predictions = (
        predicted == 0
    )

    if sell_predictions.sum() > 0:

        sell_accuracy = (
            actual[
                sell_predictions
            ] == 0
        ).mean() * 100

    else:

        sell_accuracy = 0.0

    return {

        "accuracy_percent":
            float(accuracy),

        "buy_prediction_accuracy_percent":
            float(buy_accuracy),

        "sell_prediction_accuracy_percent":
            float(sell_accuracy),

        "actual_up":
            int((actual == 1).sum()),

        "actual_down":
            int((actual == 0).sum()),

        "predicted_buy":
            int((predicted == 1).sum()),

        "predicted_sell":
            int((predicted == 0).sum()),
    }


# ============================================================
# TRADE VALIDATION
# ============================================================

def calculate_trade_metrics(df):

    print()
    print("=" * 80)
    print("TRADE PERFORMANCE VALIDATION")
    print("=" * 80)

    trades = df[
        df["trade"]
        != "NO TRADE"
    ].copy()

    total_trades = len(
        trades
    )

    if total_trades == 0:

        return {

            "total_trades": 0,

            "winning_trades": 0,

            "losing_trades": 0,

            "win_rate_percent": 0.0,

            "average_trade_return_percent":
                0.0,

            "total_trade_return_percent":
                0.0,

            "profit_factor": 0.0,

            "buy_trades": 0,

            "sell_trades": 0,
        }

    wins = trades[
        trades[
            "strategy_return"
        ] > 0
    ]

    losses = trades[
        trades[
            "strategy_return"
        ] < 0
    ]

    winning_trades = len(
        wins
    )

    losing_trades = len(
        losses
    )

    win_rate = (
        winning_trades
        /
        total_trades
        * 100
    )

    average_return = (
        trades[
            "strategy_return"
        ].mean()
        * 100
    )

    total_return = (
        trades[
            "strategy_return"
        ].sum()
        * 100
    )

    gross_profit = (
        wins[
            "strategy_return"
        ].sum()
    )

    gross_loss = (
        losses[
            "strategy_return"
        ].sum()
    )

    if gross_loss < 0:

        profit_factor = (
            gross_profit
            /
            abs(gross_loss)
        )

    else:

        profit_factor = 0.0

    return {

        "total_trades":
            int(total_trades),

        "winning_trades":
            int(winning_trades),

        "losing_trades":
            int(losing_trades),

        "win_rate_percent":
            float(win_rate),

        "average_trade_return_percent":
            float(
                average_return
            ),

        "total_trade_return_percent":
            float(
                total_return
            ),

        "profit_factor":
            float(
                profit_factor
            ),

        "buy_trades":
            int(
                (
                    trades["trade"]
                    == "BUY"
                ).sum()
            ),

        "sell_trades":
            int(
                (
                    trades["trade"]
                    == "SELL"
                ).sum()
            ),
    }


# ============================================================
# EQUITY VALIDATION
# ============================================================

def calculate_equity_metrics(df):

    print()
    print("=" * 80)
    print("EQUITY CURVE VALIDATION")
    print("=" * 80)

    if "equity" not in df.columns:

        return {

            "initial_equity": 0.0,

            "final_equity": 0.0,

            "total_return_percent": 0.0,

            "maximum_drawdown_percent": 0.0,

            "sharpe_ratio": 0.0,
        }

    equity = (
        pd.to_numeric(
            df["equity"],
            errors="coerce"
        )
        .dropna()
    )

    if equity.empty:

        return {

            "initial_equity": 0.0,

            "final_equity": 0.0,

            "total_return_percent": 0.0,

            "maximum_drawdown_percent": 0.0,

            "sharpe_ratio": 0.0,
        }

    initial_equity = float(
        equity.iloc[0]
    )

    final_equity = float(
        equity.iloc[-1]
    )

    if initial_equity != 0:

        total_return = (
            final_equity
            /
            initial_equity
            - 1
        ) * 100

    else:

        total_return = 0.0

    running_max = (
        equity
        .cummax()
    )

    drawdown = (
        equity
        /
        running_max
        - 1
    )

    maximum_drawdown = (
        drawdown.min()
        * 100
    )

    returns = (
        equity
        .pct_change()
        .dropna()
    )

    if (
        len(returns) > 1
        and
        returns.std() > 0
    ):

        sharpe = (
            returns.mean()
            /
            returns.std()
        ) * np.sqrt(252)

    else:

        sharpe = 0.0

    return {

        "initial_equity":
            initial_equity,

        "final_equity":
            final_equity,

        "total_return_percent":
            float(total_return),

        "maximum_drawdown_percent":
            float(maximum_drawdown),

        "sharpe_ratio":
            float(sharpe),
    }


# ============================================================
# BUY / SELL ANALYSIS
# ============================================================

def direction_analysis(df):

    print()
    print("=" * 80)
    print("BUY / SELL ANALYSIS")
    print("=" * 80)

    result = {}

    for direction in [
        "BUY",
        "SELL",
    ]:

        subset = df[
            df["trade"]
            == direction
        ]

        if subset.empty:

            result[direction] = {

                "trades": 0,

                "win_rate_percent": 0.0,

                "average_return_percent":
                    0.0,

                "total_return_percent":
                    0.0,
            }

            continue

        wins = (
            subset[
                "strategy_return"
            ] > 0
        ).sum()

        win_rate = (
            wins
            /
            len(subset)
            * 100
        )

        average_return = (
            subset[
                "strategy_return"
            ].mean()
            * 100
        )

        total_return = (
            subset[
                "strategy_return"
            ].sum()
            * 100
        )

        result[direction] = {

            "trades":
                int(len(subset)),

            "win_rate_percent":
                float(win_rate),

            "average_return_percent":
                float(
                    average_return
                ),

            "total_return_percent":
                float(
                    total_return
                ),
        }

    return result


# ============================================================
# SYMBOL ANALYSIS
# ============================================================

def symbol_analysis(df):

    print()
    print("=" * 80)
    print("PER-SYMBOL ANALYSIS")
    print("=" * 80)

    rows = []

    for symbol, group in (
        df.groupby("Symbol")
    ):

        trades = group[
            group["trade"]
            != "NO TRADE"
        ]

        if trades.empty:

            continue

        wins = (
            trades[
                "strategy_return"
            ] > 0
        ).sum()

        win_rate = (
            wins
            /
            len(trades)
            * 100
        )

        total_return = (
            trades[
                "strategy_return"
            ].sum()
            * 100
        )

        average_return = (
            trades[
                "strategy_return"
            ].mean()
            * 100
        )

        rows.append({

            "Symbol":
                symbol,

            "trades":
                len(trades),

            "wins":
                int(wins),

            "losses":
                int(
                    len(trades)
                    - wins
                ),

            "win_rate_percent":
                round(
                    win_rate,
                    4
                ),

            "average_return_percent":
                round(
                    average_return,
                    6
                ),

            "total_return_percent":
                round(
                    total_return,
                    6
                ),
        })

    if not rows:

        return pd.DataFrame(
            columns=[
                "Symbol",
                "trades",
                "wins",
                "losses",
                "win_rate_percent",
                "average_return_percent",
                "total_return_percent",
            ]
        )

    result = pd.DataFrame(
        rows
    )

    result = (
        result
        .sort_values(
            "total_return_percent",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# VALIDATION DECISION
# ============================================================

def make_decision(
    classification,
    trades,
    equity
):

    print()
    print("=" * 80)
    print("VALIDATING STRATEGY")
    print("=" * 80)

    accuracy = (
        classification[
            "accuracy_percent"
        ]
    )

    win_rate = (
        trades[
            "win_rate_percent"
        ]
    )

    profit_factor = (
        trades[
            "profit_factor"
        ]
    )

    total_trades = (
        trades[
            "total_trades"
        ]
    )

    drawdown = (
        equity[
            "maximum_drawdown_percent"
        ]
    )

    sharpe = (
        equity[
            "sharpe_ratio"
        ]
    )

    total_return = (
        equity[
            "total_return_percent"
        ]
    )

    passed = 0
    failed = 0
    checks = []

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    if accuracy >= MIN_ACCURACY:

        passed += 1

        checks.append(
            "PASS: ML accuracy"
        )

    else:

        failed += 1

        checks.append(
            "FAIL: ML accuracy"
        )

    # --------------------------------------------------------
    # Win rate
    # --------------------------------------------------------

    if win_rate >= MIN_WIN_RATE:

        passed += 1

        checks.append(
            "PASS: Win rate"
        )

    else:

        failed += 1

        checks.append(
            "FAIL: Win rate"
        )

    # --------------------------------------------------------
    # Profit factor
    # --------------------------------------------------------

    if profit_factor >= MIN_PROFIT_FACTOR:

        passed += 1

        checks.append(
            "PASS: Profit factor"
        )

    else:

        failed += 1

        checks.append(
            "FAIL: Profit factor"
        )

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    if drawdown >= MAX_DRAWDOWN:

        passed += 1

        checks.append(
            "PASS: Maximum drawdown"
        )

    else:

        failed += 1

        checks.append(
            "FAIL: Maximum drawdown"
        )

    # --------------------------------------------------------
    # Number of trades
    # --------------------------------------------------------

    if total_trades >= MIN_TOTAL_TRADES:

        passed += 1

        checks.append(
            "PASS: Minimum trade count"
        )

    else:

        failed += 1

        checks.append(
            "FAIL: Minimum trade count"
        )

    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    if sharpe >= MIN_SHARPE:

        passed += 1

        checks.append(
            "PASS: Sharpe ratio"
        )

    else:

        failed += 1

        checks.append(
            "FAIL: Sharpe ratio"
        )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    if (
        failed == 0
        and
        total_return > 0
    ):

        decision = "PASS"

    elif (
        passed >= 4
        and
        total_return >= 0
    ):

        decision = "REVIEW"

    else:

        decision = "FAIL"

    print()

    for check in checks:

        print(
            f"  {check}"
        )

    print()
    print(
        f"Checks passed : {passed}"
    )

    print(
        f"Checks failed : {failed}"
    )

    print()
    print(
        f"FINAL DECISION: {decision}"
    )

    return {

        "decision":
            decision,

        "checks_passed":
            passed,

        "checks_failed":
            failed,

        "total_return_percent":
            total_return,

        "accuracy_percent":
            accuracy,

        "win_rate_percent":
            win_rate,

        "profit_factor":
            profit_factor,

        "maximum_drawdown_percent":
            drawdown,

        "sharpe_ratio":
            sharpe,

        "total_trades":
            total_trades,

        "checks":
            checks,
    }


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    symbol_report,
    metrics
):

    print()
    print("=" * 80)
    print("SAVING VALIDATION REPORT")
    print("=" * 80)

    symbol_report.to_csv(
        VALIDATION_FILE,
        index=False
    )

    with open(
        VALIDATION_METRICS_FILE,
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
        f"Report saved: "
        f"{VALIDATION_FILE}"
    )

    print(
        f"Metrics saved: "
        f"{VALIDATION_METRICS_FILE}"
    )


# ============================================================
# DISPLAY FINAL SUMMARY
# ============================================================

def display_summary(
    classification,
    trades,
    equity,
    decision
):

    print()
    print("=" * 80)
    print("STAGE 20.7 - VALIDATION SUMMARY")
    print("=" * 80)

    print()

    print(
        f"ML Accuracy          : "
        f"{classification['accuracy_percent']:.2f}%"
    )

    print(
        f"Win Rate             : "
        f"{trades['win_rate_percent']:.2f}%"
    )

    print(
        f"Profit Factor        : "
        f"{trades['profit_factor']:.4f}"
    )

    print(
        f"Total Trades         : "
        f"{trades['total_trades']}"
    )

    print(
        f"Total Return         : "
        f"{equity['total_return_percent']:.2f}%"
    )

    print(
        f"Maximum Drawdown     : "
        f"{equity['maximum_drawdown_percent']:.2f}%"
    )

    print(
        f"Sharpe Ratio         : "
        f"{equity['sharpe_ratio']:.4f}"
    )

    print()

    print(
        f"FINAL DECISION       : "
        f"{decision['decision']}"
    )

    print()

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("STAGE 20.7 - BACKTEST VALIDATION")
    print("=" * 80)

    try:

        # ----------------------------------------------------
        # 1. Load
        # ----------------------------------------------------

        df = load_backtest_results()

        backtest_metrics = (
            load_backtest_metrics()
        )

        # ----------------------------------------------------
        # 2. Prepare
        # ----------------------------------------------------

        df = prepare_data(
            df
        )

        # ----------------------------------------------------
        # 3. Classification
        # ----------------------------------------------------

        classification = (
            calculate_classification_metrics(
                df
            )
        )

        # ----------------------------------------------------
        # 4. Trade performance
        # ----------------------------------------------------

        trades = (
            calculate_trade_metrics(
                df
            )
        )

        # ----------------------------------------------------
        # 5. Equity
        # ----------------------------------------------------

        equity = (
            calculate_equity_metrics(
                df
            )
        )

        # ----------------------------------------------------
        # 6. Direction
        # ----------------------------------------------------

        direction = (
            direction_analysis(
                df
            )
        )

        # ----------------------------------------------------
        # 7. Symbol analysis
        # ----------------------------------------------------

        symbol_report = (
            symbol_analysis(
                df
            )
        )

        # ----------------------------------------------------
        # 8. Decision
        # ----------------------------------------------------

        decision = (
            make_decision(
                classification,
                trades,
                equity
            )
        )

        # ----------------------------------------------------
        # 9. Combined metrics
        # ----------------------------------------------------

        metrics = {

            "stage":
                "20.7",

            "classification":
                classification,

            "trade_performance":
                trades,

            "equity":
                equity,

            "direction_analysis":
                direction,

            "decision":
                decision,

            "source_backtest_metrics":
                backtest_metrics,

            "validation_note":
                (
                    "This is a historical "
                    "validation of the existing "
                    "Stage 20.6 backtest. "
                    "It is not a guarantee of "
                    "future trading performance."
                ),
        }

        # ----------------------------------------------------
        # 10. Save
        # ----------------------------------------------------

        save_report(
            symbol_report,
            metrics
        )

        # ----------------------------------------------------
        # 11. Summary
        # ----------------------------------------------------

        display_summary(
            classification,
            trades,
            equity,
            decision
        )

        # ----------------------------------------------------
        # 12. Important data-quality warning
        # ----------------------------------------------------

        unique_dates = (
            df["date"]
            .nunique()
        )

        if unique_dates <= 1:

            print()
            print("=" * 80)
            print("⚠️ DATA QUALITY WARNING")
            print("=" * 80)

            print()
            print(
                "The backtest results contain "
                f"only {unique_dates} unique test date."
            )

            print(
                "Do NOT treat the performance "
                "statistics as a reliable "
                "multi-day strategy evaluation."
            )

            print(
                "The historical dataset should "
                "be checked before using this "
                "for live/paper trading."
            )

        print()
        print("=" * 80)
        print("STAGE 20.7 VALIDATION COMPLETE")
        print("=" * 80)

    except Exception as e:

        print()
        print("=" * 80)
        print("STAGE 20.7 VALIDATION FAILED")
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