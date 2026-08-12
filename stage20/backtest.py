# ============================================================
# STAGE 20.6 - ML BACKTEST
# stage20/backtest.py
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/stage20"

TRAINING_FILE = os.path.join(
    DATA_DIR,
    "stage20_training_data.csv"
)

FEATURE_FILE = os.path.join(
    "models",
    "feature_columns.json"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "stage20_backtest_results.csv"
)

METRICS_FILE = os.path.join(
    DATA_DIR,
    "stage20_backtest_metrics.json"
)

INITIAL_CAPITAL = 100000.0

POSITION_SIZE = 0.10

PROBABILITY_THRESHOLD = 0.55

TRANSACTION_COST = 0.001

TEST_RATIO = 0.20


# ============================================================
# LOAD FEATURE LIST
# ============================================================

def load_feature_columns():

    print()
    print("=" * 80)
    print("LOADING MODEL FEATURE LIST")
    print("=" * 80)

    if not os.path.exists(FEATURE_FILE):

        raise FileNotFoundError(
            f"Feature file not found:\n"
            f"{FEATURE_FILE}"
        )

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    # --------------------------------------------------------
    # Support different JSON structures
    # --------------------------------------------------------

    if isinstance(data, list):

        features = data

    elif isinstance(data, dict):

        possible_keys = [
            "features",
            "feature_columns",
            "columns",
            "feature_list",
        ]

        features = None

        for key in possible_keys:

            if key in data:

                features = data[key]

                break

        if features is None:

            raise ValueError(
                "Could not find feature list in "
                f"{FEATURE_FILE}"
            )

    else:

        raise ValueError(
            "Invalid feature_columns.json format."
        )

    features = [
        str(feature)
        for feature in features
    ]

    if not features:

        raise ValueError(
            "Feature list is empty."
        )

    print()
    print(
        f"Features loaded: {len(features)}"
    )

    for feature in features:

        print(
            f"  - {feature}"
        )

    return features


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_data():

    print()
    print("=" * 80)
    print("LOADING STAGE 20 TRAINING DATA")
    print("=" * 80)

    if not os.path.exists(
        TRAINING_FILE
    ):

        raise FileNotFoundError(
            f"Training file not found:\n"
            f"{TRAINING_FILE}"
        )

    df = pd.read_csv(
        TRAINING_FILE
    )

    print()
    print(
        f"File: {TRAINING_FILE}"
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    print()
    print("Actual columns:")

    for column in df.columns:

        print(
            f"  - {column}"
        )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    df,
    feature_columns
):

    print()
    print("=" * 80)
    print("PREPARING BACKTEST DATA")
    print("=" * 80)

    df = df.copy()

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "date",
        "Symbol",
        "target",
        "future_return",
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_required:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f" - {column}"
                for column in missing_required
            )
        )

    # --------------------------------------------------------
    # Feature validation
    # --------------------------------------------------------

    missing_features = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:

        print()
        print(
            "WARNING: Some model features are "
            "not present in the training CSV:"
        )

        for column in missing_features:

            print(
                f"  - {column}"
            )

        raise ValueError(
            "Feature list from "
            "feature_columns.json does not match "
            "stage20_training_data.csv."
        )

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Convert features to numeric
    # --------------------------------------------------------

    numeric_columns = (
        feature_columns
        + [
            "target",
            "future_return",
        ]
    )

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=(
            feature_columns
            + [
                "date",
                "target",
                "future_return",
            ]
        )
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

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

    df["target"] = (
        df["target"]
        .astype(int)
    )

    print()
    print(
        f"Valid rows: {len(df)}"
    )

    print(
        f"Symbols: {df['Symbol'].nunique()}"
    )

    print(
        f"Start date: {df['date'].min()}"
    )

    print(
        f"End date: {df['date'].max()}"
    )

    return df


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def split_data(df):

    print()
    print("=" * 80)
    print("CHRONOLOGICAL TRAIN / TEST SPLIT")
    print("=" * 80)

    unique_dates = (
        df["date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(unique_dates) < 10:

        raise ValueError(
            "Not enough dates for backtesting."
        )

    split_index = int(
        len(unique_dates)
        * (1 - TEST_RATIO)
    )

    split_index = max(
        1,
        min(
            split_index,
            len(unique_dates) - 1
        )
    )

    split_date = unique_dates.iloc[
        split_index
    ]

    train_df = df[
        df["date"] < split_date
    ].copy()

    test_df = df[
        df["date"] >= split_date
    ].copy()

    print()
    print(
        f"Training rows : {len(train_df)}"
    )

    print(
        f"Testing rows  : {len(test_df)}"
    )

    print()
    print(
        f"Training until: "
        f"{train_df['date'].max()}"
    )

    print(
        f"Testing from  : "
        f"{test_df['date'].min()}"
    )

    print(
        f"Testing until : "
        f"{test_df['date'].max()}"
    )

    return train_df, test_df


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    train_df,
    feature_columns
):

    print()
    print("=" * 80)
    print("TRAINING BACKTEST MODEL")
    print("=" * 80)

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        "target"
    ]

    print()
    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    print(
        f"UP target: "
        f"{int((y_train == 1).sum())}"
    )

    print(
        f"DOWN target: "
        f"{int((y_train == 0).sum())}"
    )

    if y_train.nunique() < 2:

        raise ValueError(
            "Training data contains only "
            "one target class."
        )

    model = RandomForestClassifier(

        n_estimators=300,

        max_depth=8,

        min_samples_leaf=5,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    print()
    print(
        "Backtest model trained successfully."
    )

    return model


# ============================================================
# PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    test_df,
    feature_columns
):

    print()
    print("=" * 80)
    print("GENERATING OUT-OF-SAMPLE PREDICTIONS")
    print("=" * 80)

    result = test_df.copy()

    X_test = result[
        feature_columns
    ]

    probabilities = (
        model.predict_proba(
            X_test
        )
    )

    classes = (
        model.classes_
        .tolist()
    )

    if 1 in classes:

        up_index = (
            classes.index(1)
        )

        probability_up = (
            probabilities[
                :,
                up_index
            ]
        )

    else:

        probability_up = np.zeros(
            len(result)
        )

    result[
        "probability_up"
    ] = probability_up

    result[
        "probability_down"
    ] = (
        1
        - result[
            "probability_up"
        ]
    )

    result[
        "prediction"
    ] = np.where(

        result[
            "probability_up"
        ] >= 0.50,

        1,

        0
    )

    result[
        "prediction_direction"
    ] = np.where(

        result[
            "prediction"
        ] == 1,

        "BUY",

        "SELL"
    )

    result[
        "confidence"
    ] = np.maximum(

        result[
            "probability_up"
        ],

        result[
            "probability_down"
        ]
    )

    # --------------------------------------------------------
    # Trade filter
    # --------------------------------------------------------

    result[
        "trade"
    ] = "NO TRADE"

    buy_mask = (
        result[
            "probability_up"
        ]
        >= PROBABILITY_THRESHOLD
    )

    sell_mask = (
        result[
            "probability_down"
        ]
        >= PROBABILITY_THRESHOLD
    )

    result.loc[
        buy_mask,
        "trade"
    ] = "BUY"

    result.loc[
        sell_mask,
        "trade"
    ] = "SELL"

    return result


# ============================================================
# TRADE RETURNS
# ============================================================

def calculate_trade_returns(
    df
):

    print()
    print("=" * 80)
    print("CALCULATING STRATEGY RETURNS")
    print("=" * 80)

    result = df.copy()

    result[
        "strategy_return"
    ] = 0.0

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    buy_mask = (
        result["trade"]
        == "BUY"
    )

    result.loc[
        buy_mask,
        "strategy_return"
    ] = (
        result.loc[
            buy_mask,
            "future_return"
        ]
        - TRANSACTION_COST
    )

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    sell_mask = (
        result["trade"]
        == "SELL"
    )

    result.loc[
        sell_mask,
        "strategy_return"
    ] = (
        -result.loc[
            sell_mask,
            "future_return"
        ]
        - TRANSACTION_COST
    )

    result[
        "strategy_return"
    ] = result[
        "strategy_return"
    ].fillna(0)

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result[
        "trade_result"
    ] = "NO TRADE"

    active = (
        result["trade"]
        != "NO TRADE"
    )

    result.loc[
        active
        &
        (
            result[
                "strategy_return"
            ] > 0
        ),
        "trade_result"
    ] = "WIN"

    result.loc[
        active
        &
        (
            result[
                "strategy_return"
            ] <= 0
        ),
        "trade_result"
    ] = "LOSS"

    return result


# ============================================================
# EQUITY CURVE
# ============================================================

def calculate_equity_curve(
    df
):

    result = (
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

    capital = INITIAL_CAPITAL

    equity = []

    for _, row in result.iterrows():

        strategy_return = float(
            row[
                "strategy_return"
            ]
        )

        position_return = (
            strategy_return
            * POSITION_SIZE
        )

        capital *= (
            1
            + position_return
        )

        equity.append(
            capital
        )

    result[
        "equity"
    ] = equity

    return result


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    df,
    train_df
):

    print()
    print("=" * 80)
    print("CALCULATING PERFORMANCE METRICS")
    print("=" * 80)

    trades = df[
        df["trade"]
        != "NO TRADE"
    ]

    total_trades = len(
        trades
    )

    wins = int(
        (
            trades[
                "trade_result"
            ]
            == "WIN"
        ).sum()
    )

    losses = int(
        (
            trades[
                "trade_result"
            ]
            == "LOSS"
        ).sum()
    )

    if total_trades > 0:

        win_rate = (
            wins
            /
            total_trades
            * 100
        )

    else:

        win_rate = 0.0

    final_equity = float(
        df[
            "equity"
        ].iloc[-1]
    )

    total_return = (
        (
            final_equity
            /
            INITIAL_CAPITAL
        )
        - 1
    ) * 100

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    rolling_max = (
        df["equity"]
        .cummax()
    )

    drawdown = (
        df["equity"]
        /
        rolling_max
        - 1
    )

    max_drawdown = (
        drawdown.min()
        * 100
    )

    # --------------------------------------------------------
    # Daily returns
    # --------------------------------------------------------

    daily_returns = (
        df
        .groupby("date")[
            "strategy_return"
        ]
        .mean()
    )

    if (
        len(daily_returns) > 1
        and
        daily_returns.std() > 0
    ):

        sharpe = (
            daily_returns.mean()
            /
            daily_returns.std()
        ) * np.sqrt(252)

    else:

        sharpe = 0.0

    # --------------------------------------------------------
    # Profit factor
    # --------------------------------------------------------

    profits = trades.loc[
        trades[
            "strategy_return"
        ] > 0,
        "strategy_return"
    ].sum()

    losses_value = trades.loc[
        trades[
            "strategy_return"
        ] < 0,
        "strategy_return"
    ].sum()

    if losses_value != 0:

        profit_factor = (
            profits
            /
            abs(losses_value)
        )

    else:

        profit_factor = 0.0

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        df["target"],
        df["prediction"]
    )

    precision = precision_score(
        df["target"],
        df["prediction"],
        zero_division=0
    )

    recall = recall_score(
        df["target"],
        df["prediction"],
        zero_division=0
    )

    cm = confusion_matrix(
        df["target"],
        df["prediction"],
        labels=[0, 1]
    )

    metrics = {

        "initial_capital":
            INITIAL_CAPITAL,

        "final_equity":
            round(
                final_equity,
                2
            ),

        "total_return_percent":
            round(
                total_return,
                4
            ),

        "maximum_drawdown_percent":
            round(
                max_drawdown,
                4
            ),

        "sharpe_ratio":
            round(
                float(sharpe),
                4
            ),

        "profit_factor":
            round(
                float(profit_factor),
                4
            ),

        "total_trades":
            int(total_trades),

        "winning_trades":
            int(wins),

        "losing_trades":
            int(losses),

        "win_rate_percent":
            round(
                float(win_rate),
                4
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

        "accuracy_percent":
            round(
                float(
                    accuracy * 100
                ),
                4
            ),

        "precision_percent":
            round(
                float(
                    precision * 100
                ),
                4
            ),

        "recall_percent":
            round(
                float(
                    recall * 100
                ),
                4
            ),

        "train_rows":
            int(len(train_df)),

        "test_rows":
            int(len(df)),

        "test_start":
            str(df["date"].min()),

        "test_end":
            str(df["date"].max()),

        "probability_threshold":
            PROBABILITY_THRESHOLD,

        "transaction_cost":
            TRANSACTION_COST,

        "confusion_matrix": {

            "true_negative":
                int(cm[0][0]),

            "false_positive":
                int(cm[0][1]),

            "false_negative":
                int(cm[1][0]),

            "true_positive":
                int(cm[1][1]),
        },
    }

    return metrics


# ============================================================
# SAVE
# ============================================================

def save_results(
    results,
    metrics
):

    print()
    print("=" * 80)
    print("SAVING BACKTEST RESULTS")
    print("=" * 80)

    results.to_csv(
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
        f"Results saved: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Metrics saved: "
        f"{METRICS_FILE}"
    )


# ============================================================
# SUMMARY
# ============================================================

def display_summary(
    metrics
):

    print()
    print("=" * 80)
    print("STAGE 20.6 BACKTEST RESULTS")
    print("=" * 80)

    print()

    print(
        f"Initial capital       : "
        f"₹{metrics['initial_capital']:,.2f}"
    )

    print(
        f"Final equity          : "
        f"₹{metrics['final_equity']:,.2f}"
    )

    print(
        f"Total return          : "
        f"{metrics['total_return_percent']:.2f}%"
    )

    print(
        f"Maximum drawdown      : "
        f"{metrics['maximum_drawdown_percent']:.2f}%"
    )

    print(
        f"Sharpe ratio          : "
        f"{metrics['sharpe_ratio']:.4f}"
    )

    print(
        f"Profit factor         : "
        f"{metrics['profit_factor']:.4f}"
    )

    print()

    print(
        f"Total trades          : "
        f"{metrics['total_trades']}"
    )

    print(
        f"Winning trades        : "
        f"{metrics['winning_trades']}"
    )

    print(
        f"Losing trades         : "
        f"{metrics['losing_trades']}"
    )

    print(
        f"Win rate              : "
        f"{metrics['win_rate_percent']:.2f}%"
    )

    print()

    print(
        f"BUY trades            : "
        f"{metrics['buy_trades']}"
    )

    print(
        f"SELL trades           : "
        f"{metrics['sell_trades']}"
    )

    print()

    print(
        f"ML accuracy           : "
        f"{metrics['accuracy_percent']:.2f}%"
    )

    print(
        f"ML precision          : "
        f"{metrics['precision_percent']:.2f}%"
    )

    print(
        f"ML recall             : "
        f"{metrics['recall_percent']:.2f}%"
    )

    print()

    print(
        f"Test start            : "
        f"{metrics['test_start']}"
    )

    print(
        f"Test end              : "
        f"{metrics['test_end']}"
    )

    print()
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("STAGE 20.6 - ML BACKTEST")
    print("=" * 80)

    try:

        # ----------------------------------------------------
        # 1. Load exact feature list from Stage 20.2
        # ----------------------------------------------------

        feature_columns = (
            load_feature_columns()
        )

        # ----------------------------------------------------
        # 2. Load training dataset
        # ----------------------------------------------------

        df = load_data()

        # ----------------------------------------------------
        # 3. Prepare
        # ----------------------------------------------------

        df = prepare_data(
            df,
            feature_columns
        )

        # ----------------------------------------------------
        # 4. Split
        # ----------------------------------------------------

        train_df, test_df = (
            split_data(
                df
            )
        )

        # ----------------------------------------------------
        # 5. Train
        # ----------------------------------------------------

        model = train_model(
            train_df,
            feature_columns
        )

        # ----------------------------------------------------
        # 6. Predict
        # ----------------------------------------------------

        results = (
            generate_predictions(
                model,
                test_df,
                feature_columns
            )
        )

        # ----------------------------------------------------
        # 7. Calculate returns
        # ----------------------------------------------------

        results = (
            calculate_trade_returns(
                results
            )
        )

        # ----------------------------------------------------
        # 8. Equity curve
        # ----------------------------------------------------

        results = (
            calculate_equity_curve(
                results
            )
        )

        # ----------------------------------------------------
        # 9. Metrics
        # ----------------------------------------------------

        metrics = (
            calculate_metrics(
                results,
                train_df
            )
        )

        # ----------------------------------------------------
        # 10. Save
        # ----------------------------------------------------

        save_results(
            results,
            metrics
        )

        # ----------------------------------------------------
        # 11. Summary
        # ----------------------------------------------------

        display_summary(
            metrics
        )

        print()
        print("=" * 80)
        print("STAGE 20.6 BACKTEST COMPLETE")
        print("=" * 80)

    except Exception as e:

        print()
        print("=" * 80)
        print("STAGE 20.6 BACKTEST FAILED")
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