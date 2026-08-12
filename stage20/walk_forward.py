# ============================================================
# STAGE 20.8 - WALK-FORWARD VALIDATION
# stage20/walk_forward.py
# ============================================================

import os
import json
import copy
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
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

TRAINING_FILE = os.path.join(
    "data",
    "stage20",
    "stage20_training_data.csv"
)

MODEL_FILE = os.path.join(
    "models",
    "stock_classifier.pkl"
)

FEATURE_FILE = os.path.join(
    "models",
    "feature_columns.json"
)

OUTPUT_DIR = os.path.join(
    "data",
    "stage20"
)

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "stage20_walk_forward_results.csv"
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "stage20_walk_forward_metrics.json"
)


# ------------------------------------------------------------
# Walk-forward configuration
# ------------------------------------------------------------

N_FOLDS = 5

MIN_TRAIN_ROWS = 300

TEST_ROWS_PER_FOLD = None


# ============================================================
# DIRECTORY
# ============================================================

def ensure_directories():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# ============================================================
# PRINT HEADER
# ============================================================

def print_header(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():

    print_header(
        "LOADING STAGE 20 TRAINING DATA"
    )

    print(
        f"File: {TRAINING_FILE}"
    )

    if not os.path.exists(
        TRAINING_FILE
    ):

        raise FileNotFoundError(
            f"Training file not found: "
            f"{TRAINING_FILE}"
        )

    df = pd.read_csv(
        TRAINING_FILE
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    print()
    print(
        "Columns:"
    )

    for column in df.columns:

        print(
            f"  - {column}"
        )

    return df


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print_header(
        "LOADING TRAINED ML MODEL"
    )

    print(
        f"Model: {MODEL_FILE}"
    )

    if not os.path.exists(
        MODEL_FILE
    ):

        raise FileNotFoundError(
            f"Model file not found: "
            f"{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "Model loaded successfully."
    )

    print(
        f"Model type: "
        f"{type(model).__name__}"
    )

    return model


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

def load_feature_columns():

    print_header(
        "LOADING MODEL FEATURE LIST"
    )

    print(
        f"Feature file: {FEATURE_FILE}"
    )

    if not os.path.exists(
        FEATURE_FILE
    ):

        raise FileNotFoundError(
            f"Feature file not found: "
            f"{FEATURE_FILE}"
        )

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(
            file
        )

    # --------------------------------------------------------
    # Support:
    #
    # ["feature1", "feature2"]
    #
    # OR
    #
    # {"features": [...]}
    #
    # OR
    #
    # {"feature_columns": [...]}
    #
    # OR
    #
    # {"columns": [...]}
    # --------------------------------------------------------

    if isinstance(
        data,
        list
    ):

        features = data

    elif isinstance(
        data,
        dict
    ):

        if "features" in data:

            features = data[
                "features"
            ]

        elif "feature_columns" in data:

            features = data[
                "feature_columns"
            ]

        elif "columns" in data:

            features = data[
                "columns"
            ]

        else:

            raise ValueError(
                "feature_columns.json does not "
                "contain a recognized feature list."
            )

    else:

        raise ValueError(
            "Invalid feature_columns.json format."
        )

    if not features:

        raise ValueError(
            "Feature list is empty."
        )

    features = [
        str(feature)
        for feature in features
    ]

    # --------------------------------------------------------
    # Remove duplicates while preserving order
    # --------------------------------------------------------

    features = list(
        dict.fromkeys(
            features
        )
    )

    print()
    print(
        f"Features loaded: "
        f"{len(features)}"
    )

    for number, feature in enumerate(
        features,
        start=1
    ):

        print(
            f"  {number:02d}. {feature}"
        )

    return features


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    df,
    feature_columns
):

    print_header(
        "PREPARING WALK-FORWARD DATA"
    )

    data = df.copy()

    # --------------------------------------------------------
    # Required base columns
    # --------------------------------------------------------

    required_columns = [
        "date",
        "target",
    ]

    missing_base = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_base:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_base
            )
        )

    # --------------------------------------------------------
    # Check model features
    # --------------------------------------------------------

    missing_features = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    if missing_features:

        print()
        print(
            "ERROR: Model feature columns "
            "are missing from training data:"
        )

        for column in missing_features:

            print(
                f"  - {column}"
            )

        raise ValueError(
            "Missing feature columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_features
            )
        )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    data["target"] = pd.to_numeric(
        data["target"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Feature values
    # --------------------------------------------------------

    for column in feature_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    before = len(data)

    data = data.dropna(
        subset=[
            "date",
            "target",
            *feature_columns,
        ]
    )

    removed = (
        before
        -
        len(data)
    )

    print()
    print(
        f"Rows removed due to "
        f"missing values: {removed}"
    )

    # --------------------------------------------------------
    # Target must be binary
    # --------------------------------------------------------

    data = data[
        data["target"].isin(
            [0, 1]
        )
    ]

    data["target"] = (
        data["target"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    data = (
        data
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Duplicate dates are allowed because multiple stocks
    # can have the same trading date.
    # --------------------------------------------------------

    print(
        f"Prepared rows: {len(data)}"
    )

    if len(data) == 0:

        raise ValueError(
            "No valid rows remain after preparation."
        )

    print()
    print(
        f"Start date: {data['date'].min()}"
    )

    print(
        f"End date  : {data['date'].max()}"
    )

    print()
    print(
        "Target distribution:"
    )

    print(
        data["target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return data


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(
    base_model
):

    # --------------------------------------------------------
    # Prefer sklearn clone.
    # This preserves the model configuration while creating
    # a fresh unfitted estimator.
    # --------------------------------------------------------

    try:

        model = clone(
            base_model
        )

        return model

    except Exception:

        # ----------------------------------------------------
        # Fallback for models that cannot be cloned.
        # ----------------------------------------------------

        return copy.deepcopy(
            base_model
        )


# ============================================================
# GET TIME FOLDS
# ============================================================

def create_walk_forward_folds(
    data
):

    print_header(
        "CREATING WALK-FORWARD FOLDS"
    )

    unique_dates = (
        data["date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(
            drop=True
        )
    )

    number_of_dates = len(
        unique_dates
    )

    print(
        f"Unique dates: "
        f"{number_of_dates}"
    )

    if number_of_dates < 2:

        raise ValueError(
            "Not enough unique dates "
            "for walk-forward validation."
        )

    # --------------------------------------------------------
    # Automatically determine test size.
    # --------------------------------------------------------

    if TEST_ROWS_PER_FOLD is not None:

        test_dates_count = (
            TEST_ROWS_PER_FOLD
        )

    else:

        available_dates = (
            number_of_dates
            -
            1
        )

        test_dates_count = max(
            1,
            available_dates
            //
            N_FOLDS
        )

    folds = []

    for fold_number in range(
        1,
        N_FOLDS + 1
    ):

        test_start_position = (
            number_of_dates
            -
            (
                N_FOLDS
                -
                fold_number
                +
                1
            )
            *
            test_dates_count
        )

        test_end_position = min(
            number_of_dates,
            test_start_position
            +
            test_dates_count
        )

        if (
            test_start_position
            <= 0
        ):

            continue

        if (
            test_start_position
            >= number_of_dates
        ):

            continue

        train_dates = (
            unique_dates.iloc[
                :test_start_position
            ]
        )

        test_dates = (
            unique_dates.iloc[
                test_start_position:
                test_end_position
            ]
        )

        if len(
            train_dates
        ) == 0:

            continue

        if len(
            test_dates
        ) == 0:

            continue

        train_start = (
            train_dates.iloc[0]
        )

        train_end = (
            train_dates.iloc[-1]
        )

        test_start = (
            test_dates.iloc[0]
        )

        test_end = (
            test_dates.iloc[-1]
        )

        train_mask = (
            data["date"]
            <= train_end
        )

        test_mask = (
            (
                data["date"]
                >= test_start
            )
            &
            (
                data["date"]
                <= test_end
            )
        )

        train_indices = (
            data.index[
                train_mask
            ]
        )

        test_indices = (
            data.index[
                test_mask
            ]
        )

        if len(
            train_indices
        ) < MIN_TRAIN_ROWS:

            print()
            print(
                f"Fold {fold_number} skipped:"
            )

            print(
                f"  Training rows: "
                f"{len(train_indices)}"
            )

            print(
                f"  Minimum required: "
                f"{MIN_TRAIN_ROWS}"
            )

            continue

        folds.append(
            {
                "fold": fold_number,
                "train_indices": train_indices,
                "test_indices": test_indices,
                "train_start": str(
                    train_start
                ),
                "train_end": str(
                    train_end
                ),
                "test_start": str(
                    test_start
                ),
                "test_end": str(
                    test_end
                ),
            }
        )

    if not folds:

        raise ValueError(
            "No valid walk-forward folds "
            "could be created."
        )

    print()

    for fold in folds:

        print(
            f"Fold {fold['fold']}:"
        )

        print(
            f"  Train: "
            f"{fold['train_start']} "
            f"-> "
            f"{fold['train_end']}"
        )

        print(
            f"  Test : "
            f"{fold['test_start']} "
            f"-> "
            f"{fold['test_end']}"
        )

        print(
            f"  Train rows: "
            f"{len(fold['train_indices'])}"
        )

        print(
            f"  Test rows : "
            f"{len(fold['test_indices'])}"
        )

    return folds


# ============================================================
# GET PROBABILITY
# ============================================================

def get_probability(
    model,
    X
):

    # --------------------------------------------------------
    # Preferred: predict_proba
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model
            .predict_proba(X)
        )

        if probabilities.ndim == 2:

            # ------------------------------------------------
            # Binary classifier:
            # column 1 = probability of class 1
            # ------------------------------------------------

            if probabilities.shape[1] >= 2:

                return probabilities[
                    :,
                    1
                ]

            return probabilities[
                :,
                0
            ]

    # --------------------------------------------------------
    # Decision function fallback
    # --------------------------------------------------------

    if hasattr(
        model,
        "decision_function"
    ):

        decision = (
            model
            .decision_function(X)
        )

        decision = np.asarray(
            decision,
            dtype=float
        )

        # Logistic conversion
        probability = (
            1
            /
            (
                1
                +
                np.exp(
                    -np.clip(
                        decision,
                        -50,
                        50
                    )
                )
            )
        )

        return probability

    # --------------------------------------------------------
    # No probability available
    # --------------------------------------------------------

    return None


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def run_walk_forward(
    data,
    base_model,
    feature_columns,
    folds
):

    print_header(
        "STARTING WALK-FORWARD VALIDATION"
    )

    all_results = []

    fold_metrics = []

    for fold in folds:

        fold_number = (
            fold["fold"]
        )

        print()
        print(
            "=" * 80
        )

        print(
            f"FOLD {fold_number}"
        )

        print(
            "=" * 80
        )

        train_indices = (
            fold["train_indices"]
        )

        test_indices = (
            fold["test_indices"]
        )

        train_df = data.loc[
            train_indices
        ]

        test_df = data.loc[
            test_indices
        ]

        X_train = (
            train_df[
                feature_columns
            ]
        )

        y_train = (
            train_df[
                "target"
            ]
        )

        X_test = (
            test_df[
                feature_columns
            ]
        )

        y_test = (
            test_df[
                "target"
            ]
        )

        print()
        print(
            f"Training rows: "
            f"{len(X_train)}"
        )

        print(
            f"Testing rows : "
            f"{len(X_test)}"
        )

        print(
            f"Train period : "
            f"{fold['train_start']} "
            f"-> "
            f"{fold['train_end']}"
        )

        print(
            f"Test period  : "
            f"{fold['test_start']} "
            f"-> "
            f"{fold['test_end']}"
        )

        # ----------------------------------------------------
        # Check both target classes
        # ----------------------------------------------------

        unique_train_targets = (
            y_train
            .dropna()
            .unique()
        )

        if len(
            unique_train_targets
        ) < 2:

            print(
                "⚠️ Fold skipped: "
                "training data contains "
                "only one target class."
            )

            continue

        # ----------------------------------------------------
        # Fresh model
        # ----------------------------------------------------

        model = create_model(
            base_model
        )

        print()
        print(
            "Training model..."
        )

        model.fit(
            X_train,
            y_train
        )

        print(
            "Model trained."
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predictions = (
            model.predict(
                X_test
            )
        )

        predictions = np.asarray(
            predictions
        ).astype(int)

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability_up = (
            get_probability(
                model,
                X_test
            )
        )

        if probability_up is None:

            probability_up = np.full(
                len(X_test),
                np.nan
            )

        probability_up = np.asarray(
            probability_up,
            dtype=float
        )

        probability_down = (
            1
            -
            probability_up
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = (
            accuracy_score(
                y_test,
                predictions
            )
        )

        precision = (
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        )

        recall = (
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        )

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        print()
        print(
            f"Accuracy : "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"Precision: "
            f"{precision * 100:.2f}%"
        )

        print(
            f"Recall   : "
            f"{recall * 100:.2f}%"
        )

        print()
        print(
            "Confusion matrix:"
        )

        print(
            cm
        )

        # ----------------------------------------------------
        # Build result dataframe
        # ----------------------------------------------------

        fold_result = test_df[
            [
                column
                for column in [
                    "date",
                    "Symbol",
                    "close",
                    "future_return",
                    "target",
                ]
                if column in test_df.columns
            ]
        ].copy()

        fold_result[
            "prediction"
        ] = predictions

        fold_result[
            "probability_up"
        ] = probability_up

        fold_result[
            "probability_down"
        ] = probability_down

        fold_result[
            "fold"
        ] = fold_number

        fold_result[
            "train_start"
        ] = fold[
            "train_start"
        ]

        fold_result[
            "train_end"
        ] = fold[
            "train_end"
        ]

        fold_result[
            "test_start"
        ] = fold[
            "test_start"
        ]

        fold_result[
            "test_end"
        ] = fold[
            "test_end"
        ]

        all_results.append(
            fold_result
        )

        fold_metrics.append(
            {
                "fold": fold_number,
                "train_rows": int(
                    len(X_train)
                ),
                "test_rows": int(
                    len(X_test)
                ),
                "train_start": fold[
                    "train_start"
                ],
                "train_end": fold[
                    "train_end"
                ],
                "test_start": fold[
                    "test_start"
                ],
                "test_end": fold[
                    "test_end"
                ],
                "accuracy": float(
                    accuracy
                ),
                "precision": float(
                    precision
                ),
                "recall": float(
                    recall
                ),
            }
        )

    if not all_results:

        raise ValueError(
            "Walk-forward validation "
            "produced no results."
        )

    results = pd.concat(
        all_results,
        ignore_index=True
    )

    return results, fold_metrics


# ============================================================
# CALCULATE OVERALL METRICS
# ============================================================

def calculate_metrics(
    results,
    fold_metrics
):

    y_true = (
        results["target"]
        .astype(int)
    )

    y_pred = (
        results["prediction"]
        .astype(int)
    )

    accuracy = (
        accuracy_score(
            y_true,
            y_pred
        )
    )

    precision = (
        precision_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    recall = (
        recall_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Directional return
    #
    # Prediction 1 = long / positive direction
    # Prediction 0 = negative direction
    #
    # This is a simple validation statistic and is NOT a
    # transaction-cost-aware trading backtest.
    # --------------------------------------------------------

    if (
        "future_return"
        in results.columns
    ):

        future_returns = pd.to_numeric(
            results[
                "future_return"
            ],
            errors="coerce"
        )

        strategy_returns = np.where(
            y_pred == 1,
            future_returns,
            -future_returns
        )

        strategy_returns = pd.Series(
            strategy_returns
        ).dropna()

        total_return = (
            (
                1
                +
                strategy_returns
            )
            .prod()
            -
            1
        )

        if len(
            strategy_returns
        ) > 0:

            win_rate = (
                strategy_returns
                > 0
            ).mean()

        else:

            win_rate = 0.0

    else:

        total_return = 0.0
        win_rate = 0.0

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    if (
        "strategy_returns"
        not in locals()
        or
        len(strategy_returns) == 0
    ):

        max_drawdown = 0.0

    else:

        equity = (
            1
            +
            strategy_returns
        ).cumprod()

        running_max = (
            equity
            .cummax()
        )

        drawdown = (
            equity
            /
            running_max
            -
            1
        )

        max_drawdown = (
            drawdown.min()
        )

    # --------------------------------------------------------
    # Sharpe-like ratio
    # --------------------------------------------------------

    if (
        "strategy_returns"
        in locals()
        and
        len(strategy_returns) > 1
        and
        strategy_returns.std() != 0
    ):

        sharpe = (
            strategy_returns.mean()
            /
            strategy_returns.std()
        ) * np.sqrt(
            252
        )

    else:

        sharpe = 0.0

    metrics = {

        "rows_tested": int(
            len(results)
        ),

        "folds_completed": int(
            len(fold_metrics)
        ),

        "ml_accuracy": float(
            accuracy
        ),

        "ml_precision": float(
            precision
        ),

        "ml_recall": float(
            recall
        ),

        "win_rate": float(
            win_rate
        ),

        "total_return": float(
            total_return
        ),

        "maximum_drawdown": float(
            max_drawdown
        ),

        "sharpe_ratio": float(
            sharpe
        ),

        "fold_metrics": fold_metrics,
    }

    return metrics


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    metrics
):

    print_header(
        "SAVING WALK-FORWARD RESULTS"
    )

    results.to_csv(
        RESULT_FILE,
        index=False
    )

    print(
        f"Results saved:"
    )

    print(
        f"  {RESULT_FILE}"
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
        f"Metrics saved:"
    )

    print(
        f"  {METRICS_FILE}"
    )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_summary(
    metrics
):

    print_header(
        "STAGE 20.8 WALK-FORWARD SUMMARY"
    )

    print(
        f"Rows tested       : "
        f"{metrics['rows_tested']}"
    )

    print(
        f"Folds completed   : "
        f"{metrics['folds_completed']}"
    )

    print()

    print(
        f"ML Accuracy       : "
        f"{metrics['ml_accuracy'] * 100:.2f}%"
    )

    print(
        f"ML Precision      : "
        f"{metrics['ml_precision'] * 100:.2f}%"
    )

    print(
        f"ML Recall         : "
        f"{metrics['ml_recall'] * 100:.2f}%"
    )

    print()

    print(
        f"Win Rate          : "
        f"{metrics['win_rate'] * 100:.2f}%"
    )

    print(
        f"Total Return      : "
        f"{metrics['total_return'] * 100:.2f}%"
    )

    print(
        f"Maximum Drawdown  : "
        f"{metrics['maximum_drawdown'] * 100:.2f}%"
    )

    print(
        f"Sharpe Ratio      : "
        f"{metrics['sharpe_ratio']:.4f}"
    )

    # --------------------------------------------------------
    # Simple validation decision
    # --------------------------------------------------------

    if (
        metrics["folds_completed"] >= 2
        and
        metrics["ml_accuracy"] >= 0.50
    ):

        decision = "PASS"

    else:

        decision = "REVIEW"

    print()

    print(
        f"FINAL DECISION    : "
        f"{decision}"
    )

    return decision


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("STAGE 20.8 - WALK-FORWARD VALIDATION")
    print("=" * 80)

    try:

        ensure_directories()

        # ----------------------------------------------------
        # Load data
        # ----------------------------------------------------

        data = (
            load_training_data()
        )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        base_model = (
            load_model()
        )

        # ----------------------------------------------------
        # Load EXACT feature list used during training
        # ----------------------------------------------------

        feature_columns = (
            load_feature_columns()
        )

        # ----------------------------------------------------
        # Prepare data
        # ----------------------------------------------------

        data = prepare_data(
            data,
            feature_columns
        )

        # ----------------------------------------------------
        # Create folds
        # ----------------------------------------------------

        folds = (
            create_walk_forward_folds(
                data
            )
        )

        # ----------------------------------------------------
        # Run validation
        # ----------------------------------------------------

        results, fold_metrics = (
            run_walk_forward(
                data,
                base_model,
                feature_columns,
                folds
            )
        )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        metrics = (
            calculate_metrics(
                results,
                fold_metrics
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_results(
            results,
            metrics
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        decision = (
            display_summary(
                metrics
            )
        )

        print()
        print("=" * 80)
        print("STAGE 20.8 WALK-FORWARD VALIDATION COMPLETE")
        print("=" * 80)

        print()
        print(
            f"Results : {RESULT_FILE}"
        )

        print(
            f"Metrics : {METRICS_FILE}"
        )

        print()
        print(
            f"Decision: {decision}"
        )

        return results

    except Exception as e:

        print()
        print("=" * 80)
        print("STAGE 20.8 WALK-FORWARD VALIDATION FAILED")
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