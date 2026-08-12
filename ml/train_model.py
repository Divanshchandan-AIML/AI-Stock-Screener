import os
import joblib
import pandas as pd

from api.historical_data import get_historical_data
from ml.features import create_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOLS = [
    "SBIN-EQ",
    "SUZLON-EQ",
    "IRFC-EQ",
    "TATAMOTORS-EQ",
    "RELIANCE-EQ",
    "ITC-EQ"
]

FEATURE_COLUMNS = [
    "SMMA20",
    "SMMA120",
    "RSI14",
    "Price_Change",
    "Price_vs_SMMA20",
    "SMMA_Difference"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():

    all_data = []

    for symbol in SYMBOLS:

        print(f"Loading {symbol}...")

        try:

            df = get_historical_data(symbol)

            if df is None or len(df) < 150:
                print(
                    f"Skipping {symbol}: "
                    "not enough data"
                )
                continue

            feature_df = create_features(df)

            feature_df["Stock"] = symbol.replace(
                "-EQ",
                ""
            )

            all_data.append(feature_df)

            print(
                f"{symbol}: "
                f"{len(feature_df)} rows loaded"
            )

        except Exception as e:

            print(
                f"Error loading {symbol}: {e}"
            )

    if not all_data:
        return None

    combined_df = pd.concat(
        all_data,
        ignore_index=True
    )

    return combined_df


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    print("\n================================")
    print("      ML MODEL TRAINING")
    print("================================\n")

    df = load_training_data()

    if df is None or len(df) == 0:

        print(
            "No training data available."
        )

        return None

    print(
        f"\nTotal training rows: {len(df)}"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X = df[FEATURE_COLUMNS]

    # Target:
    #  1  = BUY
    #  0  = HOLD
    # -1  = SELL

    y = df["Target"]

    # --------------------------------------------------------
    # TIME-BASED TRAIN / TEST SPLIT
    # --------------------------------------------------------

    split_index = int(
        len(df) * 0.80
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows: {len(X_test)}"
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    print("\nTraining Random Forest...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nModel Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    print(
        "\nFeature Importance:"
    )

    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    print(importance)

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    os.makedirs(
        "ml/models",
        exist_ok=True
    )

    model_path = (
        "ml/models/"
        "stock_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nModel saved successfully:"
    )

    print(model_path)

    return model


# ============================================================
# RUN TRAINING
# ============================================================

if __name__ == "__main__":

    train_model()