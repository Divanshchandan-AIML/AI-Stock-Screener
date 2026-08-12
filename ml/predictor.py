import os
import joblib
import pandas as pd

from ml.features import create_features


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "stock_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# PREDICT STOCK
# ============================================================

def predict_stock(df):

    if df is None or len(df) < 120:
        return None

    # Create same features used during training
    features_df = create_features(df)

    if features_df is None or features_df.empty:
        return None

    # Features used by the trained model
    feature_columns = [
    "SMMA20",
    "SMMA120",
    "RSI14",
    "Price_Change",
    "Price_vs_SMMA20",
    "SMMA_Difference"
]

    # Make sure all features exist
    missing = [
        column
        for column in feature_columns
        if column not in features_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    latest = features_df.iloc[[-1]]

    X = latest[feature_columns]

    # Load trained model
    model = load_model()

    # Prediction
    prediction = model.predict(X)[0]

    # Probability if model supports it
    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)[0]

        confidence = float(
            max(probabilities) * 100
        )

    return {
        "Prediction": int(prediction),
        "Confidence": (
            round(confidence, 2)
            if confidence is not None
            else None
        )
    }