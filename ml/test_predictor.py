from api.historical_data import get_historical_data
from ml.predictor import predict_stock


# ============================================================
# TEST ML PREDICTION
# ============================================================

symbol = "SBIN-EQ"

print("\n==============================")
print("   ML STOCK PREDICTION TEST")
print("==============================\n")

print(f"Loading data for {symbol}...")

df = get_historical_data(symbol)

if df is None:
    print("❌ No historical data available.")
else:

    print(f"Historical rows: {len(df)}")

    result = predict_stock(df)

    if result is None:

        print("❌ Prediction could not be generated.")

    else:

        print("\n==============================")
        print("        PREDICTION")
        print("==============================")

        print(
            "Prediction:",
            result["Prediction"]
        )

        print(
            "Confidence:",
            result["Confidence"],
            "%"
        )

        print("\n==============================")