from api.historical_data import get_historical_data
from indicators.smma import add_smma_indicators
from indicators.crossover import get_crossover_signals


symbol = "SBIN-EQ"

print("=" * 70)
print(f"TESTING CROSSOVER: {symbol}")
print("=" * 70)

# --------------------------------------------------
# 1. Historical data
# --------------------------------------------------

df = get_historical_data(
    symbol,
    days=250
)

if df is None or df.empty:
    print("ERROR: No historical data")
    raise SystemExit

print(f"Historical rows: {len(df)}")


# --------------------------------------------------
# 2. SMMA
# --------------------------------------------------

df = add_smma_indicators(df)

print("\nSMMA indicators added successfully.")


# --------------------------------------------------
# 3. Crossover
# --------------------------------------------------

df = get_crossover_signals(df)

print("Crossover detection completed.")


# --------------------------------------------------
# 4. Signal counts
# --------------------------------------------------

print("\n" + "=" * 70)
print("SIGNAL COUNTS")
print("=" * 70)

print(df["Signal"].value_counts())


# --------------------------------------------------
# 5. Show actual BUY / SELL signals
# --------------------------------------------------

signals = df[df["Signal"].isin(["BUY", "SELL"])]

print("\n" + "=" * 70)
print("CROSSOVER SIGNALS")
print("=" * 70)

if signals.empty:

    print("No BUY or SELL crossover found in available data.")

else:

    print(
        signals[
            [
                "date",
                "close",
                "SMMA20",
                "SMMA120",
                "Signal"
            ]
        ].to_string(index=False)
    )


# --------------------------------------------------
# 6. Latest values
# --------------------------------------------------

latest = df.iloc[-1]

print("\n" + "=" * 70)
print("LATEST")
print("=" * 70)

print(f"Date    : {latest['date']}")
print(f"Price   : ₹{latest['close']:.2f}")
print(f"SMMA20  : {latest['SMMA20']:.2f}")
print(f"SMMA120 : {latest['SMMA120']:.2f}")
print(f"Signal  : {latest['Signal']}")


# --------------------------------------------------
# 7. Difference
# --------------------------------------------------

difference = latest["SMMA20"] - latest["SMMA120"]

print(f"Difference: {difference:.2f}")

if difference > 0:
    print("SMMA20 is ABOVE SMMA120")

elif difference < 0:
    print("SMMA20 is BELOW SMMA120")

else:
    print("SMMA20 and SMMA120 are equal")


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)