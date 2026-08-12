import pandas as pd

from indicators.smma import calculate_smma
from indicators.rsi import calculate_rsi


def create_features(df):
    """
    Create ML features from historical stock data.
    """

    df = df.copy()

    # Make sure data is sorted
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    # ==================================================
    # TECHNICAL INDICATORS
    # ==================================================

    df["SMMA20"] = calculate_smma(
        df["Close"],
        20
    )

    df["SMMA120"] = calculate_smma(
        df["Close"],
        120
    )

    df["RSI14"] = calculate_rsi(
        df["Close"],
        14
    )

    # ==================================================
    # PRICE FEATURES
    # ==================================================

    df["Price_Change"] = (
        df["Close"].pct_change() * 100
    )

    df["Price_vs_SMMA20"] = (
        (df["Close"] - df["SMMA20"])
        / df["SMMA20"]
        * 100
    )

    df["SMMA_Difference"] = (
        (df["SMMA20"] - df["SMMA120"])
        / df["SMMA120"]
        * 100
    )

    # ==================================================
    # FUTURE PRICE CHANGE
    # ==================================================

    # Used only for creating the training target.
    # It looks one day into the future.

    df["Future_Change"] = (
        df["Close"].shift(-1)
        / df["Close"]
        - 1
    ) * 100

    # ==================================================
    # CREATE TARGET
    # ==================================================

    def create_target(change):

        if pd.isna(change):
            return None

        # BUY
        if change > 1:
            return 1

        # SELL
        elif change < -1:
            return -1

        # HOLD
        else:
            return 0

    df["Target"] = (
        df["Future_Change"]
        .apply(create_target)
    )

    # ==================================================
    # REMOVE INVALID ROWS
    # ==================================================

    df = df.dropna().reset_index(
        drop=True
    )

    return df