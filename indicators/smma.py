import pandas as pd


def calculate_smma(series, period):
    """
    Calculate Smoothed Moving Average (SMMA).

    Parameters:
        series: pandas Series containing prices
        period: SMMA period

    Returns:
        pandas Series containing SMMA values
    """

    series = pd.Series(series, dtype="float64")

    smma = pd.Series(index=series.index, dtype="float64")

    # Need enough data
    if len(series) < period:
        return smma

    # First SMMA = SMA
    smma.iloc[period - 1] = (
        series.iloc[:period].mean()
    )

    # Remaining SMMA values
    for i in range(period, len(series)):

        smma.iloc[i] = (
            (smma.iloc[i - 1] * (period - 1))
            + series.iloc[i]
        ) / period

    return smma


def add_smma_indicators(df):

    """
    Add SMMA20 and SMMA120 columns.

    DataFrame must contain a 'close' column.
    """

    if "close" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'close' column"
        )

    df = df.copy()

    df["SMMA20"] = calculate_smma(
        df["close"],
        20
    )

    df["SMMA120"] = calculate_smma(
        df["close"],
        120
    )

    return df