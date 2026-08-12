import pandas as pd
import plotly.graph_objects as go

from indicators.smma import calculate_smma


def prepare_chart_data(df):
    """
    Prepare historical price data with SMMA20,
    SMMA120 and crossover signals.
    """

    if df is None or df.empty:
        return None

    df = df.copy()

    # ---------------------------------------------------------
    # Make sure Date exists
    # ---------------------------------------------------------

    if "Date" not in df.columns:
        return None

    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove invalid dates
    df = df.dropna(subset=["Date"])

    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    # ---------------------------------------------------------
    # Make sure Close exists
    # ---------------------------------------------------------

    if "Close" not in df.columns:
        return None

    # Convert Close to numeric
    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df = df.dropna(subset=["Close"])

    # ---------------------------------------------------------
    # Calculate SMMA
    # ---------------------------------------------------------

    df["SMMA20"] = calculate_smma(
        df["Close"],
        20
    )

    df["SMMA120"] = calculate_smma(
        df["Close"],
        120
    )

    # ---------------------------------------------------------
    # Previous values
    # ---------------------------------------------------------

    df["Previous_SMMA20"] = df["SMMA20"].shift(1)

    df["Previous_SMMA120"] = df["SMMA120"].shift(1)

    # ---------------------------------------------------------
    # BUY CROSSOVER
    #
    # Previous:
    # SMMA20 <= SMMA120
    #
    # Current:
    # SMMA20 > SMMA120
    # ---------------------------------------------------------

    df["BUY_CROSSOVER"] = (
        (df["Previous_SMMA20"] <= df["Previous_SMMA120"])
        &
        (df["SMMA20"] > df["SMMA120"])
    )

    # ---------------------------------------------------------
    # SELL CROSSOVER
    #
    # Previous:
    # SMMA20 >= SMMA120
    #
    # Current:
    # SMMA20 < SMMA120
    # ---------------------------------------------------------

    df["SELL_CROSSOVER"] = (
        (df["Previous_SMMA20"] >= df["Previous_SMMA120"])
        &
        (df["SMMA20"] < df["SMMA120"])
    )

    return df


def create_historical_chart(df, symbol):
    """
    Create interactive Plotly historical price chart
    with SMMA20, SMMA120, BUY and SELL markers.
    """

    df = prepare_chart_data(df)

    if df is None or df.empty:
        return None

    # ---------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------

    fig = go.Figure()

    # ---------------------------------------------------------
    # PRICE
    # ---------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Price",
            line=dict(
                width=2
            )
        )
    )

    # ---------------------------------------------------------
    # SMMA20
    # ---------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["SMMA20"],
            mode="lines",
            name="SMMA20",
            line=dict(
                width=2
            )
        )
    )

    # ---------------------------------------------------------
    # SMMA120
    # ---------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["SMMA120"],
            mode="lines",
            name="SMMA120",
            line=dict(
                width=2
            )
        )
    )

    # ---------------------------------------------------------
    # BUY MARKERS
    # ---------------------------------------------------------

    buy_data = df[
        df["BUY_CROSSOVER"]
    ]

    if not buy_data.empty:

        fig.add_trace(
            go.Scatter(
                x=buy_data["Date"],
                y=buy_data["Close"],
                mode="markers",
                name="BUY Crossover",
                marker=dict(
                    symbol="triangle-up",
                    size=12
                ),
                text=[
                    f"BUY<br>"
                    f"Price: ₹{price:.2f}<br>"
                    f"SMMA20: {smma20:.2f}<br>"
                    f"SMMA120: {smma120:.2f}"
                    for price, smma20, smma120
                    in zip(
                        buy_data["Close"],
                        buy_data["SMMA20"],
                        buy_data["SMMA120"]
                    )
                ],
                hovertemplate=(
                    "%{text}"
                    "<extra></extra>"
                )
            )
        )

    # ---------------------------------------------------------
    # SELL MARKERS
    # ---------------------------------------------------------

    sell_data = df[
        df["SELL_CROSSOVER"]
    ]

    if not sell_data.empty:

        fig.add_trace(
            go.Scatter(
                x=sell_data["Date"],
                y=sell_data["Close"],
                mode="markers",
                name="SELL Crossover",
                marker=dict(
                    symbol="triangle-down",
                    size=12
                ),
                text=[
                    f"SELL<br>"
                    f"Price: ₹{price:.2f}<br>"
                    f"SMMA20: {smma20:.2f}<br>"
                    f"SMMA120: {smma120:.2f}"
                    for price, smma20, smma120
                    in zip(
                        sell_data["Close"],
                        sell_data["SMMA20"],
                        sell_data["SMMA120"]
                    )
                ],
                hovertemplate=(
                    "%{text}"
                    "<extra></extra>"
                )
            )
        )

    # ---------------------------------------------------------
    # CHART LAYOUT
    # ---------------------------------------------------------

    fig.update_layout(
        title=f"{symbol} - Historical Price & SMMA",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode="x unified",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    return fig, df