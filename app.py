import streamlit as st
import pandas as pd

from api.historical_data import get_historical_data
from indicators.smma import add_smma_indicators
from indicators.crossover import get_crossover_signals
from backtest.backtest import run_all_backtests

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Stock Screener",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# STOCKS
# ============================================================

STOCKS = [
    "SBIN-EQ",
    "SUZLON-EQ",
    "IRFC-EQ",
    "TATAMOTORS-EQ",
    "RELIANCE-EQ",
    "ITC-EQ",
]


# ============================================================
# TITLE
# ============================================================

st.title("📈 AI Stock Screener")

st.markdown(
    """
    **SMMA20 / SMMA120 Crossover Strategy**

    This dashboard uses historical NSE data to calculate:
    - SMMA20
    - SMMA120
    - BUY / SELL crossover signals
    - Current price
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Settings")

selected_stock = st.sidebar.selectbox(
    "Select Stock",
    STOCKS
)

days = st.sidebar.slider(
    "Historical Data",
    min_value=120,
    max_value=500,
    value=250,
    step=10
)

run_button = st.sidebar.button(
    "🔍 Analyze Stock"
)


# ============================================================
# GET STOCK DATA
# ============================================================

@st.cache_data(ttl=300)
def load_stock_data(symbol, days):

    df = get_historical_data(
        symbol,
        days=days
    )

    if df is None or df.empty:
        return None

    df = add_smma_indicators(df)

    df = get_crossover_signals(df)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.sort_values(
            "date"
        )

    return df


# ============================================================
# ANALYZE
# ============================================================

if run_button:

    with st.spinner(
        f"Getting data for {selected_stock}..."
    ):

        try:

            df = load_stock_data(
                selected_stock,
                days
            )

        except Exception as e:

            st.error(
                f"Error getting stock data: {e}"
            )

            df = None


    if df is None or df.empty:

        st.error(
            "No historical data available."
        )

    else:

        # ----------------------------------------------------
        # Latest row
        # ----------------------------------------------------

        latest = df.iloc[-1]

        price = float(
            latest["close"]
        )

        smma20 = float(
            latest["SMMA20"]
        )

        smma120 = float(
            latest["SMMA120"]
        )

        signal = str(
            latest["Signal"]
        )

        # ----------------------------------------------------
        # Signal counts
        # ----------------------------------------------------

        buy_count = int(
            (df["Signal"] == "BUY").sum()
        )

        sell_count = int(
            (df["Signal"] == "SELL").sum()
        )

        no_signal_count = int(
            (df["Signal"] == "NO SIGNAL").sum()
        )

        # ====================================================
        # METRICS
        # ====================================================

        st.subheader(
            f"📊 {selected_stock.replace('-EQ', '')}"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Current Price",
                f"₹{price:.2f}"
            )

        with col2:

            st.metric(
                "SMMA20",
                f"{smma20:.2f}"
            )

        with col3:

            st.metric(
                "SMMA120",
                f"{smma120:.2f}"
            )

        with col4:

            if signal == "BUY":

                st.success(
                    "🟢 BUY"
                )

            elif signal == "SELL":

                st.error(
                    "🔴 SELL"
                )

            else:

                st.warning(
                    "⚪ NO SIGNAL"
                )


        # ====================================================
        # SIGNAL SUMMARY
        # ====================================================

        st.subheader(
            "Signal Summary"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "BUY Signals",
                buy_count
            )

        with c2:

            st.metric(
                "SELL Signals",
                sell_count
            )

        with c3:

            st.metric(
                "NO SIGNAL",
                no_signal_count
            )


        # ====================================================
        # SMMA CHART
        # ====================================================

        st.subheader(
            "SMMA20 vs SMMA120"
        )

        chart_df = df[
            [
                "date",
                "close",
                "SMMA20",
                "SMMA120"
            ]
        ].copy()

        chart_df = chart_df.set_index(
            "date"
        )

        st.line_chart(
            chart_df[
                [
                    "close",
                    "SMMA20",
                    "SMMA120"
                ]
            ]
        )


        # ====================================================
        # CROSSOVER SIGNALS
        # ====================================================

        st.subheader(
            "Crossover Signals"
        )

        signal_df = df[
            df["Signal"].isin(
                ["BUY", "SELL"]
            )
        ].copy()

        if signal_df.empty:

            st.info(
                "No BUY or SELL crossover found in the available data."
            )

        else:

            display_columns = [
                "date",
                "close",
                "SMMA20",
                "SMMA120",
                "Signal"
            ]

            st.dataframe(
                signal_df[
                    display_columns
                ].tail(20),
                use_container_width=True
            )


        # ====================================================
        # LATEST DATA
        # ====================================================

        st.subheader(
            "Latest Data"
        )

        latest_columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "SMMA20",
            "SMMA120",
            "Signal"
        ]

        available_columns = [
            col
            for col in latest_columns
            if col in df.columns
        ]

        st.dataframe(
            df[
                available_columns
            ].tail(20),
            use_container_width=True
        )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "Select a stock from the sidebar and click "
        "'Analyze Stock' to begin."
    )

    st.markdown(
        """
        ### Strategy

        **BUY:** SMMA20 crosses above SMMA120.

        **SELL:** SMMA20 crosses below SMMA120.

        **NO SIGNAL:** No crossover on the latest candle.
        """
    )
        # ============================================================
# BACKTEST
# ============================================================

st.markdown("---")

st.subheader("📊 Strategy Backtest")

st.write(
    "Historical performance of the SMMA20 / SMMA120 crossover strategy."
)

if st.button("🚀 Run Backtest", width="stretch"):

    with st.spinner("Running backtest for all stocks..."):

        try:

            backtest_results = run_all_backtests()

            if backtest_results:

                st.success("✅ Backtest completed successfully.")

                results_df = pd.DataFrame(backtest_results)

                # ------------------------------------------------
                # RESULTS TABLE
                # ------------------------------------------------

                st.markdown("### 📋 Backtest Results")

                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # SUMMARY
                # ------------------------------------------------

                st.markdown("### 📈 Performance Summary")

                # Find column names safely
                profit_col = next(
                    (
                        c for c in results_df.columns
                        if c.lower().replace(" ", "").replace("%", "")
                        == "profit"
                    ),
                    None
                )

                profit_pct_col = next(
                    (
                        c for c in results_df.columns
                        if "profit" in c.lower()
                        and "%" in c
                    ),
                    None
                )

                trades_col = next(
                    (
                        c for c in results_df.columns
                        if "trade" in c.lower()
                    ),
                    None
                )

                # --------------------------------------------
                # Total profit
                # --------------------------------------------

                total_profit = 0.0

                if profit_col:
                    total_profit = pd.to_numeric(
                        results_df[profit_col],
                        errors="coerce"
                    ).fillna(0).sum()

                # --------------------------------------------
                # Total trades
                # --------------------------------------------

                total_trades = 0

                if trades_col:
                    total_trades = int(
                        pd.to_numeric(
                            results_df[trades_col],
                            errors="coerce"
                        ).fillna(0).sum()
                    )

                # --------------------------------------------
                # Best / Worst stock
                # --------------------------------------------

                best_stock = "N/A"
                worst_stock = "N/A"

                if profit_col and "Stock" in results_df.columns:

                    profit_values = pd.to_numeric(
                        results_df[profit_col],
                        errors="coerce"
                    ).fillna(0)

                    best_index = profit_values.idxmax()
                    worst_index = profit_values.idxmin()

                    best_stock = str(
                        results_df.loc[best_index, "Stock"]
                    )

                    worst_stock = str(
                        results_df.loc[worst_index, "Stock"]
                    )

                # --------------------------------------------
                # Metric cards
                # --------------------------------------------

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "💰 Total Profit",
                        f"₹{total_profit:,.2f}"
                    )

                with col2:
                    st.metric(
                        "📊 Total Trades",
                        total_trades
                    )

                with col3:
                    st.metric(
                        "🏆 Best Stock",
                        best_stock
                    )

                with col4:
                    st.metric(
                        "⚠️ Worst Stock",
                        worst_stock
                    )

                # --------------------------------------------
                # Profit chart
                # --------------------------------------------

                if profit_col and "Stock" in results_df.columns:

                    st.markdown("### 💰 Profit by Stock")

                    chart_df = results_df[
                        ["Stock", profit_col]
                    ].copy()

                    chart_df[profit_col] = pd.to_numeric(
                        chart_df[profit_col],
                        errors="coerce"
                    ).fillna(0)

                    chart_df = chart_df.set_index("Stock")

                    st.bar_chart(chart_df)

            else:

                st.warning(
                    "⚠️ No backtest results were returned."
                )

        except Exception as e:

            st.error(
                f"❌ Backtest failed: {e}"
            )