import pandas as pd

from api.historical_data import get_historical_data
from indicators.smma import add_smma_indicators
from indicators.crossover import get_crossover_signals


# ============================================================
# BACKTEST SETTINGS
# ============================================================

INITIAL_CAPITAL = 100000

# Estimated transaction cost per side
# 0.0005 = 0.05%
TRANSACTION_COST = 0.0005

# Estimated slippage per side
# 0.0005 = 0.05%
SLIPPAGE = 0.0005


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
# PREPARE DATA
# ============================================================

def prepare_data(symbol):

    print(f"\nPreparing {symbol}...")

    # ========================================================
    # GET HISTORICAL DATA
    # ========================================================

    df = get_historical_data(
        symbol,
        days=500
    )

    if df is None or df.empty:

        print(
            f"{symbol}: No historical data"
        )

        return None

    print(
        f"Historical rows: {len(df)}"
    )

    # ========================================================
    # CONVERT DATE
    # ========================================================

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["date"]
        ).copy()

        # IMPORTANT:
        # Historical data must be oldest -> newest
        # BEFORE calculating indicators.

        df = df.sort_values(
            "date",
            ascending=True
        ).reset_index(
            drop=True
        )

    # ========================================================
    # REMOVE DUPLICATE DATES
    # ========================================================

    if "date" in df.columns:

        df = df.drop_duplicates(
            subset=["date"],
            keep="last"
        ).reset_index(
            drop=True
        )

    # ========================================================
    # ADD SMMA INDICATORS
    # ========================================================

    df = add_smma_indicators(df)

    print(
        "SMMA indicators added successfully."
    )

    # ========================================================
    # DETECT CROSSOVER SIGNALS
    # ========================================================

    df = get_crossover_signals(df)

    print(
        "Crossover detection completed."
    )

    # ========================================================
    # REMOVE INVALID ROWS
    # ========================================================

    df = df.dropna(
        subset=[
            "SMMA20",
            "SMMA120",
            "close"
        ]
    ).copy()

    # ========================================================
    # CLEAN SIGNAL COLUMN
    # ========================================================

    if "Signal" not in df.columns:

        df["Signal"] = "NO SIGNAL"

    df["Signal"] = (
        df["Signal"]
        .fillna("NO SIGNAL")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # ========================================================
    # FINAL CHRONOLOGICAL ORDER
    # ========================================================

    if "date" in df.columns:

        df = df.sort_values(
            "date",
            ascending=True
        ).reset_index(
            drop=True
        )

    # ========================================================
    # INFORMATION
    # ========================================================

    print(
        f"Usable rows: {len(df)}"
    )

    print(
        f"BUY signals: "
        f"{(df['Signal'] == 'BUY').sum()}"
    )

    print(
        f"SELL signals: "
        f"{(df['Signal'] == 'SELL').sum()}"
    )

    print(
        f"NO SIGNAL: "
        f"{(df['Signal'] == 'NO SIGNAL').sum()}"
    )

    return df


# ============================================================
# RUN BACKTEST
# ============================================================

def run_backtest(symbol):

    df = prepare_data(symbol)

    # --------------------------------------------------------
    # EMPTY DATA RESULT
    # --------------------------------------------------------

    if df is None or df.empty:

        return {
            "Stock": symbol.replace("-EQ", ""),
            "Profit": 0.0,
            "Profit %": 0.0,
            "Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "Win Rate %": 0.0,
            "Drawdown %": 0.0,
            "Costs": 0.0,
            "Buy & Hold %": 0.0,
        }

    # ========================================================
    # INITIAL STATE
    # ========================================================

    cash = float(INITIAL_CAPITAL)

    shares = 0

    entry_price = None

    trades = []

    equity_curve = []

    total_costs = 0.0

    # ========================================================
    # BUY & HOLD BENCHMARK
    # ========================================================

    first_price = float(
        df.iloc[0]["close"]
    )

    last_price = float(
        df.iloc[-1]["close"]
    )

    if first_price > 0:

        buy_hold_return = (
            (last_price - first_price)
            / first_price
        ) * 100

    else:

        buy_hold_return = 0.0

    # ========================================================
    # PROCESS EVERY CANDLE
    # ========================================================

    for i in range(len(df)):

        row = df.iloc[i]

        # ----------------------------------------------------
        # MARKET PRICE
        # ----------------------------------------------------

        try:

            market_price = float(
                row["close"]
            )

        except Exception:

            continue

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        signal = str(
            row["Signal"]
        ).upper().strip()

        # ====================================================
        # BUY
        # ====================================================

        if signal == "BUY" and shares == 0:

            # ------------------------------------------------
            # APPLY SLIPPAGE
            # ------------------------------------------------
            # Actual buying price is slightly higher.

            execution_price = (
                market_price
                * (1 + SLIPPAGE)
            )

            # ------------------------------------------------
            # CALCULATE TRANSACTION COST
            # ------------------------------------------------

            cost_per_share = (
                execution_price
                * TRANSACTION_COST
            )

            total_price_per_share = (
                execution_price
                + cost_per_share
            )

            # ------------------------------------------------
            # CALCULATE NUMBER OF SHARES
            # ------------------------------------------------

            if total_price_per_share > 0:

                shares = int(
                    cash
                    // total_price_per_share
                )

            # ------------------------------------------------
            # EXECUTE BUY
            # ------------------------------------------------

            if shares > 0:

                trade_value = (
                    shares
                    * execution_price
                )

                transaction_cost = (
                    trade_value
                    * TRANSACTION_COST
                )

                total_buy_cost = (
                    trade_value
                    + transaction_cost
                )

                cash -= total_buy_cost

                total_costs += (
                    transaction_cost
                )

                entry_price = (
                    execution_price
                )

                print(
                    f"{symbol}: BUY "
                    f"{shares} shares @ "
                    f"₹{execution_price:.2f} "
                    f"Cost: "
                    f"₹{transaction_cost:.2f}"
                )

        # ====================================================
        # SELL
        # ====================================================

        elif signal == "SELL" and shares > 0:

            # ------------------------------------------------
            # APPLY SLIPPAGE
            # ------------------------------------------------
            # Actual selling price is slightly lower.

            execution_price = (
                market_price
                * (1 - SLIPPAGE)
            )

            # ------------------------------------------------
            # SELL VALUE
            # ------------------------------------------------

            sell_value = (
                shares
                * execution_price
            )

            # ------------------------------------------------
            # TRANSACTION COST
            # ------------------------------------------------

            transaction_cost = (
                sell_value
                * TRANSACTION_COST
            )

            net_sell_value = (
                sell_value
                - transaction_cost
            )

            # ------------------------------------------------
            # UPDATE CASH
            # ------------------------------------------------

            cash += net_sell_value

            total_costs += (
                transaction_cost
            )

            # ------------------------------------------------
            # TRADE PROFIT
            # ------------------------------------------------

            gross_profit = (
                execution_price
                - entry_price
            ) * shares

            profit = (
                gross_profit
                - transaction_cost
            )

            # ------------------------------------------------
            # SAVE TRADE
            # ------------------------------------------------

            trades.append({

                "entry_price":
                    entry_price,

                "exit_price":
                    execution_price,

                "shares":
                    shares,

                "profit":
                    profit,

                "cost":
                    transaction_cost,
            })

            print(
                f"{symbol}: SELL "
                f"{shares} shares @ "
                f"₹{execution_price:.2f} "
                f"Profit: "
                f"₹{profit:.2f} "
                f"Cost: "
                f"₹{transaction_cost:.2f}"
            )

            # ------------------------------------------------
            # RESET POSITION
            # ------------------------------------------------

            shares = 0

            entry_price = None

        # ====================================================
        # CALCULATE CURRENT EQUITY
        # ====================================================

        current_equity = (
            cash
            + (
                shares
                * market_price
            )
        )

        equity_curve.append(
            current_equity
        )

    # ========================================================
    # CLOSE OPEN POSITION
    # ========================================================

    if shares > 0:

        market_price = float(
            df.iloc[-1]["close"]
        )

        # ----------------------------------------------------
        # APPLY SELL SLIPPAGE
        # ----------------------------------------------------

        execution_price = (
            market_price
            * (1 - SLIPPAGE)
        )

        # ----------------------------------------------------
        # SELL VALUE
        # ----------------------------------------------------

        sell_value = (
            shares
            * execution_price
        )

        # ----------------------------------------------------
        # TRANSACTION COST
        # ----------------------------------------------------

        transaction_cost = (
            sell_value
            * TRANSACTION_COST
        )

        # ----------------------------------------------------
        # UPDATE CASH
        # ----------------------------------------------------

        cash += (
            sell_value
            - transaction_cost
        )

        total_costs += (
            transaction_cost
        )

        # ----------------------------------------------------
        # FINAL TRADE PROFIT
        # ----------------------------------------------------

        gross_profit = (
            execution_price
            - entry_price
        ) * shares

        profit = (
            gross_profit
            - transaction_cost
        )

        trades.append({

            "entry_price":
                entry_price,

            "exit_price":
                execution_price,

            "shares":
                shares,

            "profit":
                profit,

            "cost":
                transaction_cost,
        })

        print(
            f"{symbol}: FINAL SELL "
            f"{shares} shares @ "
            f"₹{execution_price:.2f} "
            f"Profit: "
            f"₹{profit:.2f}"
        )

        shares = 0

        entry_price = None

    # ========================================================
    # FINAL CAPITAL
    # ========================================================

    final_capital = float(
        cash
    )

    # ========================================================
    # TOTAL PROFIT
    # ========================================================

    profit = (
        final_capital
        - INITIAL_CAPITAL
    )

    # ========================================================
    # PROFIT %
    # ========================================================

    profit_percent = (
        profit
        / INITIAL_CAPITAL
    ) * 100

    # ========================================================
    # TRADE STATISTICS
    # ========================================================

    total_trades = len(
        trades
    )

    # --------------------------------------------------------
    # WINS
    # --------------------------------------------------------

    wins = sum(
        1
        for trade in trades
        if trade["profit"] > 0
    )

    # --------------------------------------------------------
    # LOSSES
    # --------------------------------------------------------

    losses = sum(
        1
        for trade in trades
        if trade["profit"] <= 0
    )

    # --------------------------------------------------------
    # WIN RATE
    # --------------------------------------------------------

    if total_trades > 0:

        win_rate = (
            wins
            / total_trades
        ) * 100

    else:

        win_rate = 0.0

    # ========================================================
    # MAXIMUM DRAWDOWN
    # ========================================================

    max_drawdown = 0.0

    if equity_curve:

        equity_series = pd.Series(
            equity_curve
        )

        # Highest equity reached
        running_max = (
            equity_series.cummax()
        )

        # Drawdown percentage
        drawdown = (
            (
                equity_series
                - running_max
            )
            / running_max
        ) * 100

        # Maximum drawdown
        max_drawdown = abs(
            float(drawdown.min())
        )

    # ========================================================
    # PRINT STOCK RESULT
    # ========================================================

    print("\n")
    print(
        "-" * 70
    )

    print(
        f"{symbol} BACKTEST RESULT"
    )

    print(
        "-" * 70
    )

    print(
        f"Initial Capital : "
        f"₹{INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Final Capital   : "
        f"₹{final_capital:,.2f}"
    )

    print(
        f"Profit          : "
        f"₹{profit:,.2f}"
    )

    print(
        f"Profit %        : "
        f"{profit_percent:.2f}%"
    )

    print(
        f"Trades          : "
        f"{total_trades}"
    )

    print(
        f"Wins            : "
        f"{wins}"
    )

    print(
        f"Losses          : "
        f"{losses}"
    )

    print(
        f"Win Rate        : "
        f"{win_rate:.2f}%"
    )

    print(
        f"Drawdown        : "
        f"{max_drawdown:.2f}%"
    )

    print(
        f"Trading Costs   : "
        f"₹{total_costs:,.2f}"
    )

    print(
        f"Buy & Hold      : "
        f"{buy_hold_return:.2f}%"
    )

    print(
        "-" * 70
    )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "Stock":
            symbol.replace(
                "-EQ",
                ""
            ),

        "Profit":
            profit,

        "Profit %":
            profit_percent,

        "Trades":
            total_trades,

        "Wins":
            wins,

        "Losses":
            losses,

        "Win Rate %":
            win_rate,

        "Drawdown %":
            max_drawdown,

        "Costs":
            total_costs,

        "Buy & Hold %":
            buy_hold_return,
    }


# ============================================================
# RUN ALL BACKTESTS
# ============================================================

def run_all_backtests():

    results = []

    print("\n")
    print("=" * 90)
    print(
        "                 SMMA CROSSOVER BACKTEST"
    )
    print("=" * 90)

    print(
        f"Initial Capital : "
        f"₹{INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Transaction Cost: "
        f"{TRANSACTION_COST * 100:.3f}% per side"
    )

    print(
        f"Slippage        : "
        f"{SLIPPAGE * 100:.3f}% per side"
    )

    print("=" * 90)

    # ========================================================
    # TEST EACH STOCK
    # ========================================================

    for symbol in STOCKS:

        try:

            result = run_backtest(
                symbol
            )

            results.append(
                result
            )

        except Exception as e:

            print(
                f"\n{symbol}: ERROR - {e}"
            )

            # ------------------------------------------------
            # ERROR RESULT
            # ------------------------------------------------

            results.append({

                "Stock":
                    symbol.replace(
                        "-EQ",
                        ""
                    ),

                "Profit":
                    0.0,

                "Profit %":
                    0.0,

                "Trades":
                    0,

                "Wins":
                    0,

                "Losses":
                    0,

                "Win Rate %":
                    0.0,

                "Drawdown %":
                    0.0,

                "Costs":
                    0.0,

                "Buy & Hold %":
                    0.0,
            })

    return results


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(results):

    print("\n")
    print("=" * 110)
    print(
        "                              FINAL RESULTS"
    )
    print("=" * 110)

    # ========================================================
    # TABLE HEADER
    # ========================================================

    print(
        f"{'Stock':<15}"
        f"{'Profit':>14}"
        f"{'Profit %':>12}"
        f"{'Trades':>10}"
        f"{'Win Rate %':>14}"
        f"{'Drawdown %':>14}"
        f"{'Costs':>14}"
        f"{'BuyHold %':>14}"
    )

    print(
        "-" * 110
    )

    # ========================================================
    # PRINT EACH STOCK
    # ========================================================

    for result in results:

        print(
            f"{result['Stock']:<15}"
            f"₹{result['Profit']:>12.2f}"
            f"{result['Profit %']:>11.2f}"
            f"{result['Trades']:>10}"
            f"{result['Win Rate %']:>13.2f}"
            f"{result['Drawdown %']:>13.2f}"
            f"₹{result['Costs']:>12.2f}"
            f"{result['Buy & Hold %']:>13.2f}"
        )

    print(
        "-" * 110
    )

    # ========================================================
    # BEST AND WORST STOCK
    # ========================================================

    if results:

        best = max(
            results,
            key=lambda x: x["Profit"]
        )

        worst = min(
            results,
            key=lambda x: x["Profit"]
        )

        # ====================================================
        # BEST STOCK
        # ====================================================

        print("\n")
        print(
            "BEST STOCK"
        )

        print(
            "-" * 50
        )

        print(
            f"Stock       : "
            f"{best['Stock']}"
        )

        print(
            f"Profit      : "
            f"₹{best['Profit']:,.2f}"
        )

        print(
            f"Profit %    : "
            f"{best['Profit %']:.2f}%"
        )

        print(
            f"Trades      : "
            f"{best['Trades']}"
        )

        print(
            f"Wins        : "
            f"{best['Wins']}"
        )

        print(
            f"Losses      : "
            f"{best['Losses']}"
        )

        print(
            f"Win Rate    : "
            f"{best['Win Rate %']:.2f}%"
        )

        print(
            f"Drawdown    : "
            f"{best['Drawdown %']:.2f}%"
        )

        print(
            f"Costs       : "
            f"₹{best['Costs']:,.2f}"
        )

        print(
            f"Buy & Hold  : "
            f"{best['Buy & Hold %']:.2f}%"
        )

        # ====================================================
        # WORST STOCK
        # ====================================================

        print("\n")
        print(
            "WORST STOCK"
        )

        print(
            "-" * 50
        )

        print(
            f"Stock       : "
            f"{worst['Stock']}"
        )

        print(
            f"Profit      : "
            f"₹{worst['Profit']:,.2f}"
        )

        print(
            f"Profit %    : "
            f"{worst['Profit %']:.2f}%"
        )

        print(
            f"Trades      : "
            f"{worst['Trades']}"
        )

        print(
            f"Wins        : "
            f"{worst['Wins']}"
        )

        print(
            f"Losses      : "
            f"{worst['Losses']}"
        )

        print(
            f"Win Rate    : "
            f"{worst['Win Rate %']:.2f}%"
        )

        print(
            f"Drawdown    : "
            f"{worst['Drawdown %']:.2f}%"
        )

        print(
            f"Costs       : "
            f"₹{worst['Costs']:,.2f}"
        )

        print(
            f"Buy & Hold  : "
            f"{worst['Buy & Hold %']:.2f}%"
        )

    # ========================================================
    # OVERALL STATISTICS
    # ========================================================

    if results:

        total_profit = sum(
            result["Profit"]
            for result in results
        )

        total_trades = sum(
            result["Trades"]
            for result in results
        )

        total_wins = sum(
            result["Wins"]
            for result in results
        )

        total_losses = sum(
            result["Losses"]
            for result in results
        )

        total_costs = sum(
            result["Costs"]
            for result in results
        )

        if total_trades > 0:

            overall_win_rate = (
                total_wins
                / total_trades
            ) * 100

        else:

            overall_win_rate = 0.0

        print("\n")
        print(
            "OVERALL STATISTICS"
        )

        print(
            "-" * 50
        )

        print(
            f"Stocks Tested    : "
            f"{len(results)}"
        )

        print(
            f"Total Trades     : "
            f"{total_trades}"
        )

        print(
            f"Total Wins       : "
            f"{total_wins}"
        )

        print(
            f"Total Losses     : "
            f"{total_losses}"
        )

        print(
            f"Overall Win Rate : "
            f"{overall_win_rate:.2f}%"
        )

        print(
            f"Total Profit     : "
            f"₹{total_profit:,.2f}"
        )

        print(
            f"Total Costs      : "
            f"₹{total_costs:,.2f}"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 90)
    print(
        "                         BACKTEST COMPLETE"
    )
    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = run_all_backtests()

    print_summary(
        results
    )