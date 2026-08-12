"""
============================================================
STAGE 9 - PORTFOLIO VALIDATION
============================================================

Locked strategy:
    SMMA 20 / 120
    Holding period = 60 trading days
    Entry slippage = 0.10%
    Exit slippage = 0.10%
    Transaction cost = 0.05% per side

Risk management:
    Initial capital = Rs. 100,000
    Risk per trade = 1%
    Stop loss = 5%
    Take profit = 10%
    Maximum position = 20%

Portfolio:
    Maximum open positions = 5
    Maximum exposure = 100%

IMPORTANT:
    SMMA 20/120 remains locked.
    No parameter optimization is performed.
"""

import math
from collections import defaultdict

import pandas as pd

from api.historical_data import get_historical_data
from indicators.smma import calculate_smma


# ============================================================
# CONFIGURATION
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
# LOCKED STRATEGY PARAMETERS
# ============================================================

FAST_SMMA = 20
SLOW_SMMA = 120

HISTORICAL_DAYS = 1000
HOLDING_PERIOD = 60

ENTRY_SLIPPAGE_PERCENT = 0.10
EXIT_SLIPPAGE_PERCENT = 0.10
TRANSACTION_COST_PERCENT = 0.05

INITIAL_CAPITAL = 100000.0

RISK_PER_TRADE_PERCENT = 1.00
STOP_LOSS_PERCENT = 5.00
TAKE_PROFIT_PERCENT = 10.00
MAX_POSITION_PERCENT = 20.00


# ============================================================
# PORTFOLIO PARAMETERS
# ============================================================

MAX_OPEN_POSITIONS = 5
MAX_PORTFOLIO_EXPOSURE_PERCENT = 100.00


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_columns(df):

    if df is None or df.empty:
        return None

    df = df.copy()

    rename_map = {}

    for column in df.columns:

        name = str(column).strip().lower()

        if name in ("date", "datetime"):
            rename_map[column] = "Date"

        elif name == "open":
            rename_map[column] = "Open"

        elif name == "high":
            rename_map[column] = "High"

        elif name == "low":
            rename_map[column] = "Low"

        elif name == "close":
            rename_map[column] = "Close"

        elif name == "volume":
            rename_map[column] = "Volume"

    return df.rename(columns=rename_map)


# ============================================================
# LOAD AND PREPARE STOCK DATA
# ============================================================

def prepare_data(symbol):

    print()
    print("=" * 80)
    print(f"GETTING HISTORICAL DATA FOR {symbol}")
    print("=" * 80)

    try:

        df = get_historical_data(
            symbol,
            days=HISTORICAL_DAYS
        )

    except Exception as error:

        print(
            f"{symbol}: historical data error: {error}"
        )

        return None

    if df is None or df.empty:

        print(
            f"{symbol}: No historical data."
        )

        return None

    print(
        f"{symbol}: Historical rows loaded: {len(df)}"
    )

    df = normalize_columns(df)

    if df is None:
        return None

    required = ["Date", "Close"]

    for column in required:

        if column not in df.columns:

            print(
                f"{symbol}: Missing required column '{column}'"
            )

            return None

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date", "Close"]
    ).copy()

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(df) < SLOW_SMMA + 2:

        print(
            f"{symbol}: Not enough data for SMMA {SLOW_SMMA}"
        )

        return None

    # --------------------------------------------------------
    # SMMA
    # --------------------------------------------------------

    df["SMMA20"] = calculate_smma(
        df["Close"],
        FAST_SMMA
    )

    df["SMMA120"] = calculate_smma(
        df["Close"],
        SLOW_SMMA
    )

    df["Previous_SMMA20"] = df["SMMA20"].shift(1)
    df["Previous_SMMA120"] = df["SMMA120"].shift(1)

    # --------------------------------------------------------
    # BUY CROSSOVER
    # --------------------------------------------------------

    df["BUY_CROSSOVER"] = (
        (df["Previous_SMMA20"] <= df["Previous_SMMA120"])
        &
        (df["SMMA20"] > df["SMMA120"])
    )

    # --------------------------------------------------------
    # SELL CROSSOVER
    # --------------------------------------------------------

    df["SELL_CROSSOVER"] = (
        (df["Previous_SMMA20"] >= df["Previous_SMMA120"])
        &
        (df["SMMA20"] < df["SMMA120"])
    )

    df = df.dropna(
        subset=["SMMA20", "SMMA120"]
    ).reset_index(drop=True)

    print(
        f"{symbol}: Usable rows: {len(df)}"
    )

    print(
        f"{symbol}: BUY crossovers: "
        f"{int(df['BUY_CROSSOVER'].sum())}"
    )

    print(
        f"{symbol}: SELL crossovers: "
        f"{int(df['SELL_CROSSOVER'].sum())}"
    )

    return df


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_position_size(
    capital,
    entry_price
):

    if capital <= 0 or entry_price <= 0:
        return 0

    max_position_value = (
        capital
        * MAX_POSITION_PERCENT
        / 100.0
    )

    max_risk_amount = (
        capital
        * RISK_PER_TRADE_PERCENT
        / 100.0
    )

    risk_per_share = (
        entry_price
        * STOP_LOSS_PERCENT
        / 100.0
    )

    if risk_per_share <= 0:
        return 0

    shares_by_risk = int(
        max_risk_amount / risk_per_share
    )

    shares_by_position = int(
        max_position_value / entry_price
    )

    return max(
        0,
        min(
            shares_by_risk,
            shares_by_position
        )
    )


# ============================================================
# ENTRY SLIPPAGE
# ============================================================

def apply_entry_slippage(
    price,
    signal
):

    if signal == "BUY":

        return price * (
            1 + ENTRY_SLIPPAGE_PERCENT / 100
        )

    if signal == "SELL":

        return price * (
            1 - ENTRY_SLIPPAGE_PERCENT / 100
        )

    return price


# ============================================================
# EXIT SLIPPAGE
# ============================================================

def apply_exit_slippage(
    price,
    signal
):

    if signal == "BUY":

        return price * (
            1 - EXIT_SLIPPAGE_PERCENT / 100
        )

    if signal == "SELL":

        return price * (
            1 + EXIT_SLIPPAGE_PERCENT / 100
        )

    return price


# ============================================================
# TRANSACTION COST
# ============================================================

def calculate_transaction_cost(
    entry_value,
    exit_value
):

    return (
        entry_value + exit_value
    ) * TRANSACTION_COST_PERCENT / 100


# ============================================================
# FIND EXIT
# ============================================================

def find_exit(
    df,
    entry_index,
    signal,
    entry_price
):

    if signal == "BUY":

        stop_price = (
            entry_price
            * (1 - STOP_LOSS_PERCENT / 100)
        )

        target_price = (
            entry_price
            * (1 + TAKE_PROFIT_PERCENT / 100)
        )

    elif signal == "SELL":

        stop_price = (
            entry_price
            * (1 + STOP_LOSS_PERCENT / 100)
        )

        target_price = (
            entry_price
            * (1 - TAKE_PROFIT_PERCENT / 100)
        )

    else:

        return None

    exit_index = (
        entry_index + HOLDING_PERIOD
    )

    if exit_index >= len(df):

        return None

    for i in range(
        entry_index + 1,
        exit_index + 1
    ):

        close_price = float(
            df["Close"].iloc[i]
        )

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        if signal == "BUY":

            if close_price <= stop_price:

                return {
                    "index": i,
                    "price": close_price,
                    "reason": "STOP_LOSS"
                }

            if close_price >= target_price:

                return {
                    "index": i,
                    "price": close_price,
                    "reason": "TAKE_PROFIT"
                }

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        elif signal == "SELL":

            if close_price >= stop_price:

                return {
                    "index": i,
                    "price": close_price,
                    "reason": "STOP_LOSS"
                }

            if close_price <= target_price:

                return {
                    "index": i,
                    "price": close_price,
                    "reason": "TAKE_PROFIT"
                }

    # --------------------------------------------------------
    # TIME EXIT
    # --------------------------------------------------------

    return {
        "index": exit_index,
        "price": float(
            df["Close"].iloc[exit_index]
        ),
        "reason": "TIME_EXIT"
    }


# ============================================================
# CALCULATE TRADE
# ============================================================

def calculate_trade(
    symbol,
    df,
    signal_index,
    signal,
    capital
):

    raw_entry_price = float(
        df["Close"].iloc[signal_index]
    )

    entry_price = apply_entry_slippage(
        raw_entry_price,
        signal
    )

    shares = calculate_position_size(
        capital,
        entry_price
    )

    if shares <= 0:
        return None

    entry_value = (
        shares * entry_price
    )

    exit_data = find_exit(
        df,
        signal_index,
        signal,
        entry_price
    )

    if exit_data is None:
        return None

    exit_index = exit_data["index"]

    raw_exit_price = float(
        exit_data["price"]
    )

    exit_reason = exit_data["reason"]

    exit_price = apply_exit_slippage(
        raw_exit_price,
        signal
    )

    exit_value = (
        shares * exit_price
    )

    # --------------------------------------------------------
    # PNL
    # --------------------------------------------------------

    if signal == "BUY":

        gross_pnl = (
            exit_value - entry_value
        )

    else:

        gross_pnl = (
            entry_value - exit_value
        )

    gross_return = (
        gross_pnl
        / entry_value
        * 100
    )

    transaction_cost = (
        calculate_transaction_cost(
            entry_value,
            exit_value
        )
    )

    net_pnl = (
        gross_pnl
        - transaction_cost
    )

    net_return = (
        net_pnl
        / entry_value
        * 100
    )

    if net_return > 0:

        result = "PROFITABLE"

    elif net_return < 0:

        result = "FAILED"

    else:

        result = "BREAKEVEN"

    # ========================================================
    # IMPORTANT FIX
    # ========================================================
    #
    # "Symbol" is explicitly stored here.
    #
    # This prevents:
    #
    # KeyError: 'Symbol'
    #
    # when the portfolio later executes:
    #
    # position["Symbol"]
    #
    # ========================================================

    trade = {

        "Stock": symbol.replace(
            "-EQ",
            ""
        ),

        "Symbol": symbol,

        "Signal": signal,

        "Entry Index": signal_index,

        "Exit Index": exit_index,

        "Entry Date": df["Date"].iloc[
            signal_index
        ],

        "Exit Date": df["Date"].iloc[
            exit_index
        ],

        "Raw Entry": raw_entry_price,

        "Entry Price": entry_price,

        "Raw Exit": raw_exit_price,

        "Exit Price": exit_price,

        "Shares": shares,

        "Position Value": entry_value,

        "Exit Value": exit_value,

        "Gross PnL": gross_pnl,

        "Gross Return %": gross_return,

        "Transaction Cost": transaction_cost,

        "Net PnL": net_pnl,

        "Net Return %": net_return,

        "Result": result,

        "Exit Reason": exit_reason,

        "Holding Bars": (
            exit_index - signal_index
        )
    }

    return trade


# ============================================================
# GENERATE SIGNAL EVENTS
# ============================================================

def generate_signal_events(data):

    events = []

    for symbol, df in data.items():

        for i in range(len(df)):

            signal = None

            if bool(
                df["BUY_CROSSOVER"].iloc[i]
            ):

                signal = "BUY"

            elif bool(
                df["SELL_CROSSOVER"].iloc[i]
            ):

                signal = "SELL"

            if signal is None:
                continue

            events.append({

                "Date": df["Date"].iloc[i],

                "Symbol": symbol,

                "Index": i,

                "Signal": signal

            })

    # --------------------------------------------------------
    # Chronological ordering
    # --------------------------------------------------------

    stock_order = {
        symbol: index
        for index, symbol in enumerate(STOCKS)
    }

    events.sort(
        key=lambda event: (
            event["Date"],
            stock_order.get(
                event["Symbol"],
                999
            )
        )
    )

    return events


# ============================================================
# MARK PORTFOLIO TO MARKET
# ============================================================

def mark_to_market(
    cash,
    positions,
    data,
    date
):

    equity = cash

    for position in positions:

        # Safe Symbol access
        symbol = position.get(
            "Symbol"
        )

        if not symbol:
            continue

        if symbol not in data:
            continue

        df = data[symbol]

        rows = df.index[
            df["Date"] == date
        ].tolist()

        if not rows:
            continue

        price = float(
            df["Close"].iloc[
                rows[0]
            ]
        )

        shares = position.get(
            "Shares",
            0
        )

        if position.get(
            "Signal"
        ) == "BUY":

            equity += (
                shares * price
            )

        else:

            equity -= (
                shares * price
            )

    return equity


# ============================================================
# EXPOSURE
# ============================================================

def calculate_exposure(
    positions
):

    total = 0.0

    for position in positions:

        total += float(
            position.get(
                "Position Value",
                0
            )
        )

    return total


# ============================================================
# CLOSE POSITIONS
# ============================================================

def close_due_positions(
    current_date,
    positions,
    cash,
    closed_trades
):

    remaining = []

    for position in positions:

        exit_date = position.get(
            "Exit Date"
        )

        if exit_date is None:

            remaining.append(
                position
            )

            continue

        if exit_date > current_date:

            remaining.append(
                position
            )

            continue

        exit_value = float(
            position.get(
                "Exit Value",
                0
            )
        )

        exit_cost = float(
            position.get(
                "Exit Cost",
                0
            )
        )

        if position.get(
            "Signal"
        ) == "BUY":

            cash += (
                exit_value
                - exit_cost
            )

        else:

            cash -= (
                exit_value
                + exit_cost
            )

        closed_trades.append(
            position
        )

    return remaining, cash


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def run_portfolio(data):

    events = generate_signal_events(
        data
    )

    cash = INITIAL_CAPITAL

    positions = []

    closed_trades = []

    equity_curve = []

    skipped_signals = []

    for event in events:

        current_date = event["Date"]

        symbol = event["Symbol"]

        # ----------------------------------------------------
        # Close old positions
        # ----------------------------------------------------

        positions, cash = close_due_positions(

            current_date,

            positions,

            cash,

            closed_trades
        )

        # ----------------------------------------------------
        # Current occupied symbols
        # ----------------------------------------------------

        occupied_symbols = {

            position.get(
                "Symbol"
            )

            for position in positions

            if position.get(
                "Symbol"
            )
        }

        # ----------------------------------------------------
        # Current equity
        # ----------------------------------------------------

        equity = mark_to_market(

            cash,

            positions,

            data,

            current_date
        )

        equity_curve.append({

            "Date": current_date,

            "Equity": equity

        })

        # ----------------------------------------------------
        # Same stock already open
        # ----------------------------------------------------

        if symbol in occupied_symbols:

            skipped_signals.append({

                **event,

                "Reason":
                    "STOCK_ALREADY_IN_POSITION"

            })

            continue

        # ----------------------------------------------------
        # Maximum open positions
        # ----------------------------------------------------

        if len(positions) >= MAX_OPEN_POSITIONS:

            skipped_signals.append({

                **event,

                "Reason":
                    "MAX_OPEN_POSITIONS"

            })

            continue

        # ----------------------------------------------------
        # Maximum exposure
        # ----------------------------------------------------

        exposure = calculate_exposure(
            positions
        )

        max_exposure = (

            equity
            * MAX_PORTFOLIO_EXPOSURE_PERCENT
            / 100.0

        )

        if exposure >= max_exposure:

            skipped_signals.append({

                **event,

                "Reason":
                    "MAX_PORTFOLIO_EXPOSURE"

            })

            continue

        # ----------------------------------------------------
        # Calculate candidate trade
        # ----------------------------------------------------

        trade = calculate_trade(

            symbol=symbol,

            df=data[symbol],

            signal_index=event["Index"],

            signal=event["Signal"],

            capital=equity

        )

        if trade is None:

            skipped_signals.append({

                **event,

                "Reason":
                    "NO_COMPLETED_TRADE"

            })

            continue

        # ----------------------------------------------------
        # Candidate exposure
        # ----------------------------------------------------

        candidate_value = float(
            trade["Position Value"]
        )

        if (
            exposure
            + candidate_value
            > max_exposure + 1e-9
        ):

            skipped_signals.append({

                **event,

                "Reason":
                    "EXPOSURE_LIMIT"

            })

            continue

        # ----------------------------------------------------
        # Entry cost
        # ----------------------------------------------------

        entry_cost = (

            trade["Position Value"]

            * TRANSACTION_COST_PERCENT
            / 100.0

        )

        trade["Entry Cost"] = (
            entry_cost
        )

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        if trade["Signal"] == "BUY":

            required_cash = (

                trade["Position Value"]

                + entry_cost

            )

            if required_cash > cash:

                skipped_signals.append({

                    **event,

                    "Reason":
                        "INSUFFICIENT_CASH"

                })

                continue

            cash -= required_cash

        # ----------------------------------------------------
        # SELL / SHORT
        # ----------------------------------------------------

        else:

            cash += (

                trade["Position Value"]

                - entry_cost

            )

        # ----------------------------------------------------
        # Exit cost
        # ----------------------------------------------------

        exit_cost = (

            trade["Exit Value"]

            * TRANSACTION_COST_PERCENT
            / 100.0

        )

        trade["Exit Cost"] = (
            exit_cost
        )

        # ----------------------------------------------------
        # Add position
        # ----------------------------------------------------

        positions.append(
            trade
        )

    # ========================================================
    # CLOSE ALL REMAINING POSITIONS
    # ========================================================

    for position in positions:

        exit_value = float(
            position.get(
                "Exit Value",
                0
            )
        )

        exit_cost = float(
            position.get(
                "Exit Cost",
                0
            )
        )

        if position.get(
            "Signal"
        ) == "BUY":

            cash += (
                exit_value
                - exit_cost
            )

        else:

            cash -= (
                exit_value
                + exit_cost
            )

        closed_trades.append(
            position
        )

    # --------------------------------------------------------
    # Sort trades by exit date
    # --------------------------------------------------------

    closed_trades.sort(
        key=lambda trade:
            trade["Exit Date"]
    )

    # ========================================================
    # REALIZED EQUITY CURVE
    # ========================================================

    portfolio_capital = (
        INITIAL_CAPITAL
    )

    equity_points = [{

        "Date": None,

        "Equity":
            portfolio_capital

    }]

    for trade in closed_trades:

        portfolio_capital += float(
            trade["Net PnL"]
        )

        equity_points.append({

            "Date":
                trade["Exit Date"],

            "Equity":
                portfolio_capital

        })

    return {

        "Trades":
            closed_trades,

        "Skipped Signals":
            skipped_signals,

        "Equity Curve":
            equity_points,

        "Ending Capital":
            portfolio_capital,

        "Final Cash":
            cash

    }


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(
    trades
):

    total_trades = len(
        trades
    )

    wins = sum(

        1

        for trade in trades

        if trade["Net PnL"] > 0

    )

    losses = sum(

        1

        for trade in trades

        if trade["Net PnL"] < 0

    )

    gross_pnl = sum(

        trade["Gross PnL"]

        for trade in trades

    )

    net_pnl = sum(

        trade["Net PnL"]

        for trade in trades

    )

    total_costs = sum(

        trade["Transaction Cost"]

        for trade in trades

    )

    gross_profit = sum(

        trade["Gross PnL"]

        for trade in trades

        if trade["Gross PnL"] > 0

    )

    gross_loss = sum(

        trade["Gross PnL"]

        for trade in trades

        if trade["Gross PnL"] < 0

    )

    win_rate = (

        wins
        / total_trades
        * 100

        if total_trades > 0

        else 0.0

    )

    gross_return = (

        gross_pnl
        / INITIAL_CAPITAL
        * 100

    )

    net_return = (

        net_pnl
        / INITIAL_CAPITAL
        * 100

    )

    average_net_return = (

        sum(
            trade["Net Return %"]
            for trade in trades
        )
        / total_trades

        if total_trades > 0

        else 0.0

    )

    average_win = (

        sum(
            trade["Net PnL"]
            for trade in trades
            if trade["Net PnL"] > 0
        )
        / wins

        if wins > 0

        else 0.0

    )

    average_loss = (

        sum(
            trade["Net PnL"]
            for trade in trades
            if trade["Net PnL"] < 0
        )
        / losses

        if losses > 0

        else 0.0

    )

    if gross_loss < 0:

        profit_factor = (
            gross_profit
            / abs(gross_loss)
        )

    elif gross_profit > 0:

        profit_factor = math.inf

    else:

        profit_factor = 0.0

    return {

        "Trades":
            total_trades,

        "Wins":
            wins,

        "Losses":
            losses,

        "Win Rate %":
            win_rate,

        "Gross PnL":
            gross_pnl,

        "Gross Return %":
            gross_return,

        "Net PnL":
            net_pnl,

        "Net Return %":
            net_return,

        "Average Net Return %":
            average_net_return,

        "Average Win":
            average_win,

        "Average Loss":
            average_loss,

        "Profit Factor":
            profit_factor,

        "Total Costs":
            total_costs

    }


# ============================================================
# DRAWDOWN
# ============================================================

def calculate_drawdown(
    equity_curve
):

    if not equity_curve:

        return {

            "Peak Capital":
                INITIAL_CAPITAL,

            "Maximum Drawdown":
                0.0,

            "Maximum Drawdown %":
                0.0

        }

    peak = float(
        equity_curve[0]["Equity"]
    )

    max_drawdown = 0.0

    max_drawdown_percent = 0.0

    for point in equity_curve:

        equity = float(
            point["Equity"]
        )

        if equity > peak:

            peak = equity

        drawdown = (
            equity - peak
        )

        if drawdown < max_drawdown:

            max_drawdown = (
                drawdown
            )

        if peak > 0:

            drawdown_percent = (

                drawdown
                / peak
                * 100

            )

            if (
                drawdown_percent
                < max_drawdown_percent
            ):

                max_drawdown_percent = (
                    drawdown_percent
                )

    return {

        "Peak Capital":
            peak,

        "Maximum Drawdown":
            max_drawdown,

        "Maximum Drawdown %":
            max_drawdown_percent

    }


# ============================================================
# PRINT TRADE DETAILS
# ============================================================

def print_trade_details(
    trades
):

    print()
    print("=" * 140)
    print("PORTFOLIO TRADE DETAILS")
    print("=" * 140)

    if not trades:

        print(
            "No completed portfolio trades."
        )

        return

    print(

        f"{'Stock':<12}"
        f"{'Signal':<8}"
        f"{'Entry':>11}"
        f"{'Exit':>11}"
        f"{'Shares':>8}"
        f"{'Net PnL':>12}"
        f"{'Net %':>10}"
        f"{'Result':>14}"
        f"{'Exit Reason':>15}"

    )

    print("-" * 140)

    for trade in trades:

        print(

            f"{trade['Stock']:<12}"
            f"{trade['Signal']:<8}"
            f"{trade['Entry Price']:>11.2f}"
            f"{trade['Exit Price']:>11.2f}"
            f"{trade['Shares']:>8}"
            f"{trade['Net PnL']:>12.2f}"
            f"{trade['Net Return %']:>9.2f}%"
            f"{trade['Result']:>14}"
            f"{trade['Exit Reason']:>15}"

        )


# ============================================================
# STOCK CONTRIBUTION
# ============================================================

def print_stock_contribution(
    trades
):

    grouped = defaultdict(list)

    for trade in trades:

        grouped[
            trade["Stock"]
        ].append(trade)

    print()
    print("=" * 100)
    print("STOCK CONTRIBUTION")
    print("=" * 100)

    print(

        f"{'Stock':<15}"
        f"{'Trades':>10}"
        f"{'Wins':>10}"
        f"{'Losses':>10}"
        f"{'Win Rate':>12}"
        f"{'Net PnL':>15}"
        f"{'Net %':>12}"

    )

    print("-" * 100)

    for stock in STOCKS:

        name = stock.replace(
            "-EQ",
            ""
        )

        stock_trades = (
            grouped.get(
                name,
                []
            )
        )

        stats = calculate_statistics(
            stock_trades
        )

        print(

            f"{name:<15}"
            f"{stats['Trades']:>10}"
            f"{stats['Wins']:>10}"
            f"{stats['Losses']:>10}"
            f"{stats['Win Rate %']:>11.2f}%"
            f"{stats['Net PnL']:>15.2f}"
            f"{stats['Net Return %']:>11.2f}%"

        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print("STAGE 9 - PORTFOLIO VALIDATION")
    print("=" * 110)

    print()
    print("LOCKED STRATEGY")
    print("-" * 60)

    print(
        f"Fast SMMA              : {FAST_SMMA}"
    )

    print(
        f"Slow SMMA              : {SLOW_SMMA}"
    )

    print(
        f"Holding Period         : "
        f"{HOLDING_PERIOD} trading days"
    )

    print(
        f"Historical Days        : "
        f"{HISTORICAL_DAYS}"
    )

    print(
        f"Entry Slippage         : "
        f"{ENTRY_SLIPPAGE_PERCENT:.2f}%"
    )

    print(
        f"Exit Slippage          : "
        f"{EXIT_SLIPPAGE_PERCENT:.2f}%"
    )

    print(
        f"Transaction Cost       : "
        f"{TRANSACTION_COST_PERCENT:.2f}% per side"
    )

    print()
    print("RISK MANAGEMENT")
    print("-" * 60)

    print(
        f"Initial Capital        : "
        f"Rs. {INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Risk Per Trade         : "
        f"{RISK_PER_TRADE_PERCENT:.2f}%"
    )

    print(
        f"Stop Loss              : "
        f"{STOP_LOSS_PERCENT:.2f}%"
    )

    print(
        f"Take Profit            : "
        f"{TAKE_PROFIT_PERCENT:.2f}%"
    )

    print(
        f"Maximum Position       : "
        f"{MAX_POSITION_PERCENT:.2f}%"
    )

    print()
    print("PORTFOLIO RULES")
    print("-" * 60)

    print(
        f"Maximum Open Positions : "
        f"{MAX_OPEN_POSITIONS}"
    )

    print(
        f"Maximum Exposure       : "
        f"{MAX_PORTFOLIO_EXPOSURE_PERCENT:.2f}%"
    )

    print()
    print("SMMA 20/120 IS LOCKED.")
    print("NO PARAMETER OPTIMIZATION IS PERFORMED.")
    print("=" * 110)

    # ========================================================
    # LOAD STOCKS
    # ========================================================

    data = {}

    for symbol in STOCKS:

        try:

            df = prepare_data(
                symbol
            )

            if df is not None:

                data[symbol] = df

        except Exception as error:

            print(
                f"{symbol}: ERROR: {error}"
            )

    print()
    print("=" * 100)

    print(
        f"Stocks successfully loaded: "
        f"{len(data)}/{len(STOCKS)}"
    )

    print("=" * 100)

    if not data:

        print()
        print("RESULT: INCONCLUSIVE")
        print(
            "No stock data was available."
        )

        return

    # ========================================================
    # RUN PORTFOLIO
    # ========================================================

    portfolio = run_portfolio(
        data
    )

    trades = portfolio[
        "Trades"
    ]

    statistics = calculate_statistics(
        trades
    )

    drawdown = calculate_drawdown(
        portfolio[
            "Equity Curve"
        ]
    )

    # ========================================================
    # TRADE DETAILS
    # ========================================================

    print_trade_details(
        trades
    )

    # ========================================================
    # STOCK CONTRIBUTION
    # ========================================================

    print_stock_contribution(
        trades
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 90)
    print("STAGE 9 - PORTFOLIO RESULTS")
    print("=" * 90)

    print(
        f"Starting Capital       : "
        f"Rs. {INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Ending Capital         : "
        f"Rs. {portfolio['Ending Capital']:,.2f}"
    )

    print(
        f"Total Trades           : "
        f"{statistics['Trades']}"
    )

    print(
        f"Total Wins             : "
        f"{statistics['Wins']}"
    )

    print(
        f"Total Losses           : "
        f"{statistics['Losses']}"
    )

    print(
        f"Overall Win Rate       : "
        f"{statistics['Win Rate %']:.2f}%"
    )

    print(
        f"Gross PnL              : "
        f"Rs. {statistics['Gross PnL']:.2f}"
    )

    print(
        f"Gross Return           : "
        f"{statistics['Gross Return %']:.2f}%"
    )

    print(
        f"Net PnL                : "
        f"Rs. {statistics['Net PnL']:.2f}"
    )

    print(
        f"Net Return             : "
        f"{statistics['Net Return %']:.2f}%"
    )

    print(
        f"Average Net Trade      : "
        f"{statistics['Average Net Return %']:.2f}%"
    )

    print(
        f"Average Winning Trade  : "
        f"Rs. {statistics['Average Win']:.2f}"
    )

    print(
        f"Average Losing Trade   : "
        f"Rs. {statistics['Average Loss']:.2f}"
    )

    if math.isinf(
        statistics["Profit Factor"]
    ):

        print(
            "Profit Factor          : INF"
        )

    else:

        print(
            f"Profit Factor          : "
            f"{statistics['Profit Factor']:.2f}"
        )

    print(
        f"Total Transaction Cost : "
        f"Rs. {statistics['Total Costs']:.4f}"
    )

    print()
    print(
        f"Peak Capital           : "
        f"Rs. {drawdown['Peak Capital']:.2f}"
    )

    print(
        f"Maximum Drawdown       : "
        f"Rs. {drawdown['Maximum Drawdown']:.2f}"
    )

    print(
        f"Maximum Drawdown       : "
        f"{drawdown['Maximum Drawdown %']:.2f}%"
    )

    # ========================================================
    # SKIPPED SIGNALS
    # ========================================================

    skipped = portfolio[
        "Skipped Signals"
    ]

    print()
    print("=" * 90)
    print("PORTFOLIO SIGNAL FILTERING")
    print("=" * 90)

    print(
        f"Total Signals Skipped  : "
        f"{len(skipped)}"
    )

    reason_counts = defaultdict(
        int
    )

    for item in skipped:

        reason_counts[
            item["Reason"]
        ] += 1

    for reason, count in sorted(
        reason_counts.items()
    ):

        print(
            f"{reason:<30}: {count}"
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 90)
    print("STAGE 9 VALIDATION")
    print("=" * 90)

    if statistics["Trades"] == 0:

        print(
            "RESULT: INCONCLUSIVE"
        )

        print()

        print(
            "No completed portfolio trades "
            "were available for validation."
        )

    elif statistics["Net Return %"] > 0:

        print(
            "RESULT: POSITIVE"
        )

        print()

        print(
            "The locked SMMA 20/120 strategy "
            "produced a positive portfolio-level "
            "net return after costs and slippage."
        )

    else:

        print(
            "RESULT: NEGATIVE"
        )

        print()

        print(
            "The portfolio-level strategy was not "
            "profitable under the locked assumptions."
        )

    print()
    print("=" * 110)
    print("STAGE 9 COMPLETE")
    print("=" * 110)


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()