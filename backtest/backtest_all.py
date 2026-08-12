from api.historical_data import get_historical_data
from indicators.smma import calculate_smma
from indicators.rsi import calculate_rsi
from signals.smma_signal import generate_signal


SYMBOLS = [
    "SBIN-EQ",
    "SUZLON-EQ",
    "IRFC-EQ",
    "TATAMOTORS-EQ",
    "RELIANCE-EQ",
    "ITC-EQ"
]

INITIAL_CAPITAL = 100000


def backtest_stock(symbol):

    df = get_historical_data(symbol)

    if df is None or len(df) < 120:
        return None

    close_prices = df["Close"]

    # Indicators
    smma20 = calculate_smma(close_prices, 20)
    smma120 = calculate_smma(close_prices, 120)
    rsi = calculate_rsi(close_prices, 14)

    capital = INITIAL_CAPITAL
    shares = 0

    trades = []
    winning_trades = 0
    losing_trades = 0

    for i in range(120, len(df)):

        price = close_prices.iloc[i]

        current_smma20 = smma20.iloc[i]
        current_smma120 = smma120.iloc[i]
        current_rsi = rsi.iloc[i]

        if (
            pd_is_valid(price)
            and pd_is_valid(current_smma20)
            and pd_is_valid(current_smma120)
            and pd_is_valid(current_rsi)
        ):

            signal_result = generate_signal(
                price,
                current_smma20,
                current_smma120,
                current_rsi
            )

            signal = signal_result["Signal"]
            date = df["Date"].iloc[i]

            # BUY
            if signal == "BUY" and shares == 0:

                shares = int(capital // price)

                if shares > 0:
                    capital -= shares * price

                    trades.append({
                        "Date": date,
                        "Type": "BUY",
                        "Price": float(price),
                        "Shares": shares
                    })

            # SELL
            elif signal == "SELL" and shares > 0:

                sell_value = shares * price
                buy_price = trades[-1]["Price"]

                profit = (price - buy_price) * shares

                capital += sell_value

                if profit > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1

                trades.append({
                    "Date": date,
                    "Type": "SELL",
                    "Price": float(price),
                    "Shares": shares,
                    "Profit": float(profit)
                })

                shares = 0

    # Close open position at latest price
    if shares > 0:

        final_price = close_prices.iloc[-1]

        profit = (final_price - trades[-1]["Price"]) * shares

        capital += shares * final_price

        if profit > 0:
            winning_trades += 1
        else:
            losing_trades += 1

        trades.append({
            "Date": df["Date"].iloc[-1],
            "Type": "FINAL SELL",
            "Price": float(final_price),
            "Shares": shares,
            "Profit": float(profit)
        })

        shares = 0

    final_capital = capital

    profit = final_capital - INITIAL_CAPITAL

    profit_percent = (
        profit / INITIAL_CAPITAL
    ) * 100

    total_trades = winning_trades + losing_trades

    if total_trades > 0:
        win_rate = (
            winning_trades / total_trades
        ) * 100
    else:
        win_rate = 0

    return {
        "Stock": symbol.replace("-EQ", ""),
        "Initial Capital": round(INITIAL_CAPITAL, 2),
        "Final Capital": round(final_capital, 2),
        "Profit": round(profit, 2),
        "Profit %": round(profit_percent, 2),
        "Total Trades": total_trades,
        "Winning Trades": winning_trades,
        "Losing Trades": losing_trades,
        "Win Rate %": round(win_rate, 2),
        "Trades": trades
    }


def pd_is_valid(value):
    return value == value

def run_all_backtests():

    symbols = [
        "SBIN-EQ",
        "SUZLON-EQ",
        "IRFC-EQ",
        "TATAMOTORS-EQ",
        "RELIANCE-EQ",
        "ITC-EQ"
    ]

    results = []

    for symbol in symbols:

        print(f"Testing {symbol}...")

        result = backtest_stock(symbol)

        if result is not None:
            results.append(result)

    return results

if __name__ == "__main__":

    print("\n========== ALL STOCK BACKTEST ==========\n")

    results = run_all_backtests()

    print(
        f"{'Stock':<12}"
        f"{'Profit %':>12}"
        f"{'Trades':>10}"
        f"{'Win Rate %':>14}"
    )

    print("-" * 50)

    for result in results:

        print(
            f"{result['Stock']:<12}"
            f"{result['Profit %']:>12.2f}"
            f"{result['Total Trades']:>10}"
            f"{result['Win Rate %']:>14.2f}"
        )