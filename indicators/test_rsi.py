from api.historical_data import get_historical_data
from indicators.rsi import calculate_rsi


df = get_historical_data("SBIN-EQ")

if df is not None:

    close_prices = df["Close"]

    rsi = calculate_rsi(close_prices, 14)

    print("\n========== RSI RESULTS ==========")
    print("Latest Close:", close_prices.iloc[-1])
    print("RSI14:", round(rsi.iloc[-1], 2))