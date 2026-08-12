import pandas as pd

def screen_stocks():
    df = pd.read_csv("data/stocks.csv")

    filtered = df[
        (df["LTP"] >= 30) &
        (df["LTP"] <= 500) &
        (df["BidQty"] > 1000000) &
        (df["AskQty"] > 1000000)
    ]

    return filtered