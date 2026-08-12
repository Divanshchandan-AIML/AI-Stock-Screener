def get_signal(smma20, smma120):
    if isinstance(smma120, str):
        return "No Signal"

    if smma20 > smma120:
        return "BUY"

    elif smma20 < smma120:
        return "SELL"

    else:
        return "HOLD"