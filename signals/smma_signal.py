def generate_signal(close, smma20, smma120, rsi):
    score = 0
    reasons = []

    # ==================================================
    # 1. PRICE vs SMMA20
    # ==================================================

    if close > smma20:
        score += 30
        reasons.append("Price is above SMMA20")
    else:
        score -= 30
        reasons.append("Price is below SMMA20")

    # ==================================================
    # 2. SMMA20 vs SMMA120 — TREND
    # ==================================================

    if smma20 > smma120:
        score += 30
        reasons.append("SMMA20 is above SMMA120")
    else:
        score -= 30
        reasons.append("SMMA20 is below SMMA120")

    # ==================================================
    # 3. RSI — MOMENTUM
    # ==================================================

    if 55 <= rsi < 70:

        score += 25
        reasons.append(
            "RSI shows healthy bullish momentum"
        )

    elif 50 <= rsi < 55:

        score += 10
        reasons.append(
            "RSI shows mild bullish momentum"
        )

    elif 70 <= rsi:

        score -= 10
        reasons.append(
            "RSI is overbought"
        )

    elif 30 <= rsi < 50:

        score -= 10
        reasons.append(
            "RSI shows weak momentum"
        )

    else:

        score -= 25
        reasons.append(
            "RSI is strongly oversold"
        )

    # ==================================================
    # 4. FINAL SIGNAL
    # ==================================================

    if score >= 60:

        signal = "BUY"

    elif score <= -50:

        signal = "SELL"

    else:

        signal = "HOLD"

    # ==================================================
    # 5. CONFIDENCE
    # ==================================================

    confidence = min(abs(score), 100)

    return {
        "Signal": signal,
        "Confidence": confidence,
        "Reasons": reasons
    }