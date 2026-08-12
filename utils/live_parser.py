def parse_live_market_data(symbol, data):
    """
    Convert Angel One FULL market-data response
    into a simple dictionary for the stock screener.
    """

    if not data:
        return None

    try:
        # ============================================================
        # 1. VALIDATE API RESPONSE
        # ============================================================

        if not isinstance(data, dict):
            return None

        if not data.get("status"):
            return None

        response_data = data.get("data")

        if not isinstance(response_data, dict):
            return None

        fetched = response_data.get("fetched", [])

        if not fetched:
            return None

        # ============================================================
        # 2. GET STOCK DATA
        # ============================================================

        stock = fetched[0]

        if not isinstance(stock, dict):
            return None

        # ============================================================
        # 3. BASIC MARKET DATA
        # ============================================================

        ltp = float(stock.get("ltp") or 0)
        avg_price = float(stock.get("avgPrice") or 0)
        trade_volume = int(float(stock.get("tradeVolume") or 0))

        total_buy_quantity = float(
            stock.get("totalBuyQuantity") or 0
        )

        total_sell_quantity = float(
            stock.get("totalSellQuantity") or 0
        )

        # ============================================================
        # 4. MARKET DEPTH
        # ============================================================

        depth = stock.get("depth") or {}

        buy_depth = depth.get("buy") or []
        sell_depth = depth.get("sell") or []

        # ------------------------------------------------------------
        # BEST BID
        # ------------------------------------------------------------

        bid_price = 0.0
        bid_quantity = 0

        if buy_depth:

            best_bid = buy_depth[0]

            bid_price = float(
                best_bid.get("price") or 0
            )

            bid_quantity = int(
                float(best_bid.get("quantity") or 0)
            )

        # ------------------------------------------------------------
        # BEST ASK
        # ------------------------------------------------------------

        ask_price = 0.0
        ask_quantity = 0

        if sell_depth:

            best_ask = sell_depth[0]

            ask_price = float(
                best_ask.get("price") or 0
            )

            ask_quantity = int(
                float(best_ask.get("quantity") or 0)
            )

        # ============================================================
        # 5. RETURN CLEAN DATA
        # ============================================================

        return {
            "Stock": symbol.replace("-EQ", ""),
            "Symbol": symbol,

            "LTP": round(ltp, 2),

            "Average Price": round(
                avg_price,
                2
            ),

            "Trade Volume": trade_volume,

            "Total Buy Quantity": round(
                total_buy_quantity,
                2
            ),

            "Total Sell Quantity": round(
                total_sell_quantity,
                2
            ),

            "Bid Price": round(
                bid_price,
                2
            ),

            "Bid Quantity": bid_quantity,

            "Ask Price": round(
                ask_price,
                2
            ),

            "Ask Quantity": ask_quantity
        }

    except (TypeError, ValueError, KeyError, IndexError) as e:

        print(
            f"Error parsing {symbol}: {e}"
        )

        return None