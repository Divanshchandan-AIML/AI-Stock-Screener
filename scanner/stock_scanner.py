#
# Flow:
#   1. Load all NSE equity symbols/tokens from utils.token_map.TOKENS
#   2. Fetch FULL market quotes in batches of <= 50
#   3. Apply live LTP filter: Rs.30 <= LTP <= Rs.500
#   4. Fetch DAILY data only for LTP-qualified stocks
#   5. Calculate SMMA20 / SMMA120 / RSI
#   6. Fetch ONE_MINUTE data only for LTP-qualified stocks
#   7. Calculate ETQ 5m/20m/60m and average LTP 20m/60m
#   8. Extract best bid/ask market depth
#   9. Apply liquidity:
#          Bid Quantity > 1,000,000
#          AND
#          Ask Quantity > 1,000,000
#  10. Save normalized and filtered CSV files
#
# IMPORTANT:
#   Do not use this as an order-execution system.
#   It is a screening/analysis assignment.
# ============================================================

import os
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from api.historical_data import (
    get_historical_data,
    get_intraday_data,
    get_smartapi_object,
)
from indicators.smma import calculate_smma
from indicators.rsi import calculate_rsi
from signals.smma_signal import generate_signal
from utils.token_map import TOKENS


# ============================================================
# CONFIGURATION
# ============================================================

MIN_LTP = 30.00
MAX_LTP = 500.00

MIN_BID_QUANTITY = 1_000_000
MIN_ASK_QUANTITY = 1_000_000

QUOTE_BATCH_SIZE = 50
QUOTE_DELAY = 1.10

INTRADAY_MINUTES = 120
HISTORICAL_DAYS = 250

OUTPUT_DIR = os.path.join(
    "data",
    "stage17"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def normalize_symbol(symbol):
    if symbol is None:
        return ""

    symbol = str(symbol).strip().upper()

    if symbol.endswith("-EQ"):
        return symbol

    return f"{symbol}-EQ"


def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

    except Exception:
        pass

    try:
        value = str(value).replace(",", "").strip()

        if value == "":
            return default

        return float(value)

    except Exception:
        return default


# ============================================================
# STOCK UNIVERSE
# ============================================================

def load_stock_universe():
    """
    Load ALL NSE equity symbols directly from the instrument
    token map.

    This avoids the previous six-stock fallback when the
    token master already contains the NSE universe.
    """

    if not isinstance(TOKENS, dict) or not TOKENS:
        print("ERROR: TOKENS is empty.")
        return []

    symbols = []

    for symbol, token in TOKENS.items():

        symbol = normalize_symbol(symbol)

        if not symbol:
            continue

        if token is None:
            continue

        token_text = str(token).strip()

        if not token_text:
            continue

        symbols.append(symbol)

    symbols = list(dict.fromkeys(symbols))

    print()
    print("=" * 80)
    print("NSE STOCK UNIVERSE")
    print("=" * 80)
    print(f"All NSE equity stocks loaded: {len(symbols)}")

    return symbols


# ============================================================
# TOKEN MAP
# ============================================================

def build_token_map():
    """
    Build normalized symbol -> token mapping from the already
    downloaded Angel One instrument master.
    """

    mapping = {}

    if not isinstance(TOKENS, dict):
        return mapping

    for symbol, token in TOKENS.items():

        symbol = normalize_symbol(symbol)

        if not symbol:
            continue

        if token is None:
            continue

        token = str(token).strip()

        if not token:
            continue

        mapping[symbol] = token

    print(
        f"Token mappings available: {len(mapping)}"
    )

    return mapping


# ============================================================
# LIVE MARKET DATA
# ============================================================

def get_live_market_data(symbols):
    """
    Fetch FULL market data for all supplied NSE symbols.

    Angel One FULL quote requests are sent in batches of
    at most 50 tokens.

    Returns:
        {
            "SBIN-EQ": {...},
            ...
        }
    """

    api_obj = get_smartapi_object()

    if api_obj is None:
        print("ERROR: SmartAPI connection unavailable.")
        return {}

    if not hasattr(api_obj, "getMarketData"):
        print("ERROR: SmartAPI object has no getMarketData().")
        return {}

    token_map = build_token_map()

    resolved = []

    for symbol in symbols:

        symbol = normalize_symbol(symbol)

        token = token_map.get(symbol)

        if token:
            resolved.append(
                (symbol, token)
            )

    print()
    print("=" * 80)
    print("LIVE MARKET DATA")
    print("=" * 80)
    print(f"Symbols requested : {len(symbols)}")
    print(f"Tokens resolved   : {len(resolved)}")
    print(
        f"Quote batches     : "
        f"{int(np.ceil(len(resolved) / QUOTE_BATCH_SIZE))}"
    )

    market_data = {}

    for start in range(
        0,
        len(resolved),
        QUOTE_BATCH_SIZE
    ):

        batch = resolved[
            start:start + QUOTE_BATCH_SIZE
        ]

        exchange_tokens = {
            "NSE": [
                token
                for _, token in batch
            ]
        }

        batch_number = (
            start // QUOTE_BATCH_SIZE
        ) + 1

        total_batches = int(
            np.ceil(
                len(resolved)
                / QUOTE_BATCH_SIZE
            )
        )

        print()
        print(
            f"[QUOTE BATCH {batch_number}/{total_batches}] "
            f"Requesting {len(batch)} NSE stocks..."
        )

        try:

            response = api_obj.getMarketData(
                "FULL",
                exchange_tokens
            )

            if not isinstance(
                response,
                dict
            ):

                print(
                    "Invalid market-data response."
                )

                continue

            if response.get(
                "status"
            ) is not True:

                print(
                    "Market-data request failed:"
                )

                print(
                    response.get(
                        "message",
                        response
                    )
                )

                continue

            data = response.get(
                "data",
                {}
            )

            if not isinstance(
                data,
                dict
            ):

                print(
                    "Invalid market-data data object."
                )

                continue

            fetched = data.get(
                "fetched",
                []
            )

            if not isinstance(
                fetched,
                list
            ):

                print(
                    "Invalid fetched list."
                )

                continue

            print(
                f"Quotes received: {len(fetched)}"
            )

            token_to_symbol = {
                str(token).strip(): symbol
                for symbol, token in batch
            }

            for item in fetched:

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                raw_token = item.get(
                    "symbolToken",
                    item.get(
                        "symboltoken",
                        ""
                    )
                )

                token = str(
                    raw_token
                ).strip()

                symbol = token_to_symbol.get(
                    token
                )

                if symbol is None:

                    raw_symbol = item.get(
                        "tradingSymbol",
                        item.get(
                            "tradingsymbol",
                            ""
                        )
                    )

                    if raw_symbol:
                        symbol = normalize_symbol(
                            raw_symbol
                        )

                if symbol:
                    market_data[symbol] = item

        except Exception as e:

            print(
                f"Quote batch error: "
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

        if (
            start + QUOTE_BATCH_SIZE
            < len(resolved)
        ):

            time.sleep(
                QUOTE_DELAY
            )

    print()
    print(
        f"Total live quotes received: "
        f"{len(market_data)}"
    )

    return market_data


# ============================================================
# LTP EXTRACTION
# ============================================================

def extract_ltp(quote):

    if not isinstance(
        quote,
        dict
    ):
        return np.nan

    possible_fields = [
        "ltp",
        "LTP",
        "lastTradedPrice",
        "lastTradedPrice",
    ]

    for field in possible_fields:

        value = safe_float(
            quote.get(field)
        )

        if not pd.isna(value):
            return value

    return np.nan


# ============================================================
# DEPTH EXTRACTION
# ============================================================

def extract_depth(quote):

    bid_price = np.nan
    bid_quantity = np.nan

    ask_price = np.nan
    ask_quantity = np.nan

    if not isinstance(
        quote,
        dict
    ):
        return (
            bid_price,
            bid_quantity,
            ask_price,
            ask_quantity
        )

    # --------------------------------------------------------
    # Direct fields
    # --------------------------------------------------------

    bid_price = safe_float(
        quote.get(
            "bestBidPrice",
            quote.get(
                "bidPrice",
                quote.get(
                    "buyPrice",
                    np.nan
                )
            )
        )
    )

    bid_quantity = safe_float(
        quote.get(
            "bestBidQty",
            quote.get(
                "bestBidQuantity",
                quote.get(
                    "bidQty",
                    quote.get(
                        "bidQuantity",
                        quote.get(
                            "buyQty",
                            np.nan
                        )
                    )
                )
            )
        )
    )

    ask_price = safe_float(
        quote.get(
            "bestAskPrice",
            quote.get(
                "askPrice",
                quote.get(
                    "sellPrice",
                    np.nan
                )
            )
        )
    )

    ask_quantity = safe_float(
        quote.get(
            "bestAskQty",
            quote.get(
                "bestAskQuantity",
                quote.get(
                    "askQty",
                    quote.get(
                        "askQuantity",
                        quote.get(
                            "sellQty",
                            np.nan
                        )
                    )
                )
            )
        )
    )

    # --------------------------------------------------------
    # Angel One FULL depth
    # --------------------------------------------------------

    depth = quote.get(
        "depth",
        {}
    )

    if isinstance(
        depth,
        dict
    ):

        buy_depth = depth.get(
            "buy",
            []
        )

        sell_depth = depth.get(
            "sell",
            []
        )

        if (
            isinstance(
                buy_depth,
                list
            )
            and buy_depth
        ):

            first_buy = buy_depth[0]

            if isinstance(
                first_buy,
                dict
            ):

                price = safe_float(
                    first_buy.get(
                        "price",
                        np.nan
                    )
                )

                quantity = safe_float(
                    first_buy.get(
                        "quantity",
                        first_buy.get(
                            "qty",
                            np.nan
                        )
                    )
                )

                if not pd.isna(price):
                    bid_price = price

                if not pd.isna(quantity):
                    bid_quantity = quantity

        if (
            isinstance(
                sell_depth,
                list
            )
            and sell_depth
        ):

            first_sell = sell_depth[0]

            if isinstance(
                first_sell,
                dict
            ):

                price = safe_float(
                    first_sell.get(
                        "price",
                        np.nan
                    )
                )

                quantity = safe_float(
                    first_sell.get(
                        "quantity",
                        first_sell.get(
                            "qty",
                            np.nan
                        )
                    )
                )

                if not pd.isna(price):
                    ask_price = price

                if not pd.isna(quantity):
                    ask_quantity = quantity

    # --------------------------------------------------------
    # Top-level fallback lists
    # --------------------------------------------------------

    if (
        pd.isna(bid_price)
        or pd.isna(bid_quantity)
    ):

        top_buy = quote.get("buy")

        if (
            isinstance(
                top_buy,
                list
            )
            and top_buy
        ):

            first_buy = top_buy[0]

            if isinstance(
                first_buy,
                dict
            ):

                price = safe_float(
                    first_buy.get(
                        "price",
                        np.nan
                    )
                )

                quantity = safe_float(
                    first_buy.get(
                        "quantity",
                        first_buy.get(
                            "qty",
                            np.nan
                        )
                    )
                )

                if not pd.isna(price):
                    bid_price = price

                if not pd.isna(quantity):
                    bid_quantity = quantity

    if (
        pd.isna(ask_price)
        or pd.isna(ask_quantity)
    ):

        top_sell = quote.get("sell")

        if (
            isinstance(
                top_sell,
                list
            )
            and top_sell
        ):

            first_sell = top_sell[0]

            if isinstance(
                first_sell,
                dict
            ):

                price = safe_float(
                    first_sell.get(
                        "price",
                        np.nan
                    )
                )

                quantity = safe_float(
                    first_sell.get(
                        "quantity",
                        first_sell.get(
                            "qty",
                            np.nan
                        )
                    )
                )

                if not pd.isna(price):
                    ask_price = price

                if not pd.isna(quantity):
                    ask_quantity = quantity

    return (
        bid_price,
        bid_quantity,
        ask_price,
        ask_quantity
    )


# ============================================================
# RECENT INTRADAY METRICS
# ============================================================

def calculate_recent_metrics(df):

    result = {
        "ETQ 5m": np.nan,
        "ETQ 20m": np.nan,
        "ETQ 60m": np.nan,
        "Average LTP 20m": np.nan,
        "Average LTP 60m": np.nan,
    }

    if df is None or df.empty:
        return result

    close_col = None

    for col in [
        "close",
        "Close",
        "CLOSE",
    ]:

        if col in df.columns:
            close_col = col
            break

    volume_col = None

    for col in [
        "volume",
        "Volume",
        "VOLUME",
    ]:

        if col in df.columns:
            volume_col = col
            break

    if close_col is None:
        return result

    close = pd.to_numeric(
        df[close_col],
        errors="coerce"
    )

    close = close.dropna()

    # --------------------------------------------------------
    # Average LTP
    # --------------------------------------------------------

    if len(close) >= 20:

        result[
            "Average LTP 20m"
        ] = float(
            close.iloc[-20:].mean()
        )

    if len(close) >= 60:

        result[
            "Average LTP 60m"
        ] = float(
            close.iloc[-60:].mean()
        )

    # --------------------------------------------------------
    # Exchange Traded Quantity
    # --------------------------------------------------------

    if volume_col is not None:

        volume = pd.to_numeric(
            df[volume_col],
            errors="coerce"
        ).fillna(0)

        if len(volume) >= 5:

            result[
                "ETQ 5m"
            ] = float(
                volume.iloc[-5:].sum()
            )

        if len(volume) >= 20:

            result[
                "ETQ 20m"
            ] = float(
                volume.iloc[-20:].sum()
            )

        if len(volume) >= 60:

            result[
                "ETQ 60m"
            ] = float(
                volume.iloc[-60:].sum()
            )

    return result


# ============================================================
# CROSSOVER DETECTION
# ============================================================

def detect_latest_crossover(
    close_prices,
    smma20,
    smma120,
):
    """
    Detect the most recent completed SMMA crossover.

    BUY:
        SMMA20 crosses from <= SMMA120 to > SMMA120

    SELL:
        SMMA20 crosses from >= SMMA120 to < SMMA120
    """

    temp = pd.DataFrame({
        "close": pd.Series(
            close_prices
        ).reset_index(drop=True),

        "smma20": pd.Series(
            smma20
        ).reset_index(drop=True),

        "smma120": pd.Series(
            smma120
        ).reset_index(drop=True),
    })

    temp = temp.dropna(
        subset=[
            "smma20",
            "smma120"
        ]
    ).reset_index(
        drop=True
    )

    if len(temp) < 2:

        return {
            "Crossover": "NONE",
            "Crossover Index": np.nan,
            "Crossover Price": np.nan,
        }

    diff = (
        temp["smma20"]
        - temp["smma120"]
    )

    crossover_type = "NONE"
    crossover_index = np.nan
    crossover_price = np.nan

    for i in range(
        1,
        len(temp)
    ):

        previous_diff = diff.iloc[i - 1]
        current_diff = diff.iloc[i]

        if (
            previous_diff <= 0
            and current_diff > 0
        ):

            crossover_type = "BUY"

            crossover_index = i

            crossover_price = (
                temp["close"].iloc[i]
            )

        elif (
            previous_diff >= 0
            and current_diff < 0
        ):

            crossover_type = "SELL"

            crossover_index = i

            crossover_price = (
                temp["close"].iloc[i]
            )

    return {
        "Crossover": crossover_type,
        "Crossover Index": crossover_index,
        "Crossover Price": crossover_price,
    }


# ============================================================
# SCAN ONE STOCK
# ============================================================

def scan_stock(
    symbol,
    market_quote
):

    try:

        symbol = normalize_symbol(symbol)

        live_ltp = extract_ltp(
            market_quote
        )

        if pd.isna(live_ltp):

            print(
                f"{symbol}: LTP unavailable."
            )

            return None

        # ====================================================
        # LTP FILTER
        # ====================================================

        if not (
            MIN_LTP
            <= live_ltp
            <= MAX_LTP
        ):

            return None

        print()
        print(
            f"Scanning {symbol}..."
        )

        print(
            f"{symbol}: Live LTP = {live_ltp:.2f}"
        )

        # ====================================================
        # DAILY HISTORICAL DATA
        # ====================================================

        daily_df = get_historical_data(
            symbol,
            days=HISTORICAL_DAYS
        )

        if (
            daily_df is None
            or daily_df.empty
        ):

            print(
                f"{symbol}: Daily data unavailable."
            )

            return None

        close_column = None

        for column in [
            "close",
            "Close",
            "CLOSE",
        ]:

            if column in daily_df.columns:

                close_column = column
                break

        if close_column is None:

            print(
                f"{symbol}: Close column missing."
            )

            return None

        close_prices = pd.to_numeric(
            daily_df[close_column],
            errors="coerce"
        ).dropna()

        if len(close_prices) < 120:

            print(
                f"{symbol}: Only "
                f"{len(close_prices)} daily rows; "
                f"SMMA120 unavailable."
            )

            return None

        # ====================================================
        # SMMA
        # ====================================================

        smma20 = calculate_smma(
            close_prices,
            20
        )

        smma120 = calculate_smma(
            close_prices,
            120
        )

        # ====================================================
        # RSI
        # ====================================================

        rsi = calculate_rsi(
            close_prices,
            14
        )

        latest_smma20 = safe_float(
            smma20.iloc[-1]
        )

        latest_smma120 = safe_float(
            smma120.iloc[-1]
        )

        latest_rsi = safe_float(
            rsi.iloc[-1]
        )

        if any(
            pd.isna(value)
            for value in [
                latest_smma20,
                latest_smma120,
                latest_rsi
            ]
        ):

            print(
                f"{symbol}: Indicator values unavailable."
            )

            return None

        # ====================================================
        # CROSSOVER
        # ====================================================

        crossover = detect_latest_crossover(
            close_prices,
            smma20,
            smma120
        )

        # ====================================================
        # CURRENT SIGNAL
        # ====================================================

        signal_result = generate_signal(
            live_ltp,
            latest_smma20,
            latest_smma120,
            latest_rsi
        )

        if isinstance(
            signal_result,
            dict
        ):

            signal = signal_result.get(
                "Signal",
                "HOLD"
            )

            confidence = signal_result.get(
                "Confidence",
                np.nan
            )

            reasons = signal_result.get(
                "Reasons",
                ""
            )

        else:

            signal = str(
                signal_result
            )

            confidence = np.nan
            reasons = ""

        # ====================================================
        # MARKET DEPTH
        # ====================================================

        (
            bid_price,
            bid_quantity,
            ask_price,
            ask_quantity
        ) = extract_depth(
            market_quote
        )

        total_buy_quantity = safe_float(
            market_quote.get(
                "totBuyQuan",
                market_quote.get(
                    "totalBuyQuantity",
                    np.nan
                )
            )
            if isinstance(
                market_quote,
                dict
            )
            else np.nan
        )

        total_sell_quantity = safe_float(
            market_quote.get(
                "totSellQuan",
                market_quote.get(
                    "totalSellQuantity",
                    np.nan
                )
            )
            if isinstance(
                market_quote,
                dict
            )
            else np.nan
        )

        trade_volume = safe_float(
            market_quote.get(
                "tradeVolume",
                market_quote.get(
                    "volume",
                    np.nan
                )
            )
            if isinstance(
                market_quote,
                dict
            )
            else np.nan
        )

        last_trade_quantity = safe_float(
            market_quote.get(
                "lastTradeQty",
                market_quote.get(
                    "lastTradeQuantity",
                    np.nan
                )
            )
            if isinstance(
                market_quote,
                dict
            )
            else np.nan
        )

        average_price = safe_float(
            market_quote.get(
                "avgPrice",
                market_quote.get(
                    "averagePrice",
                    np.nan
                )
            )
            if isinstance(
                market_quote,
                dict
            )
            else np.nan
        )

        # ====================================================
        # INTRADAY DATA
        # ====================================================

        print(
            f"{symbol}: Getting ONE_MINUTE data..."
        )

        intraday_df = get_intraday_data(
            symbol,
            minutes=INTRADAY_MINUTES
        )

        recent_metrics = (
            calculate_recent_metrics(
                intraday_df
            )
        )

        intraday_rows = (
            0
            if intraday_df is None
            else len(intraday_df)
        )

        print(
            f"{symbol}: Intraday rows = "
            f"{intraday_rows}"
        )

        # ====================================================
        # LIQUIDITY
        # ====================================================

        bid_ok = (
            not pd.isna(bid_quantity)
            and bid_quantity
            > MIN_BID_QUANTITY
        )

        ask_ok = (
            not pd.isna(ask_quantity)
            and ask_quantity
            > MIN_ASK_QUANTITY
        )

        liquidity_pass = (
            bid_ok
            and ask_ok
        )

        if pd.isna(bid_quantity):
            bid_text = "NA"
        else:
            bid_text = f"{bid_quantity:,.0f}"

        if pd.isna(ask_quantity):
            ask_text = "NA"
        else:
            ask_text = f"{ask_quantity:,.0f}"

        print(
            f"{symbol}: "
            f"Bid Qty={bid_text}, "
            f"Ask Qty={ask_text}, "
            f"Liquidity="
            f"{'PASS' if liquidity_pass else 'FAIL'}"
        )

        # ====================================================
        # RESULT
        # ====================================================

        result = {

            "Stock": symbol.replace(
                "-EQ",
                ""
            ),

            "Symbol": symbol,

            "LTP": live_ltp,

            "Price": live_ltp,

            "Bid Price": bid_price,

            "Bid Quantity": bid_quantity,

            "Ask Price": ask_price,

            "Ask Quantity": ask_quantity,

            "Total Buy Quantity":
                total_buy_quantity,

            "Total Sell Quantity":
                total_sell_quantity,

            "Last Trade Quantity":
                last_trade_quantity,

            "Trade Volume":
                trade_volume,

            "Average Price":
                average_price,

            "SMMA20":
                latest_smma20,

            "SMMA120":
                latest_smma120,

            "RSI":
                latest_rsi,

            "ETQ 5m":
                recent_metrics[
                    "ETQ 5m"
                ],

            "ETQ 20m":
                recent_metrics[
                    "ETQ 20m"
                ],

            "ETQ 60m":
                recent_metrics[
                    "ETQ 60m"
                ],

            "Average LTP 20m":
                recent_metrics[
                    "Average LTP 20m"
                ],

            "Average LTP 60m":
                recent_metrics[
                    "Average LTP 60m"
                ],

            "Crossover":
                crossover[
                    "Crossover"
                ],

            "Crossover Price":
                crossover[
                    "Crossover Price"
                ],

            "Signal":
                str(
                    signal
                ).upper(),

            "Confidence":
                safe_float(
                    confidence
                ),

            "Liquidity Pass":
                liquidity_pass,

            "LTP Pass":
                True,

            "Reasons":
                str(
                    reasons
                ),

            "Scan Time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
        }

        return result

    except Exception as e:

        print()
        print(
            f"ERROR scanning {symbol}: "
            f"{type(e).__name__}: {e}"
        )

        traceback.print_exc()

        return None


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    result_df,
    filtered_df
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    normalized_path = os.path.join(
        OUTPUT_DIR,
        "stage17_normalized.csv"
    )

    filtered_path = os.path.join(
        OUTPUT_DIR,
        "stage17_filtered.csv"
    )

    timestamp_path = os.path.join(
        OUTPUT_DIR,
        f"stage17_{timestamp}.csv"
    )

    result_df.to_csv(
        normalized_path,
        index=False
    )

    filtered_df.to_csv(
        filtered_path,
        index=False
    )

    result_df.to_csv(
        timestamp_path,
        index=False
    )

    print()
    print(
        f"Saved: {normalized_path}"
    )

    print(
        f"Saved: {filtered_path}"
    )

    print(
        f"Saved: {timestamp_path}"
    )


# ============================================================
# MAIN SCANNER
# ============================================================

def scan_all_stocks():

    print()
    print("=" * 80)
    print("STAGE 17 - ALL NSE LIVE STOCK SCREENING")
    print("=" * 80)

    # ========================================================
    # LOAD ALL NSE STOCKS
    # ========================================================

    symbols = load_stock_universe()

    if not symbols:

        print(
            "No NSE stocks available."
        )

        return pd.DataFrame()

    total_stocks = len(symbols)

    # ========================================================
    # FETCH LIVE QUOTES FOR ALL NSE STOCKS
    # ========================================================

    market_data = get_live_market_data(
        symbols
    )

    if not market_data:

        print(
            "No live market data received."
        )

        return pd.DataFrame()

    # ========================================================
    # PRE-SCREEN LTP BEFORE HISTORICAL API
    # ========================================================

    print()
    print("=" * 80)
    print("PRE-SCREENING LIVE LTP")
    print("=" * 80)

    ltp_candidates = []

    for symbol, quote in market_data.items():

        ltp = extract_ltp(
            quote
        )

        if pd.isna(ltp):
            continue

        if (
            MIN_LTP
            <= ltp
            <= MAX_LTP
        ):

            ltp_candidates.append(
                symbol
            )

    print(
        f"Stocks within LTP range "
        f"Rs.{MIN_LTP:.2f}-Rs.{MAX_LTP:.2f}: "
        f"{len(ltp_candidates)}"
    )

    if not ltp_candidates:

        print(
            "No stocks passed the LTP filter."
        )

        empty_df = pd.DataFrame()

        save_results(
            empty_df,
            empty_df
        )

        return empty_df

    # ========================================================
    # SCAN ONLY LTP QUALIFIED STOCKS
    # ========================================================

    results = []

    failed = 0

    for index, symbol in enumerate(
        ltp_candidates,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(ltp_candidates)}] "
            f"{symbol}"
        )

        quote = market_data.get(
            symbol
        )

        result = scan_stock(
            symbol,
            quote
        )

        if result is not None:

            results.append(
                result
            )

        else:

            failed += 1

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    if results:

        result_df = pd.DataFrame(
            results
        )

    else:

        result_df = pd.DataFrame()

    # ========================================================
    # LIQUIDITY FILTER
    # ========================================================

    print()
    print("=" * 80)
    print("LIQUIDITY FILTER")
    print("=" * 80)

    print(
        f"Required Bid Quantity > "
        f"{MIN_BID_QUANTITY:,}"
    )

    print(
        f"Required Ask Quantity > "
        f"{MIN_ASK_QUANTITY:,}"
    )

    if result_df.empty:

        filtered_df = pd.DataFrame()

    else:

        filtered_df = result_df[
            (
                result_df[
                    "Bid Quantity"
                ]
                > MIN_BID_QUANTITY
            )
            &
            (
                result_df[
                    "Ask Quantity"
                ]
                > MIN_ASK_QUANTITY
            )
        ].copy()

    # ========================================================
    # DISPLAY DEPTH DIAGNOSTICS
    # ========================================================

    if not result_df.empty:

        for _, row in result_df.iterrows():

            print()
            print(
                f"{row['Stock']}:"
            )

            print(
                f"  Bid Price      : "
                f"{row['Bid Price']}"
            )

            print(
                f"  Bid Quantity   : "
                f"{row['Bid Quantity']}"
            )

            print(
                f"  Ask Price      : "
                f"{row['Ask Price']}"
            )

            print(
                f"  Ask Quantity   : "
                f"{row['Ask Quantity']}"
            )

            print(
                f"  Liquidity      : "
                f"{'PASS' if row['Liquidity Pass'] else 'FAIL'}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    save_results(
        result_df,
        filtered_df
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 80)
    print("TECHNICAL + INTRADAY ANALYSIS")
    print("=" * 80)

    if filtered_df.empty:

        print(
            "No stocks passed the complete "
            "LTP + liquidity screening criteria."
        )

    else:

        display_columns = [
            "Stock",
            "LTP",
            "Bid Price",
            "Bid Quantity",
            "Ask Price",
            "Ask Quantity",
            "SMMA20",
            "SMMA120",
            "RSI",
            "ETQ 5m",
            "ETQ 20m",
            "ETQ 60m",
            "Average LTP 20m",
            "Average LTP 60m",
            "Crossover",
            "Signal",
            "Confidence",
            "Liquidity Pass",
            "Reasons",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in filtered_df.columns
        ]

        print(
            filtered_df[
                display_columns
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("STAGE 17 SCAN COMPLETE")
    print("=" * 80)

    print(
        f"All NSE stocks        : {total_stocks}"
    )

    print(
        f"Live quotes received  : {len(market_data)}"
    )

    print(
        f"LTP-qualified stocks  : "
        f"{len(ltp_candidates)}"
    )

    print(
        f"Valid analysis results: "
        f"{len(result_df)}"
    )

    print(
        f"Qualified stocks      : "
        f"{len(filtered_df)}"
    )

    print(
        f"Failed/skipped        : "
        f"{failed}"
    )

    print()
    print("=" * 80)
    print("STAGE 17 COMPLETE")
    print("=" * 80)

    return filtered_df


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    try:

        results = scan_all_stocks()

        print()

        if results.empty:

            print(
                "No results returned."
            )

        else:

            print(
                f"Successfully qualified "
                f"{len(results)} stocks."
            )

    except KeyboardInterrupt:

        print()
        print(
            "Scanner stopped by user."
        )

    except Exception as e:

        print()
        print(
            "FATAL SCANNER ERROR:"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        traceback.print_exc()