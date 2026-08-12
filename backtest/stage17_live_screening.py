# ============================================================
# STAGE 17 - LIVE STOCK SCREENING
# Assignment 1 - AI/ML Quantitative Programming
# ============================================================

import os
import sys
import time
import importlib
import traceback
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INITIAL_CAPITAL = 100000.00

SMMA_FAST = 20
SMMA_SLOW = 120

HOLDING_PERIOD = 60

STOP_LOSS_PERCENT = 5.00
TAKE_PROFIT_PERCENT = 10.00

MIN_LTP = 30.00
MAX_LTP = 500.00

# Recruiter screening rules
MIN_BID_QUANTITY = 1_000_000
MIN_ASK_QUANTITY = 1_000_000

REFRESH_SECONDS = 30

OUTPUT_DIR = "data/stage17"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

def print_header():

    print()
    print("=" * 100)
    print("STAGE 17 - LIVE STOCK SCREENING")
    print("=" * 100)

    print()
    print("PAPER / ANALYSIS ONLY")
    print("NO REAL ORDERS WILL BE PLACED")

    print()
    print("=" * 100)
    print("LOCKED STRATEGY")
    print("=" * 100)

    print(
        f"SMMA Fast       : {SMMA_FAST}"
    )

    print(
        f"SMMA Slow       : {SMMA_SLOW}"
    )

    print(
        f"Holding Period  : "
        f"{HOLDING_PERIOD} trading days"
    )

    print(
        f"Stop Loss       : "
        f"{STOP_LOSS_PERCENT:.2f}%"
    )

    print(
        f"Take Profit     : "
        f"{TAKE_PROFIT_PERCENT:.2f}%"
    )

    print(
        f"Initial Capital : "
        f"Rs. {INITIAL_CAPITAL:,.2f}"
    )

    print()
    print("=" * 100)
    print("RECRUITER SCREENING RULES")
    print("=" * 100)

    print(
        f"LTP Range       : "
        f"Rs. {MIN_LTP:.2f} - Rs. {MAX_LTP:.2f}"
    )

    print(
        f"Bid Quantity    : "
        f"> {MIN_BID_QUANTITY:,}"
    )

    print(
        f"Ask Quantity    : "
        f"> {MIN_ASK_QUANTITY:,}"
    )

    print(
        f"Refresh         : "
        f"{REFRESH_SECONDS} seconds"
    )

    print()
    print(
        f"Output directory : "
        f"{OUTPUT_DIR}"
    )


# ============================================================
# LOAD STOCK SCANNER
# ============================================================

def load_stock_scanner():

    print()
    print("=" * 100)
    print("LOADING STOCK SCANNER")
    print("=" * 100)

    module_name = "scanner.stock_scanner"

    # --------------------------------------------------------
    # Detect accidental circular import
    # --------------------------------------------------------

    if module_name in sys.modules:

        existing_module = sys.modules[
            module_name
        ]

        if not hasattr(
            existing_module,
            "scan_all_stocks"
        ):

            print()
            print(
                "ERROR: CIRCULAR IMPORT DETECTED"
            )

            print()
            print(
                "scanner.stock_scanner is partially "
                "initialized."
            )

            print()
            print(
                "Do NOT import Stage 17 from "
                "scanner/stock_scanner.py."
            )

            print()
            print(
                "Remove imports such as:"
            )

            print(
                "from backtest.stage17_live_screening "
                "import ..."
            )

            print(
                "import backtest.stage17_live_screening"
            )

            return None

    # --------------------------------------------------------
    # Import scanner
    # --------------------------------------------------------

    try:

        module = importlib.import_module(
            module_name
        )

    except Exception as e:

        print()
        print(
            "ERROR importing "
            "scanner.stock_scanner"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        print()
        print(
            "Full traceback:"
        )

        traceback.print_exc()

        return None

    print(
        "Scanner module imported successfully."
    )

    # --------------------------------------------------------
    # Find scan_all_stocks
    # --------------------------------------------------------

    scan_function = getattr(
        module,
        "scan_all_stocks",
        None
    )

    if scan_function is None:

        print()
        print(
            "ERROR: scan_all_stocks() was not found "
            "inside scanner/stock_scanner.py"
        )

        print()
        print(
            "Available public functions:"
        )

        for name in dir(module):

            if not name.startswith("_"):

                attribute = getattr(
                    module,
                    name,
                    None
                )

                if callable(attribute):

                    print(
                        f"  - {name}"
                    )

        return None

    if not callable(
        scan_function
    ):

        print(
            "ERROR: scan_all_stocks exists "
            "but is not callable."
        )

        return None

    print(
        "scan_all_stocks() found successfully."
    )

    return scan_function


# ============================================================
# CONVERT SCANNER OUTPUT TO DATAFRAME
# ============================================================

def convert_to_dataframe(data):

    if data is None:

        return pd.DataFrame()

    if isinstance(
        data,
        pd.DataFrame
    ):

        return data.copy()

    if isinstance(
        data,
        list
    ):

        if len(data) == 0:

            return pd.DataFrame()

        return pd.DataFrame(
            data
        )

    if isinstance(
        data,
        tuple
    ):

        if len(data) == 0:

            return pd.DataFrame()

        return pd.DataFrame(
            list(data)
        )

    if isinstance(
        data,
        dict
    ):

        return pd.DataFrame(
            [data]
        )

    raise TypeError(
        "Unsupported scanner output type: "
        f"{type(data)}"
    )


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    df,
    names
):

    if df.empty:

        return None

    normalized = {}

    for column in df.columns:

        key = (
            str(column)
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        normalized[key] = column

    for name in names:

        key = (
            str(name)
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        if key in normalized:

            return normalized[key]

    return None


# ============================================================
# GET VALUE
# ============================================================

def get_value(
    row,
    names,
    default=np.nan
):

    for name in names:

        column = find_column(
            pd.DataFrame([row]),
            [name]
        )

        if column is None:

            continue

        try:

            value = row[column]

        except Exception:

            continue

        if value is None:

            continue

        try:

            if pd.isna(value):

                continue

        except Exception:

            pass

        return value

    return default


# ============================================================
# NUMBER
# ============================================================

def to_number(
    value,
    default=np.nan
):

    if value is None:

        return default

    try:

        if pd.isna(value):

            return default

    except Exception:

        pass

    try:

        if isinstance(
            value,
            str
        ):

            value = (
                value
                .replace(",", "")
                .replace("%", "")
                .strip()
            )

            if value == "":

                return default

        return float(
            value
        )

    except Exception:

        return default


# ============================================================
# TEXT
# ============================================================

def to_text(
    value,
    default=""
):

    if value is None:

        return default

    try:

        if pd.isna(value):

            return default

    except Exception:

        pass

    text = str(
        value
    ).strip()

    if not text:

        return default

    return text


# ============================================================
# NORMALIZE SCANNER DATA
# ============================================================

def normalize_data(df):

    if df.empty:

        return pd.DataFrame()

    print()
    print("=" * 100)
    print("NORMALIZING SCANNER DATA")
    print("=" * 100)

    print(
        "Scanner columns:"
    )

    for column in df.columns:

        print(
            f"  - {column}"
        )

    records = []

    for _, row in df.iterrows():

        # ----------------------------------------------------
        # STOCK
        # ----------------------------------------------------

        stock = to_text(
            get_value(
                row,
                [
                    "Stock",
                    "Symbol",
                    "TradingSymbol",
                    "Trading Symbol",
                    "Ticker",
                    "Scrip"
                ],
                ""
            )
        )

        if not stock:

            continue

        # ----------------------------------------------------
        # LTP
        # ----------------------------------------------------

        ltp = to_number(
            get_value(
                row,
                [
                    "LTP",
                    "Price",
                    "LastPrice",
                    "Last Price",
                    "Last Traded Price",
                    "Close"
                ]
            )
        )

        if pd.isna(
            ltp
        ):

            print(
                f"Skipping {stock}: "
                f"LTP unavailable"
            )

            continue

        # ----------------------------------------------------
        # SMMA
        # ----------------------------------------------------

        smma20 = to_number(
            get_value(
                row,
                [
                    "SMMA20",
                    "SMMA 20",
                    "SMMA_20"
                ]
            )
        )

        smma120 = to_number(
            get_value(
                row,
                [
                    "SMMA120",
                    "SMMA 120",
                    "SMMA_120"
                ]
            )
        )

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        rsi = to_number(
            get_value(
                row,
                ["RSI"]
            )
        )

        # ----------------------------------------------------
        # BEST BID
        # ----------------------------------------------------

        bid_price = to_number(
            get_value(
                row,
                [
                    "Bid Price",
                    "BidPrice",
                    "Best Bid Price",
                    "bestBidPrice",
                    "bid_price",
                    "Bid"
                ]
            )
        )

        best_bid_quantity = to_number(
            get_value(
                row,
                [
                    "Bid Quantity",
                    "BidQuantity",
                    "Best Bid Quantity",
                    "Best Bid Qty",
                    "bestBidQty",
                    "bid_quantity",
                    "Bid Qty",
                    "BidQty"
                ]
            )
        )

        # ----------------------------------------------------
        # BEST ASK
        # ----------------------------------------------------

        ask_price = to_number(
            get_value(
                row,
                [
                    "Ask Price",
                    "AskPrice",
                    "Best Ask Price",
                    "bestAskPrice",
                    "ask_price",
                    "Ask"
                ]
            )
        )

        best_ask_quantity = to_number(
            get_value(
                row,
                [
                    "Ask Quantity",
                    "AskQuantity",
                    "Best Ask Quantity",
                    "Best Ask Qty",
                    "bestAskQty",
                    "ask_quantity",
                    "Ask Qty",
                    "AskQty"
                ]
            )
        )

        # ----------------------------------------------------
        # TOTAL BUY / SELL QUANTITY
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # These are separate from best bid/ask quantities.
        #
        # Angel One FULL market data can provide:
        #
        # totBuyQuan
        # totSellQuan
        #
        # These represent aggregate buy/sell quantities.
        #
        # Stage 17 liquidity screening uses these fields.
        # ----------------------------------------------------

        total_buy_quantity = to_number(
            get_value(
                row,
                [
                    "Total Buy Quantity",
                    "TotalBuyQuantity",
                    "totBuyQuan",
                    "Total Buy Qty",
                    "Buy Quantity Total"
                ]
            )
        )

        total_sell_quantity = to_number(
            get_value(
                row,
                [
                    "Total Sell Quantity",
                    "TotalSellQuantity",
                    "totSellQuan",
                    "Total Sell Qty",
                    "Sell Quantity Total"
                ]
            )
        )

        # ----------------------------------------------------
        # LAST TRADE QUANTITY
        # ----------------------------------------------------

        last_trade_quantity = to_number(
            get_value(
                row,
                [
                    "Last Trade Quantity",
                    "LastTradeQuantity",
                    "lastTradeQty",
                    "Last Trade Qty"
                ]
            )
        )

        # ----------------------------------------------------
        # TRADE VOLUME
        # ----------------------------------------------------

        trade_volume = to_number(
            get_value(
                row,
                [
                    "Trade Volume",
                    "TradeVolume",
                    "tradeVolume",
                    "Volume"
                ]
            )
        )

        # ----------------------------------------------------
        # AVERAGE PRICE
        # ----------------------------------------------------

        average_price = to_number(
            get_value(
                row,
                [
                    "Average Price",
                    "AveragePrice",
                    "avgPrice"
                ]
            )
        )

        # ----------------------------------------------------
        # EXCHANGE TRADED QUANTITY
        # ----------------------------------------------------

        etq5 = to_number(
            get_value(
                row,
                [
                    "ETQ 5m",
                    "ETQ5m",
                    "ETQ_5m",
                    "Volume 5m",
                    "Exchange Traded Quantity 5m"
                ]
            )
        )

        etq20 = to_number(
            get_value(
                row,
                [
                    "ETQ 20m",
                    "ETQ20m",
                    "ETQ_20m",
                    "Volume 20m",
                    "Exchange Traded Quantity 20m"
                ]
            )
        )

        etq60 = to_number(
            get_value(
                row,
                [
                    "ETQ 60m",
                    "ETQ60m",
                    "ETQ_60m",
                    "Volume 60m",
                    "Exchange Traded Quantity 60m"
                ]
            )
        )

        # ----------------------------------------------------
        # AVERAGE LTP
        # ----------------------------------------------------

        avg20 = to_number(
            get_value(
                row,
                [
                    "Average LTP 20m",
                    "Average_LTP_20m",
                    "Avg LTP 20m",
                    "Average Price 20m",
                    "Average LTP 20 Minutes"
                ]
            )
        )

        avg60 = to_number(
            get_value(
                row,
                [
                    "Average LTP 60m",
                    "Average_LTP_60m",
                    "Avg LTP 60m",
                    "Average Price 60m",
                    "Average LTP 60 Minutes"
                ]
            )
        )

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        signal = to_text(
            get_value(
                row,
                [
                    "Signal",
                    "signal"
                ],
                "HOLD"
            ),
            "HOLD"
        ).upper()

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = to_number(
            get_value(
                row,
                [
                    "Confidence",
                    "Probability",
                    "Probability %",
                    "Success Probability",
                    "AI Probability"
                ]
            )
        )

        # ----------------------------------------------------
        # REASONS
        # ----------------------------------------------------

        reasons = to_text(
            get_value(
                row,
                [
                    "Reasons",
                    "Reason",
                    "AI Reason",
                    "Explanation",
                    "AI Explanation"
                ],
                ""
            )
        )

        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        records.append({

            "Stock": stock.replace(
                "-EQ",
                ""
            ),

            "Symbol": stock,

            "LTP": ltp,

            # Best market depth
            "Bid Price": bid_price,
            "Bid Quantity": best_bid_quantity,

            "Ask Price": ask_price,
            "Ask Quantity": best_ask_quantity,

            # Aggregate liquidity
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

            "SMMA20": smma20,
            "SMMA120": smma120,

            "RSI": rsi,

            "ETQ 5m": etq5,
            "ETQ 20m": etq20,
            "ETQ 60m": etq60,

            "Average LTP 20m": avg20,
            "Average LTP 60m": avg60,

            "Signal": signal,

            "Confidence": confidence,

            "Reasons": reasons,

            "Scan Time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        })

    return pd.DataFrame(
        records
    )


# ============================================================
# APPLY LTP FILTER
# ============================================================

def apply_ltp_filter(df):

    if df.empty:

        return df.copy()

    result = df.copy()

    result = result[
        result["LTP"].notna()
    ]

    result = result[
        (result["LTP"] >= MIN_LTP)
        &
        (result["LTP"] <= MAX_LTP)
    ]

    return result.reset_index(
        drop=True
    )


# ============================================================
# APPLY LIQUIDITY FILTER
# ============================================================
# ============================================================
# APPLY LIQUIDITY FILTER
# ASSIGNMENT REQUIREMENT:
#
# Bid Quantity > 10,00,000
# AND
# Ask Quantity > 10,00,000
# ============================================================

def apply_liquidity_filter(df):

    if df.empty:
        return df.copy()

    result = df.copy()

    print()
    print("=" * 100)
    print("LIQUIDITY FILTER")
    print("=" * 100)

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "Bid Quantity",
        "Ask Quantity"
    ]

    for column in required_columns:

        if column not in result.columns:

            print(
                f"[ERROR] Required column missing: {column}"
            )

            return pd.DataFrame()

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    result["Bid Quantity"] = pd.to_numeric(
        result["Bid Quantity"],
        errors="coerce"
    )

    result["Ask Quantity"] = pd.to_numeric(
        result["Ask Quantity"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Diagnostic
    # --------------------------------------------------------

    for _, row in result.iterrows():

        stock = row.get(
            "Stock",
            "UNKNOWN"
        )

        bid_quantity = row["Bid Quantity"]
        ask_quantity = row["Ask Quantity"]

        bid_pass = (
            pd.notna(bid_quantity)
            and bid_quantity > MIN_BID_QUANTITY
        )

        ask_pass = (
            pd.notna(ask_quantity)
            and ask_quantity > MIN_ASK_QUANTITY
        )

        print()
        print(f"{stock}:")

        print(
            f"  Bid Quantity : {bid_quantity}"
        )

        print(
            f"  Required     : > {MIN_BID_QUANTITY:,}"
        )

        print(
            f"  Bid condition: "
            f"{'PASS' if bid_pass else 'FAIL'}"
        )

        print(
            f"  Ask Quantity : {ask_quantity}"
        )

        print(
            f"  Required     : > {MIN_ASK_QUANTITY:,}"
        )

        print(
            f"  Ask condition: "
            f"{'PASS' if ask_pass else 'FAIL'}"
        )

    # --------------------------------------------------------
    # ACTUAL ASSIGNMENT FILTER
    # --------------------------------------------------------

    result = result[
        result["Bid Quantity"].notna()
        &
        result["Ask Quantity"].notna()
        &
        (result["Bid Quantity"] > MIN_BID_QUANTITY)
        &
        (result["Ask Quantity"] > MIN_ASK_QUANTITY)
    ]

    return result.reset_index(drop=True)


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(df):

    print()
    print("=" * 180)
    print("LIVE STOCK SCREENING RESULTS")
    print("=" * 180)

    if df.empty:

        print(
            "No stocks passed the complete "
            "screening criteria."
        )

        return

    columns = [

        "Stock",

        "LTP",

        "Bid Price",
        "Bid Quantity",

        "Ask Price",
        "Ask Quantity",

        "Total Buy Quantity",
        "Total Sell Quantity",

        "SMMA20",
        "SMMA120",

        "ETQ 5m",
        "ETQ 20m",
        "ETQ 60m",

        "Average LTP 20m",
        "Average LTP 60m",

        "RSI",

        "Signal",
        "Confidence"
    ]

    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    display_df = df[
        columns
    ].copy()

    print(
        display_df.to_string(
            index=False
        )
    )


# ============================================================
# DISPLAY DATA AVAILABILITY
# ============================================================

def display_data_availability(df):

    print()
    print("=" * 100)
    print("ASSIGNMENT DATA AVAILABILITY")
    print("=" * 100)

    required = [

        "LTP",

        "Bid Price",
        "Bid Quantity",

        "Ask Price",
        "Ask Quantity",

        "Total Buy Quantity",
        "Total Sell Quantity",

        "SMMA20",
        "SMMA120",

        "ETQ 5m",
        "ETQ 20m",
        "ETQ 60m",

        "Average LTP 20m",
        "Average LTP 60m"
    ]

    for field in required:

        if field not in df.columns:

            print(
                f"[MISSING] {field}"
            )

            continue

        count = df[
            field
        ].notna().sum()

        if count > 0:

            print(
                f"[OK]      "
                f"{field} ({count} rows)"
            )

        else:

            print(
                f"[MISSING] {field}"
            )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    normalized_df,
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

    history_path = os.path.join(
        OUTPUT_DIR,
        f"stage17_{timestamp}.csv"
    )

    try:

        normalized_df.to_csv(
            normalized_path,
            index=False
        )

        filtered_df.to_csv(
            filtered_path,
            index=False
        )

        filtered_df.to_csv(
            history_path,
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
            f"Saved: {history_path}"
        )

    except Exception as e:

        print()
        print(
            "ERROR saving Stage 17 output:"
        )

        print(
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# ONE LIVE SCAN
# ============================================================

def run_one_scan():

    print_header()

    print()
    print("=" * 100)
    print("STEP 1 - LIVE MARKET SCAN")
    print("=" * 100)

    print()
    print(
        "RUNNING LIVE STOCK SCANNER"
    )

    # --------------------------------------------------------
    # Load scanner
    # --------------------------------------------------------

    scan_all_stocks = load_stock_scanner()

    if scan_all_stocks is None:

        print()
        print("=" * 100)
        print("STAGE 17 STOPPED")
        print("=" * 100)

        print(
            "Scanner could not be loaded."
        )

        return None

    # --------------------------------------------------------
    # Run scanner
    # --------------------------------------------------------

    try:

        scanner_result = scan_all_stocks()

    except TypeError as e:

        print()
        print(
            "SCANNER CALL FAILED"
        )

        print(
            f"TypeError: {e}"
        )

        print()
        print(
            "The scanner function exists but "
            "requires different arguments."
        )

        traceback.print_exc()

        return None

    except Exception as e:

        print()
        print(
            "SCANNER EXECUTION FAILED"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        traceback.print_exc()

        return None

    # --------------------------------------------------------
    # Convert scanner output
    # --------------------------------------------------------

    try:

        raw_df = convert_to_dataframe(
            scanner_result
        )

    except Exception as e:

        print()
        print(
            "ERROR converting scanner output:"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        return None

    print()
    print(
        f"Stocks returned by scanner: "
        f"{len(raw_df)}"
    )

    if raw_df.empty:

        print()
        print(
            "No live market data returned."
        )

        return None

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized_df = normalize_data(
        raw_df
    )

    print()
    print(
        f"Normalized stocks: "
        f"{len(normalized_df)}"
    )

    if normalized_df.empty:

        print()
        print(
            "No usable stock data "
            "after normalization."
        )

        return None

    # --------------------------------------------------------
    # Data availability
    # --------------------------------------------------------

    display_data_availability(
        normalized_df
    )

    # --------------------------------------------------------
    # LTP filter
    # --------------------------------------------------------

    ltp_df = apply_ltp_filter(
        normalized_df
    )

    print()
    print(
        f"Stocks after LTP filter: "
        f"{len(ltp_df)}"
    )

    # --------------------------------------------------------
    # Liquidity filter
    # --------------------------------------------------------

    filtered_df = apply_liquidity_filter(
        ltp_df
    )

    print()
    print(
        f"Stocks after liquidity filter: "
        f"{len(filtered_df)}"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_results(
        filtered_df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        normalized_df,
        filtered_df
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("STAGE 17 SCAN COMPLETE")
    print("=" * 100)

    print(
        f"Stocks scanned   : "
        f"{len(raw_df)}"
    )

    print(
        f"Valid results    : "
        f"{len(normalized_df)}"
    )

    print(
        f"Qualified stocks : "
        f"{len(filtered_df)}"
    )

    return filtered_df


# ============================================================
# CONTINUOUS LIVE MODE
# ============================================================

def live_mode():

    print()
    print("=" * 100)
    print("STAGE 17 CONTINUOUS LIVE MODE")
    print("=" * 100)

    print(
        f"Refresh interval: "
        f"{REFRESH_SECONDS} seconds"
    )

    print(
        "Press CTRL+C to stop."
    )

    while True:

        try:

            run_one_scan()

            print()
            print(
                f"Waiting "
                f"{REFRESH_SECONDS} seconds..."
            )

            time.sleep(
                REFRESH_SECONDS
            )

        except KeyboardInterrupt:

            print()
            print(
                "STAGE 17 STOPPED BY USER"
            )

            break

        except Exception as e:

            print()
            print(
                "Unexpected Stage 17 error:"
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

            time.sleep(
                REFRESH_SECONDS
            )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # DEBUG MODE
    # --------------------------------------------------------
    # Run only ONE scan until Stage 17 works correctly.
    # --------------------------------------------------------

    result = run_one_scan()

    if result is None:

        print()
        print("=" * 100)
        print("STAGE 17 STOPPED")
        print("=" * 100)

        return

    print()
    print("=" * 100)
    print("STAGE 17 COMPLETE")
    print("=" * 100)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()