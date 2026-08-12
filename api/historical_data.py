# ============================================================
# HISTORICAL DATA PROVIDER
# api/historical_data.py
# ============================================================

import time
import datetime
import pandas as pd

from api.broker import connect
from utils.token_map import TOKENS


# ============================================================
# GLOBAL SMARTAPI SESSION
# ============================================================

_SMARTAPI = None


# ============================================================
# GET / REUSE SMARTAPI OBJECT
# ============================================================

def get_smartapi():

    global _SMARTAPI

    if _SMARTAPI is not None:
        return _SMARTAPI

    try:

        print()
        print("=" * 70)
        print("CONNECTING TO ANGEL ONE")
        print("=" * 70)

        _SMARTAPI = connect()

    except Exception as e:

        print(
            f"API connection failed: "
            f"{type(e).__name__}: {e}"
        )

        _SMARTAPI = None

        return None

    if _SMARTAPI is None:

        print(
            "connect() returned None."
        )

        return None

    print(
        "SmartAPI object created successfully."
    )

    return _SMARTAPI


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================

def get_smartapi_object():

    return get_smartapi()


# ============================================================
# INTERNAL CANDLE FETCHER
# ============================================================

def _get_candle_data(
    symbol,
    interval,
    from_date,
    to_date,
    max_retries=3
):

    token = TOKENS.get(symbol)

    if not token:

        print(
            f"Stock token not found for {symbol}"
        )

        return None

    api = get_smartapi()

    if api is None:

        print(
            f"API object unavailable for {symbol}"
        )

        return None

    params = {

        "exchange": "NSE",

        "symboltoken": str(token),

        "interval": interval,

        "fromdate": from_date,

        "todate": to_date,
    }

    response = None

    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            time.sleep(0.8)

            print(
                f"Getting {interval} data for "
                f"{symbol} "
                f"(attempt {attempt}/{max_retries})..."
            )

            response = api.getCandleData(
                params
            )

            if not isinstance(
                response,
                dict
            ):

                print(
                    f"Invalid API response for "
                    f"{symbol}"
                )

                continue

            if response.get(
                "status"
            ) is True:

                data = response.get(
                    "data"
                )

                if data:

                    return _candles_to_dataframe(
                        data
                    )

                print(
                    f"No {interval} candle data "
                    f"returned for {symbol}"
                )

                return None

            message = response.get(
                "message",
                "Unknown API error"
            )

            print(
                f"Historical API error for "
                f"{symbol}: {message}"
            )

            error_text = str(
                message
            ).lower()

            if (
                "rate"
                in error_text
                or "access denied"
                in error_text
                or "exceed"
                in error_text
            ):

                wait_time = 5 * attempt

            else:

                wait_time = 2 * attempt

            time.sleep(
                wait_time
            )

        except Exception as e:

            print(
                f"Error getting {interval} "
                f"data for {symbol}: {e}"
            )

            if attempt < max_retries:

                time.sleep(
                    2 * attempt
                )

    print(
        f"Failed to get {interval} data "
        f"for {symbol}"
    )

    return None


# ============================================================
# DATAFRAME CONVERTER
# ============================================================

def _candles_to_dataframe(data):

    if not data:

        return None

    try:

        df = pd.DataFrame(
            data,
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.dropna(
            subset=[
                "date",
                "open",
                "high",
                "low",
                "close",
            ]
        )

        df = df.sort_values(
            "date"
        )

        df = df.drop_duplicates(
            subset=["date"],
            keep="last"
        )

        df = df.reset_index(
            drop=True
        )

        return df

    except Exception as e:

        print(
            f"Error creating candle DataFrame: "
            f"{e}"
        )

        return None


# ============================================================
# DAILY HISTORICAL DATA
# ============================================================

def get_historical_data(
    symbol,
    days=250
):

    to_date = datetime.datetime.now()

    from_date = (
        to_date
        - datetime.timedelta(
            days=days
        )
    )

    return _get_candle_data(

        symbol,

        "ONE_DAY",

        from_date.strftime(
            "%Y-%m-%d 09:15"
        ),

        to_date.strftime(
            "%Y-%m-%d 15:30"
        ),
    )


# ============================================================
# INTRADAY ONE-MINUTE DATA
# ============================================================

def get_intraday_data(
    symbol,
    minutes=120
):

    """
    Fetch recent ONE_MINUTE candles.

    Used by Stage 17 for:

        ETQ 5m
        ETQ 20m
        ETQ 60m
        Average LTP 20m
        Average LTP 60m
    """

    to_date = datetime.datetime.now()

    from_date = (
        to_date
        - datetime.timedelta(
            minutes=minutes
        )
    )

    return _get_candle_data(

        symbol,

        "ONE_MINUTE",

        from_date.strftime(
            "%Y-%m-%d %H:%M"
        ),

        to_date.strftime(
            "%Y-%m-%d %H:%M"
        ),
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("HISTORICAL DATA TEST")
    print("=" * 70)

    symbol = "SBIN-EQ"

    df = get_historical_data(
        symbol,
        days=250
    )

    if df is not None:

        print()
        print(
            f"Daily rows: {len(df)}"
        )

        print(
            df.tail().to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("INTRADAY DATA TEST")
    print("=" * 70)

    intraday = get_intraday_data(
        symbol,
        minutes=120
    )

    if intraday is not None:

        print()
        print(
            f"Intraday rows: "
            f"{len(intraday)}"
        )

        print(
            intraday.tail().to_string(
                index=False
            )
        )

    else:

        print(
            "Intraday data unavailable."
        )

    print()
    print("=" * 70)