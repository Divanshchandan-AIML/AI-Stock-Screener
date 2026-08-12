# ============================================================
# ANGEL ONE TOKEN MAP
# utils/token_map.py
# ============================================================

import json
import urllib.request


# ============================================================
# ANGEL ONE INSTRUMENT MASTER
# ============================================================

INSTRUMENT_URL = (
    "https://margincalculator.angelbroking.com/"
    "OpenAPI_File/files/OpenAPIScripMaster.json"
)


# ============================================================
# LOAD NSE TOKENS
# ============================================================

def load_nse_tokens():

    """
    Download Angel One instrument master and create:

        NSE equity symbol -> token

    Example:

        SBIN-EQ      -> 3045
        RELIANCE-EQ  -> 2885
    """

    print()
    print("=" * 70)
    print("DOWNLOADING NSE INSTRUMENT LIST")
    print("=" * 70)

    try:

        request = urllib.request.Request(
            INSTRUMENT_URL,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=30
        ) as response:

            raw_data = response.read()

        instruments = json.loads(
            raw_data.decode("utf-8")
        )

    except Exception as e:

        print(
            "❌ Error downloading instrument master:"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        return {}

    if not isinstance(
        instruments,
        list
    ):

        print(
            "❌ Instrument master returned "
            "unexpected data."
        )

        return {}

    tokens = {}

    # ========================================================
    # PROCESS INSTRUMENTS
    # ========================================================

    for item in instruments:

        if not isinstance(
            item,
            dict
        ):

            continue

        # ----------------------------------------------------
        # Only NSE
        # ----------------------------------------------------

        if str(
            item.get(
                "exch_seg",
                ""
            )
        ).upper() != "NSE":

            continue

        # ----------------------------------------------------
        # Symbol
        # ----------------------------------------------------

        symbol = str(
            item.get(
                "symbol",
                ""
            )
        ).strip().upper()

        # ----------------------------------------------------
        # Token
        # ----------------------------------------------------

        token = item.get(
            "token"
        )

        if not symbol:

            continue

        if token is None:

            continue

        token = str(
            token
        ).strip()

        if not token:

            continue

        # ----------------------------------------------------
        # Only equity symbols
        # ----------------------------------------------------

        if not symbol.endswith(
            "-EQ"
        ):

            continue

        tokens[symbol] = token

    print()
    print(
        f"✅ NSE equity stocks loaded: "
        f"{len(tokens)}"
    )

    return tokens


# ============================================================
# LOAD TOKENS WHEN MODULE IS IMPORTED
# ============================================================

TOKENS = load_nse_tokens()


# ============================================================
# TATA MOTORS COMPATIBILITY
# ============================================================

if "TMPV-EQ" in TOKENS:

    TOKENS[
        "TATAMOTORS-EQ"
    ] = TOKENS[
        "TMPV-EQ"
    ]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("NSE TOKEN TEST")
    print("=" * 70)

    print(
        f"Total NSE stocks: "
        f"{len(TOKENS)}"
    )

    print()
    print(
        "First 20 stocks:"
    )

    for i, (
        symbol,
        token
    ) in enumerate(
        TOKENS.items()
    ):

        print(
            f"{symbol:<25} {token}"
        )

        if i >= 19:

            break

    print()
    print("=" * 70)
    print("TEST KNOWN STOCKS")
    print("=" * 70)

    known_stocks = [

        "SBIN-EQ",
        "RELIANCE-EQ",
        "ITC-EQ",
        "SUZLON-EQ",
        "IRFC-EQ",
        "TATAMOTORS-EQ",

    ]

    for symbol in known_stocks:

        print(
            f"{symbol:<20}: "
            f"{TOKENS.get(symbol, 'NOT FOUND')}"
        )