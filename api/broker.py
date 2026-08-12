# ============================================================
# ANGEL ONE SMARTAPI BROKER CONNECTION
# api/broker.py
# ============================================================

from SmartApi import SmartConnect
import pyotp
import config


# ============================================================
# GLOBAL SESSION
# ============================================================

_SMART_API = None


# ============================================================
# CONNECT TO ANGEL ONE
# ============================================================

def connect():

    global _SMART_API

    # --------------------------------------------------------
    # Reuse existing connection
    # --------------------------------------------------------

    if _SMART_API is not None:

        return _SMART_API

    print()
    print("=" * 70)
    print("CREATING SMARTAPI OBJECT")
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # Create SmartAPI object
        # ----------------------------------------------------

        smart_api = SmartConnect(
            api_key=config.API_KEY
        )

        print(
            "SmartAPI object created."
        )

        # ----------------------------------------------------
        # Generate TOTP
        # ----------------------------------------------------

        otp = pyotp.TOTP(
            config.TOTP_SECRET
        ).now()

        print(
            "TOTP generated successfully."
        )

        # ----------------------------------------------------
        # Login
        # ----------------------------------------------------

        session = smart_api.generateSession(
            config.CLIENT_ID,
            config.PIN,
            otp
        )

        # ----------------------------------------------------
        # Validate response
        # ----------------------------------------------------

        if not isinstance(
            session,
            dict
        ):

            print(
                "❌ Invalid login response."
            )

            return None

        print()
        print(
            "SmartAPI login response:"
        )

        print(
            session
        )

        if session.get("status") is not True:

            print()
            print(
                "❌ SmartAPI Login Failed."
            )

            print(
                f"Message: "
                f"{session.get('message', 'Unknown error')}"
            )

            return None

        # ----------------------------------------------------
        # Save session
        # ----------------------------------------------------

        _SMART_API = smart_api

        print()
        print(
            "✅ Login Successful"
        )

        # ----------------------------------------------------
        # Get feed token
        # ----------------------------------------------------

        try:

            feed_token = smart_api.getfeedToken()

            if feed_token:

                print(
                    "✅ Feed token obtained."
                )

        except Exception as e:

            print(
                f"⚠️ Feed token could not be obtained: {e}"
            )

        return _SMART_API

    except Exception as e:

        print()
        print(
            "❌ SmartAPI connection failed."
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        _SMART_API = None

        return None


# ============================================================
# GET EXISTING SESSION
# ============================================================

def get_connection():

    return _SMART_API


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SMARTAPI CONNECTION TEST")
    print("=" * 70)

    api = connect()

    print()

    if api is not None:

        print(
            "✅ SmartAPI connection is ready."
        )

    else:

        print(
            "❌ SmartAPI connection failed."
        )