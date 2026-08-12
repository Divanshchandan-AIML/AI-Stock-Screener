from api.live_data import get_live_market_data
from utils.live_parser import parse_live_market_data


if __name__ == "__main__":

    symbol = "SBIN-EQ"

    print("\n========== LIVE MARKET DATA TEST ==========\n")

    raw_data = get_live_market_data(symbol)

    if raw_data is None:

        print("Failed to get live data.")

    else:

        parsed_data = parse_live_market_data(
            symbol,
            raw_data
        )

        if parsed_data is None:

            print("Failed to parse live data.")

        else:

            print("========== PARSED DATA ==========")

            for key, value in parsed_data.items():

                print(
                    f"{key:<25}: {value}"
                )