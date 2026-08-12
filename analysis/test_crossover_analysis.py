from api.historical_data import get_historical_data
from analysis.crossover_analysis import (
    analyze_crossovers,
    get_crossover_summary
)


symbol = "SBIN-EQ"

print("\n======================================")
print("CROSSOVER PROFITABILITY TEST")
print("======================================\n")

print(
    f"Loading historical data for {symbol}..."
)

df = get_historical_data(symbol)

if df is None or df.empty:

    print("No historical data available.")

else:

    results = analyze_crossovers(
        df,
        holding_period=20
    )

    print(
        f"\nTotal evaluated crossovers: "
        f"{len(results)}"
    )

    print("\n--------------------------------------")

    for result in results:

        print(
            f"Date: {result['Entry Date']}"
        )

        print(
            f"Signal: {result['Signal']}"
        )

        print(
            f"Entry Price: "
            f"{result['Entry Price']}"
        )

        print(
            f"Exit Price: "
            f"{result['Exit Price']}"
        )

        print(
            f"Return: "
            f"{result['Return %']}%"
        )

        print(
            f"Result: "
            f"{result['Result']}"
        )

        print(
            f"Max Favorable: "
            f"{result['Max Favorable %']}%"
        )

        print(
            f"Max Adverse: "
            f"{result['Max Adverse %']}%"
        )

        print("--------------------------------------")

    summary = get_crossover_summary(
        results
    )

    print("\n======================================")
    print("SUMMARY")
    print("======================================")

    print(
        f"Total Crossovers: "
        f"{summary['Total Crossovers']}"
    )

    print(
        f"Profitable: "
        f"{summary['Profitable']}"
    )

    print(
        f"Failed: "
        f"{summary['Failed']}"
    )

    print(
        f"Win Rate: "
        f"{summary['Win Rate %']}%"
    )

    print(
        f"Average Return: "
        f"{summary['Average Return %']}%"
    )