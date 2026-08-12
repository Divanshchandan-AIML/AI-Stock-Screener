# AI Stock Screener

An AI/ML-based stock screening and paper-trading system designed to evaluate
stock predictions, portfolio behavior, risk controls, and production-readiness
through a staged validation pipeline.

## Project Overview

The system processes historical stock data, generates ML-based predictions,
evaluates trading signals, performs paper trading, applies risk controls, and
runs automated validation and release audits.

The pipeline contains multiple stages covering:

- Data processing
- Feature generation
- ML model training
- Model validation
- Strategy optimization
- Portfolio allocation
- Risk validation
- Paper trading
- Stability testing
- Risk-controlled evaluation
- Final release auditing

## Machine Learning

The current model configuration uses:

- Random Forest Classifier
- Python
- pandas
- NumPy
- scikit-learn

## Trading Mode

The system is configured for:

**PAPER TRADING ONLY**

Live trading remains disabled.

## Final Validation Results

- Win Rate: 57.69%
- Profit Factor: 1.4701
- Sharpe Ratio: 2.2446
- Maximum Drawdown: -24.75%
- Stage 25 Checks: 11/11
- Stage 26 Checks: 44/44
- Stage 26 Pass Rate: 100%
- Promotion Decision: PROMOTION_READY
- Audit Decision: RELEASE_READY

## Final Release Audit

The final audit confirmed:

- 18 artifacts found
- 0 artifacts missing
- 44/44 Stage 26 checks passed
- Paper trading only
- Live trading disabled

## Project Structure

```text
AI_Stock_Screener/
│
├── data/
│   ├── stage20/
│   ├── stage21/
│   ├── stage22/
│   ├── stage23/
│   ├── stage24/
│   ├── stage25/
│   └── stage26/
│
├── scripts/
├── src/
├── tests/
├── requirements.txt
├── README.md
└── ...