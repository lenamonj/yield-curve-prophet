<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://xgboost.readthedocs.io/"><img src="https://img.shields.io/badge/XGBoost-EC6C37?logo=xgboost&logoColor=white" alt="XGBoost"></a>
  <a href="https://fred.stlouisfed.org/docs/api/"><img src="https://img.shields.io/badge/Data-FRED%20API-green" alt="Data: FRED API"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="License: MIT"></a>
</p>

<h1 align="center">Yield Curve Prophet</h1>

<p align="center">
  <strong>A PyTorch LSTM + XGBoost ensemble that predicts weekly directional moves in US Treasury yields (2Y, 10Y, 2s10s spread), trained on 20 years of FRED macro data with walk-forward validation and SHAP interpretability.</strong>
</p>

<p align="center">
  <em>No Bloomberg terminal. No paid data feeds. Just a free API key and PyTorch.</em>
</p>

---

## Why This Exists

Yield curve prediction is core to fixed income portfolio management. Where rates go next drives duration positioning, curve trades, and spread sector allocation. Most forecasting in practice is discretionary - portfolio managers reading tea leaves from dot plots and payrolls prints.

This project tests whether machines can beat a coin flip on weekly rate direction using nothing but free public data from FRED. No proprietary feeds, no Bloomberg, no vendor lock-in. If a simple LSTM + XGBoost ensemble can reliably predict direction even 55-60% of the time, that is an edge worth exploring.

## What It Produces

| Stage | Description |
|---|---|
| **Data Collection** | 15 FRED macro series, 2005-2025 |
| **Feature Engineering** | ~80-100 features (rolling changes, volatility, z-scores, momentum) |
| **XGBoost Baseline** | Optuna-tuned gradient boosting (100 trials per target) |
| **LSTM** | 2-layer LSTM (128->64), 60-day lookback, early stopping |
| **Ensemble** | Weighted blend optimized on validation set |
| **Evaluation** | Accuracy, ROC-AUC, Brier score, SHAP, calibration plots, hypothetical P&L |

## Quick Start

```bash
git clone https://github.com/lenamonj/yield-curve-prophet.git
cd yield-curve-prophet
pip install -r requirements.txt
```

Set your FRED API key (free from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)):

```bash
export FRED_API_KEY="your_key_here"
```

Run the full pipeline:

```bash
jupyter notebook yield_curve_prophet.ipynb
```

## Data Source

All data comes from the [FRED API](https://fred.stlouisfed.org/docs/api/) - free with registration.

| Series ID | Description |
|---|---|
| `DGS2` | 2-Year Treasury Constant Maturity Rate |
| `DGS10` | 10-Year Treasury Constant Maturity Rate |
| `DGS30` | 30-Year Treasury Constant Maturity Rate |
| `DFEDTARU` | Federal Funds Target Rate (Upper) |
| `T10Y2Y` | 10Y-2Y Treasury Spread |
| `T10Y3M` | 10Y-3M Treasury Spread |
| `BAMLH0A0HYM2` | ICE BofA US High Yield OAS |
| `BAMLC0A4CBBB` | ICE BofA BBB Corporate OAS |
| `VIXCLS` | CBOE Volatility Index (VIX) |
| `DTWEXBGS` | Trade-Weighted USD Index (Broad) |
| `CPIAUCSL` | Consumer Price Index (All Urban) |
| `UNRATE` | Unemployment Rate |
| `PAYEMS` | Total Nonfarm Payrolls |
| `UMCSENT` | University of Michigan Consumer Sentiment |
| `GDPC1` | Real GDP (Quarterly) |

## Design Decisions

- **Walk-forward temporal validation** - Train 2005-2018, validation 2019-2020, test 2021-2025. No future leakage.
- **XGBoost as baseline** - Proves whether the LSTM actually adds value over a strong tree-based model.
- **Rolling standardization with train-only statistics** - Features are standardized using rolling windows fit only on training data to prevent look-ahead bias.
- **Optuna over grid search** - 100-trial Bayesian optimization per target beats brute-force grid search on both runtime and performance.
- **Binary classification (direction) over regression (magnitude)** - Predicting "up or down" is more actionable for duration positioning than predicting exact basis point moves.
- **SHAP for interpretability** - Every prediction is explainable. Feature importance is not a black box.

## Project Structure

```
yield-curve-prophet/
├── yield_curve_prophet.ipynb   # Full pipeline notebook
├── src/
│   ├── data.py                 # FRED data collection and cleaning
│   ├── features.py             # Feature engineering pipeline
│   ├── models.py               # XGBoost and LSTM model definitions
│   ├── ensemble.py             # Weighted ensemble logic
│   └── evaluation.py           # Metrics, SHAP, calibration, P&L
├── config/
│   └── params.yaml             # Hyperparameters and FRED series config
├── requirements.txt
├── LICENSE
└── README.md
```

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | LSTM model architecture and training |
| `xgboost` | Gradient boosting baseline |
| `optuna` | Bayesian hyperparameter optimization |
| `pandas` | Data manipulation and time series |
| `numpy` | Numerical computation |
| `scikit-learn` | Metrics, preprocessing, validation splits |
| `shap` | Model interpretability |
| `fredapi` | FRED data access |
| `matplotlib` | Visualization and calibration plots |

## License

MIT

<p align="center">
  <em>Built with PyTorch, XGBoost, and the FRED API. No proprietary data feeds required.</em>
</p>
