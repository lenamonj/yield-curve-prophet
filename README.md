<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EC6C37?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/Data-FRED%20API-6DB33F?style=for-the-badge&logo=api&logoColor=white" alt="FRED API">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License">
</p>

<h1 align="center">Yield Curve Prophet</h1>

<p align="center">
  <strong>A PyTorch LSTM + XGBoost ensemble that predicts weekly directional moves in US Treasury yields (2Y, 10Y, 2s10s spread), built on 19 years of FRED macro data with walk-forward validation, SHAP interpretability, and statistical significance testing.</strong>
</p>

<p align="center">
  <em>No Bloomberg terminal. No paid data feeds. Just a free API key and PyTorch.</em>
</p>

---

## Key Result

The 2s10s spread direction (steepening vs. flattening) is predictable at **60.1% accuracy** (p = 0.000005), statistically significant at the 99.9% confidence level. Outright yield direction carries weaker but detectable signal (2Y at 54.0%, 10Y at 53.6%).

| Target | XGBoost | LSTM | Ensemble | p-value | Significant? |
|--------|---------|------|----------|---------|-------------|
| 2Y Yield Direction | **54.0%** | 47.3% | 47.3% | 0.0431 | Yes (95%) |
| 10Y Yield Direction | **53.6%** | 50.5% | 50.5% | 0.0624 | Marginal (90%) |
| 2s10s Spread Direction | 54.4% | 58.0% | **60.1%** | **0.000005** | **Yes (99.9%)** |

Baseline (coin flip): 50.0%

---

## Why This Exists

The yield curve is the most watched indicator in fixed income. Its shape drives portfolio positioning, curve trades, and macro regime classification. Most forecasting is still discretionary - PMs reading dot plots and payrolls prints.

This project tests whether a machine can beat a coin flip at predicting the direction of weekly yield moves using only free public data. The answer: outright yield direction is hard, but curve spread direction carries statistically significant signal that the ensemble captures at 60.1% accuracy.

---

## What It Produces

A single Jupyter notebook that runs the complete pipeline:

| Stage | Description |
|-------|-------------|
| **Data Collection** | 12 FRED macro series, 2007-2026 |
| **Feature Engineering** | 90 features (rolling changes, volatility, z-scores, momentum) |
| **XGBoost Baseline** | Optuna-tuned gradient boosting (100 trials per target), SHAP interpretability |
| **LSTM** | 2-layer LSTM (128->64), 60-day lookback, early stopping |
| **Ensemble** | Weighted blend optimized per target on validation set |
| **Significance Test** | Binomial test on each target (H0: accuracy = 50%) |
| **Evaluation** | ROC curves, confusion matrices, rolling accuracy, calibration, hypothetical P&L |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/lenamonj/yield-curve-prophet.git
cd yield-curve-prophet
pip install -r requirements.txt
```

### 2. Set your API key

Register for a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

```bash
export FRED_API_KEY="your-key-here"
```

Or on Windows PowerShell:

```powershell
$env:FRED_API_KEY = "your-key-here"
```

### 3. Run

```bash
jupyter notebook yield_curve_prophet.ipynb
```

Full execution takes 10-15 minutes (Optuna tuning + LSTM training).

---

## Data Source

All data comes from the [FRED API](https://fred.stlouisfed.org/docs/api/fred/) (free, 120 requests/minute).

| Category | Series ID | Description | Frequency |
|----------|-----------|-------------|-----------|
| Treasury Yields | DGS2, DGS5, DGS10, DGS30 | Constant maturity yields | Daily |
| Monetary Policy | DFF | Effective federal funds rate | Daily |
| Inflation | T10YIE | 10-year breakeven inflation | Daily |
| Inflation | T5YIFR | 5Y5Y forward inflation expectation | Daily |
| Labor | UNRATE | Unemployment rate | Monthly |
| Growth | INDPRO | Industrial production index | Monthly |
| Sentiment | UMCSENT | Consumer sentiment | Monthly |
| Volatility | VIXCLS | CBOE VIX | Daily |
| Dollar | DTWEXBGS | Trade-weighted dollar index | Daily |

See `final_features.txt` for the complete feature dictionary (90 engineered features with definitions).

---

## Design Decisions

- **70/15/15 walk-forward split** - Train (70%), validation (15%), test (15%) with 63-day leakage gaps between each split. No random shuffling, no future information leakage.
- **XGBoost as point-in-time baseline** - Sees only today's feature snapshot. Proves whether the LSTM's sequential modeling adds value.
- **LSTM as sequential model** - 60-day lookback window captures momentum shifts, volatility clustering, and regime transitions.
- **Ensemble** - Weighted blend of both models, optimized per target. The combination outperforms either model alone on the 2s10s spread.
- **Optuna over grid search** - 100-trial Bayesian optimization per target for XGBoost hyperparameters.
- **Binary classification over regression** - Predicting "up or down" is more actionable than predicting exact basis point moves.
- **SHAP for interpretability** - Every XGBoost prediction is explainable via feature attribution.
- **Binomial significance test** - Statistical rigor: the null hypothesis (50% accuracy) is formally tested for each target.
- **SEED=42 locked** - numpy, PyTorch, and Optuna seeds fixed for full reproducibility.

---

## Project Structure

```
yield-curve-prophet/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- .gitignore
|-- final_features.txt              # Complete feature dictionary (90 features)
|-- yield_curve_prophet.ipynb       # Full pipeline - one notebook
+-- *.png                           # Generated charts (8 figures)
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation and time series alignment |
| `numpy` | Numerical computation |
| `requests` | FRED API calls |
| `torch` | LSTM neural network |
| `torchinfo` | Model architecture summary |
| `xgboost` | Gradient boosting baseline |
| `optuna` | Bayesian hyperparameter optimization |
| `shap` | Model interpretability (feature attribution) |
| `scikit-learn` | Metrics, preprocessing, calibration |
| `matplotlib` | Visualization |
| `scipy` | Statistical significance testing (binomial test) |
| `seaborn` | Statistical plots |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built with PyTorch, XGBoost, and the FRED API. No proprietary data feeds required.</sub>
</p>
