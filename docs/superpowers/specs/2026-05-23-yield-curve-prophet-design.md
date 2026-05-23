# yield-curve-prophet Design Spec

**Date:** 2026-05-23
**Status:** Approved

## Overview

A single-notebook project that predicts weekly (5-trading-day) directional moves in the US Treasury yield curve using an LSTM neural network ensembled with XGBoost. Built on 20 years of free FRED macro data. Three binary classification targets: 2Y yield direction, 10Y yield direction, and 2s10s spread direction (steepening vs flattening).

The XGBoost baseline establishes whether temporal structure matters. The LSTM captures sequential patterns. The ensemble combines both for maximum predictive power. SHAP interpretability on the XGBoost leg makes the model explainable.

## Targets

Three binary classifiers, each predicting direction over 5 trading days:

| Target | Label = 1 | Label = 0 |
|--------|-----------|-----------|
| 2Y yield | Yield rises (rate up) | Yield falls (rate down) |
| 10Y yield | Yield rises (rate up) | Yield falls (rate down) |
| 2s10s spread | Spread widens (steepening) | Spread narrows (flattening) |

Flat moves (exactly 0 bp change) are dropped from training and evaluation.

## Data Pipeline

### Source

FRED API. Free tier (120 requests/minute, no daily cap). Single `requests.get()` per series. Env var `FRED_API_KEY` for authentication.

### History

2005-2025 (~20 years, ~5,000 trading days). Start at 2005 to capture pre-GFC, GFC, recovery, COVID, and hiking cycle regimes.

### Raw Series (~15-20 from FRED)

| Category | Series ID | Description |
|----------|-----------|-------------|
| Treasury Yields | DGS2, DGS5, DGS10, DGS30 | Constant maturity Treasury yields |
| Fed Funds | DFF | Effective federal funds rate |
| Inflation | T10YIE | 10Y breakeven inflation |
| Inflation | T5YIFR | 5Y5Y forward inflation expectation |
| Labor | UNRATE | Unemployment rate (monthly, ffill) |
| Labor | ICSA | Initial jobless claims (weekly, ffill) |
| Growth | INDPRO | Industrial production index (monthly, ffill) |
| Sentiment | UMCSENT | U of Michigan consumer sentiment (monthly, ffill) |
| Volatility | VIXCLS | CBOE VIX |
| Dollar | DTWEXBGS | Trade-weighted US dollar index |
| Credit Stress | TEDRATE | TED spread (3M LIBOR - 3M T-bill) |
| Credit Stress | BAMLH0A0HYM2 | ICE BofA HY OAS (limited to ~3Y history post-April 2026) |

Monthly/weekly series are forward-filled to daily frequency and aligned to the trading day calendar.

### Engineered Features (per series)

- 5-day change (1-week delta)
- 21-day change (1-month delta)
- 63-day change (1-quarter delta)
- 21-day rolling standard deviation (realized vol)
- Z-score vs 252-day rolling mean (where is this reading vs its 1Y history)
- Momentum signal: 5-day MA vs 21-day MA crossover (binary)

### Derived Series

- 2s10s spread: DGS10 - DGS2
- 5s30s spread: DGS30 - DGS5
- Real 10Y yield: DGS10 - T10YIE
- Curve slope change features (engineered from the derived spreads)

Total feature count: ~80-100 after engineering.

### LSTM Input Shape

- Lookback window: 60 trading days (~3 months)
- Input tensor: (batch_size, 60, N_features)
- All features standardized with rolling 252-day mean/std (no future leakage)

## Model Architecture

### Model A: XGBoost Baseline

- One `XGBClassifier` per target (3 models)
- Input: single-row feature vector (current day's engineered features, no sequence)
- Hyperparameter tuning: Optuna, 100 trials, optimizing ROC-AUC
- Search space: max_depth [3-8], learning_rate [0.01-0.3], n_estimators [100-1000], subsample [0.6-1.0], colsample_bytree [0.6-1.0], min_child_weight [1-10]
- SHAP values computed on test set for interpretability

### Model B: LSTM

- One LSTM classifier per target (3 models)
- Framework: PyTorch
- Architecture:
  - Input: (batch, 60, N_features)
  - LSTM layer 1: 128 hidden units, dropout 0.3
  - LSTM layer 2: 64 hidden units, dropout 0.3
  - Take final hidden state
  - Dense: 64 -> ReLU -> Dropout(0.3) -> 1 -> Sigmoid
- Loss: Binary cross-entropy
- Optimizer: Adam, lr=1e-3
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Early stopping: patience=10 on validation loss
- Batch size: 64
- Max epochs: 100

### Model C: Ensemble

- Weighted average: `p_ensemble = w * p_xgb + (1-w) * p_lstm`
- Weight `w` optimized on validation set by grid search [0.0, 0.05, ..., 1.0] maximizing accuracy
- One weight per target (3 weights total)

## Validation Strategy

### Walk-Forward (Expanding Window)

| Split | Period | Trading Days | Purpose |
|-------|--------|-------------|---------|
| Train | 2005-2018 | ~3,500 | Initial training |
| Validation | 2019-2020 | ~500 | Hyperparameter tuning, early stopping, ensemble weights |
| Test | 2021-2025 | ~1,250 | Final evaluation, all reported metrics |

- No random shuffling. Temporal integrity enforced throughout.
- Rolling standardization uses only past data (252-day lookback).
- Target labels use 5-day forward return - computed before splitting to avoid alignment bugs, but only used in training with appropriate lag.

## Evaluation

### Metrics (per model x per target = 9 combinations)

- Accuracy vs 50% naive baseline
- Precision / Recall / F1
- ROC-AUC
- Brier score (probability calibration quality)
- Rolling 63-day accuracy (does the model degrade over time?)

### Visualizations (~10-12 figures)

1. Confusion matrices (3x3 grid: target x model)
2. ROC curves overlaid (XGBoost vs LSTM vs Ensemble, one plot per target)
3. Rolling 63-day hit rate over time (line chart per target)
4. SHAP summary plot (top 15 features, XGBoost, beeswarm)
5. SHAP bar plot per target (mean absolute SHAP values)
6. Feature importance comparison (XGBoost gain vs LSTM gradient attribution)
7. Calibration curve (predicted probability vs observed frequency)
8. Cumulative hypothetical P&L (long/short duration based on 10Y signal)
9. Prediction confidence distribution (histogram of predicted probabilities)
10. Model comparison summary table (all metrics, all targets, all models)

## Notebook Structure

Single Jupyter notebook: `yield_curve_prophet.ipynb`

| Section | Content |
|---------|---------|
| 1. Why This Matters | Markdown intro connecting yield curve prediction to portfolio management |
| 2. Setup & Configuration | Imports, FRED API key, constants |
| 3. Data Collection | Pull all FRED series, handle frequencies, align to trading calendar |
| 4. Feature Engineering | Rolling changes, vol, z-scores, momentum, derived spreads |
| 5. Target Construction | 5-day forward returns, binary labels, class balance check |
| 6. Train/Val/Test Split | Walk-forward split, rolling standardization |
| 7. XGBoost Baseline | Optuna tuning, train, evaluate, SHAP |
| 8. LSTM | Build PyTorch model, train loop, evaluate |
| 9. Ensemble | Optimize blend weights, evaluate combined model |
| 10. Head-to-Head Comparison | Side-by-side metrics table, overlay charts |
| 11. Key Findings | What worked, what the models learned, market implications |

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas | Data manipulation, time series alignment |
| numpy | Numerical computation |
| requests | FRED API calls |
| torch | LSTM neural network |
| xgboost | Gradient boosting baseline |
| optuna | Hyperparameter optimization |
| shap | Model interpretability |
| scikit-learn | Metrics, preprocessing, calibration |
| matplotlib | Visualization |
| seaborn | Statistical plots |

## Project Deliverables

```
yield-curve-prophet/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- .gitignore
+-- yield_curve_prophet.ipynb
```

README follows the same professional style as other portfolio repos: centered shield badges, "Why This Exists" section, results table, design decisions, quick start.

## What This Does NOT Do

- Does not predict magnitude (only direction)
- Does not trade or connect to any brokerage
- Does not use paid data (FRED free tier only)
- Does not claim to be a trading strategy (hypothetical P&L is illustrative only)
- Does not use intraday data
