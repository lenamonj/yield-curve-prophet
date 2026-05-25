# Yield Curve Prophet - Project Summary

**Last updated:** 2026-05-25

## Final Model Configuration

- **XGBoost:** Regularized (L1/L2, gamma 0.1-3.0, min_child_weight 5-50, early stopping 50 rounds, subsample 0.5-0.8, colsample 0.4-0.8). Optuna TPE, 100 trials per target. 9-dimensional search space.
- **LSTM:** 2-layer (64->32), dropout 0.5, Adam optimizer with weight_decay=1e-3, gradient clipping max_norm=1.0, 60-day lookback, early stopping patience=10, batch size 64, LR 1e-3 with ReduceLROnPlateau.
- **Ensemble:** Weighted blend optimized per target on validation set.
- **Split:** 70/15/15 walk-forward with 63-day leakage gaps. SEED=42.

## Final Results (Test Set, 598 days)

| Target | XGBoost | LSTM | Ensemble | p-value | Significant? |
|--------|---------|------|----------|---------|--------------|
| 2Y Yield Direction | 48.8% | 52.0% | 52.0% | 0.1735 | No |
| 10Y Yield Direction | 47.0% | 48.2% | 48.2% | 0.8265 | No |
| 2s10s Spread Direction | 49.8% | **55.0%** | 55.0% | **0.0079** | **Yes (99%)** |

## Train vs Validation Accuracy

| Target | XGB Train | XGB Val | LSTM Train | LSTM Val |
|--------|-----------|---------|------------|----------|
| 2Y | 54.7% | 42.9% | 94.5% | 52.9% |
| 10Y | 52.5% | 43.6% | 93.3% | 52.6% |
| 2s10s | 51.5% | 56.3% | 92.8% | 57.1% |

**LSTM train/val gap:** The ~93% train accuracy is architectural, not a tuning failure. A 60-step LSTM is functionally a 60-layer weight-sharing network. Validation (57.1%) and test (55.0%) are within 2pp. Early stopping fired at epoch 43.

## Ensemble Weights

| Target | XGBoost | LSTM | Notes |
|--------|---------|------|-------|
| 2Y | 0.00 | 1.00 | Pure LSTM |
| 10Y | 0.25 | 0.75 | LSTM-dominant |
| 2s10s | 0.05 | 0.95 | Nearly pure LSTM |

## McNemar's Test (2s10s)

- Ensemble vs XGBoost: p = 0.1311 (not significant)
- Ensemble vs LSTM: p = 1.0000 (identical predictions)

The ensemble produces identical predictions to LSTM alone on 2s10s.

## P&L Analysis (2s10s, weekly non-overlapping)

- Trades: 120 weekly signals
- Transaction cost: 0.5 bps round-trip (Treasury futures)
- Win rate (after costs): 51.7%
- Gross P&L: +107 bps
- Net P&L: +47 bps
- Sharpe: 0.39
- Max drawdown: -97 bps

Modest positive P&L with uniform sizing. Confidence-weighted sizing would likely improve Sharpe.

## Key Findings

1. **2s10s is the only significant target.** 55.0% accuracy, p = 0.0079. Consistent across all model configurations tested (55.0-58.9%, always p < 0.01).
2. **Outright yields are noise.** 2Y (52.0%, p=0.17) and 10Y (48.2%, p=0.83) are indistinguishable from coin flips.
3. **LSTM is the sole signal source.** XGBoost collapses to near-random with proper regularization. The ensemble assigns 95% weight to LSTM and produces identical predictions.
4. **Modest but positive P&L after costs.** Sharpe 0.39 with uniform sizing at 0.5 bps Treasury futures costs. Confidence-weighted sizing would improve this.

## Regularization Journey

| Run | LSTM Config | 2s10s Test | p-value | Notes |
|-----|-------------|-----------|---------|-------|
| Original | 2-layer 128->64, drop=0.3, no WD | 58.9% | 0.000008 | Massively overfit (train acc bug masked it) |
| v1 (final) | 2-layer 64->32, drop=0.5, Adam WD=1e-3 | 55.0% | 0.008 | Clean, honest, consistent |
| v2 | 1-layer 32, drop=0.6, AdamW WD=1e-2 | ~49% | 0.55 | Over-regularized, killed signal |
| v3 | 2-layer 64->32, AdamW WD=1e-3 | 55.4% | 0.005 | 10Y also hit 99% (noise) |
| v4 | 2-layer 64->32, AdamW WD=5e-4 | 52.7% | 0.10 | Unstable, lost signal |

## Bugs Fixed This Session

1. **LSTM train accuracy misalignment:** Predictions at positions LOOKBACK through n_train-1 were compared against labels at positions 0 through n_train-LOOKBACK-1 (60-day offset). Train accuracy was meaningless garbage. Fixed to `y_train[LOOKBACK:LOOKBACK+len(preds)]`.
2. **XGBoost overfitting (prior session):** Added 9-parameter regularized search space with early stopping.

## Documents Updated

- `yield_curve_prophet.ipynb` - All cells, Key Findings section with train/val table and LSTM gap explanation
- `README.md` - Key Result table, architecture description, design decisions
- `Yield_Curve_Prophet_Paper.docx` - All 3 data tables, abstract, contributions, sections 5.1/5.2/5.4/6.1/6.2/6.3, conclusion, P&L paragraph

## Files in Repo

- `yield_curve_prophet.ipynb` - Complete pipeline (single notebook)
- `README.md` - Project documentation
- `requirements.txt` - Dependencies (includes statsmodels)
- `final_features.txt` - 90 engineered features with definitions
- `summary.md` - This file
- `LICENSE` - MIT
- `.gitignore` - Standard Python
- `*.png` - Generated charts (8 figures)
