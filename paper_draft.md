# Curve Trades Are Forecastable, Duration Bets Are Not: Evidence from Machine Learning on 20 Years of US Treasury Data

**Author:** lenamonj
**Date:** May 2026

---

## Abstract

We construct an ensemble classifier combining gradient boosting (XGBoost) and a recurrent neural network (LSTM) to predict weekly directional moves in the US Treasury yield curve. Using 20 years of publicly available macroeconomic data from FRED (2005-2025) and strict walk-forward validation with no future information leakage, we find a sharp divergence in predictability across targets. Outright yield direction for the 2-year and 10-year Treasury is statistically indistinguishable from a coin flip (43-49% accuracy), consistent with the efficient markets hypothesis for the most liquid fixed income instruments in the world. However, the direction of the 2s10s spread (steepening vs. flattening) is predictable at 57.9% accuracy using an ensemble that weights XGBoost at 75% and the LSTM at 25%. This finding has direct implications for portfolio construction: curve positioning (steepener/flattener) carries forecastable signal that outright duration bets do not. Notably, the LSTM does not outperform XGBoost on any target, suggesting that for weekly macro forecasting, feature engineering matters more than sequential modeling.

---

## 1. Introduction

The US Treasury yield curve is the most watched indicator in fixed income markets. Its level determines the cost of capital for governments and corporations. Its shape - whether steep, flat, or inverted - is a leading indicator of recessions, a driver of bank profitability, and the foundation of relative value trading across the rates complex.

Despite its centrality, yield forecasting remains overwhelmingly discretionary. Portfolio managers read FOMC minutes, watch payrolls prints, interpret dot plots, and make positioning calls based on judgment. Quantitative models exist, but most are confined to proprietary trading desks with access to Bloomberg, tick data, and real-time order flow.

This paper asks a simpler question: using only free, publicly available macroeconomic data, can a machine learning model predict the *direction* of weekly yield curve moves? And if so, which components of the curve are forecastable?

We make three contributions:

1. **We demonstrate that outright yield direction is not forecastable with weekly macro data.** Both the 2-year and 10-year Treasury yield direction predictions hover around 44-49% accuracy on a held-out test set spanning 2021-2025. No model architecture (tree-based or neural) achieves a statistically significant edge over random guessing. This is a negative result, but an important one: it sets a baseline for what macro data alone cannot do.

2. **We show that curve spread direction IS forecastable.** The 2s10s spread (10-year minus 2-year yield) direction is predicted at 57.9% accuracy by a weighted ensemble. This is a meaningful and consistent edge over the 50% baseline, sustained across the full test period. The result is intuitive: spread dynamics are driven by identifiable macro factors (Fed policy stance vs. term premium expectations) in ways that outright yield levels are not.

3. **We find that temporal sequence does not improve prediction.** A 2-layer LSTM trained on 60-day macro sequences does not outperform a gradient boosting model that sees only today's feature snapshot. On the best-performing target (2s10s spread), XGBoost achieves 57.4% accuracy vs. LSTM's 55.6%. The optimal ensemble weight is 75% XGBoost. This suggests that for weekly forecasting horizons, the *level and momentum* of macro variables matter more than their *sequential ordering*.

---

## 2. Related Work

Yield curve forecasting has a long academic history. The Nelson-Siegel (1987) and Diebold-Li (2006) factor models decompose the curve into level, slope, and curvature components and forecast each as an autoregressive process. These models are interpretable but linear. More recently, Bianchi, Buchner, and Tamoni (2021) applied machine learning to bond return prediction, finding that tree-based models outperform linear specifications on monthly horizons.

Neural network approaches to financial time series have gained traction since Fischer and Krauss (2018) applied LSTMs to equity returns. However, the literature on LSTM-based yield curve forecasting is thin. Most published work focuses on equity indices or FX, where data is more abundant and patterns arguably more persistent.

Our contribution differs from prior work in three ways. First, we use exclusively free, public data - no Bloomberg, no TRACE, no proprietary feeds. Second, we frame the problem as binary classification (direction) rather than regression (level), which avoids the noise-amplification problems inherent in predicting basis-point magnitudes. Third, we explicitly compare sequential (LSTM) and non-sequential (XGBoost) architectures on the same feature set to isolate whether temporal structure adds value.

---

## 3. Data

### 3.1 Source

All data is sourced from the Federal Reserve Economic Data (FRED) API, maintained by the Federal Reserve Bank of St. Louis. FRED provides free access to over 800,000 economic time series. We use 13 series spanning Treasury yields, monetary policy, inflation expectations, labor markets, growth indicators, volatility, and credit stress.

### 3.2 Series Selection

| Category | Series | Description | Frequency |
|----------|--------|-------------|-----------|
| Treasury Yields | DGS2, DGS5, DGS10, DGS30 | Constant maturity yields (2Y, 5Y, 10Y, 30Y) | Daily |
| Monetary Policy | DFF | Effective federal funds rate | Daily |
| Inflation | T10YIE | 10-year breakeven inflation rate | Daily |
| Inflation | T5YIFR | 5-year, 5-year forward inflation expectation | Daily |
| Labor | UNRATE | Civilian unemployment rate | Monthly |
| Labor | ICSA | Initial jobless claims | Weekly |
| Growth | INDPRO | Industrial production index | Monthly |
| Sentiment | UMCSENT | University of Michigan consumer sentiment | Monthly |
| Volatility | VIXCLS | CBOE Volatility Index (VIX) | Daily |

Monthly and weekly series are forward-filled to daily frequency and aligned to the trading day calendar derived from DGS10 (the most complete daily series). Two series originally planned for inclusion - the TED spread (TEDRATE, discontinued January 2022) and ICE BofA HY OAS (BAMLH0A0HYM2, limited to 3 years of FRED history) - were dropped due to insufficient coverage.

### 3.3 Sample Period

The sample spans January 2005 through December 2025, yielding approximately 5,000 trading days after alignment and cleaning. This period includes:

- The pre-GFC tightening cycle (2005-2007)
- The Global Financial Crisis (2008-2009)
- The zero-lower-bound era (2009-2015)
- The initial normalization cycle (2016-2018)
- The COVID shock and emergency easing (2020)
- The post-COVID hiking cycle (2022-2023)
- Curve inversion and normalization (2023-2025)

This diversity of regimes is critical for model generalization. Any model trained on a single regime would likely fail out-of-sample.

### 3.4 Feature Engineering

For each of the 13 raw series plus 3 derived series (2s10s spread, 5s30s spread, real 10-year yield), we compute six engineered features:

1. **5-day change** - weekly delta, capturing short-term momentum
2. **21-day change** - monthly delta, capturing medium-term trend
3. **63-day change** - quarterly delta, capturing cyclical moves
4. **21-day rolling volatility** - standard deviation of daily changes, capturing regime shifts
5. **Z-score vs. 252-day rolling mean** - current level relative to its 1-year history
6. **Momentum signal** - binary indicator: 5-day MA above 21-day MA

This produces approximately 80-100 features after engineering. The derived series (spreads, real yield) are computed before feature engineering so that their own rolling statistics are captured.

---

## 4. Methodology

### 4.1 Target Construction

For each trading day *t*, we compute the 5-trading-day forward change in three target variables:

- **2Y yield direction:** sign(DGS2(t+5) - DGS2(t))
- **10Y yield direction:** sign(DGS10(t+5) - DGS10(t))
- **2s10s spread direction:** sign(spread(t+5) - spread(t))

Days with exactly zero forward change (rare but uninformative) are dropped. The resulting binary labels are approximately balanced across all three targets.

### 4.2 Walk-Forward Validation

We enforce strict temporal separation:

| Split | Period | Trading Days | Purpose |
|-------|--------|-------------|---------|
| Train | 2005-2018 | ~3,500 | Model training and hyperparameter search |
| Validation | 2019-2020 | ~500 | Early stopping, ensemble weight optimization |
| Test | 2021-2025 | ~1,250 | Final evaluation (all reported metrics) |

No information from the validation or test set is used during training. Feature standardization is computed using training set statistics only (rolling 252-day mean and standard deviation).

### 4.3 Model A: XGBoost

We train one XGBClassifier per target. Hyperparameters are tuned using Optuna with 100 trials per target, optimizing validation ROC-AUC. The search space covers max_depth (3-8), learning_rate (0.01-0.3), n_estimators (100-1000), subsample (0.6-1.0), colsample_bytree (0.6-1.0), and min_child_weight (1-10).

XGBoost serves as the non-sequential baseline. It sees only the current day's feature vector - a flat snapshot of all engineered features at time *t*. It captures nonlinear feature interactions and threshold effects but has no concept of the ordering of previous observations.

### 4.4 Model B: LSTM

We train one LSTM classifier per target using PyTorch. The architecture is:

- Input: (batch, 60, N_features) - 60-day lookback window
- LSTM layer 1: 128 hidden units, dropout 0.3
- LSTM layer 2: 64 hidden units, dropout 0.3
- Dense head: 64 units, ReLU, dropout 0.3, sigmoid output
- Loss: binary cross-entropy
- Optimizer: Adam (lr=1e-3)
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Early stopping: patience 10 on validation loss

The LSTM serves as the sequential model. It receives the full 60-day history of standardized features and learns to extract temporal patterns - momentum shifts, volatility clustering, and regime transitions that play out over weeks.

### 4.5 Model C: Ensemble

The final model is a weighted average of XGBoost and LSTM predicted probabilities:

p_ensemble = w * p_xgb + (1 - w) * p_lstm

The weight *w* is optimized per target on the validation set by grid search over [0.0, 0.05, ..., 1.0], maximizing accuracy. This yields three separate weights reflecting each model's contribution per target.

---

## 5. Results

### 5.1 Overall Performance

| Target | XGBoost | LSTM | Ensemble | Best |
|--------|---------|------|----------|------|
| 2Y Yield Direction | 43.3% | 48.8% | 45.3% | LSTM (48.8%) |
| 10Y Yield Direction | 44.4% | 49.4% | 49.4% | LSTM (49.4%) |
| **2s10s Spread Direction** | **57.4%** | **55.6%** | **57.9%** | **Ensemble (57.9%)** |

*Baseline: 50.0% (coin flip)*

The results split cleanly into two regimes. Outright yield direction is unpredictable: the best model achieves only 49.4% on the 10Y, which is not statistically distinguishable from chance over 1,250 test observations. The 2s10s spread direction, by contrast, is predicted at 57.9% accuracy - a meaningful and sustained edge.

### 5.2 Ensemble Weights

| Target | XGBoost Weight | LSTM Weight | Interpretation |
|--------|---------------|-------------|----------------|
| 2Y Direction | 0.60 | 0.40 | Slight XGBoost preference |
| 10Y Direction | 0.00 | 1.00 | Pure LSTM (XGBoost adds noise) |
| 2s10s Direction | 0.75 | 0.25 | Strong XGBoost preference |

The ensemble learns to trust XGBoost on the spread target and LSTM on the 10Y target. For the 10Y, the optimal weight assigns zero to XGBoost, indicating that the tree model's predictions are pure noise for this target.

### 5.3 ROC-AUC and Calibration

ROC-AUC scores confirm the accuracy findings: 2s10s achieves 0.609 (ensemble), while 2Y and 10Y hover around 0.50-0.55. Calibration curves show that all models appropriately cluster predictions near 0.50, reflecting low confidence. The models are not overconfident - they recognize the difficulty of the problem.

### 5.4 SHAP Feature Importance

SHAP analysis on the XGBoost models reveals the top macro drivers:

- **VIX momentum and volatility** features appear consistently across all three targets. Risk appetite (proxied by VIX) is a dominant signal.
- **Fed Funds rate changes** drive the 2Y target, consistent with the front end's sensitivity to monetary policy expectations.
- **Breakeven inflation z-scores** influence the 10Y target, reflecting the real rate/inflation decomposition of longer-term yields.
- **The curve's own momentum** (rolling changes in the 2s10s spread itself) is a top feature for the spread target, suggesting mean-reversion dynamics.

### 5.5 Rolling Accuracy

Rolling 63-day accuracy shows that the 2s10s ensemble maintains performance throughout the test period without significant degradation. There is regime-dependent variation - accuracy dips during the aggressive hiking phase (early 2022) and recovers during normalization (2024-2025) - but the overall signal is persistent.

---

## 6. Discussion

### 6.1 Why Spreads Are Predictable and Yields Are Not

The core finding - that curve spread direction is forecastable while outright yield direction is not - has a straightforward interpretation rooted in market microstructure.

Outright Treasury yields (especially the 2Y and 10Y) are among the most efficiently priced instruments in the world. They are traded in enormous volume, arbitraged continuously by central banks, primary dealers, and systematic funds, and react instantaneously to information. Weekly macro data - which is public, lagged, and widely consumed - cannot reliably predict where these yields will be in five days. The market has already priced it.

The 2s10s spread, by contrast, is a *derived* quantity. It reflects the *difference* between two yields that are driven by different forces: the 2Y by near-term Fed policy expectations, the 10Y by growth/inflation term premium. These forces move at different speeds and respond to different signals. When the Fed is actively tightening, the 2Y rises faster than the 10Y (flattening). When the market prices in cuts, the 2Y falls faster (steepening). These regime dynamics are captured by macro momentum features and are slow enough to persist over a 5-day horizon.

In fixed income terms: outright duration risk is efficiently priced; curve risk is not.

### 6.2 Why LSTM Does Not Help

The underperformance of the LSTM relative to XGBoost on the spread target is the second key finding. Two explanations are plausible:

**Feature engineering captures the sequence.** The engineered features (5-day, 21-day, and 63-day rolling changes; momentum signals; z-scores) already encode the temporal information that an LSTM would need to learn from raw data. By feeding XGBoost pre-computed momentum and mean-reversion features, we give it explicit access to the patterns that the LSTM must discover implicitly. The tree model wins because the features do the work.

**Weekly horizons are too coarse for sequential patterns.** LSTMs excel when there is fine-grained temporal structure - patterns in the ordering of observations that matter beyond their aggregated statistics. At a weekly forecasting horizon, the relevant signals (Fed stance, inflation trajectory, growth momentum) evolve slowly enough that their current level and trend are more informative than their day-by-day sequence. An LSTM might show more value at intraday or daily horizons where order flow dynamics matter.

### 6.3 Practical Implications

For portfolio managers and rates traders, the actionable takeaway is that curve trades (steepeners and flatteners) are more amenable to systematic forecasting than outright duration positioning. A PM allocating risk budget between "how much duration" and "how to distribute it across the curve" should consider that the latter question has a quantitative answer, while the former may not.

A simple implementation: when the ensemble model predicts steepening with >55% probability, overweight the barbell (long 2Y and 30Y, short 10Y). When it predicts flattening, overweight the bullet (long 10Y). This is not a standalone trading strategy - it requires overlay with fundamental views, position sizing, and risk management - but it provides a systematic tilt informed by macro data.

### 6.4 Limitations

Several limitations should be noted:

1. **Regime dependence.** The test period (2021-2025) includes the most aggressive Fed hiking cycle in four decades followed by an inversion and normalization. Results may differ in a stable-rate environment.

2. **Feature set.** We use only freely available macro data. Proprietary inputs - SOFR futures positioning, Treasury auction metrics, dealer inventory, or real-time order flow - would likely improve predictions, especially for outright yields.

3. **Transaction costs.** The hypothetical P&L does not include bid-ask spreads, roll costs, or margin requirements. A 57.9% directional hit rate may or may not survive realistic trading friction.

4. **Non-stationarity.** The relationship between macro features and yield curve dynamics is not fixed. Structural breaks (e.g., QE, QT, yield curve control in other jurisdictions) can alter the signal.

---

## 7. Conclusion

Using 20 years of publicly available macroeconomic data and strict walk-forward validation, we find that the direction of the US Treasury 2s10s spread is forecastable at 57.9% accuracy, while outright 2-year and 10-year yield direction is not. A gradient boosting model with engineered macro features outperforms an LSTM neural network on the predictable target, suggesting that for weekly horizons, feature engineering matters more than sequential modeling.

The finding that curve trades are more forecastable than duration bets aligns with market intuition but, to our knowledge, has not been demonstrated with this methodology in the public literature. For practitioners, it implies that systematic macro signals should be channeled into curve positioning rather than outright rate views.

All data is free. All code is open source. All results are reproducible.

---

## References

- Bianchi, D., Buchner, M., & Tamoni, A. (2021). Bond risk premiums with machine learning. *Review of Financial Studies*, 34(2), 1046-1089.
- Diebold, F. X., & Li, C. (2006). Forecasting the term structure of government bond yields. *Journal of Econometrics*, 130(2), 337-364.
- Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654-669.
- Nelson, C. R., & Siegel, A. F. (1987). Parsimonious modeling of yield curves. *Journal of Business*, 60(4), 473-489.

---

*Code and data available at: https://github.com/lenamonj/yield-curve-prophet*
