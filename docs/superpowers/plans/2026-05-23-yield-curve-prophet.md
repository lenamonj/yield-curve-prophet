# yield-curve-prophet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-notebook project that predicts weekly directional moves in 2Y, 10Y, and 2s10s spread using an LSTM + XGBoost ensemble on 20 years of FRED macro data.

**Architecture:** FRED API pulls ~15 macro series, engineers ~80-100 features (rolling changes, vol, z-scores, momentum), trains XGBoost baseline + LSTM per target, blends into ensemble. Walk-forward validation (train 2005-2018, val 2019-2020, test 2021-2025). Single Jupyter notebook.

**Tech Stack:** Python, PyTorch, XGBoost, Optuna, SHAP, pandas, FRED API

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `yield_curve_prophet.ipynb` (empty notebook with metadata)

- [ ] **Step 1: Create requirements.txt**

```
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
torch>=2.0.0
xgboost>=2.0.0
optuna>=3.0.0
shap>=0.42.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.py[cod]
.venv/
venv/
.ipynb_checkpoints/
.env
*.egg-info/
dist/
build/
.DS_Store
Thumbs.db
```

- [ ] **Step 3: Create MIT LICENSE**

Standard MIT license, copyright 2025 lenamonj.

- [ ] **Step 4: Create empty notebook**

Create `yield_curve_prophet.ipynb` with Python 3 kernel metadata and zero cells. All subsequent tasks will append cells to this notebook.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore LICENSE yield_curve_prophet.ipynb
git commit -m "Scaffold yield-curve-prophet project"
```

---

### Task 2: Notebook Intro & Setup (Sections 1-2)

**Files:**
- Modify: `yield_curve_prophet.ipynb`

- [ ] **Step 1: Add markdown cell - "Why This Matters"**

```markdown
# Yield Curve Prophet

## Why This Matters

The yield curve is the single most watched indicator in fixed income. Its shape dictates portfolio positioning, relative value trades, and macro regime classification. Every basis point move in the 2Y or 10Y triggers billions in hedging flow.

Most yield forecasting on Wall Street is still discretionary - PMs read the dot plot, watch payrolls, and make a call. This notebook asks whether a machine can do better than a coin flip at predicting the *direction* of weekly yield moves.

We build three classifiers - one for the 2Y (front-end, Fed-sensitive), one for the 10Y (term premium, growth expectations), and one for the 2s10s spread (curve shape, recession signal). Each model sees 60 trading days of macro history and predicts whether the next 5 trading days will move up or down.

**The ensemble (LSTM + XGBoost) lets us answer two questions:**
1. Does temporal sequence matter, or is today's snapshot sufficient? (LSTM vs XGBoost)
2. Can we beat 50%? By how much, and in which regimes?

All data is free via the FRED API. No Bloomberg required.
```

- [ ] **Step 2: Add code cell - imports and configuration**

```python
import os
import warnings
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

import xgboost as xgb
import optuna
import shap
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
    classification_report, roc_curve, calibration_curve,
)
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)

FRED_API_KEY = os.environ["FRED_API_KEY"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"PyTorch device: {DEVICE}")
print(f"Setup complete.")
```

- [ ] **Step 3: Add code cell - FRED API helper and series config**

```python
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

def fetch_fred_series(series_id: str, start: str = "2004-01-01",
                      end: str = "2025-12-31") -> pd.Series:
    """Pull a single FRED series and return as a pandas Series with datetime index."""
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    resp = requests.get(BASE_URL, params=params)
    resp.raise_for_status()
    data = resp.json()["observations"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    series = df.set_index("date")["value"].dropna()
    series.name = series_id
    time.sleep(0.1)  # Gentle throttle
    return series


# Series configuration: (series_id, description, frequency)
FRED_SERIES = [
    ("DGS2", "2Y Treasury Yield", "daily"),
    ("DGS5", "5Y Treasury Yield", "daily"),
    ("DGS10", "10Y Treasury Yield", "daily"),
    ("DGS30", "30Y Treasury Yield", "daily"),
    ("DFF", "Fed Funds Effective Rate", "daily"),
    ("T10YIE", "10Y Breakeven Inflation", "daily"),
    ("T5YIFR", "5Y5Y Forward Inflation", "daily"),
    ("UNRATE", "Unemployment Rate", "monthly"),
    ("ICSA", "Initial Jobless Claims", "weekly"),
    ("INDPRO", "Industrial Production Index", "monthly"),
    ("UMCSENT", "Consumer Sentiment", "monthly"),
    ("VIXCLS", "CBOE VIX", "daily"),
    ("DTWEXBGS", "Trade-Weighted Dollar Index", "daily"),
    ("TEDRATE", "TED Spread", "daily"),
    ("BAMLH0A0HYM2", "HY OAS", "daily"),
]

# Walk-forward split dates
TRAIN_END = "2018-12-31"
VAL_END = "2020-12-31"
# Test: 2021-01-01 onward

LOOKBACK = 60       # Trading days for LSTM input
HORIZON = 5         # 5-day forward prediction
TARGETS = ["dgs2_dir", "dgs10_dir", "spread_2s10s_dir"]
TARGET_LABELS = {
    "dgs2_dir": "2Y Yield Direction",
    "dgs10_dir": "10Y Yield Direction",
    "spread_2s10s_dir": "2s10s Spread Direction",
}

print(f"Configured {len(FRED_SERIES)} FRED series")
print(f"Train: 2005 - {TRAIN_END}")
print(f"Val: {TRAIN_END} - {VAL_END}")
print(f"Test: {VAL_END} - 2025")
```

- [ ] **Step 4: Commit**

```bash
git add yield_curve_prophet.ipynb
git commit -m "Add intro, imports, and FRED configuration"
```

---

### Task 3: Data Collection (Section 3)

**Files:**
- Modify: `yield_curve_prophet.ipynb`

- [ ] **Step 1: Add markdown cell**

```markdown
## 2. Data Collection

Pull all FRED series, align to a common daily trading calendar, and forward-fill lower-frequency series (monthly, weekly) to daily.
```

- [ ] **Step 2: Add code cell - pull all series**

```python
print("Pulling FRED data...")
print("=" * 50)

raw_series = {}
for series_id, desc, freq in FRED_SERIES:
    try:
        s = fetch_fred_series(series_id)
        raw_series[series_id] = s
        print(f"  {series_id:20s} | {desc:35s} | {len(s):,} obs | {s.index[0].date()} to {s.index[-1].date()}")
    except Exception as e:
        print(f"  {series_id:20s} | FAILED: {e}")

print(f"\nPulled {len(raw_series)} / {len(FRED_SERIES)} series successfully.")
```

- [ ] **Step 3: Add code cell - align to daily trading calendar**

```python
# Build a common trading day index from DGS10 (most complete daily series)
trading_days = raw_series["DGS10"].index

# Combine all series into a single DataFrame, reindex to trading days
df = pd.DataFrame(index=trading_days)
for series_id, s in raw_series.items():
    df[series_id] = s.reindex(df.index)

# Forward-fill lower-frequency series (monthly/weekly data)
df = df.ffill()

# Trim to 2005+ (need 2004 data for some rolling calcs that look back)
df = df.loc["2005-01-01":]

# Drop any rows with NaNs remaining at the start
df = df.dropna()

print(f"Aligned dataset: {df.shape[0]:,} trading days x {df.shape[1]} series")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"\nMissing values per series:")
print(df.isnull().sum())
```

- [ ] **Step 4: Add code cell - data quality validation**

```python
# Sanity checks
assert df.shape[0] > 4000, f"Expected ~5000 trading days, got {df.shape[0]}"
assert df.isnull().sum().sum() == 0, "No NaNs should remain after ffill + dropna"
assert "DGS2" in df.columns and "DGS10" in df.columns, "Must have 2Y and 10Y yields"

# Quick visual: yield levels over time
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Raw FRED Series: Yield Curve Inputs", fontsize=14, fontweight="bold")

ax = axes[0, 0]
for tenor in ["DGS2", "DGS5", "DGS10", "DGS30"]:
    ax.plot(df.index, df[tenor], label=tenor, linewidth=1)
ax.set_title("Treasury Yields")
ax.set_ylabel("Yield (%)")
ax.legend()

ax = axes[0, 1]
ax.plot(df.index, df["DFF"], color="darkred", linewidth=1)
ax.set_title("Fed Funds Rate")
ax.set_ylabel("Rate (%)")

ax = axes[1, 0]
ax.plot(df.index, df["VIXCLS"], color="purple", linewidth=0.8)
ax.set_title("VIX")
ax.set_ylabel("Level")

ax = axes[1, 1]
if "BAMLH0A0HYM2" in df.columns:
    ax.plot(df.index, df["BAMLH0A0HYM2"], color="orange", linewidth=1)
ax.set_title("HY OAS (bps)")
ax.set_ylabel("OAS")

plt.tight_layout()
plt.savefig("raw_fred_series.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: raw_fred_series.png")
```

- [ ] **Step 5: Commit**

```bash
git add yield_curve_prophet.ipynb
git commit -m "Add FRED data collection and alignment"
```

---

### Task 4: Feature Engineering (Section 4)

**Files:**
- Modify: `yield_curve_prophet.ipynb`

- [ ] **Step 1: Add markdown cell**

```markdown
## 3. Feature Engineering

For each raw series, compute rolling changes (5d, 21d, 63d), realized volatility (21d), z-scores (vs 252d rolling mean), and momentum signals. Also derive curve spreads (2s10s, 5s30s, real 10Y).

Total features: ~80-100 after engineering.
```

- [ ] **Step 2: Add code cell - derived series**

```python
# Derived curve measures
df["spread_2s10s"] = df["DGS10"] - df["DGS2"]
df["spread_5s30s"] = df["DGS30"] - df["DGS5"]
df["real_10y"] = df["DGS10"] - df["T10YIE"]

print("Derived series added: spread_2s10s, spread_5s30s, real_10y")
print(f"Current 2s10s: {df['spread_2s10s'].iloc[-1]:.2f}%")
print(f"Current real 10Y: {df['real_10y'].iloc[-1]:.2f}%")
```

- [ ] **Step 3: Add code cell - feature engineering function**

```python
def engineer_features(df: pd.DataFrame, series_cols: list[str]) -> pd.DataFrame:
    """
    Engineer rolling features for each series column.
    Returns a new DataFrame with original + engineered columns.
    """
    feat = df.copy()

    for col in series_cols:
        s = feat[col]

        # Rolling changes (absolute, in same units as original)
        feat[f"{col}_chg5d"] = s.diff(5)
        feat[f"{col}_chg21d"] = s.diff(21)
        feat[f"{col}_chg63d"] = s.diff(63)

        # Realized volatility (21-day rolling std of daily changes)
        daily_chg = s.diff(1)
        feat[f"{col}_vol21d"] = daily_chg.rolling(21).std()

        # Z-score vs 252-day rolling stats
        roll_mean = s.rolling(252).mean()
        roll_std = s.rolling(252).std()
        feat[f"{col}_zscore"] = (s - roll_mean) / roll_std.replace(0, np.nan)

        # Momentum: 5d MA vs 21d MA (1 = bullish, 0 = bearish)
        ma5 = s.rolling(5).mean()
        ma21 = s.rolling(21).mean()
        feat[f"{col}_momentum"] = (ma5 > ma21).astype(int)

    return feat


# Apply to all raw + derived series
feature_cols = [sid for sid, _, _ in FRED_SERIES] + ["spread_2s10s", "spread_5s30s", "real_10y"]
df_feat = engineer_features(df, feature_cols)

# Drop rows with NaN from rolling calcs (first ~252 days)
df_feat = df_feat.dropna()

print(f"Feature matrix: {df_feat.shape[0]:,} rows x {df_feat.shape[1]} columns")
print(f"Date range after engineering: {df_feat.index[0].date()} to {df_feat.index[-1].date()}")
print(f"\nSample feature names (first 20):")
for c in df_feat.columns[:20]:
    print(f"  {c}")
print(f"  ... and {len(df_feat.columns) - 20} more")
```

- [ ] **Step 4: Commit**

```bash
git add yield_curve_prophet.ipynb
git commit -m "Add feature engineering: rolling changes, vol, z-scores, momentum"
```

---

### Task 5: Target Construction (Section 5)

**Files:**
- Modify: `yield_curve_prophet.ipynb`

- [ ] **Step 1: Add markdown cell**

```markdown
## 4. Target Construction

Create binary labels for 5-day forward yield/spread direction. Drop flat moves (exactly 0 change). Check class balance.
```

- [ ] **Step 2: Add code cell - build targets**

```python
# 5-day forward changes
df_feat["dgs2_fwd5d"] = df_feat["DGS2"].shift(-HORIZON) - df_feat["DGS2"]
df_feat["dgs10_fwd5d"] = df_feat["DGS10"].shift(-HORIZON) - df_feat["DGS10"]
df_feat["spread_2s10s_fwd5d"] = df_feat["spread_2s10s"].shift(-HORIZON) - df_feat["spread_2s10s"]

# Binary direction labels (1 = up/steepening, 0 = down/flattening)
df_feat["dgs2_dir"] = (df_feat["dgs2_fwd5d"] > 0).astype(int)
df_feat["dgs10_dir"] = (df_feat["dgs10_fwd5d"] > 0).astype(int)
df_feat["spread_2s10s_dir"] = (df_feat["spread_2s10s_fwd5d"] > 0).astype(int)

# Drop rows where forward change is exactly 0 (rare but uninformative)
for target in TARGETS:
    fwd_col = target.replace("_dir", "_fwd5d")
    zero_mask = df_feat[fwd_col] == 0
    n_zeros = zero_mask.sum()
    if n_zeros > 0:
        print(f"Dropping {n_zeros} flat moves for {target}")
        df_feat = df_feat[~zero_mask]

# Drop rows with NaN targets (last 5 rows due to forward shift)
df_feat = df_feat.dropna(subset=TARGETS)

# Class balance
print("\nClass balance (% label=1):")
for target in TARGETS:
    pct = df_feat[target].mean() * 100
    print(f"  {TARGET_LABELS[target]:30s}: {pct:.1f}% up | {100-pct:.1f}% down | n={len(df_feat[target]):,}")
```

- [ ] **Step 3: Commit**

```bash
git add yield_curve_prophet.ipynb
git commit -m "Add target construction: 5-day forward direction labels"
```

---

### Task 6: Train/Val/Test Split (Section 6)

**Files:**
- Modify: `yield_curve_prophet.ipynb`

- [ ] **Step 1: Add markdown cell**

```markdown
## 5. Train / Validation / Test Split

Walk-forward temporal split. No random shuffling. Rolling standardization uses only past data to prevent future leakage.

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 2005-2018 | Model training |
| Validation | 2019-2020 | Hyperparameter tuning, early stopping, ensemble weights |
| Test | 2021-2025 | Final evaluation (all reported metrics) |
```

- [ ] **Step 2: Add code cell - split and standardize**

```python
# Identify feature columns (everything except raw series, targets, and forward columns)
exclude_suffixes = ["_fwd5d", "_dir"]
exclude_cols = [sid for sid, _, _ in FRED_SERIES] + ["spread_2s10s", "spread_5s30s", "real_10y"]
feature_names = [c for c in df_feat.columns
                 if c not in exclude_cols
                 and not any(c.endswith(s) for s in exclude_suffixes)]

print(f"Feature count: {len(feature_names)}")

# Temporal split
train_mask = df_feat.index <= TRAIN_END
val_mask = (df_feat.index > TRAIN_END) & (df_feat.index <= VAL_END)
test_mask = df_feat.index > VAL_END

X_train_raw = df_feat.loc[train_mask, feature_names]
X_val_raw = df_feat.loc[val_mask, feature_names]
X_test_raw = df_feat.loc[test_mask, feature_names]

y_train = df_feat.loc[train_mask, TARGETS]
y_val = df_feat.loc[val_mask, TARGETS]
y_test = df_feat.loc[test_mask, TARGETS]

print(f"\nSplit sizes:")
print(f"  Train: {len(X_train_raw):,} ({X_train_raw.index[0].date()} to {X_train_raw.index[-1].date()})")
print(f"  Val:   {len(X_val_raw):,} ({X_val_raw.index[0].date()} to {X_val_raw.index[-1].date()})")
print(f"  Test:  {len(X_test_raw):,} ({X_test_raw.index[0].date()} to {X_test_raw.index[-1].date()})")

# Standardize using training set statistics only
scaler = StandardScaler()
scaler.fit(X_train_raw)

X_train = pd.DataFrame(scaler.transform(X_train_raw), index=X_train_raw.index, columns=feature_names)
X_val = pd.DataFrame(scaler.transform(X_val_raw), index=X_val_raw.index, columns=feature_names)
X_test = pd.DataFrame(scaler.transform(X_test_raw), index=X_test_raw.index, columns=feature_names)

print(f"\nStandardized. Train mean ~0: {X_train.mean().mean():.4f}, std ~1: {X_train.std().mean():.4f}")
```

- [ ] **Step 3: Commit**

```bash
git add yield_curve_prophet.ipynb
git commit -m "Add walk-forward train/val/test split with standardization"
```

---

### Task 7: XGBoost Baseline (Section 7)

**Files:**
- Modify: `yield_curve_prophet.ipynb`

- [ ] **Step 1: Add markdown cell**

```markdown
## 6. XGBoost Baseline

One XGBClassifier per target, tuned with Optuna (100 trials, ROC-AUC objective). This is the "does temporal structure matter?" benchmark - XGBoost sees only today's feature snapshot, not the sequence.

SHAP values reveal which macro features drive predictions.
```

- [ ] **Step 2: Add code cell - Optuna tuning + training**

```python
def tune_xgboost(X_tr: pd.DataFrame, y_tr: pd.Series,
                 X_v: pd.DataFrame, y_v: pd.Series,
                 n_trials: int = 100) -> dict:
    """Tune XGBoost hyperparameters with Optuna."""
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "eval_metric": "auc",
            "random_state": 42,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
        y_prob = model.predict_proba(X_v)[:, 1]
        return roc_auc_score(y_v, y_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


# Train one XGBoost model per target
xgb_models = {}
xgb_val_probs = {}
xgb_test_probs = {}

print("Training XGBoost models...")
print("=" * 60)

for target in TARGETS:
    print(f"\n--- {TARGET_LABELS[target]} ---")

    # Tune
    best_params = tune_xgboost(X_train, y_train[target], X_val, y_val[target], n_trials=100)
    best_params["eval_metric"] = "auc"
    best_params["random_state"] = 42
    print(f"Best params: max_depth={best_params['max_depth']}, lr={best_params['learning_rate']:.3f}, n_est={best_params['n_estimators']}")

    # Train final model on train set
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train[target], eval_set=[(X_val, y_val[target])], verbose=False)

    # Store predictions
    xgb_val_probs[target] = model.predict_proba(X_val.values)[:, 1]
    xgb_test_probs[target] = model.predict_proba(X_test.values)[:, 1]
    xgb_models[target] = model

    # Quick eval
    val_acc = accuracy_score(y_val[target], (xgb_val_probs[target] > 0.5).astype(int))
    test_acc = accuracy_score(y_test[target], (xgb_test_probs[target] > 0.5).astype(int))
    test_auc = roc_auc_score(y_test[target], xgb_test_probs[target])
    print(f"Val acc: {val_acc:.1%} | Test acc: {test_acc:.1%} | Test AUC: {test_auc:.3f}")

print("\nXGBoost training complete.")
```

- [ ] **Step 3: Add code cell - SHAP analysis**

```python
# SHAP values for XGBoost interpretability
print("Computing SHAP values...")

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle("XGBoost Feature Importance (SHAP)", fontsize=14, fontweight="bold")

for idx, target in enumerate(TARGETS):
    explainer = shap.TreeExplainer(xgb_models[target])
    shap_values = explainer.shap_values(X_test)

    ax = axes[idx]
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15,
                      show=False, ax=ax)
    ax.set_title(TARGET_LABELS[target])

plt.tight_layout()
plt.savefig("xgb_shap_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: xgb_shap_importance.png")
```

- [ ] **Step 4: Commit**

```bash
git add yield_curve_prophet.ipynb
git commit -m "Add XGBoost baseline with Optuna tuning and SHAP"
```

---

### Task 8: LSTM Model (Section 8)

**Files:**
- Modify: `yield_curve_prophet.ipynb`

- [ ] **Step 1: Add markdown cell**

```markdown
## 7. LSTM Neural Network

Two-layer LSTM that ingests 60-day sequences of macro features. Unlike XGBoost, which sees only today's snapshot, the LSTM learns temporal patterns - momentum shifts, volatility clustering, and regime transitions that play out over weeks.

Architecture: LSTM(128) -> LSTM(64) -> Dense(64) -> Sigmoid
```

- [ ] **Step 2: Add code cell - PyTorch Dataset and Model**

```python
class YieldDataset(Dataset):
    """Sliding window dataset for LSTM input."""
    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int = 60):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.lookback = lookback

    def __len__(self):
        return len(self.X) - self.lookback

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.lookback]
        y_val = self.y[idx + self.lookback]
        return x_seq, y_val


class YieldLSTM(nn.Module):
    """Two-layer LSTM classifier for yield direction prediction."""
    def __init__(self, input_size: int, hidden1: int = 128, hidden2: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # Take final hidden state
        return self.head(out).squeeze(-1)


print(f"LSTM architecture defined.")
print(f"Input features: {len(feature_names)}")
print(f"Lookback window: {LOOKBACK} days")
print(f"Device: {DEVICE}")
```

- [ ] **Step 3: Add code cell - LSTM training loop**

```python
def train_lstm(X_tr: np.ndarray, y_tr: np.ndarray,
               X_v: np.ndarray, y_v: np.ndarray,
               input_size: int, target_name: str,
               max_epochs: int = 100, patience: int = 10,
               batch_size: int = 64, lr: float = 1e-3) -> nn.Module:
    """Train LSTM with early stopping on validation loss."""
    train_ds = YieldDataset(X_tr, y_tr, LOOKBACK)
    val_ds = YieldDataset(X_v, y_v, LOOKBACK)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = YieldLSTM(input_size).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(xb)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model


def predict_lstm(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Generate predictions from LSTM model on a full array."""
    ds = YieldDataset(X, np.zeros(len(X)), LOOKBACK)
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(DEVICE)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds)
```

- [ ] **Step 4: Add code cell - train LSTM for all targets**

```python
# Prepare sequential data: concat train+val+test for proper windowing
X_all = pd.concat([X_train, X_val, X_test]).values
y_all_dict = {t: pd.concat([y_train[t], y_val[t], y_test[t]]).values for t in TARGETS}

# Index boundaries (accounting for lookback offset)
n_train = len(X_train)
n_val = len(X_val)
n_test = len(X_test)

lstm_models = {}
lstm_val_probs = {}
lstm_test_probs = {}

print("Training LSTM models...")
print("=" * 60)

for target in TARGETS:
    print(f"\n--- {TARGET_LABELS[target]} ---")

    # Build training arrays (train period)
    X_tr_seq = X_all[:n_train + LOOKBACK]
    y_tr_seq = y_all_dict[target][:n_train + LOOKBACK]

    # Build validation arrays (train + val period for proper windowing)
    X_v_seq = X_all[:n_train + n_val + LOOKBACK]
    y_v_seq = y_all_dict[target][:n_train + n_val + LOOKBACK]

    model = train_lstm(
        X_tr_seq, y_tr_seq,
        X_v_seq, y_v_seq,
        input_size=len(feature_names),
        target_name=target,
    )
    lstm_models[target] = model

    # Predictions on val and test
    # Val predictions: use data up through val period
    val_preds = predict_lstm(model, X_all[:n_train + n_val])
    lstm_val_probs[target] = val_preds[n_train - LOOKBACK:]

    # Test predictions: use all data
    test_preds = predict_lstm(model, X_all)
    lstm_test_probs[target] = test_preds[n_train + n_val - LOOKBACK:]

    # Trim to match y lengths
    lstm_val_probs[target] = lstm_val_probs[target][:n_val]
    lstm_test_probs[target] = lstm_test_probs[target][:n_test]

    val_acc = accuracy_score(y_val[target].values, (lstm_val_probs[target] > 0.5).astype(int))
    test_acc = accuracy_score(y_test[target].values[:len(lstm_test_probs[target])],
                              (lstm_test_probs[target] > 0.5).astype(int))
    print(f"  Val acc: {val_acc:.1%} | Test acc: {test_acc:.1%}")

print("\nLSTM training complete.")
```

- [ ] **Step 5: Commit**

```bash
git add yield_curve_prophet.ipynb
git commit -m "Add LSTM model: architecture, training loop, all targets"
```

---

### Task 9: Ensemble (Section 9)

**Files:**
- Modify: `yield_curve_prophet.ipynb`

- [ ] **Step 1: Add markdown cell**

```markdown
## 8. Ensemble

Weighted average of XGBoost and LSTM probabilities. Weight optimized per target on validation set accuracy via grid search.

`p_ensemble = w * p_xgb + (1-w) * p_lstm`
```

- [ ] **Step 2: Add code cell - optimize ensemble weights**

```python
ensemble_weights = {}
ensemble_test_probs = {}

print("Optimizing ensemble weights...")
print("=" * 60)

for target in TARGETS:
    best_w = 0.5
    best_acc = 0

    # Trim to common length
    n_common_val = min(len(xgb_val_probs[target]), len(lstm_val_probs[target]))
    xgb_vp = xgb_val_probs[target][:n_common_val]
    lstm_vp = lstm_val_probs[target][:n_common_val]
    y_v = y_val[target].values[:n_common_val]

    for w in np.arange(0, 1.05, 0.05):
        blended = w * xgb_vp + (1 - w) * lstm_vp
        acc = accuracy_score(y_v, (blended > 0.5).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_w = w

    ensemble_weights[target] = best_w

    # Apply to test set
    n_common_test = min(len(xgb_test_probs[target]), len(lstm_test_probs[target]))
    ensemble_test_probs[target] = (
        best_w * xgb_test_probs[target][:n_common_test]
        + (1 - best_w) * lstm_test_probs[target][:n_common_test]
    )

    test_acc = accuracy_score(
        y_test[target].values[:n_common_test],
        (ensemble_test_probs[target] > 0.5).astype(int)
    )
    print(f"  {TARGET_LABELS[target]:30s} | w_xgb={best_w:.2f} | Val acc={best_acc:.1%} | Test acc={test_acc:.1%}")

print("\nEnsemble optimization complete.")
```

- [ ] **Step 3: Commit**

```bash
git add yield_curve_prophet.ipynb
git commit -m "Add ensemble: weighted blend of XGBoost + LSTM"
```

---

### Task 10: Evaluation & Visualizations (Section 10)

**Files:**
- Modify: `yield_curve_prophet.ipynb`

- [ ] **Step 1: Add markdown cell**

```markdown
## 9. Head-to-Head Comparison

Full evaluation of all three models (XGBoost, LSTM, Ensemble) across all three targets. Metrics: accuracy, precision, recall, F1, ROC-AUC, Brier score.
```

- [ ] **Step 2: Add code cell - comprehensive metrics table**

```python
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all classification metrics."""
    y_pred = (y_prob > 0.5).astype(int)
    n = min(len(y_true), len(y_prob))
    y_true, y_pred, y_prob = y_true[:n], y_pred[:n], y_prob[:n]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
    }


# Build results table
results = []
all_probs = {
    "XGBoost": xgb_test_probs,
    "LSTM": lstm_test_probs,
    "Ensemble": ensemble_test_probs,
}

for model_name, probs_dict in all_probs.items():
    for target in TARGETS:
        y_true = y_test[target].values
        y_prob = probs_dict[target]
        metrics = compute_metrics(y_true, y_prob)
        metrics["Model"] = model_name
        metrics["Target"] = TARGET_LABELS[target]
        results.append(metrics)

results_df = pd.DataFrame(results)
results_df = results_df[["Model", "Target", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Brier"]]

print("=" * 90)
print("MODEL COMPARISON - TEST SET (2021-2025)")
print("=" * 90)
print(results_df.to_string(index=False))
print(f"\nBaseline (coin flip): 50.0% accuracy")
```

- [ ] **Step 3: Add code cell - ROC curves**

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("ROC Curves: XGBoost vs LSTM vs Ensemble", fontsize=14, fontweight="bold")

colors = {"XGBoost": "#2ecc71", "LSTM": "#3498db", "Ensemble": "#e74c3c"}

for idx, target in enumerate(TARGETS):
    ax = axes[idx]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (0.500)")

    for model_name, probs_dict in all_probs.items():
        y_true = y_test[target].values
        y_prob = probs_dict[target]
        n = min(len(y_true), len(y_prob))
        fpr, tpr, _ = roc_curve(y_true[:n], y_prob[:n])
        auc = roc_auc_score(y_true[:n], y_prob[:n])
        ax.plot(fpr, tpr, color=colors[model_name], linewidth=2,
                label=f"{model_name} ({auc:.3f})")

    ax.set_title(TARGET_LABELS[target])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: roc_curves.png")
```

- [ ] **Step 4: Add code cell - confusion matrices**

```python
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle("Confusion Matrices (Test Set)", fontsize=14, fontweight="bold")

for col, (model_name, probs_dict) in enumerate(all_probs.items()):
    for row, target in enumerate(TARGETS):
        ax = axes[row][col]
        y_true = y_test[target].values
        y_prob = probs_dict[target]
        n = min(len(y_true), len(y_prob))
        y_pred = (y_prob[:n] > 0.5).astype(int)

        cm = confusion_matrix(y_true[:n], y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
        acc = accuracy_score(y_true[:n], y_pred)
        ax.set_title(f"{model_name} - {TARGET_LABELS[target]}\nAcc: {acc:.1%}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: confusion_matrices.png")
```

- [ ] **Step 5: Add code cell - rolling accuracy**

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Rolling 63-Day Accuracy (Test Set)", fontsize=14, fontweight="bold")

for idx, target in enumerate(TARGETS):
    ax = axes[idx]
    test_dates = y_test.index

    for model_name, probs_dict in all_probs.items():
        y_true = y_test[target].values
        y_prob = probs_dict[target]
        n = min(len(y_true), len(y_prob))
        correct = (y_true[:n] == (y_prob[:n] > 0.5).astype(int)).astype(float)
        rolling_acc = pd.Series(correct, index=test_dates[:n]).rolling(63).mean()
        ax.plot(rolling_acc.index, rolling_acc.values, label=model_name,
                color=colors[model_name], linewidth=1.5)

    ax.axhline(0.5, color="black", linestyle="--", alpha=0.3, label="50% baseline")
    ax.set_title(TARGET_LABELS[target])
    ax.set_ylabel("63-Day Rolling Accuracy")
    ax.legend()
    ax.set_ylim(0.3, 0.7)

plt.tight_layout()
plt.savefig("rolling_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: rolling_accuracy.png")
```

- [ ] **Step 6: Add code cell - calibration curve**

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Probability Calibration (Test Set)", fontsize=14, fontweight="bold")

for idx, target in enumerate(TARGETS):
    ax = axes[idx]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")

    for model_name, probs_dict in all_probs.items():
        y_true = y_test[target].values
        y_prob = probs_dict[target]
        n = min(len(y_true), len(y_prob))
        prob_true, prob_pred = calibration_curve(y_true[:n], y_prob[:n], n_bins=10)
        brier = brier_score_loss(y_true[:n], y_prob[:n])
        ax.plot(prob_pred, prob_true, "o-", color=colors[model_name], linewidth=2,
                label=f"{model_name} (Brier={brier:.3f})")

    ax.set_title(TARGET_LABELS[target])
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.legend()

plt.tight_layout()
plt.savefig("calibration_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: calibration_curves.png")
```

- [ ] **Step 7: Add code cell - hypothetical P&L**

```python
# Hypothetical: go long 10Y when model predicts "up", short when "down"
# P&L = direction_bet * actual_weekly_change_in_bps
target = "dgs10_dir"

fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle("Hypothetical P&L: Long/Short 10Y Based on Model Signal (Test Set)",
             fontsize=14, fontweight="bold")

test_dates = y_test.index
fwd_changes = df_feat.loc[test_mask, "dgs10_fwd5d"].values

for model_name, probs_dict in all_probs.items():
    y_prob = probs_dict[target]
    n = min(len(fwd_changes), len(y_prob))
    signal = np.where(y_prob[:n] > 0.5, 1, -1)
    # P&L: if we predict "up" and rates go up, we made money (long duration loses, but
    # we're betting on direction, so +1 * positive_change = positive P&L)
    pnl = signal * fwd_changes[:n] * 100  # Convert to bps
    cum_pnl = np.cumsum(pnl)
    ax.plot(test_dates[:n], cum_pnl, color=colors[model_name], linewidth=1.5,
            label=f"{model_name} ({cum_pnl[-1]:+.0f} bps)")

ax.axhline(0, color="black", linestyle="-", alpha=0.2)
ax.set_ylabel("Cumulative P&L (bps)")
ax.set_xlabel("Date")
ax.legend()

plt.tight_layout()
plt.savefig("hypothetical_pnl.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: hypothetical_pnl.png")
```

- [ ] **Step 8: Add code cell - prediction confidence histogram**

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Prediction Confidence Distribution (Test Set)", fontsize=14, fontweight="bold")

for idx, target in enumerate(TARGETS):
    ax = axes[idx]
    for model_name, probs_dict in all_probs.items():
        ax.hist(probs_dict[target], bins=30, alpha=0.4, color=colors[model_name],
                label=model_name, density=True)
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.3)
    ax.set_title(TARGET_LABELS[target])
    ax.set_xlabel("Predicted Probability (Up)")
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()
plt.savefig("prediction_confidence.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: prediction_confidence.png")
```

- [ ] **Step 9: Commit**

```bash
git add yield_curve_prophet.ipynb
git commit -m "Add full evaluation: metrics table, ROC, confusion matrices, rolling accuracy, calibration, P&L"
```

---

### Task 11: Key Findings (Section 11)

**Files:**
- Modify: `yield_curve_prophet.ipynb`

- [ ] **Step 1: Add markdown cell - key findings**

```markdown
## 10. Key Findings

**Model Performance Summary**

The results table above shows how each model performed on the held-out test set (2021-2025). Key takeaways:

1. **Does temporal structure matter?** Compare LSTM vs XGBoost accuracy. If LSTM consistently outperforms, the sequence of macro readings over the past 60 days contains signal beyond what today's snapshot captures.

2. **Does the ensemble help?** The weighted blend should match or beat both individual models. The optimal weights reveal which model contributes more per target.

3. **Which tenor is most predictable?** The 2Y is more Fed-driven (policy-sensitive), while the 10Y is more market-driven (term premium, growth expectations). The 2s10s spread combines both signals.

4. **What drives predictions?** SHAP values show which macro features the XGBoost models rely on. Common suspects: VIX (risk appetite), Fed Funds (policy stance), breakeven inflation (real rate expectations), and the yield curve's own momentum.

5. **Calibration matters.** A model that says "60% chance of up" should be right 60% of the time. The calibration curves show whether the models are well-calibrated or overconfident.

**Caveats:**
- This is a directional classification exercise, not a trading strategy
- Past performance in backtest does not guarantee future results
- The walk-forward split is honest (no future leakage), but the test period (2021-2025) includes a specific regime (hiking cycle + inversion + normalization) that may not generalize
- Real-world implementation would require transaction costs, slippage, and position sizing
```

- [ ] **Step 2: Commit**

```bash
git add yield_curve_prophet.ipynb
git commit -m "Add key findings and caveats section"
```

---

### Task 12: README, Execute, and Push

**Files:**
- Create: `README.md`
- Modify: `yield_curve_prophet.ipynb` (execute in-place)

- [ ] **Step 1: Create README.md**

Professional README matching the user's portfolio style: centered shield badges (Python, PyTorch, XGBoost, FRED API, MIT License), centered title, "Why This Exists" section, results summary table (populated after notebook execution), notebooks section, data source table, design decisions, dependencies table, quick start, project structure, license footer.

Title: `<h1 align="center">Yield Curve Prophet</h1>`

Tagline: `A PyTorch LSTM + XGBoost ensemble that predicts weekly directional moves in US Treasury yields, trained on 20 years of FRED macro data with walk-forward validation and SHAP interpretability.`

No em dashes anywhere.

- [ ] **Step 2: Execute notebook**

```bash
export FRED_API_KEY="your-key"
jupyter nbconvert --to notebook --execute --inplace yield_curve_prophet.ipynb --ExecutePreprocessor.timeout=1800
```

This will take 10-20 minutes (Optuna tuning + LSTM training). All outputs (tables, charts, PNGs) will be saved into the notebook.

- [ ] **Step 3: Create GitHub repo and push**

```bash
gh repo create yield-curve-prophet --public --description "LSTM + XGBoost ensemble for weekly yield curve direction prediction. 20 years of FRED macro data, walk-forward validation, SHAP interpretability."
git add .
git commit -m "Add executed notebook with full outputs and README"
git remote add origin https://github.com/lenamonj/yield-curve-prophet.git
git push -u origin main
```
