# Roadmap: Recreating the Özcan 2025 Ensemble RUL Study

> **Paper:** "Interpretable ensemble remaining useful life prediction enables dynamic maintenance scheduling for aircraft engines" — Özcan, *Scientific Reports* (2025)
>
> **Goal:** Reproduce the key results (RMSE ≈ 6.62 on FD001) using the same C-MAPSS data and ensemble methods, then compare against Rudder's geometry-based approach.

---

## Phase 0 — Environment Setup

**What you need:**
- Python 3.10+ with pip
- A code editor (VS Code recommended)
- ~500 MB disk space

**Install the libraries:**
```bash
pip install pandas numpy scikit-learn lightgbm catboost xgboost shap matplotlib seaborn
```

**Why these?** The paper uses LightGBM, CatBoost, Gradient Boosting (scikit-learn), and XGBoost as its core models. SHAP is used for interpretability. Everything else is standard data science tooling.

---

## Phase 1 — Get the Data

**Source:** NASA C-MAPSS dataset (free, public)
- Download: https://data.nasa.gov/download/xaut-bemq/application%2Fzip
- GitHub mirror of the code: https://github.com/hkmtcn/interpretable-rul-maintenance

**What's inside the zip:**
| File | What it is |
|------|-----------|
| `train_FD001.txt` | Training data — 100 engines run to failure |
| `test_FD001.txt` | Test data — 100 engines, cut short |
| `RUL_FD001.txt` | Ground truth RUL for test engines (one per line) |
| FD002, FD003, FD004 | Same format, harder scenarios |

**Each row has 26 columns** (no headers):
1. `unit_id` — which engine
2. `cycle` — time step (1, 2, 3...)
3. `op_1, op_2, op_3` — 3 operating conditions
4. `s1` through `s21` — 21 sensor readings

**NOTE:** You already have `fetchers/cmapss_fetcher.py` in your project that handles downloading and parsing. You can use that or load manually.

---

## Phase 2 — Data Preprocessing

This is where you prepare the raw data for the ML models. Follow these steps in order:

### Step 2a: Load the data
```python
import pandas as pd
import numpy as np

# Column names (no headers in the file)
cols = ['unit_id', 'cycle'] + [f'op_{i}' for i in range(1,4)] + [f's_{i}' for i in range(1,22)]

train = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=cols)
test = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=cols)
rul_true = pd.read_csv('RUL_FD001.txt', header=None, names=['RUL'])
```

### Step 2b: Create the RUL target column
For training data, RUL = (max cycle for that engine) − (current cycle):
```python
# Training RUL
max_cycles = train.groupby('unit_id')['cycle'].max().reset_index()
max_cycles.columns = ['unit_id', 'max_cycle']
train = train.merge(max_cycles, on='unit_id')
train['RUL'] = train['max_cycle'] - train['cycle']
train.drop('max_cycle', axis=1, inplace=True)
```

For test data, you only predict the RUL at the *last* cycle of each engine, then compare to `rul_true`.

### Step 2c: Normalize the features
The paper uses min-max scaling and z-score standardization:
```python
from sklearn.preprocessing import MinMaxScaler

feature_cols = [f'op_{i}' for i in range(1,4)] + [f's_{i}' for i in range(1,22)]

scaler = MinMaxScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols] = scaler.transform(test[feature_cols])  # Use SAME scaler
```

### Step 2d: Prepare train/test splits
```python
X_train = train[feature_cols + ['cycle']]  # cycle is a feature too
y_train = train['RUL']

# For test: get last cycle per engine
test_last = test.groupby('unit_id').last().reset_index()
X_test = test_last[feature_cols + ['cycle']]
y_test = rul_true['RUL'].values
```

---

## Phase 3 — Train Individual Models

The paper trains these models individually first, then combines them.

### Step 3a: LightGBM (the star performer)
```python
import lightgbm as lgb

lgbm_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=-1,        # No limit
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgbm_model.fit(X_train, y_train)
lgbm_preds = lgbm_model.predict(X_test)
```

### Step 3b: CatBoost
```python
from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=0
)
cat_model.fit(X_train, y_train)
cat_preds = cat_model.predict(X_test)
```

### Step 3c: Gradient Boosting (scikit-learn)
```python
from sklearn.ensemble import GradientBoostingRegressor

gbr_model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
gbr_model.fit(X_train, y_train)
gbr_preds = gbr_model.predict(X_test)
```

### Step 3d: XGBoost
```python
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.2,
    max_depth=4,
    n_estimators=200,
    subsample=1.0,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
```

---

## Phase 4 — Build the Ensembles

The paper's best results come from combining models. Start simple:

### Step 4a: Two-model ensemble (LGBM + CatBoost)
```python
# Simple average — this is what the paper uses
ensemble_2_preds = (lgbm_preds + cat_preds) / 2
```

### Step 4b: Three-model ensemble (LGBM + CatBoost + GBR)
```python
ensemble_3_preds = (lgbm_preds + cat_preds + gbr_preds) / 3
```

### Step 4c: Stacking (advanced — optional)
The ablation study shows stacking beats simple averaging:
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

# Get out-of-fold predictions for training data
lgbm_oof = cross_val_predict(lgbm_model, X_train, y_train, cv=5)
cat_oof = cross_val_predict(cat_model, X_train, y_train, cv=5)

# Train meta-learner on OOF predictions
meta_X_train = np.column_stack([lgbm_oof, cat_oof])
meta_X_test = np.column_stack([lgbm_preds, cat_preds])

meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X_train, y_train)
stacked_preds = meta_model.predict(meta_X_test)
```

---

## Phase 5 — Evaluate with the Paper's Metrics

The paper uses four metrics. Here's how to compute all of them:

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate(y_true, y_pred, label="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # PHM08 RUL Score (asymmetric — penalizes late predictions more)
    d = y_pred - y_true
    rul_score = np.sum(np.where(d < 0, np.exp(-d/10), np.exp(d/13)))

    print(f"--- {label} ---")
    print(f"  R²:        {r2:.4f}")
    print(f"  RMSE:      {rmse:.2f}")
    print(f"  MSE:       {mse:.2f}")
    print(f"  MAE:       {mae:.2f}")
    print(f"  RUL Score: {rul_score:.2f}")
    return {'r2': r2, 'rmse': rmse, 'mse': mse, 'mae': mae, 'rul_score': rul_score}

# Run evaluations
evaluate(y_test, lgbm_preds, "LightGBM")
evaluate(y_test, cat_preds, "CatBoost")
evaluate(y_test, ensemble_2_preds, "LGBM + CatBoost")
evaluate(y_test, ensemble_3_preds, "LGBM + Cat + GBR")
```

### Target numbers to hit (from Table 5, FD001):

| Model | R² | RMSE | MSE | RUL Score |
|-------|-----|------|-----|-----------|
| LightGBM alone | 0.9894 | 6.95 | 48.25 | 3,236 |
| LGBM + CatBoost | **0.9904** | **6.62** | **43.79** | **2,951** |
| LGBM + Cat + GBR | 0.9901 | 6.73 | 45.26 | 3,073 |

**Important:** These numbers are on the *full* training data (all cycles, not just last-cycle). The paper trains on every row, not just the last cycle per engine. Make sure your `X_train` and `y_train` use ALL rows.

---

## Phase 6 — SHAP Interpretability Analysis

```python
import shap

# SHAP for LightGBM
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_test)

# Global summary plot
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('shap_global.png', dpi=150, bbox_inches='tight')

# Single prediction explanation (local)
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

**What to look for:** The paper found `s3`, `s8`, `s21`, and `cycle` as the dominant features. Your SHAP plots should show the same pattern.

---

## Phase 7 — Dynamic Maintenance Scheduling

This is the simpler part — it's just a threshold rule:

```python
def maintenance_alert(predicted_rul, tau=15):
    """Algorithm 1 from the paper, simplified."""
    if predicted_rul <= tau:
        return "MAINTENANCE REQUIRED (HIGH)"
    else:
        return "No maintenance needed"

# Apply to test predictions
for i, pred in enumerate(ensemble_2_preds[:10]):
    print(f"Engine {i+1}: RUL={pred:.1f} → {maintenance_alert(pred)}")
```

---

## Phase 8 — Run All Four Datasets

Repeat Phases 2-6 for FD002, FD003, and FD004. Key differences:

| Dataset | Engines (Train/Test) | Operating Conditions | Fault Modes | Difficulty |
|---------|---------------------|---------------------|-------------|-----------|
| FD001 | 100/100 | 1 | 1 | Easiest |
| FD002 | 260/259 | 6 | 1 | Harder |
| FD003 | 100/100 | 1 | 2 | Medium |
| FD004 | 248/249 | 6 | 2 | Hardest |

**Tip:** Wrap everything in functions so you can call `run_experiment("FD001")` through `run_experiment("FD004")`.

---

## Phase 9 — Statistical Validation (Optional but Impressive)

The paper includes rigorous stats. If you want to match that:

1. **5-fold cross-validation** — Get R² scores across folds
2. **Shapiro-Wilk test** — Confirm R² scores are normally distributed (`scipy.stats.shapiro`)
3. **Levene's test** — Confirm equal variances (`scipy.stats.levene`)
4. **One-way ANOVA** — Test if models differ significantly (`scipy.stats.f_oneway`)
5. **Tukey HSD** — Find which pairs differ (`statsmodels.stats.multicomp.pairwise_tukeyhsd`)
6. **Bootstrap MSE** — 10,000 resamples for confidence intervals

---

## Phase 10 — Compare Against Rudder

This is where it gets interesting for your work. The paper achieves RMSE ≈ 6.62 using raw sensor features. Your Rudder framework uses geometry-based features (effective dimension, coherence, etc.) and achieved 11.03 RMSE while being **domain-agnostic**.

**Key comparison points:**
- Their approach is domain-specific (tuned hyperparameters, sensor selection)
- Your `effective_dim` alone explains 63% of importance — they need 24 features
- Their SHAP shows sensor-specific knowledge; your geometry features are universal
- They beat you on raw accuracy; you beat them on generalizability

**To make a fair comparison:**
```python
# Their way: raw sensors → ensemble
# Your way: raw sensors → PRISM geometry → single model
# Hybrid: raw sensors → PRISM geometry → ensemble (best of both?)
```

---

## Checklist

- [ ] Environment set up, libraries installed
- [ ] C-MAPSS data downloaded and loaded
- [ ] Preprocessing: RUL labels created, features normalized
- [ ] LightGBM trained and evaluated
- [ ] CatBoost trained and evaluated
- [ ] Gradient Boosting trained and evaluated
- [ ] XGBoost trained and evaluated
- [ ] 2-model ensemble (LGBM + Cat) evaluated
- [ ] 3-model ensemble (LGBM + Cat + GBR) evaluated
- [ ] SHAP analysis completed
- [ ] Maintenance scheduling threshold implemented
- [ ] Repeated for FD002, FD003, FD004
- [ ] Results compared to paper's Table 5-8
- [ ] Results compared to Rudder's geometry approach

---

## Quick Reference: Paper's Best Results

| Dataset | Best Model | RMSE | R² |
|---------|-----------|------|-----|
| FD001 | LGBM + Cat | 6.62 | 0.9904 |
| FD002 | LGBM + Cat + GBR | 10.15 | 0.9780 |
| FD003 | LGBM + Cat + GBR | 9.71 | 0.9904 |
| FD004 | LightGBM alone | 11.70 | 0.9830 |

**Code repo:** https://github.com/hkmtcn/interpretable-rul-maintenance
