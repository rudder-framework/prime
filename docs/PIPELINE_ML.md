# ML Pipeline — Cycle-Level Stacking Ensemble

## Overview

Predicts Remaining Useful Life (RUL) at every cycle using a stacking ensemble
of gradient boosting models. Domain-agnostic — same code for any C-MAPSS dataset.

```
observations.parquet
        │
        ▼
[Stage 34] Fleet Baseline ──→ cohort_baseline.parquet (1 row, from training fleet)
        │
        ▼
[Stage 35] Observation Geometry ──→ observation_geometry.parquet (per-cycle scores)
        │
        ▼
[cycle_features.py] Feature Builder ──→ cycle_features.parquet (538 cols)
        │
        ▼
[run_clean_ensemble.py] Train + Predict ──→ RMSE, PHM08, predictions
```

## Feature Engineering (7 Layers)

### Layer 1: Raw Sensors (17 features)
Pivot observations to wide format. One row per (cohort, cycle).
Drop constant sensors (zero variance). Keep operational settings.

### Layer 2: Cycle Number (1 feature)
Sequential cycle index per engine. **#1 most important feature.**

### Layer 3: Rolling Statistics (375 features)
```
Sensors:  15 varying (drop constants)
Windows:  [5, 10, 15, 20, 30] cycles
Stats:    [mean, std, min, max, range]
Total:    15 × 5 × 5 = 375
```
Computed with `polars.rolling_*(...).over('cohort')`.
30-cycle window features dominate importance.

### Layer 4: Deltas (15 features)
First differences for all varying sensors: `delta_X = X(I) - X(I-1)`.
Captures instantaneous degradation rate.

### Layer 5: Real-Time Geometry (5 features) — from Stage 35
```
rt_centroid_dist      = ||x_norm(I)||           Distance from fleet healthy baseline
rt_centroid_dist_norm = cd / sqrt(n_signals)    Normalized distance
rt_pc1_projection     = x_norm(I) · PC1        Primary degradation axis
rt_pc2_projection     = x_norm(I) · PC2        Lateral displacement
rt_sensor_norm        = ||x(I)||               Total magnitude (not from stage 35)
```
**rt_centroid_dist is #2 most important feature.** Zero overfit gap.

### Layer 6: Interpolated Geometry (33 features) — EXCLUDED
Per-engine geometry dynamics from Manifold windows (velocity, acceleration,
jerk, curvature of effective_dim, eigenvalues, total_variance).
**Causes +24 RMSE overfit gap. DO NOT USE IN ML.**

### Layer 7: Gaussian Features (92 features) — EXCLUDED
Per-engine gaussian fingerprint (56) + similarity (36). Static per engine.
**Encodes engine identity. Causes +26 RMSE overfit gap. DO NOT USE IN ML.**

## Clean Feature Set

413 features used in production:
```
sensors:   17
cycle:      1
rolling:  375
delta:     15
rt:         5
───────────
TOTAL:    413
```

## Model Architecture

### Base Learners

| Model | Library | Key Params |
|-------|---------|------------|
| LightGBM | lightgbm | n_estimators=500, max_depth=6, lr=0.05, num_leaves=31 |
| XGBoost | xgboost | n_estimators=500, max_depth=6, lr=0.05 |
| HistGBR | sklearn | max_iter=500, max_depth=6, lr=0.05 |

### Meta-Learner
```python
RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
```
Selected alpha: 100.0. Weights: LGB=0.460, XGB=0.037, Hist=0.513.

### Cross-Validation
- 5-fold GroupKFold (entire engines held out per fold)
- OOF predictions used for meta-learner training
- Test predictions averaged across folds

### Prediction
```python
# Test: take LAST cycle per engine
test_last = test_df.group_by('cohort').agg([pl.all().sort_by('I').last()])
pred = np.clip(meta.predict(stack_test), 0, 125)
```

## Preprocessing

```python
# 1. Impute NaN (from rolling window edges)
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)   # Fitted on TRAIN only

# 2. Replace Inf
X_train = np.where(np.isinf(X_train), 0, X_train)

# 3. Standardize
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)  # Fitted on TRAIN only
```

## Metrics

```python
# RMSE (primary)
RMSE = sqrt(mean((pred - true)²))

# PHM08 Score (asymmetric — late predictions penalized more)
for d in (pred - true):
    if d < 0: score += exp(-d/13) - 1   # early: gentle
    else:     score += exp(d/10) - 1     # late: harsh

# RUL Cap
y_train = np.minimum(y_train, 125)       # Clip during training
y_true_capped = np.minimum(y_true, 125)  # Clip for evaluation
```

## File Locations

```
prime/
├── prime/ml/entry_points/cycle_features.py  Main feature builder
├── build_cycle_features.py                  Repo-root wrapper
├── run_clean_ensemble.py                    Production ensemble
├── run_ablation.py                          Feature ablation study
├── run_full_ensemble.py                     Full (leaking) ensemble
├── experiments_geometry_ladder.py            Geometry-only experiments
└── docs/
    ├── FD001_RESULTS.md                     Full results
    ├── STAGES_34_35_36.md                   New Manifold stages
    ├── MANIFOLD_DERIVED_SQL.md              SQL views
    └── PIPELINE_ML.md                       This file

manifold/engines/entry_points/
├── stage_34_cohort_baseline.py              Fleet baseline SVD
├── stage_35_observation_geometry.py          Per-cycle health scoring
└── stage_36_gaussian_similarity.py          Distributional distance
```

## Quick Start

```bash
# Prerequisites
cd ~/prime && ./venv/bin/pip install lightgbm xgboost scikit-learn polars

# 1. Fleet baseline (from training data)
cd ~/manifold
./venv/bin/python -c "
from engines.entry_points.stage_34_cohort_baseline import run
run('~/data/FD001/train/observations.parquet',
    '~/data/FD001/fleet_baseline.parquet', mode='fleet')
"

# 2. Score against fleet baseline
./venv/bin/python -c "
from engines.entry_points.stage_35_observation_geometry import run
run('~/data/FD001/train/observations.parquet', '~/data/FD001/fleet_baseline.parquet',
    '~/data/FD001/train/output/observation_geometry.parquet')
run('~/data/FD001/test/observations.parquet', '~/data/FD001/fleet_baseline.parquet',
    '~/data/FD001/test/output/observation_geometry.parquet')
"

# 3. Build features
cd ~/prime
./venv/bin/python build_cycle_features.py build \
    --obs ~/data/FD001/train/observations.parquet \
    --manifold ~/data/FD001/train/output \
    --output ~/data/FD001/train/cycle_features.parquet

./venv/bin/python build_cycle_features.py build \
    --obs ~/data/FD001/test/observations.parquet \
    --manifold ~/data/FD001/test/output \
    --output ~/data/FD001/test/cycle_features.parquet

# 4. Train + evaluate
./venv/bin/python run_clean_ensemble.py
```
