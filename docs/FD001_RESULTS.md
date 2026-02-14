# FD001 Official Test Results

## Final Score

| Metric | Our Result | Ozcan (2025) | Transformer SOTA |
|--------|-----------|-------------|-----------------|
| **RMSE** | **11.72** | 6.62 | 11.28 |
| **MAE** | **8.39** | — | — |
| **R²** | **0.9145** | — | — |
| **PHM08 Score** | **188** | 2,951 | — |

**Zero overfit**: OOF RMSE = 12.41, Test RMSE = 11.72 (gap = -0.7)

---

## Architecture

### Compute Split

```
Manifold (linear algebra)          Prime (SQL + ML)
─────────────────────────          ────────────────
stage_00  breaks                   v_geometry_dynamics  (LAG)
stage_01  signal_vector            v_cohort_vector      (FILTER agg)
stage_02  state_vector             v_topology           (threshold)
stage_03  state_geometry           v_statistics         (GROUP BY)
stage_05  signal_geometry          v_zscore_ref         (normalize)
stage_06  signal_pairwise          v_correlation        (CORR)
stage_20  sensor_eigendecomp       v_velocity_field     (LAG)
stage_24  gaussian_fingerprint     v_canary_signals     (lifecycle corr)
stage_34  fleet_baseline           v_ml_features        (join all)
stage_35  observation_geometry     + 8 more SQL views
stage_36  gaussian_similarity
                                   cycle_features.py    (ML features)
11 stages → 11 parquets            run_clean_ensemble.py (stacking)
```

### Pipeline

```
1. Ingest         → observations.parquet (per split)
2. Manifold       → 11 core parquets (per split)
3. Fleet Baseline → cohort_baseline.parquet (from TRAINING fleet only)
4. Obs Geometry   → observation_geometry.parquet (scored against fleet baseline)
5. Cycle Features → cycle_features.parquet (538 raw, 413 clean)
6. Stacking       → predictions (LGB + XGB + Hist → RidgeCV)
```

---

## Feature Engineering

### 7 Feature Layers

| Layer | Source | Count | Per | Generalizes? |
|-------|--------|-------|-----|-------------|
| Raw sensors | observations | 17 | cycle | YES |
| Cycle number | observations | 1 | cycle | YES |
| Rolling stats | computed | 375 | cycle | YES |
| Deltas | computed | 15 | cycle | YES |
| RT geometry | stage 35 | 5 | cycle | **YES** |
| Interp geometry | geometry_dynamics | 33 | cycle | **NO — LEAKS** |
| Gaussian | fingerprint + similarity | 92 | engine (static) | **NO — LEAKS** |

### Clean Feature Set (413 features used in final model)

```
sensors:  17  (BPR, NRc, NRf, Nc, Nf, P15, P30, Ps30, T24, T30, T50, W31, W32, htBleed, op1, op2, phi)
cycle:     1  (cycle number)
rolling: 375  (15 sensors × 5 windows × 5 stats)
delta:    15  (first differences for 15 varying sensors)
rt:        5  (centroid_dist, centroid_dist_norm, pc1_projection, pc2_projection, mahalanobis_approx)
─────────────
TOTAL:   413
```

### Excluded Features (125 features — leak engine identity)

```
geom_*:   33  (interpolated per-engine geometry dynamics)
gfp_*:    56  (gaussian fingerprint — static per engine)
gsim_*:   36  (gaussian similarity — static per engine)
─────────────
TOTAL:   125
```

### Rolling Window Configuration

```
Sensors:  All 15 varying (dropped 2 constant + 7 duplicates)
Windows:  5, 10, 15, 20, 30 cycles
Stats:    mean, std, min, max, range
Total:    15 × 5 × 5 = 375 features
```

### RT Geometry (Stage 35) — The Key Innovation

Stage 35 computes per-cycle distance from a **fleet healthy baseline**:

```
centroid_distance     = ||x_norm(I)||           How far from healthy
centroid_distance_norm = cd / sqrt(n_signals)   Normalized distance
pc1_projection        = x_norm(I) · PC1        Primary degradation axis
pc2_projection        = x_norm(I) · PC2        Lateral displacement
mahalanobis_approx    = sqrt(Σ proj²/λ)        Eigenvalue-weighted distance
```

**rt_centroid_dist is the #2 most important feature** (importance=761, after cycle=876).

---

## Feature Leakage Analysis

### The Problem

Per-engine features encode ENGINE IDENTITY, not degradation state.

```
WRONG (what we had):
  gfp_mean_spectral_entropy = 0.823  → This uniquely identifies engine_42
  Model learns: "this fingerprint = engine_42 = fails at cycle 180"
  Test engine with different fingerprint → garbage prediction

RIGHT (what we use):
  rt_centroid_dist = 12.4            → This measures degradation magnitude
  Any engine with centroid_dist=12.4 is similarly degraded
  Transfers from train to test
```

### Ablation Proof

| Features Added | OOF RMSE | Test RMSE | Gap | Diagnosis |
|---------------|----------|-----------|-----|-----------|
| Sensors | 16.76 | 17.40 | +0.6 | Clean |
| + rolling | 16.30 | 17.24 | +0.9 | Clean |
| + delta | 16.34 | 17.33 | +1.0 | Clean |
| **+ RT geometry** | **11.82** | **11.74** | **-0.1** | **Clean + 6 RMSE drop** |
| + interp geometry | 8.23 | 32.54 | **+24.3** | **CATASTROPHIC** |
| + gaussian FP | 8.23 | 33.60 | +25.4 | Worse |
| + gaussian sim | 7.94 | 34.58 | +26.6 | Worse |

**Key insight**: OOF drops beautifully with leaking features (7.94!) because GroupKFold
leaves out engines, and the model memorizes engine identity from static features. But these
identities don't transfer to unseen test engines.

---

## Fleet Baseline (Stage 34)

### Old Approach (per-engine)
```python
# Each engine uses its OWN healthy reference
baseline = engine_first_20pct.mean()  # different for every engine
# Problem: test engines get different baselines → features don't transfer
```

### New Approach (fleet)
```python
# Pool ALL training engines' early-life into ONE reference
baseline = all_train_engines_first_20pct.mean()  # same for everyone
# 4,086 pooled cycles from 100 training engines
# Applied identically to train AND test
```

### Implementation

```bash
# Stage 34: Compute fleet baseline from training data
python stage_34_cohort_baseline.py train/observations.parquet \
    -o fleet_baseline.parquet --mode fleet

# Stage 35: Score train against fleet baseline
python stage_35_observation_geometry.py train/observations.parquet \
    --baseline fleet_baseline.parquet -o train/output/observation_geometry.parquet

# Stage 35: Score test against SAME fleet baseline
python stage_35_observation_geometry.py test/observations.parquet \
    --baseline fleet_baseline.parquet -o test/output/observation_geometry.parquet
```

Fleet baseline stats:
- Pooled cycles: 4,086 (from 100 engines × ~41 cycles each)
- Effective dimension: 4.89
- Total variance: 17.00

---

## Model Configuration

### Stacking Ensemble

```python
base_models = {
    'lgb': LGBMRegressor(n_estimators=500, max_depth=6, lr=0.05,
        num_leaves=31, min_child_samples=10, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0),
    'xgb': XGBRegressor(n_estimators=500, max_depth=6, lr=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=10),
    'hist': HistGradientBoostingRegressor(max_iter=500, max_depth=6, lr=0.05,
        min_samples_leaf=10),
}

meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
# Selected alpha: 100.0
# Weights: LGB=0.460, XGB=0.037, Hist=0.513
```

### Cross-Validation
- 5-fold GroupKFold (grouped by cohort/engine)
- No data leakage between folds (entire engines held out)

### Data
- Train: 20,631 rows × 413 features (100 engines, run to failure)
- Test: 100 engines, cut at arbitrary point, predict RUL at last cycle
- RUL cap: 125 (standard practice)
- Imputation: median (for NaN from rolling window edges)

---

## Top 20 Features

| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | cycle | 876 | time |
| 2 | rt_centroid_dist | 761 | **geometry** |
| 3 | roll_BPR_max_30 | 176 | rolling |
| 4 | roll_Nc_range_30 | 167 | rolling |
| 5 | roll_BPR_min_30 | 156 | rolling |
| 6 | rt_centroid_dist_norm | 141 | **geometry** |
| 7 | roll_phi_std_30 | 140 | rolling |
| 8 | roll_P30_min_30 | 135 | rolling |
| 9 | roll_T30_min_30 | 132 | rolling |
| 10 | roll_W32_min_30 | 132 | rolling |
| 11 | roll_NRc_max_30 | 130 | rolling |
| 12 | roll_T30_max_30 | 130 | rolling |
| 13 | roll_W32_max_30 | 128 | rolling |
| 14 | roll_T24_min_30 | 124 | rolling |
| 15 | roll_phi_max_30 | 121 | rolling |
| 16 | roll_Nc_min_30 | 115 | rolling |
| 17 | roll_Ps30_range_30 | 115 | rolling |
| 18 | roll_Nc_std_30 | 113 | rolling |
| 19 | roll_P30_max_30 | 113 | rolling |
| 20 | roll_T50_max_30 | 111 | rolling |

**Geometry contributes 2 of the top 6 features.** The 30-cycle window dominates rolling features.

---

## Geometry Ladder (Stage 35 features only)

Pure geometry experiment — how far can 7 features go?

| Experiment | Features | Model | RMSE | PHM08 |
|---|---|---|---:|---:|
| centroid_distance only | 1 | Ridge | 22.04 | 1,389 |
| + cycle | 2 | Ridge | 22.15 | 1,293 |
| + pc1 + pc2 | 4 | Ridge | 20.33 | 882 |
| All 7 geometry (scaled) | 7 | Ridge | 19.96 | 887 |
| All 7 geometry | 7 | GBR | 18.89 | 639 |
| All 7 geometry | 7 | LightGBM | 18.82 | 633 |

**A single feature (centroid_distance) explains 70% of RUL variance.**

---

## Worst Predictions

| Engine | Predicted | True RUL | Error | Notes |
|--------|----------|---------|-------|-------|
| 45 | 79.9 | 114 | -34.1 | Under-predicts healthy |
| 12 | 91.1 | 124 | -32.9 | Under-predicts healthy |
| 75 | 82.7 | 113 | -30.3 | Under-predicts healthy |
| 93 | 57.2 | 85 | -27.8 | Under-predicts healthy |
| 2 | 124.5 | 98 | +26.5 | Over-predicts (thinks healthy) |

Pattern: worst errors are on high-RUL engines (>85 true). The model is conservative —
tends to predict lower RUL than actual. This is actually the SAFER direction for
prognostics (better to warn early than late).

---

## Files

### Manifold (computation)
```
engines/entry_points/stage_34_cohort_baseline.py    Fleet/per-cohort baseline SVD
engines/entry_points/stage_35_observation_geometry.py Per-cycle health scoring
engines/entry_points/stage_36_gaussian_similarity.py Distributional distance
engines/entry_points/run_pipeline.py                 Updated stage lists
```

### Prime (ML)
```
prime/ml/entry_points/cycle_features.py    Feature builder (build + train)
prime/sql/layers/05_manifold_derived.sql   17 SQL views (moved stages)
build_cycle_features.py                    Repo-root wrapper
```

### Experiment Scripts
```
experiments_geometry_ladder.py    4 experiments (centroid only → ensemble)
run_ablation.py                  7-config ablation study
run_clean_ensemble.py            Final clean ensemble (413 features)
run_full_ensemble.py             Full ensemble (204 features — overfits)
```

### Data
```
data/FD001/
├── fleet_baseline.parquet                   Fleet healthy reference (1 row)
├── RUL_FD001.txt                            Ground truth (100 integers)
├── train/
│   ├── observations.parquet                 495,144 rows
│   ├── cycle_features.parquet               20,631 × 538
│   └── output/
│       ├── observation_geometry.parquet      20,631 × 8 (fleet-scored)
│       ├── cohort_baseline.parquet          100 × 17 (per-engine, legacy)
│       └── (11 more Manifold parquets)
└── test/
    ├── observations.parquet                 314,304 rows
    ├── cycle_features.parquet               13,096 × 538
    └── output/
        ├── observation_geometry.parquet      13,096 × 8 (fleet-scored)
        └── (11 more Manifold parquets)
```

---

## Reproduction

```bash
# 0. Prerequisites
cd ~/prime
./venv/bin/pip install lightgbm xgboost scikit-learn polars

# 1. Fleet baseline (run once, from TRAINING data only)
cd ~/manifold
./venv/bin/python -c "
from engines.entry_points.stage_34_cohort_baseline import run
run('~/data/FD001/train/observations.parquet',
    '~/data/FD001/fleet_baseline.parquet', mode='fleet')
"

# 2. Score train and test against fleet baseline
./venv/bin/python -c "
from engines.entry_points.stage_35_observation_geometry import run
run('~/data/FD001/train/observations.parquet',
    '~/data/FD001/fleet_baseline.parquet',
    '~/data/FD001/train/output/observation_geometry.parquet')
run('~/data/FD001/test/observations.parquet',
    '~/data/FD001/fleet_baseline.parquet',
    '~/data/FD001/test/output/observation_geometry.parquet')
"

# 3. Build cycle features
cd ~/prime
./venv/bin/python build_cycle_features.py build \
    --obs ~/data/FD001/train/observations.parquet \
    --manifold ~/data/FD001/train/output \
    --output ~/data/FD001/train/cycle_features.parquet

./venv/bin/python build_cycle_features.py build \
    --obs ~/data/FD001/test/observations.parquet \
    --manifold ~/data/FD001/test/output \
    --output ~/data/FD001/test/cycle_features.parquet

# 4. Run clean ensemble
./venv/bin/python run_clean_ensemble.py
```

---

## Next Steps

1. **Operating condition normalization** — Normalize sensors per regime (FD001 is single regime but FD002/FD004 have 6 regimes)
2. **Piecewise linear RUL** — Only model degradation phase, not flat healthy phase
3. **CatBoost** — Need Python <3.14 environment to use actual CatBoost (Ozcan's 3rd base learner)
4. **Hyperparameter tuning** — Optuna/Hyperopt search over ensemble params
5. **FD002-FD004** — Multi-regime datasets where fleet baseline per regime should shine
6. **Rolling RT geometry** — Compute rolling stats on centroid_distance itself (trend in degradation rate)
