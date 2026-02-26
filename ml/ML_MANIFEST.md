# ML MANIFEST — FD001 Canonical Reference

**Version:** 1.1
**Date:** 2026-02-25
**Purpose:** Single source of truth for all FD001 ML work. Every experiment references this manifest.

> Note: "ORTHON" in this document refers to the `ml/` feature export layer from the Manifold computation pipeline (legacy name, retained for ML feature naming consistency).

---

## 1. Historical Results

| Run | Date | RMSE | NASA | Gap | Best Model | Features | Key Innovation |
|-----|------|------|------|-----|------------|----------|----------------|
| **FD001 Official** | early 2026 | **11.72** | **188** | -0.7 (negative = clean) | Stacking (LGB+XGB+Hist->Ridge) | 413 | Fleet baseline RT geometry |
| **Config 4 (z-score ORTHON)** | 2026-02-25 | **14.29** | 382 | +1.24 | Stacking (LGB+XGB+Hist->Ridge) | 604 | Fleet z-score normalization: (x - fleet_mean) / fleet_std |
| Config 4 (delta ORTHON) | 2026-02-25 | 16.60 | 440 | +2.90 | Stacking | 604 | Per-cohort delta: (x - own_first) / fleet_std |
| Config 4 (raw ORTHON) | 2026-02-25 | 19.27 | — | +11.26 | Stacking | 604 | Unnormalized ORTHON (identity leak) — INVALID |
| CSV Standalone | 2026-02-25 | 12.16 | 224 | ~1.05 | XGBoost | ~1,044 | Rolling stats on raw sensors |
| ORTHON+CSV v3 | 2026-02-25 | 15.03 | 374 | 0.92 | Lasso | 38 | ml/ directory + raw sensors |
| ORTHON Alone v2 | 2026-02-25 | 16.31 | 382 | 1.07 | Lasso | 14 | Causal ml/ features only |
| ORTHON v1 (leaked) | 2026-02-25 | 51.46 | 21,666 | 5.88 | Lasso | 39 | Forward-looking derivatives — INVALID |

**The 11.72 is the benchmark. Everything must beat it or explain why.**

---

## 2. What the 11.72 Run Used

### 2.1 Feature Layers (413 total)

| Layer | Count | Source | Per | Importance | Transfers? |
|-------|-------|--------|-----|------------|------------|
| Raw sensors | 17 | observations.parquet | cycle | Medium | YES |
| Cycle number | 1 | computed | cycle | **#1 (876)** | YES |
| Rolling stats | 375 | computed (15 sensors x 5 windows x 5 stats) | cycle | Medium-High | YES |
| Delta (first diff) | 15 | computed (sensor[t] - sensor[t-1]) | cycle | Low-Medium | YES |
| RT geometry | 5 | stage_35 observation_geometry | cycle | **#2 (761)** | YES |
| ~~Interp geometry~~ | ~~33~~ | ~~geometry_dynamics~~ | ~~cycle~~ | ~~High (leak)~~ | **NO — LEAKS** |
| ~~Gaussian FP~~ | ~~56~~ | ~~gaussian_fingerprint~~ | ~~engine~~ | ~~High (leak)~~ | **NO — LEAKS** |
| ~~Gaussian sim~~ | ~~36~~ | ~~gaussian_similarity~~ | ~~engine~~ | ~~High (leak)~~ | **NO — LEAKS** |

### 2.2 RT Geometry — The Key Feature

Stage 35 computes per-cycle distance from a fleet healthy baseline:

```
Fleet baseline = pooled early-life data from ALL training engines (4,086 cycles)
                 One SVD, one centroid, one set of principal components

Per-cycle features:
  centroid_distance      = ||x_norm(I)||            How far from healthy center
  centroid_distance_norm = cd / sqrt(n_signals)     Normalized distance
  pc1_projection         = x_norm(I) . PC1          Primary degradation axis
  pc2_projection         = x_norm(I) . PC2          Lateral displacement
  mahalanobis_approx     = sqrt(sum proj^2/lambda)  Eigenvalue-weighted distance
```

**Why it works:** Every cycle of every engine is scored against the SAME fleet reference. Train and test use identical baseline. No engine identity leaks. The feature measures "how far is this engine from healthy right now" — which is literally what RUL predicts.

**The 5-point RMSE drop:** Adding RT geometry to rolling stats dropped RMSE from 16.34 -> 11.74. Nothing else in the ablation came close.

### 2.3 Rolling Window Configuration

```
Sensors:  15 varying (dropped constant sensors 5, 10, 16 + 4 duplicates)
Windows:  5, 10, 15, 20, 30 cycles
Stats:    mean, std, min, max, range
Total:    15 x 5 x 5 = 375 features
```

### 2.4 Model Architecture

```python
base_models = {
    'lgb': LGBMRegressor(n_estimators=500, max_depth=6, lr=0.05),
    'xgb': XGBRegressor(n_estimators=500, max_depth=6, lr=0.05),
    'hist': HistGradientBoostingRegressor(max_iter=500, max_depth=6, lr=0.05),
}
meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
# Selected alpha: 100.0
# Weights: LGB=0.460, XGB=0.037, Hist=0.513
```

5-fold GroupKFold. RUL cap at 125.

### 2.5 What Was Excluded and Why

| Feature Set | OOF RMSE | Test RMSE | Gap | Verdict |
|-------------|----------|-----------|-----|---------|
| Sensors only | 16.76 | 17.40 | +0.6 | Clean baseline |
| + rolling | 16.30 | 17.24 | +0.9 | Clean |
| + delta | 16.34 | 17.33 | +1.0 | Clean |
| **+ RT geometry** | **11.82** | **11.74** | **-0.1** | **CLEAN — the breakthrough** |
| + interp geometry | 8.23 | 32.54 | +24.3 | **CATASTROPHIC LEAK** |
| + gaussian FP | 8.23 | 33.60 | +25.4 | **LEAK — engine identity** |
| + gaussian sim | 7.94 | 34.58 | +26.6 | **LEAK — engine identity** |

**Lesson:** Per-engine static features (fingerprint, similarity) encode engine identity, not degradation. The model memorizes which engine it is instead of learning degradation patterns.

---

## 3. What the Current v3 Run Is Missing

### 3.1 Resolution Gap

| Run | Rows | Grain | Features see |
|-----|------|-------|-------------|
| 11.72 run | 20,631 | per-cycle | "sensor 7 at cycle 148 is X" |
| ORTHON v3 | ~800 | per-window | "sensor 7 averaged over cycles 130-160 is Y" |

The window-level ORTHON features are too coarse. An engine deteriorating at cycle 145 shows it in per-cycle data immediately. In window-level data, it's averaged with 29 other cycles.

### 3.2 Missing Fleet Baseline

The 11.72's RT geometry scores every cycle against a fleet healthy baseline. The current ORTHON pipeline computes eigendecomp and centroids per-engine-per-window. These are self-referential (each engine measured against itself) rather than fleet-referential (each engine measured against fleet healthy).

### 3.3 Missing Cycle Number

The #1 feature in the 11.72 run. Simple, powerful, no leakage.

### 3.4 Missing Delta Features

First differences of raw sensors — 15 features. Backward-looking by definition. Not computed in v3.

### 3.5 Missing Stacking

v3 used Ridge/Lasso/XGBoost (ML_PROCESS standard). The 11.72 used LGB+XGB+Hist stacking with RidgeCV meta-learner. Different architecture, better result.

---

## 4. The Path Forward: Configuration 4

### 4.1 Architecture

Combine the 11.72's proven feature architecture with ORTHON's ml/ features.

```
FEATURE MATRIX (per-cycle grain, ~20,000 rows x ~500 features)

Layer 1: Cycle number (1 feature)
  +-- cycle

Layer 2: Raw sensors (17 features)
  +-- sensor values at each cycle

Layer 3: Rolling stats (375 features)
  +-- 15 sensors x 5 windows x 5 stats

Layer 4: Delta features (15 features)
  +-- sensor[t] - sensor[t-1] per varying sensor

Layer 5: RT geometry (5 features) <-- THE KEY
  +-- centroid_dist, centroid_dist_norm, pc1_projection, pc2_projection, mahalanobis_approx
  +-- Fleet baseline: pooled training early-life SVD
  +-- Computed at EVERY CYCLE, not per window

Layer 6: ORTHON features (broadcast from ml/ to per-cycle) <-- NEW
  +-- ml_eigendecomp: eigenvalue_0, effective_dim, condition_number per window -> broadcast
  +-- ml_eigendecomp_derivatives: D1/D2 of eigendecomp metrics -> broadcast
  +-- ml_centroid: centroid position per window -> broadcast
  +-- ml_centroid_derivatives: D1/D2 of centroid drift -> broadcast
  +-- ml_coupling: coupling metrics per window -> broadcast
  +-- ml_typology: signal classifications per window -> broadcast
  +-- ml_signal_primitives: Hurst, entropy per window -> broadcast
  +-- ml_information_flow: transfer entropy per window -> broadcast
  +-- ml_persistent_homology: topological features per window -> broadcast
  +-- ml_trajectory_match: template match scores per window -> broadcast
```

### 4.2 Join Strategy

Per-cycle features (Layers 1-5) have one row per engine per cycle.
ORTHON features (Layer 6) have one row per engine per window.

Join via asof merge: for each cycle, take the most recent window that ends at or before that cycle.

```python
# Per-cycle DataFrame: (engine, cycle, sensor_1, ..., rt_centroid_dist, ...)
# ORTHON DataFrame: (engine, window, window_start_cycle, window_end_cycle, eigen_0, ...)

# asof join: each cycle gets the ORTHON features from its enclosing window
combined = cycle_df.join_asof(
    orthon_df,
    left_on="cycle",
    right_on="window_end_cycle",
    by="engine",
    strategy="backward"  # take most recent window <= current cycle
)
```

This gives per-cycle resolution (20,631 rows) with ORTHON features as slow-varying context. The per-cycle features carry fine-grained signal. ORTHON features carry structural context. Lasso decides which matter.

### 4.3 Fleet Baseline

Two options:

**Option A: Use existing stage_34/stage_35 code.** The fleet baseline pipeline already works. It produced the 11.72. Run it on the current observations, get RT geometry features.

**Option B: Compute fleet baseline from ml/ data.** Use ml_eigendecomp from the first N windows of all training engines as the "healthy" reference. Score each window against this reference. This is ORTHON-native but produces window-level features, not per-cycle.

**Recommendation: Option A.** It's proven. It works at per-cycle resolution. It produced the best result. Use it.

### 4.4 Model Architecture

Two runs:

**Run A: ML_PROCESS standard** (4 models, no tuning) — for honest comparison with v1/v2/v3.

**Run B: Stacking ensemble** (LGB+XGB+Hist->RidgeCV) — for best achievable result, matching the 11.72's architecture.

Both runs logged separately with the same feature matrix.

---

## 5. Feature Leakage Reference

### 5.1 CONFIRMED LEAKY — Never Use

| Feature | Why it leaks | Discovery |
|---------|-------------|-----------|
| Interpolated geometry dynamics | Forward-looking central differences | v1 experiment (51.46 RMSE) |
| Gaussian fingerprint (per-engine) | Encodes engine identity | 11.72 ablation (OOF 8.23, Test 33.60) |
| Gaussian similarity (per-engine) | Encodes engine identity | 11.72 ablation (OOF 7.94, Test 34.58) |
| lifecycle_fraction | Encodes total runtime | Multiple experiments |
| n_cohorts / fleet size | Constant that differs train/test | v1 shift audit |
| dominant_frequency | Shifted 8 sigma train->test | v2 shift audit |
| geometry_dynamics.parquet (analytical) | Central differences peek forward | Causality audit |
| velocity_field.parquet (analytical) | Symmetric smoothing | Causality audit |

### 5.2 CONFIRMED SAFE — Always Use

| Feature | Why it's safe | Evidence |
|---------|-------------|----------|
| RT centroid_distance (fleet baseline) | Same reference for train/test, per-cycle | 11.72: #2 feature, gap -0.1 |
| Cycle number | Observable at prediction time | 11.72: #1 feature |
| Rolling stats (backward window) | pandas rolling is backward by default | 12.16 CSV standalone |
| Delta features (first diff) | x[t]-x[t-1] is strictly causal | 11.72 ablation |
| ml/ derivative features (_d1, _d2) | Backward finite differences | Causality audit, NaN verification |
| ml/ pass-through features | Per-window, no cross-window dependency | ml_export design |

---

## 6. Top 20 Features (from 11.72 run)

| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | cycle | 876 | time |
| 2 | rt_centroid_dist | 761 | **RT geometry** |
| 3 | roll_BPR_max_30 | 176 | rolling |
| 4 | roll_Nc_range_30 | 167 | rolling |
| 5 | roll_BPR_min_30 | 156 | rolling |
| 6 | rt_centroid_dist_norm | 141 | **RT geometry** |
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

**Geometry is 2 of top 6. The 30-cycle window dominates rolling features.**

---

## 7. Geometry Ladder (RT features alone)

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

## 8. Cross-Domain Validation

The framework has been validated across multiple domains:

| Domain | Signals | Finding | ORTHON Contribution |
|--------|---------|---------|-------------------|
| C-MAPSS Turbofan | 24 sensors, 100 engines | R-tipping (no CSD), eff_dim collapse correlates with lifetime (r=0.62) | Canary sequences, eigenvalue dynamics, tipping classification |
| Global Markets | 30 indices/rates | B+R tipping, CSD detected in 43% of signals, 2026 stress mirrors 2008 | Dimensional collapse identifies fragility |
| Building Vibration | 4 sensors | Spin glass phase, coupling expected for structural dynamics | Phase classification, shock response prediction |
| Bearing Vibration | 6 signals (3 bearings x 2 axes) | Periodic (corrected from CHAOTIC), harmonic series detected | Typology classification, spectral analysis |
| CNC Milling | 6 sensors, 167 runs | AE_spindle/AE_table are canary signals, smcDC most chaotic | Canary sequence, coupling evolution |

**The analytical pipeline tells the story. ML validates it's predictive.**

---

## 9. Execution Plan for Configuration 4

### Step 1: Reproduce the 11.72 baseline
- Run stage_34 (fleet baseline from training data)
- Run stage_35 (per-cycle observation geometry for train AND test)
- Build cycle_features with: sensors + cycle + rolling + delta + RT geometry
- Run stacking ensemble
- **Verify: RMSE ~ 11.72, NASA ~ 188**
- If this doesn't reproduce, STOP and debug before adding ORTHON

### Step 2: Add ORTHON features via asof join
- Read ml/ directory from Prime output
- asof join ORTHON features to per-cycle DataFrame
- Build combined feature matrix (~500 features)
- Run ML_PROCESS standard (4 models)
- Run stacking ensemble
- **Compare: Does ORTHON+everything beat 11.72?**

### Step 3: Ablation
- Run with RT geometry removed -> quantifies RT contribution with ORTHON present
- Run with ORTHON removed -> should reproduce 11.72 (sanity check)
- Run with rolling stats removed -> quantifies raw signal contribution
- Feature importance by layer -> which ORTHON features make the cut?

### Step 4: Report
All three configurations + the combined config 4:

| Config | RMSE | NASA | Gap | Features |
|--------|------|------|-----|----------|
| CSV Standalone | 12.16 | 224 | ~1.05 | ~1,044 |
| ORTHON Alone | 16.31 | 382 | 1.07 | 14 |
| ORTHON + CSV (v3, window-level) | 15.03 | 374 | 0.92 | 38 |
| RT Geometry Baseline (11.72) | 11.72 | 188 | -0.7 | 413 |
| **RT + ORTHON (config 4)** | **?** | **?** | **?** | **~500** |

---

## 10. Files Reference

### Prime Output (Analytical)
```
output_time/
  signal/, cohort/, system/     <-- Analytical parquets (ML does not read)
  geometry_dynamics.parquet     <-- Forward-looking (ML does not read)
  sql/                          <-- 27 reports
  ml/                           <-- ML reads ONLY this for ORTHON features
    ml_manifest.yaml
    13 ml_*.parquet files
```

### Fleet Baseline (RT Geometry)
```
data/FD001/
  fleet_baseline.parquet        <-- One SVD from pooled training early-life
  train/output/
    observation_geometry.parquet  <-- Per-cycle RT features (20,631 rows)
  test/output/
    observation_geometry.parquet  <-- Per-cycle RT features (13,096 rows)
```

### ML Repo
```
ml/
  experiments/
    baselines/csv_standalone.py
    configs/
    results/
  src/ml/
    features/
      loader.py             <-- ML_PATH_MAP + ml/ detection
      builder.py            <-- Feature extraction routing
      causal_features.py    <-- Causal derivative extractors
    models/
    validation/
  ML_PROCESS.md             <-- v2.1
  ML_MANIFEST.md            <-- THIS FILE
  tests/
```

---

## 11. Configuration 4 — Results and Lessons (2026-02-25)

### 11.1 Settings

| Parameter | Value |
|-----------|-------|
| Notebook | `notebooks/fd001_rul_config4.ipynb` |
| Date | 2026-02-25 |
| Feature count | 604 (467 after NaN/constant drop) |
| Fleet baseline | PCA(10), first 30 cycles of all 100 training engines |
| ORTHON normalization | Per-cohort delta from own first window, divided by fleet_std |
| Broadcast features | prim_ and traj_ columns dropped |
| Model | Stacking: LGB+XGB+Hist → RidgeCV(alpha=100) |
| CV | 5-fold GroupKFold (by engine) |
| RUL cap | 125 cycles |

### 11.2 Results

| Run | OOF RMSE | Test RMSE | Gap | PHM08 |
|-----|----------|-----------|-----|-------|
| Run A (Ridge) | 15.08 | 17.12 | +2.04 | 432 |
| Run B LGB | — | 16.29 | — | 411 |
| Run B XGB | — | 17.02 | — | 460 |
| Run B Hist | — | 17.13 | — | 472 |
| **Run B Stacking** | **13.70** | **16.60** | **+2.90** | **440** |
| vs 11.72 benchmark | — | +4.88 | — | — |

**Beat benchmark: NO (missed by 4.88 RMSE)**

### 11.3 Ablation

| Configuration | OOF RMSE | Test RMSE | Gap |
|---------------|----------|-----------|-----|
| cycle + sensors | 16.73 | 17.45 | +0.72 |
| + rolling (CSV baseline) | 15.30 | 16.36 | +1.07 |
| + RT geometry | 15.42 | 15.15 | -0.27 |
| + ORTHON only (no RT), normalized | 13.84 | 17.36 | +3.52 |
| Config 4: + RT + ORTHON (normalized) | 13.80 | 16.48 | +2.68 |

**Key finding:** ORTHON features improve OOF significantly (13.70 vs 15.42 with RT only) but don't transfer as cleanly to test. RT geometry still shows gap -0.27 (test better than OOF).

### 11.4 Top Features (Layer Importance)

| Layer | Total Importance | Share |
|-------|-----------------|-------|
| rolling | 14,710 | 66.7% |
| ORTHON (normalized) | 5,511 | 25.0% |
| RT geometry | 399 | 1.8% |
| delta | 359 | 1.6% |
| cycle | 819 | 3.7% |
| sensors | 252 | 1.1% |

Top individual ORTHON features: `ed_condition_number_d1` (#2 overall), `ed_ratio_2_1_d2`, `ed_ratio_2_1_d1`, `ed_eigenvalue_entropy_normalized_d2`, `ed_condition_number`.

### 11.0 Z-Score Run (2026-02-25) — supersedes delta run

**Settings change:** `normalize_orthon_zscore()` replaces `normalize_orthon_per_cohort()`.
- Before: `(value - own_first_window) / fleet_std`
- After: `(value - fleet_mean) / fleet_std` where fleet_mean from first windows of all training engines

| Run | OOF RMSE | Test RMSE | Gap | PHM08 |
|-----|----------|-----------|-----|-------|
| Run A (Ridge) | 14.22 | 14.57 | +0.35 | 296 |
| Run B LGB | — | 13.89 | — | 377 |
| Run B XGB | — | 13.94 | — | 346 |
| Run B Hist | — | 14.55 | — | 383 |
| **Run B Stacking** | **13.05** | **14.29** | **+1.24** | **382** |
| vs 11.72 benchmark | — | +2.57 | — | — |

**Ablation (LGB-300 each):**

| Configuration | OOF | Test | Gap |
|---|---|---|---|
| + rolling (CSV baseline) | 15.30 | 16.36 | +1.07 |
| + RT geometry | 15.42 | 15.15 | -0.27 |
| + ORTHON only (no RT), z-scored | 13.33 | 14.59 | +1.26 |
| Config 4: RT + ORTHON, z-scored | 13.25 | 14.00 | +0.76 |

**Note:** LGB alone (13.89) beats the stack (14.29) on test — meta-learner over-weights Hist (0.668) which hurts generalization. The ablation LGB-300 (14.00) also beats the final stack.

**Normalization progression:**

| Normalization | Test RMSE | Gap | Δ vs prior |
|---|---|---|---|
| None (identity leak) | 19.27 | +11.26 | — |
| Per-cohort delta | 16.60 | +2.90 | -2.67 |
| **Fleet z-score** | **14.29** | **+1.24** | **-2.31** |

**Key insight:** Z-score is strictly better than per-cohort delta because it also captures where an engine sits in the fleet distribution at window 1. Per-cohort delta zeroes out all engines at window 1 regardless of initial health state; z-score preserves that information.

---

### 11.5 Changes Made vs Previous Run (raw ORTHON, gap +11.26)

1. **Per-cohort delta normalization**: `delta = (value - own_first_window) / fleet_std`
   - Converts absolute ORTHON values to "drift from own healthy state, scaled by fleet variance"
   - Gap dropped from +11.26 → +2.90 (8.36 improvement)
   - Test RMSE improved from 19.27 → 16.60

2. **Dropped broadcast features**: `prim_*` and `traj_*` columns removed
   - Static per-engine fingerprints with no time dimension
   - Cannot be normalized meaningfully (no "first window" to delta from)

### 11.6 Lessons Learned

1. **ORTHON features encode engine identity without normalization.**
   - Each engine has a unique eigendecomposition fingerprint at absolute scale.
   - Model memorizes training engine fingerprints (OOF 8.01) but fails on test engines (19.27).
   - This is the same pattern as Gaussian fingerprint (+25 RMSE gap in the 11.72 ablation).

2. **Per-cohort delta normalization is the correct fix for fleet features.**
   - Principle: `delta = (current - own_baseline) / fleet_scale`
   - Removes inter-engine offset, preserves intra-engine drift dynamics.
   - Same principle as RT geometry (score against fleet, not against self).
   - Gap dropped from +11.26 to +2.90. Normalization is the right answer.

3. **ORTHON features still underperform RT geometry on test generalization.**
   - RT geometry: gap -0.27 (test beats OOF). Perfectly transferable.
   - Normalized ORTHON: gap +3.52. Some identity residual remains.
   - Root cause: eigendecomp structure (effective_dim, condition_number) still has engine-level variation even after delta normalization. First-window baseline captures starting state but not systematic engine-to-engine structural differences.

4. **Rolling stats carry 67% of total importance.**
   - The 30-cycle window consistently dominates.
   - Rolling is the workhorse; everything else is marginal improvement.

5. **Next to try:**
   - Stronger z-score normalization: use fleet mean AND fleet std (not just delta/std) → standardized score
   - Feature selection: top-K ORTHON features by importance (condition_number, ratio, entropy)
   - Separate ORTHON scale: train a sub-model on ORTHON features alone, stack prediction as meta-feature

---

## 12. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-25 | Created ml_export layer in Prime | Separate causal (ML-safe) from analytical (bidirectional) derivatives |
| 2026-02-25 | Wide format for ml_ parquets | ML-ready with zero transformation |
| 2026-02-25 | Backward finite differences only | Forward-looking derivatives caused 5.88x gap ratio |
| 2026-02-25 | Three-configuration comparison mandatory | CSV vs ORTHON vs ORTHON+CSV — the comparison is the contribution |
| 2026-02-25 | Fleet baseline RT geometry is required | Single largest RMSE improvement in any ablation (5+ points) |
| 2026-02-25 | Per-cycle resolution for prediction | Window-level features lose 3+ RMSE points vs per-cycle |
| 2026-02-25 | Polyform Strict license | Protects implementation while allowing publication |
| 2026-02-25 | ORTHON features are additive, not replacement | Must combine with raw signal features, not replace them |
| 2026-02-25 | ORTHON features require per-cohort delta normalization | Absolute ORTHON values encode engine identity (gap +11.26). Delta from first window / fleet_std reduces gap to +2.90. Same principle as RT geometry. |
| 2026-02-25 | Fleet z-score is better than per-cohort delta for ORTHON | Z-score (x - fleet_mean) / fleet_std preserves initial health state info; delta zeros all engines at window 1. Gap +2.90 → +1.24. |
| 2026-02-25 | Broadcast prim_/traj_ features dropped from Config 4 | Static per-engine fingerprints (Hurst, entropy, trajectory match) have no time dimension, cannot be delta-normalized, encode identity. |
