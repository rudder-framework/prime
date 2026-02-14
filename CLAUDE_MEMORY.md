# Claude AI Memory - PRISM/Rudder Framework Architecture

**Last Updated:** 2026-02-13
**Session Summary:** Implemented ML feature matrix + ML training pipeline, replaced legacy features.py and train.py

---

## Architecture Principle

```
PRISM = Muscle (pure computation, no decisions, no classification)
Rudder Framework = Brain (orchestration, typology, classification, interpretation)

PRISM computes numbers. Rudder Framework classifies.
```

---

## What We Just Did

### 1. Refactored Entry Points (Completed)

Moved stage runners to `entry_points/` with ordered naming. Entry points are **pure orchestration** - they call engines for computation, no embedded calculations.

### 2. PRISM Entry Points Created

Location: `/Users/jasonrudder/prism/prism/entry_points/`

| Stage | File | Output | Calls Engine |
|-------|------|--------|--------------|
| 01 | `stage_01_signal_vector.py` | `signal_vector.parquet` | `engines/signal/*` via registry |
| 02 | `stage_02_state_vector.py` | `state_vector.parquet` | `engines/state/centroid.py` |
| 03 | `stage_03_state_geometry.py` | `state_geometry.parquet` | `engines/state/eigendecomp.py` |
| 04 | `stage_04_cohorts.py` | `cohorts.parquet` | Pure aggregation |

**Old stages deprecated:** `prism/pipeline/stages/` marked with DEPRECATED.md

### 3. Rudder Framework Entry Points Created

Location: `/Users/jasonrudder/framework/framework/entry_points/`

| Stage | File | Output | Calls Module |
|-------|------|--------|--------------|
| 01 | `stage_01_validate.py` | `observations_validated.parquet` | `core/validation.py` |
| 02 | `stage_02_typology.py` | `typology_raw.parquet` | `ingest/typology_raw.py` |
| 03 | `stage_03_classify.py` | `typology.parquet` | `typology/discrete_sparse.py`, `typology/level2_corrections.py` |
| 04 | `stage_04_manifest.py` | `manifest.yaml` | `manifest/generator.py` |
| 05 | `stage_05_diagnostic.py` | `diagnostic_report.txt` | `engines/*` |

### 4. Rudder Framework Engines Implemented (Previous Session)

Location: `/Users/jasonrudder/framework/framework/engines/`

| Engine | File | Purpose |
|--------|------|---------|
| Level 0 | `typology_engine.py` | System classification (ACCUMULATION/DEGRADATION/CONSERVATION) |
| Level 1 | `stationarity_engine.py` | KPSS/ADF stationarity tests |
| Level 2 | `classification_engine.py` | Signal behavior classification |
| Geometry | `signal_geometry.py` | Eigenstructure (eff_dim, alignment) |
| Mass | `mass_engine.py` | Total variance/energy |
| Structure | `structure_engine.py` | Structure = Geometry × Mass |
| Stability | `stability_engine.py` | Lyapunov, CSD detection |
| Tipping | `tipping_engine.py` | B-tipping vs R-tipping (Granger causality) |
| Spin Glass | `spin_glass.py` | Parisi framework phases |
| Report | `diagnostic_report.py` | Unified diagnostic combining all engines |

---

## WHERE WE LEFT OFF

### Missing PRISM Runners (Need to Create)

One runner per parquet file. These are missing:

| Stage | Output File | Engine Source |
|-------|-------------|---------------|
| 05 | `signal_geometry.parquet` | `engines/signal_geometry.py` |
| 06 | `signal_pairwise.parquet` | `engines/signal_pairwise.py` |
| 07 | `geometry_dynamics.parquet` | `engines/geometry_dynamics.py` |
| 08 | `lyapunov.parquet` | `engines/parallel/dynamics_runner.py` |
| 09 | `dynamics.parquet` | `engines/parallel/dynamics_runner.py` |
| 10 | `information_flow.parquet` | `engines/parallel/information_flow_runner.py` |
| 11 | `topology.parquet` | `engines/parallel/topology_runner.py` |
| 12 | `zscore.parquet` | `engines/sql/` |
| 13 | `statistics.parquet` | `engines/sql/` |
| 14 | `correlation.parquet` | `engines/sql/` |

### Task: Create these runners following the pattern:

```python
"""
Stage XX: <Name> Entry Point
============================

Pure orchestration - calls <engine> for computation.
Stages: <input> → <output.parquet>
"""

# Import engine
from prism.engines.<path> import compute

def run(...):
    # 1. Load input
    # 2. Call engine
    # 3. Write output
    pass
```

---

## Key Files Reference

### PRISM
- Entry points: `/Users/jasonrudder/prism/prism/entry_points/`
- Engines: `/Users/jasonrudder/prism/prism/engines/`
- Signal engines: `/Users/jasonrudder/prism/prism/engines/signal/`
- State engines: `/Users/jasonrudder/prism/prism/engines/state/`
- Parallel runners: `/Users/jasonrudder/prism/prism/engines/parallel/`
- Deprecated stages: `/Users/jasonrudder/prism/prism/pipeline/stages/` (DO NOT USE)

### Rudder Framework
- Entry points: `/Users/jasonrudder/framework/framework/entry_points/`
- Engines: `/Users/jasonrudder/framework/framework/engines/`
- Manifest generator: `/Users/jasonrudder/framework/framework/manifest/generator.py`
- Typology: `/Users/jasonrudder/framework/framework/typology/`
- Core validation: `/Users/jasonrudder/framework/framework/core/validation.py`

---

## Pipeline Flow

```
Rudder Framework Pipeline:
observations.parquet
    → stage_01_validate → observations_validated.parquet
    → stage_02_typology → typology_raw.parquet
    → stage_03_classify → typology.parquet
    → stage_04_manifest → manifest.yaml

PRISM Pipeline:
observations.parquet + typology.parquet + manifest.yaml
    → stage_01_signal_vector → signal_vector.parquet
    → stage_02_state_vector → state_vector.parquet
    → stage_03_state_geometry → state_geometry.parquet
    → stage_04_cohorts → cohorts.parquet
    → [MISSING: stages 05-14 for remaining parquet files]

Back to Rudder Framework:
PRISM outputs → stage_05_diagnostic → diagnostic_report.txt

ML Feature Matrix (post-Manifold):
Manifold outputs + observations.parquet
    → prime.ml.entry_points.features → machine_learning.parquet (860 x 98)
                                     → ml_column_manifest.parquet (93 x 5)

ML Training (on feature matrix):
machine_learning.parquet
    → prime.ml.entry_points.train → ml_results/predictions.parquet
                                  → ml_results/feature_importance.parquet
                                  → ml_results/cv_results.parquet
                                  → ml_results/residuals.parquet
                                  → ml_results/model_summary.json
```

---

## ML Feature Matrix (build_ml_features)

Replaces the legacy `prime/ml/entry_points/features.py` (which imported from dead `prism.db.parquet_store` and read old PRISM output names). Now reads current Manifold output format.

### Entry Point

- **Module:** `prime.ml.entry_points.features` — exposes `run()` and `main()`
- **Wrapper:** `prime/build_ml_features.py` — repo-root convenience script

### Input

Reads from Manifold output directory:

| File | Tier | Prefix | Pivot? |
|------|------|--------|--------|
| `cohort_vector.parquet` | Core (required) | `cv_` | No |
| `geometry_dynamics.parquet` | Core | `gd_` | Yes — by engine |
| `state_vector.parquet` | Core | `sv_` | No |
| `topology.parquet` | Supplemental | `tp_` | No |
| `cohort_velocity_field.parquet` | Supplemental | `vf_` | No |
| `gaussian_fingerprint.parquet` | Supplemental | `gfp_` | No — aggregated per-cohort (mean/std/min/max) |
| `gaussian_similarity.parquet` | Supplemental | `gsim_` | No — aggregated per-cohort (mean/std/min/max) |

Plus `observations.parquet` for RUL/lifecycle computation.

### Output

| File | Shape (FD001) | Description |
|------|---------------|-------------|
| `machine_learning.parquet` | 860 x 190 | One row per (cohort, I) window |
| `ml_column_manifest.parquet` | 185 x 5 | Column metadata (dtype, null_pct, mean, std) |

### Columns

- `cohort`, `I` — identity
- `RUL`, `lifecycle`, `lifecycle_pct` — targets
- `cv_*` (37) — cohort vector features
- `gd_*` (33) — geometry dynamics (velocity, acceleration, jerk, curvature)
- `sv_*` (11) — state vector (centroid distances)
- `tp_*` (6) — topology (network density, degree)
- `vf_*` (6) — cohort velocity field (speed, acceleration)
- `gfp_*` (56) — gaussian fingerprint (static per-cohort signal characterization)
- `gsim_*` (36) — gaussian similarity (static per-cohort pairwise coupling)

### Key Functions

- `load_if_exists()` — safe parquet loader (returns None if missing/empty)
- `pivot_by_engine()` — pivots (cohort, I, engine) → (cohort, I) with engine-prefixed columns
- `prefix_and_clean()` — adds prefix to feature columns, drops metadata columns
- `build_rul()` — extracts max I per cohort from observations for lifecycle/RUL computation

### Usage

```bash
# From repo root
./venv/bin/python build_ml_features.py --data ~/data/FD001/output --obs ~/data/FD001/observations.parquet

# Or as module
./venv/bin/python -m prime.ml.entry_points.features --data ~/data/FD001/output --obs ~/data/FD001/observations.parquet
```

```python
# Programmatic
from prime.ml.entry_points.features import run
out_path = run(data="~/data/FD001/output", obs="~/data/FD001/observations.parquet")
```

### Cleanup Steps

1. Drop constant columns (std < 1e-10)
2. Drop string columns (except cohort)
3. Report high-null features (>20% null from derivative computation / velocity field)

---

## ML Training (run_ml)

Replaces the legacy `prime/ml/entry_points/train.py` (which imported from dead `prism.db.parquet_store`, used simple train/test split, read old `ml_features.parquet`). Now reads `machine_learning.parquet` with GroupKFold cross-validation on cohort.

### Entry Point

- **Module:** `prime.ml.entry_points.train` — exposes `run()` and `main()`
- **Wrapper:** `prime/run_ml.py` — repo-root convenience script

### Validation Strategy

- **GroupKFold** (5-fold) on `cohort` — no data leakage between engines
- Each fold: train on ~80 cohorts, test on ~20
- RUL capped at 125 (standard C-MAPSS practice)
- Features with >30% null dropped, remaining nulls median-imputed
- StandardScaler applied per fold (fit on train, transform test)

### Models

| Model | Type | Purpose |
|-------|------|---------|
| Ridge | Linear (L2) | Baseline |
| Random Forest | Ensemble | Non-linear |
| Gradient Boosting | Ensemble | Best expected |
| Top-5 Linear | Linear (OLS) | Interpretable — uses only top 5 GB features |

### FD001 Train Results (Cross-Validation)

**Data: FD001 training set only (100 engines, 860 windows). No held-out test set evaluated yet.**

#### Current (with gaussian features, 185 features)

| Model | RMSE | +/- std | MAE | R² | Features |
|-------|------|---------|-----|-----|----------|
| gradient_boosting | **9.78** | 1.03 | 6.90 | 0.944 | 183 |
| random_forest | 10.75 | 1.24 | 7.77 | 0.932 | 183 |
| ridge | 14.14 | 2.98 | 10.04 | 0.878 | 183 |
| top5_linear | 17.23 | 0.65 | 11.94 | 0.827 | 5 |

#### Previous (without gaussian features, 93 features)

| Model | RMSE | +/- std | MAE | R² | Features |
|-------|------|---------|-----|-----|----------|
| gradient_boosting | 11.18 | 0.93 | 8.04 | 0.927 | 93 |
| random_forest | 12.46 | 0.91 | 9.07 | 0.909 | 93 |
| ridge | 11.44 | 0.74 | 8.80 | 0.923 | 93 |

#### Improvement from Gaussian Features

| Model | Before | After | Delta |
|-------|--------|-------|-------|
| gradient_boosting | 11.18 | **9.78** | **-1.40** |
| random_forest | 12.46 | 10.75 | **-1.71** |

### Top 5 Features (by GB importance)

| Feature | Importance | Source |
|---------|-----------|--------|
| `gd_shape_effective_dim_velocity` | 0.343 | geometry_dynamics |
| `gd_shape_effective_dim_jerk` | 0.203 | geometry_dynamics |
| `gd_shape_variance_velocity` | 0.094 | geometry_dynamics |
| `gd_complexity_variance_velocity` | 0.084 | geometry_dynamics |
| `gd_spectral_variance_velocity` | 0.073 | geometry_dynamics |

Top 5 remain geometry_dynamics. Gaussian features (`gsim_std_similarity`, `gsim_mean_similarity`) appear in top 15, providing ensemble diversity.

### Error by Lifecycle Phase (gradient_boosting)

| Phase | RMSE | MAE | n |
|-------|------|-----|---|
| Early (0-33%) | 7.06 | 3.85 | 181 |
| Mid (33-66%) | 13.34 | 10.60 | 331 |
| Late (66-100%) | 6.62 | 4.97 | 348 |

Mid-lifecycle is hardest to predict (RUL cap transition zone). Early and late phases are well captured.

### Output Files

| File | Description |
|------|-------------|
| `predictions.parquet` | Per-row predictions vs actual RUL for all models |
| `feature_importance.parquet` | Ranked feature importance (GB + best model) |
| `cv_results.parquet` | Per-fold cross-validation metrics |
| `residuals.parquet` | Prediction errors for best model |
| `model_summary.json` | Full summary with metrics, top features, phase errors |

### Usage

```bash
# From repo root
./venv/bin/python run_ml.py --data ~/data/FD001/output

# Or as module
./venv/bin/python -m prime.ml.entry_points.train --data ~/data/FD001/output
```

```python
# Programmatic
from prime.ml.entry_points.train import run
out_dir = run(data="~/data/FD001/output")
```

### Test Evaluation Entry Point

- **Module:** `prime.ml.entry_points.evaluate_test` — exposes `run()` and `main()`
- **Wrapper:** `prime/evaluate_test.py` — repo-root convenience script
- Trains on ALL training data (no CV), predicts last window per test engine
- Scores against `RUL_FD001.txt` ground truth: RMSE, MAE, R², PHM08 Score

```bash
./venv/bin/python evaluate_test.py --train ~/data/FD001/train/output \
                                   --test  ~/data/FD001/test/output \
                                   --rul   ~/data/FD001/RUL_FD001.txt
```

### Directory Structure

```
~/data/FD001/
├── train/
│   ├── observations.parquet
│   └── output/                ← Manifold output + machine_learning.parquet
├── test/                      ← needs Manifold run + build_ml_features
│   ├── observations.parquet   ← (needs ingest from test_FD001.txt)
│   └── output/
├── RUL_FD001.txt              ← ground truth (100 integers)
└── phm08/                     ← PHM08 challenge set (if available)
```

### Next Step

1. Ingest test_FD001.txt → test/observations.parquet
2. Run Manifold on test observations → test/output/
3. Build ML features → test/output/machine_learning.parquet
4. Run evaluate_test.py for official held-out RMSE + PHM08 Score

---

## Key Concepts

- **Structure = Geometry × Mass** - Both can fail independently
- **B-tipping** (geometry→mass): CSD provides early warning
- **R-tipping** (mass→geometry): NO early warning
- **Spin Glass phases**: Paramagnetic (healthy), Ferromagnetic (trending), Spin Glass (fragile), Mixed (critical)
- **effective_dim**: Participation ratio - 63% importance in RUL prediction

---

## Commands

```bash
# Rudder Framework pipeline
python -m framework.entry_points.stage_01_validate observations.parquet -o validated.parquet
python -m framework.entry_points.stage_02_typology observations.parquet -o typology_raw.parquet
python -m framework.entry_points.stage_03_classify typology_raw.parquet -o typology.parquet
python -m framework.entry_points.stage_04_manifest typology.parquet -o manifest.yaml

# PRISM pipeline
python -m prism.entry_points.signal_vector manifest.yaml
python -m prism.entry_points.stage_02_state_vector signal_vector.parquet typology.parquet
python -m prism.entry_points.stage_03_state_geometry signal_vector.parquet state_vector.parquet
python -m prism.entry_points.stage_04_cohorts state_vector.parquet state_geometry.parquet cohorts.parquet

# ML Feature Matrix (post-Manifold)
./venv/bin/python build_ml_features.py --data ~/data/FD001/output --obs ~/data/FD001/observations.parquet
```
