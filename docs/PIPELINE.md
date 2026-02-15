# Framework Observation Processing Pipeline

> **"Garbage in, REJECTED"** — not "garbage in, garbage out"

## Overview

The Framework pipeline validates and analyzes observations BEFORE they reach Manifold.

```
observations.parquet
        │
        ▼
┌───────────────────────────────────────┐
│  STAGE 1: VALIDATION                  │
│  ├── Remove constants (std = 0)       │
│  ├── Remove duplicates (ρ > 0.999)    │
│  ├── Flag orphans (max ρ < 0.1)       │
│  └── Fail if < 2 signals remain       │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  STAGE 2: COHORT DISCOVERY            │
│  ├── Detect cross-unit coupling       │
│  ├── Classify: constant/system/component │
│  ├── Identify coupled vs decoupled units │
│  └── Generate ML recommendations      │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  OUTPUT                               │
│  ├── observations_validated.parquet   │
│  ├── ml_signals.txt                   │
│  ├── exclude_signals.txt              │
│  └── pipeline_report.yaml             │
└───────────────────────────────────────┘
        │
        ▼
    Manifold (clean data only)
```

## Quick Start

```python
from framework.pipeline import process_observations

# Process and save
result = process_observations(
    observations_path='data/observations.parquet',
    output_dir='data/processed/',
)

# Get ML-ready signals
ml_signals = result.ml_signals        # Use these
exclude = result.exclude_signals      # Ignore these

# Access validated data
df = result.validated_df
```

## Command Line

```bash
# Full pipeline
python -m framework.pipeline observations.parquet --output ./processed/

# Permissive mode (warn but don't exclude)
python -m framework.pipeline observations.parquet --permissive

# Skip cohort discovery (validation only)
python -m framework.pipeline observations.parquet --no-cohort-discovery
```

## Validation Rules

| Check | Threshold | Default Action | Rationale |
|-------|-----------|----------------|-----------|
| **Constant** | std < 1e-10 | EXCLUDE | Zero information |
| **Near-constant** | std < 1e-6 | EXCLUDE | Negligible information |
| **Duplicate** | ρ > 0.999 | EXCLUDE (keep first) | Redundant |
| **Orphan** | max ρ < 0.1 | WARN | May be noise or unique signal |
| **Min signals** | < 2 | FAIL | Can't compute geometry |
| **Min rows** | < 10 | FAIL | Insufficient data |

## Cohort Classification

| Class | Cross-Unit ρ | Meaning | ML Action |
|-------|-------------|---------|-----------|
| **CONSTANT** | ≥ 0.99 | Operational setting | **EXCLUDE** |
| **SYSTEM** | ≥ 0.70 | Fleet-coupled | Normalize |
| **COMPONENT** | < 0.50 | Unit-specific | **USE** |
| **ORPHAN** | ≈ 0 | Uncorrelated | Investigate |

## Output Files

```
processed/
├── observations_validated.parquet  # Clean data for Manifold
├── pipeline_report.yaml            # Complete structured report
├── pipeline_report.txt             # Human-readable summary
├── ml_signals.txt                  # Signals to use (one per line)
├── exclude_signals.txt             # Signals to exclude
└── manifest_update.yaml            # Merge into existing manifest
```

## Example Report

```
======================================================================
Framework OBSERVATION PROCESSING PIPELINE
======================================================================

INPUT:  21 signals, 433,251 rows
OUTPUT: 18 signals (validated)

----------------------------------------------------------------------
STAGE 1: VALIDATION
----------------------------------------------------------------------
Excluded 3 signals:
  ✗ sensor_5: constant
  ✗ sensor_10: constant
  ✗ sensor_16: constant

----------------------------------------------------------------------
STAGE 2: COHORT DISCOVERY
----------------------------------------------------------------------
Constants (3): ['sensor_5', 'sensor_10', 'sensor_16']
Component signals (18): ['sensor_1', 'sensor_2', 'sensor_3'...]

----------------------------------------------------------------------
ML RECOMMENDATIONS
----------------------------------------------------------------------
USE (18 signals): ['sensor_1', 'sensor_2', 'sensor_3'...]
EXCLUDE (3 signals): ['sensor_5', 'sensor_10', 'sensor_16']

======================================================================
```

## Integration with Manifold

```python
from framework.pipeline import process_observations
from manifold import run_manifold

# Step 1: Process observations
result = process_observations('data/observations.parquet', 'data/processed/')

# Step 2: Run Manifold on validated data only
manifold_result = run_manifold(
    observations_path='data/processed/observations_validated.parquet',
    output_dir='data/manifold_output/',
)
```

## Configuration

### Validation Config

```python
from framework.validation import ValidationConfig, ValidationAction

# Strict mode (default)
config = ValidationConfig.strict_mode()

# Permissive mode (warn only)
config = ValidationConfig.permissive()

# Custom
config = ValidationConfig(
    constant_std=1e-8,           # Stricter constant detection
    duplicate_corr=0.995,        # Stricter duplicate detection
    orphan_max_corr=0.05,        # Stricter orphan detection
    on_constant=ValidationAction.EXCLUDE,
    on_orphan=ValidationAction.WARN,
)
```

### Cohort Discovery Thresholds

```python
from framework.pipeline import ObservationPipeline

pipeline = ObservationPipeline(
    cohort_thresholds={
        'constant': 0.95,    # Lower threshold for constants
        'system': 0.60,      # Lower threshold for system signals
        'component': 0.40,   # Lower threshold for component signals
    }
)
```

## Why This Matters

### The Problem

```
RAW DATA:
├── 21 signals
├── 3 are constants (zero variance)
├── ML model wastes capacity learning to ignore them
├── Normalization fails (std=0 → divide by zero)
├── Spurious correlations with target
```

### The Solution

```
AFTER PIPELINE:
├── 18 signals (constants removed)
├── 100% of model capacity on real signals
├── Clean normalization
├── No spurious correlations
├── Expected: 1-5% accuracy improvement
```

### The Principle

> **Telling ML what to IGNORE is as important as telling it what to use.**
> 
> Subtraction > Addition

## Validated Findings

### C-MAPSS FD001

```
INPUT:  21 signals
OUTPUT: 18 signals

EXCLUDED:
├── sensor_5: constant (ρ=1.0 across 100 engines)
├── sensor_10: constant
├── sensor_16: constant

These are operational settings, not sensors.
```

### FEMTO Bearings

```
DISCOVERED:
├── acc_y: SYSTEM (ρ=0.99 across 17 bearings)
│   └── Vibration through test rig, not bearing-specific
├── acc_x: COMPONENT (ρ<0.5 across bearings)
│   └── Bearing-specific fault signal

UNIT CLASSIFICATION:
├── Coupled (9): isotropic degradation
├── Decoupled (8): localized fault
```

## API Reference

### `process_observations()`

```python
def process_observations(
    observations_path: str,
    output_dir: Optional[str] = None,
    strict: bool = True,
    run_cohort_discovery: bool = True,
) -> PipelineResult:
    """
    Main entry point for observation processing.
    
    Args:
        observations_path: Path to observations.parquet
        output_dir: Optional output directory
        strict: If True, exclude bad signals. If False, only warn.
        run_cohort_discovery: Whether to run cohort discovery
        
    Returns:
        PipelineResult with:
            - validated_df: Clean DataFrame
            - validation_report: Validation details
            - cohort_result: Cohort discovery details
            - ml_signals: List of signals to use
            - exclude_signals: List of signals to exclude
    """
```

### `PipelineResult`

```python
@dataclass
class PipelineResult:
    validated_df: pl.DataFrame        # Clean data
    validation_report: ValidationReport
    cohort_result: CohortResult
    ml_signals: List[str]             # USE THESE
    exclude_signals: List[str]        # IGNORE THESE
    
    def summary(self) -> str: ...
    def to_dict(self) -> Dict: ...
```

---

## ML Feature Matrix (Post-Manifold)

After Manifold (engines) produces its output parquets, the ML feature builder joins them into a single ML-ready feature matrix with RUL targets.

```
Manifold output/
├── cohort_vector.parquet          ─┐
├── geometry_dynamics.parquet       │
├── state_vector.parquet            ├──► build_ml_features.py ──► machine_learning.parquet
├── topology.parquet                │                           + ml_column_manifest.parquet
├── cohort_velocity_field.parquet  ─┘
│
observations.parquet ──────────────────► (RUL / lifecycle targets)
```

### Usage

```bash
# From prime repo root
./venv/bin/python build_ml_features.py \
    --data ~/data/FD001/output \
    --obs ~/data/FD001/observations.parquet

# Or as module
./venv/bin/python -m prime.ml.entry_points.features \
    --data ~/data/FD001/output \
    --obs ~/data/FD001/observations.parquet
```

```python
# Programmatic
from prime.ml.entry_points.features import run
out_path = run(data="~/data/FD001/output", obs="~/data/FD001/observations.parquet")
```

### Join Chain

Each input parquet is loaded, prefixed, and left-joined onto the base:

| Step | Source | Prefix | Method | Features (FD001) |
|------|--------|--------|--------|-----------------|
| 1 | `cohort_vector.parquet` | `cv_` | Rename columns | 37 |
| 2 | `geometry_dynamics.parquet` | `gd_` | Pivot by engine, then join | 33 |
| 3 | `state_vector.parquet` | `sv_` | Prefix + clean (drop `n_signals`) | 11 |
| 4 | `topology.parquet` | `tp_` | Prefix + clean (drop `topology_computed`) | 6 |
| 5 | `cohort_velocity_field.parquet` | `vf_` | Select numeric only, prefix | 6 |
| 6 | RUL computation | — | From observations lifecycle | 3 (RUL, lifecycle, lifecycle_pct) |

### Target Computation

RUL (Remaining Useful Life) is computed per (cohort, I) window:

- `lifecycle` = max I per cohort + 1 (total window count)
- `RUL` = max_I - I (windows remaining)
- `lifecycle_pct` = I / max_I (normalized position 0 to 1)

### Cleanup

1. **Constant column pruning** — drops features with std < 1e-10
2. **String column removal** — drops non-numeric columns (except `cohort`)
3. **Column manifest** — writes sidecar with dtype, null_pct, mean, std per feature

### Output

| File | Shape (FD001) | Description |
|------|---------------|-------------|
| `machine_learning.parquet` | 860 x 98 | One row per (cohort, I) window. 93 features + 5 metadata/target columns |
| `ml_column_manifest.parquet` | 93 x 5 | Feature metadata: column, dtype, null_pct, mean, std |

### Feature Groups

| Prefix | Source | Count (FD001) | Description |
|--------|--------|---------------|-------------|
| `cv_` | cohort_vector | 37 | Geometry per engine group |
| `gd_` | geometry_dynamics | 33 | Velocity, acceleration, jerk, curvature |
| `sv_` | state_vector | 11 | Centroid distances |
| `tp_` | topology | 6 | Network density, degree |
| `vf_` | cohort_velocity_field | 6 | Speed, acceleration in state space |

### Notes

- `geometry_dynamics` has an `engine` column — the pivot creates per-engine columns (e.g., `gd_hurst_velocity`, `gd_spectral_curvature`)
- Velocity field features have ~20% structural nulls from first-window derivative computation
- Only `cohort_vector.parquet` is required; all other inputs are optional

---

## ML Training (Post-Feature Matrix)

Trains and evaluates ML models on `machine_learning.parquet` for RUL prediction.

```
machine_learning.parquet ──► run_ml.py ──► ml_results/
                                           ├── predictions.parquet
                                           ├── feature_importance.parquet
                                           ├── cv_results.parquet
                                           ├── residuals.parquet
                                           └── model_summary.json
```

### Usage

```bash
# From prime repo root
./venv/bin/python run_ml.py --data ~/data/FD001/output

# Or as module
./venv/bin/python -m prime.ml.entry_points.train --data ~/data/FD001/output
```

```python
# Programmatic
from prime.ml.entry_points.train import run
out_dir = run(data="~/data/FD001/output")
```

### Validation Strategy

- **GroupKFold** (5-fold) on `cohort` — no data leakage between engines
- Each fold: train on ~80 cohorts, test on ~20
- RUL capped at 125 (standard C-MAPSS practice)
- Features with >30% null dropped; remaining nulls median-imputed
- StandardScaler fit per fold (fit on train, transform on test)

### Models

| Model | Type | Purpose |
|-------|------|---------|
| Ridge | Linear (L2 regularization) | Baseline |
| Random Forest | Ensemble (bagging) | Non-linear baseline |
| Gradient Boosting | Ensemble (boosting) | Best expected performance |
| Top-5 Linear | Linear (OLS) | Interpretable — uses only top 5 GB features |

### FD001 Train Results (Cross-Validation Only)

**NOTE: These results are on the FD001 training set only (100 engines, 860 windows, internal GroupKFold CV). No held-out test set has been evaluated yet.**

#### Current — with gaussian features (185 features)

| Model | RMSE | +/- std | MAE | R² | Features |
|-------|------|---------|-----|-----|----------|
| gradient_boosting | **9.78** | 1.03 | 6.90 | 0.944 | 183 |
| random_forest | 10.75 | 1.24 | 7.77 | 0.932 | 183 |
| ridge | 14.14 | 2.98 | 10.04 | 0.878 | 183 |
| top5_linear | 17.23 | 0.65 | 11.94 | 0.827 | 5 |

#### Previous — without gaussian features (93 features)

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

### Top 5 Features (by Gradient Boosting Importance)

| Feature | Importance | Source |
|---------|-----------|--------|
| `gd_shape_effective_dim_velocity` | 0.343 | geometry_dynamics |
| `gd_shape_effective_dim_jerk` | 0.203 | geometry_dynamics |
| `gd_shape_variance_velocity` | 0.094 | geometry_dynamics |
| `gd_complexity_variance_velocity` | 0.084 | geometry_dynamics |
| `gd_spectral_variance_velocity` | 0.073 | geometry_dynamics |

Top 5 remain geometry_dynamics — the differential geometry of the manifold (how shape changes over time) is the strongest RUL signal. Gaussian similarity features (`gsim_std_similarity`, `gsim_mean_similarity`) appear in the top 15, providing cross-signal coupling information that complements the temporal dynamics.

### Error by Lifecycle Phase (Gradient Boosting)

| Phase | RMSE | MAE | n | Notes |
|-------|------|-----|---|-------|
| Early (0-33%) | 7.06 | 3.85 | 181 | Well captured |
| Mid (33-66%) | 13.34 | 10.60 | 331 | Hardest — RUL cap transition zone |
| Late (66-100%) | 6.62 | 4.97 | 348 | Well captured |

Mid-lifecycle is hardest because the RUL cap (125) creates a flat ceiling in early life, and the transition from capped to uncapped RUL introduces prediction difficulty.

### Published Benchmarks (FD001)

| Method | RMSE | PHM08 Score |
|--------|------|-------------|
| LightGBM+CatBoost ensemble (2025) | 6.62 | 2,951 |
| Transformer SOTA (2024) | 11.28 | — |
| Attention DCNN (2021) | 11.81 | 223 |
| CAELSTM (2025) | 14.44 | — |
| **Prime ML (train CV, current)** | **9.78** | **TBD** |

### Output Files

| File | Description |
|------|-------------|
| `predictions.parquet` | Per-row RUL predictions vs actual for all 4 models |
| `feature_importance.parquet` | Ranked feature importance from GB and best model |
| `cv_results.parquet` | Per-fold metrics (RMSE, MAE, R², n_train, n_test, test_cohorts) |
| `residuals.parquet` | Per-row errors for best model (cohort, I, actual, predicted, error) |
| `model_summary.json` | Full summary: all model metrics, top features, error by phase |

---

## Official Test Evaluation

Trains on ALL training data, predicts the last window per test engine, scores against `RUL_FD001.txt` ground truth.

```bash
# From prime repo root
./venv/bin/python evaluate_test.py --train ~/data/FD001/train/output \
                                   --test  ~/data/FD001/test/output \
                                   --rul   ~/data/FD001/RUL_FD001.txt
```

### Metrics

- **RMSE** — standard root mean squared error
- **MAE** — mean absolute error
- **R²** — coefficient of determination
- **PHM08 Score** — asymmetric penalty (late predictions penalized exponentially more than early)

### PHM08 Scoring Function

```
d = predicted - actual
score = sum(
    exp(-d/13) - 1   if d < 0  (early prediction — smaller penalty)
    exp( d/10) - 1   if d >= 0 (late prediction — larger penalty)
)
```

Lower is better. Late predictions are ~3x more costly than early ones. Target: < 500.

### Directory Structure

```
~/data/FD001/
├── train/
│   ├── observations.parquet     ← 100 engines, run to failure
│   └── output/
│       └── machine_learning.parquet
├── test/
│   ├── observations.parquet     ← 100 engines, CUT at arbitrary point
│   └── output/
│       └── machine_learning.parquet
└── RUL_FD001.txt                ← ground truth (100 integers, one per engine)
```

---

## See Also

- [Validation Module](./validation.md) - Signal validation details
- [Cohort Discovery](./cohort_discovery.md) - Cohort classification details
- [Manifold Integration](./manifold_integration.md) - Running Manifold on validated data
