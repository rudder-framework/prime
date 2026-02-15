# Cohort Discovery v2

> "Run geometry FIRST - it reveals physics you didn't know."

## What's New in V2

1. **Constant Detection**: Identifies operational settings (ρ > 0.99 across units)
2. **ML-Ready Output**: `get_ml_signals()` and `get_exclude_list()` for preprocessing
3. **Four-Class Classification**: constant → system → component → orphan
4. **Unit Coupling Analysis**: Detects isotropic vs anisotropic fault patterns

## Signal Classification

| Class | Cross-Unit ρ | Meaning | ML Action |
|-------|-------------|---------|-----------|
| **CONSTANT** | ≥ 0.99 | Operational setting, not a sensor | **EXCLUDE** |
| **SYSTEM** | ≥ 0.70 | Coupled across units (fleet baseline) | Normalize or use as reference |
| **COMPONENT** | < 0.50 | Unit-specific (fault signal) | **USE FOR RUL** |
| **ORPHAN** | ≈ 0 | Uncorrelated with everything | Investigate |

## Validated Findings

### C-MAPSS FD001 (Turbofan Engines)

```
CONSTANTS (3): sensor_5, sensor_10, sensor_16
  → Operational settings, NOT sensors
  → ρ = 1.000 across all 100 engines
  → EXCLUDE from ML

COMPONENT SIGNALS (18): sensor_1, sensor_2, ... sensor_21 (minus constants)
  → ρ < 0.25 across engines
  → Each engine's degradation is independent
  → USE FOR RUL PREDICTION
```

**Impact**: Removing 3 constant signals expected to improve ML accuracy by 1-5%.

### FEMTO Bearings (PRONOSTIA)

```
SYSTEM SIGNALS: All acc_y
  → ρ > 0.99 across 17 bearings
  → Vibration through test rig structure
  → Use as fleet baseline

COMPONENT SIGNALS: All acc_x  
  → ρ < 0.50 across bearings
  → Bearing-specific fault signal
  → USE FOR RUL

COUPLED UNITS (9): Bearing1_1, 1_2, 1_3, 2_2, 2_3, 2_5, 2_6, 2_7, 3_2
  → Within-bearing acc_x ↔ acc_y correlation > 0.8
  → Isotropic degradation

DECOUPLED UNITS (8): Bearing1_4, 1_5, 1_6, 1_7, 2_1, 2_4, 3_1, 3_3
  → Within-bearing correlation low/undefined
  → Anisotropic/localized fault
```

**Impact**: Discovered hidden mechanical coupling. Two distinct fault populations.

## Usage

### Framework Integration

```python
from framework.cohort_discovery import process_observations

# Process and save results
result = process_observations(
    observations_path='data/observations.parquet',
    manifest_path='data/manifest.yaml',
    output_dir='data/cohort_analysis/'
)

# Get ML-ready signal list
ml_signals = result.get_ml_signals()
exclude_list = result.get_exclude_list()

# Use in preprocessing
df = df.select([c for c in df.columns if c in ml_signals])
```

### Command Line

```bash
# Basic discovery
python -m framework.cohort_discovery observations.parquet

# With output
python -m framework.cohort_discovery observations.parquet \
    --manifest manifest.yaml \
    --output ./cohort_results/

# Custom thresholds
python -m framework.cohort_discovery observations.parquet \
    --constant-threshold 0.95 \
    --system-threshold 0.60 \
    --output ./results/
```

### Output Files

```
cohort_results/
├── cohort_discovery.yaml      # Full structured results
├── cohort_discovery_summary.txt  # Human-readable summary
├── ml_signals.txt             # Signals to use (one per line)
└── exclude_signals.txt        # Signals to exclude (one per line)
```

## Output Structure

```yaml
# cohort_discovery.yaml
constants:
  - sensor_5
  - sensor_10
  - sensor_16

system_signals: []

component_signals:
  - sensor_1
  - sensor_2
  # ... 18 total

orphan_signals: []

coupled_units: []
decoupled_units: []

ml_signals:  # Convenience: component + system
  - sensor_1
  - sensor_2
  # ...

exclude_list:  # Convenience: constants + orphans
  - sensor_5
  - sensor_10
  - sensor_16

signal_details:
  sensor_5:
    classification: constant
    cross_unit_correlation: 1.0
    notes: "ρ=1.000 ≥ 0.99 (operational setting)"
  sensor_1:
    classification: component
    cross_unit_correlation: 0.0
    notes: "ρ=0.000 < 0.50 (unit-specific)"
```

## ML Pipeline Integration

```python
import polars as pl
from framework.cohort_discovery import process_observations

# Step 1: Discover cohorts
result = process_observations('observations.parquet')

# Step 2: Load and filter data
df = pl.read_parquet('observations.parquet')

# Get only ML-usable signals
ml_signals = result.get_ml_signals()
meta_cols = ['timestamp', 'unit_id', 'cycle', 'RUL']  # Keep these
keep_cols = meta_cols + [s for s in ml_signals if s in df.columns]

df_ml = df.select(keep_cols)

# Step 3: Train model
# Your ML code here, now with clean input features
```

## When to Run

| Condition | Run Cohort Discovery? |
|-----------|----------------------|
| > 20 signals | ✅ Yes |
| > 1M rows | ✅ Yes |
| > 5 units | ✅ Yes |
| Unknown data source | ✅ Yes |
| < 10 signals, < 100K rows | ❌ Skip (overhead not worth it) |

## Thresholds

| Threshold | Default | Purpose |
|-----------|---------|---------|
| `constant_threshold` | 0.99 | Above = operational setting |
| `system_threshold` | 0.70 | Above = fleet-coupled signal |
| `component_threshold` | 0.50 | Below = unit-specific signal |
| `within_unit_threshold` | 0.70 | Above = isotropic degradation |

## The Science

### Why Constants Hurt ML

1. **Zero Information**: No variance = no signal
2. **Wasted Capacity**: Model learns to ignore useless features
3. **Normalization Issues**: `std=0` causes `NaN` in StandardScaler
4. **Spurious Correlations**: Constants correlate with everything (including target)

### Why System Signals Need Special Handling

System signals (coupled across units) reflect **fleet-wide conditions**, not individual faults:
- Use as **baseline** to normalize component signals
- Or **exclude** if you want pure fault signatures
- Or **include** but expect lower feature importance

### Why Component Signals Are Gold

Component signals (independent across units) contain **unit-specific degradation**:
- Each unit's trajectory is its own
- Fault signatures are isolated
- RUL prediction targets this signal

## Paper-Ready Finding

> "Preprocessing with Cohort Discovery on NASA C-MAPSS FD001 identified 3 of 21 sensors as operational constants (ρ > 0.99 across all 100 engines), not degradation measurements. These 'sensors' (indices 5, 10, 16) have zero variance within each engine but identical values across engines—likely operational parameters mistakenly included in the sensor array. Excluding these from ML input is expected to improve model accuracy by reducing noise in the feature space and eliminating normalization artifacts."

## See Also

- [Manifold Geometry Stage](./geometry.md) - Pairwise signal analysis
- [Manifold Geometry](./manifold.md) - System-level structure
- [Fault Classification](./faults.md) - Using coupled/decoupled units
