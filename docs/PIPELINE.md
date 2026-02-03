# ORTHON Observation Processing Pipeline

> **"Garbage in, REJECTED"** — not "garbage in, garbage out"

## Overview

The ORTHON pipeline validates and analyzes observations BEFORE they reach PRISM.

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
    PRISM (clean data only)
```

## Quick Start

```python
from orthon.pipeline import process_observations

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
python -m orthon.pipeline observations.parquet --output ./processed/

# Permissive mode (warn but don't exclude)
python -m orthon.pipeline observations.parquet --permissive

# Skip cohort discovery (validation only)
python -m orthon.pipeline observations.parquet --no-cohort-discovery
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
├── observations_validated.parquet  # Clean data for PRISM
├── pipeline_report.yaml            # Complete structured report
├── pipeline_report.txt             # Human-readable summary
├── ml_signals.txt                  # Signals to use (one per line)
├── exclude_signals.txt             # Signals to exclude
└── manifest_update.yaml            # Merge into existing manifest
```

## Example Report

```
======================================================================
ORTHON OBSERVATION PROCESSING PIPELINE
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

## Integration with PRISM

```python
from orthon.pipeline import process_observations
from prism import run_prism

# Step 1: Process observations
result = process_observations('data/observations.parquet', 'data/processed/')

# Step 2: Run PRISM on validated data only
prism_result = run_prism(
    observations_path='data/processed/observations_validated.parquet',
    output_dir='data/prism_output/',
)
```

## Configuration

### Validation Config

```python
from orthon.validation import ValidationConfig, ValidationAction

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
from orthon.pipeline import ObservationPipeline

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

## See Also

- [Validation Module](./validation.md) - Signal validation details
- [Cohort Discovery](./cohort_discovery.md) - Cohort classification details
- [PRISM Integration](./prism_integration.md) - Running PRISM on validated data
