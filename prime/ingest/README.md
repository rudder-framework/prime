# Framework Data Ingestion

## The Pipeline

```
Raw Data (any format)
        |
        v
+-------------------+
|    transform.py   |  <- Wide to long, fix indices
+---------+---------+
          |
          v
+-------------------+
|    validate.py    |  <- Check schema before Manifold
+---------+---------+
          |
          v
  observations.parquet (Manifold format)
          |
          v
+-------------------+
|      Manifold     |
+-------------------+
```

## Manifold Schema Requirements

| Column    | Type    | Requirements                           |
|-----------|---------|----------------------------------------|
| cohort    | String  | Grouping key (optional, blank is fine) |
| signal_id | String  | Signal name (temp, pressure, acc_x)    |
| signal_0  | Float64 | Coordinate axis, sorted ascending per signal |
| value     | Float64 | The measurement                        |

### Critical Rules

1. **signal_0 MUST be sorted ascending** within each (cohort, signal_id) group
2. **Each cohort MUST have >=2 signals** for pair engines
3. **signal_0 is Float64** — may represent samples, physical time, or any monotonic coordinate
4. **No nulls** in signal_id, signal_0, or value

## Usage

### Transform Data

```bash
# Generic transform
python -m prime.ingest.transform input.parquet output.parquet \
    --signals col1 col2 col3

# Dataset-specific
python -c "
from pathlib import Path
from framework.ingest.transform import transform_femto
transform_femto(Path('raw/femto.parquet'), Path('observations.parquet'))
"
```

### Validate Before Manifold

```bash
# Always validate before running Manifold
python -m prime.ingest.validate_observations observations.parquet

# Expected output:
# [OK] VALIDATION PASSED
# [OK] Safe to run Manifold
```

### Full Pipeline

```bash
# 1. Transform
python -m prime.ingest.transform raw_data.parquet observations.parquet \
    --signals acc_x acc_y temp

# 2. Validate
python -m prime.ingest.validate_observations observations.parquet

# 3. Run Prime (validates, transforms, calls Manifold)
./run ~/domains/my_dataset/train
```

## Dataset-Specific Transforms

### FEMTO Bearings

```python
from pathlib import Path
from framework.ingest.transform import transform_femto

transform_femto(
    raw_path=Path("data/benchmarks/femto/observations.parquet"),
    output_path=Path("/Users/jasonrudder/manifold/data/observations.parquet")
)
```

### SKAB

```python
from pathlib import Path
from framework.ingest.transform import transform_skab

transform_skab(
    raw_path=Path("data/benchmarks/skab/observations.parquet"),
    output_path=Path("/Users/jasonrudder/manifold/data/observations.parquet")
)
```

### C-MAPSS

```python
from pathlib import Path
from framework.ingest.transform import transform_cmapss

transform_cmapss(
    raw_path=Path("data/benchmarks/cmapss/train_FD001.parquet"),
    output_path=Path("/Users/jasonrudder/manifold/data/observations.parquet")
)
```

### Fama-French

```python
from pathlib import Path
from framework.ingest.transform import transform_fama_french

transform_fama_french(
    raw_path=Path("data/benchmarks/fama_french/data.parquet"),
    output_path=Path("/Users/jasonrudder/manifold/data/observations.parquet")
)
```

## Common Issues

### "signal_0 is not sorted ascending"

**Problem:** signal_0 values are not in ascending order within a signal group
**Solution:** Use `fix_sparse=True` in transform (default) — sorts and deduplicates signal_0

### "Only 1 signal per cohort"

**Problem:** Pair engines need >=2 signals
**Solution:** Include more signal columns in transform, or restructure data

### "dynamics.parquet is empty"

**Problem:** Usually caused by single signal or insufficient observations
**Solution:** Validate data first, fix transform

## Validation Checklist

Before running Manifold:

- [ ] `signal_id` exists and is String
- [ ] `signal_0` exists, is Float64, and sorted ascending per signal
- [ ] `value` exists and is Float64
- [ ] `cohort` exists (optional — blank is fine)
- [ ] No null values in signal_id, signal_0, or value
- [ ] Run `validate.py` and see "[OK] VALIDATION PASSED"

## The Rule

```
NO DATA REACHES Manifold WITHOUT VALIDATION.

Raw -> Transform -> Validate -> Manifold

If validate fails, fix transform.
Never run Manifold on unvalidated data.
```
