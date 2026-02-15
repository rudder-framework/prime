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
| entity_id | String  | Unique identifier per entity           |
| I         | UInt32  | Sequential 0,1,2,3... PER ENTITY       |
| signal_id | String  | Signal name (temp, pressure, acc_x)    |
| value     | Float64 | The measurement                        |

### Critical Rules

1. **I MUST be sequential** (0,1,2,3...) within each entity
2. **Each entity MUST have >=2 signals** for pair engines
3. **No sparse indices** (0,10,20... is WRONG)
4. **No nulls** in any column

## Usage

### Transform Data

```bash
# Generic transform
python -m framework.ingest.transform input.parquet output.parquet \
    --entity entity_column \
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
python -m framework.ingest.validate observations.parquet

# Expected output:
# [OK] VALIDATION PASSED
# [OK] Safe to run Manifold
```

### Full Pipeline

```bash
# 1. Transform
python -m framework.ingest.transform raw_data.parquet observations.parquet \
    --entity bearing_id \
    --signals acc_x acc_y temp

# 2. Validate
python -m framework.ingest.validate observations.parquet

# 3. Run Manifold (only if validation passes)
cd ~/manifold
./venv/bin/python -m manifold data/observations.parquet
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

### "I is not sequential"

**Problem:** I values are sparse (0, 10, 20...)
**Solution:** Use `fix_sparse=True` in transform (default)

### "Only 1 signal per entity"

**Problem:** Pair engines need >=2 signals
**Solution:** Include more signal columns in transform, or restructure data

### "dynamics.parquet is empty"

**Problem:** Usually caused by sparse I or single signal
**Solution:** Validate data first, fix transform

## Validation Checklist

Before running Manifold:

- [ ] `entity_id` exists and is String
- [ ] `I` exists and is sequential (0,1,2...)
- [ ] `signal_id` exists and has >=2 unique values per entity
- [ ] `value` exists and is Float64
- [ ] No null values
- [ ] Run `validate.py` and see "[OK] VALIDATION PASSED"

## The Rule

```
NO DATA REACHES Manifold WITHOUT VALIDATION.

Raw -> Transform -> Validate -> Manifold

If validate fails, fix transform.
Never run Manifold on unvalidated data.
```
