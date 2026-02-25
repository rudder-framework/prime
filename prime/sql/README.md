# Prime SQL

## Directory Structure

```
sql/
├── layers/              # 33 numbered SQL layers (run in order)
│   ├── 00_*.sql         # Load, config, aliases, observations, physics compat, index detection
│   ├── 01_*.sql         # Calculus (derivatives), typology
│   ├── 02_*.sql         # Geometry, statistics
│   ├── 03_*.sql         # Dynamics, signal classification
│   ├── 04_causality.sql # Causal mechanics
│   ├── 05_manifold_derived.sql
│   ├── 08_entropy.sql
│   ├── 12-18_*.sql      # Brittleness, canary, curvature, geometry, coupling, dimension, CI breach
│   ├── atlas_*.sql      # Atlas layers (analytics, breaks, FTLE, ridge, topology, velocity)
│   ├── break_classification.sql
│   ├── classification.sql
│   ├── constants_units.sql
│   └── typology_v2.sql
│
├── reports/             # 30 independent SQL reports
│   ├── 01-25_*.sql      # Typology through feature relevance
│   └── 60-63_*.sql      # Ground truth, lead time, fault signatures, thresholds
│
├── views/               # Reusable SQL views
├── stages/              # Stage-specific SQL
├── typology/            # SQL-based typology pipeline (7 SQL files + runner.py)
├── docs/                # SQL layer documentation
└── runner.py            # Python SQL runner (supports output_{axis}/ layout)
```

## Schema

**observations.parquet:**
- `cohort` (String, optional): Grouping key (e.g., engine_1, pump_A)
- `signal_id` (String, required): Signal identifier
- `signal_0` (Float64, required): Coordinate axis (sorted ascending per signal)
- `value` (Float64, required): The measurement

## Usage

```bash
# Preferred — Python runner with output_{axis}/ support
prime query ~/domains/FD004/

# Query specific view
prime query ~/domains/FD004/ --view typology

# Alternative
python -m prime.sql.runner ~/domains/FD004/
```

SQL layers are numbered and run in order. Reports are independent.
