# Framework SQL

## Directory Structure

```
sql/
├── run_all.sql      # Master runner
│
├── layers/          # Core computation layers (run in order)
│   ├── 00_load.sql
│   ├── 00_config.sql
│   ├── 01_typology.sql
│   ├── 02_geometry.sql
│   ├── 03_dynamics.sql
│   ├── 04_causality.sql
│   └── ...
│
├── views/           # Reusable views
│   ├── 04_visualization.sql
│   ├── 05_summaries.sql
│   ├── 06_general_views.sql
│   └── ...
│
├── reports/         # Analysis reports (from sql_reports/)
│   ├── 01_baseline_geometry.sql
│   ├── 02_stable_baseline.sql
│   ├── 03_drift_detection.sql
│   └── ...
│
└── ml/              # ML feature generation
    ├── 11_ml_features.sql
    └── 26_ml_feature_export.sql
```

## Schema (v2.5)

**observations.parquet:**
- `cohort` (String, optional): Grouping key (e.g., engine_1, pump_A)
- `signal_id` (String, required): Signal identifier
- `I` (UInt32, required): Observation index (sequential per signal)
- `value` (Float64, required): The measurement

**typology.parquet:**
- `signal_id`, `cohort`, `temporal_pattern`, `spectral`, `n_samples`, ...

## Execution Order

1. `layers/00_load.sql` - Load parquet files
2. `layers/00_config.sql` - Configuration
3. `layers/01_typology.sql` - Signal classification
4. `layers/02_geometry.sql` - State geometry
5. `layers/03_dynamics.sql` - Dynamical systems
6. `layers/04_causality.sql` - Causal mechanics

## Usage

```bash
# Run all layers
cd /path/to/domain
duckdb < /path/to/framework/sql/run_all.sql

# Run specific report
duckdb < /path/to/framework/sql/reports/03_drift_detection.sql
```
