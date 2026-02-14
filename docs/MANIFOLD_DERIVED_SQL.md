# Prime SQL Views — Manifold Derived Stages

## File: `prime/sql/layers/05_manifold_derived.sql`

17 DuckDB SQL views that replace 16 Manifold pipeline stages. These views operate
on Manifold's parquet outputs and produce derived analytics without any computation
in the Manifold engine itself.

## View Catalog

### Core Derived Views (replace Manifold stages)

| View | Replaces | Input Parquets | Rows (FD001) | Cols | SQL Pattern |
|------|----------|---------------|-------------|------|-------------|
| `v_geometry_dynamics` | stage_07 | state_geometry | 2,480 | 14 | LAG() for velocity/acceleration/jerk |
| `v_cohort_vector` | stage_04 | state_geometry | 860 | 41 | FILTER conditional aggregation |
| `v_topology` | stage_11 | signal_pairwise | 860 | 9 | Adaptive threshold (90th pctile) |
| `v_statistics` | stage_13 | observations | 2,400 | 14 | GROUP BY aggregates |
| `v_zscore_ref` | stage_12 | signal_vector | 1 | 7 | Normalization reference |
| `v_correlation` | stage_14 | signal_vector | 12,992 | 5 | CORR() with NaN safety |
| `v_break_sequence` | stage_16 | breaks | 1,535 | 7 | LAG() for inter-break timing |
| `v_segment_comparison` | stage_18 | signal_geometry, breaks | 300 | 11 | Segment mean comparison |
| `v_info_flow_delta` | stage_19 | information_flow | 13,600 | 11 | LAG() on transfer entropy |
| `v_velocity_field` | stage_21 | state_vector | 760 | 7 | LAG() on centroid positions |

### Cohort-Level Views

| View | Replaces | Input | Rows | Cols | Description |
|------|----------|-------|------|------|-------------|
| `v_cohort_pairwise` | — | signal_pairwise | 36,470 | 4 | Aggregated pairwise metrics |
| `v_cohort_topology` | — | signal_pairwise | 16 | 5 | Network density, degree stats |
| `v_cohort_velocity_field` | stage_31 | v_velocity_field | 760 | 7 | Velocity at cohort scale |
| `v_cohort_fingerprint` | stage_32 | gaussian_fingerprint | 100 | 27 | Gaussian summary stats |
| `v_cohort_thermodynamics` | stage_09a | state_geometry | 2,480 | 10 | Entropy from eigenvalues |

### Analysis Views

| View | Purpose | Rows | Key Finding |
|------|---------|------|-------------|
| `v_canary_signals` | Correlate signal features with lifecycle position | 34 | htBleed hurst r=0.44 |
| `v_ml_features` | Assembled ML feature matrix from all outputs | 860 | 63 columns |

## Technical Notes

### PIVOT Workaround
DuckDB views cannot use `PIVOT` with data-derived pivot elements. Solution:
```sql
-- WRONG: PIVOT (fails in view)
PIVOT state_geometry ON engine USING MAX(effective_dim)

-- RIGHT: Conditional aggregation with FILTER
SELECT
    cohort, I,
    MAX(effective_dim) FILTER (WHERE engine = 'complexity') AS complexity_effective_dim,
    MAX(effective_dim) FILTER (WHERE engine = 'spectral')   AS spectral_effective_dim,
    MAX(effective_dim) FILTER (WHERE engine = 'shape')      AS shape_effective_dim,
FROM state_geometry
GROUP BY cohort, I
```

### NaN-Safe Correlation
DuckDB's CORR/STDDEV overflow on NaN values. Must pre-filter:
```sql
-- CTE to remove NaN BEFORE computing correlation
signal_lifecycle AS (
    SELECT ...
    FROM signal_vector
    WHERE spectral_entropy IS NOT NULL
      AND NOT isnan(spectral_entropy)
      AND hurst IS NOT NULL
      AND NOT isnan(hurst)
),
-- Additional variance filter to prevent STDDEV overflow
varying_signals AS (
    SELECT signal_id FROM signal_lifecycle
    GROUP BY signal_id
    HAVING COUNT(*) > 10
       AND MAX(spectral_entropy) > MIN(spectral_entropy)
       AND MAX(lifecycle_pct) > MIN(lifecycle_pct)
)
```

### Ambiguous Column References
When joining multiple tables with same column names, use explicit aliases:
```sql
-- WRONG: USING (cohort, I)
FROM v_cohort_vector USING (cohort, I)

-- RIGHT: Explicit ON with aliases
FROM v_cohort_vector cv
JOIN v_geometry_dynamics gd ON cv.cohort = gd.cohort AND cv.I = gd.I
```

## Loading Views

```python
import duckdb

con = duckdb.connect()

# Load Manifold parquets
for f in manifold_output_dir.glob('*.parquet'):
    table_name = f.stem  # e.g., 'state_geometry'
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM '{f}'")

# Load observations
con.execute(f"CREATE TABLE observations AS SELECT * FROM '{obs_path}'")

# Execute SQL views
with open('prime/sql/layers/05_manifold_derived.sql') as f:
    con.execute(f.read())

# Query any view
result = con.execute("SELECT * FROM v_canary_signals").pl()
```
