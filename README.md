# Prime

**Signal analysis orchestrator for the Rudder Framework**

Prime classifies signals, generates compute manifests, orchestrates Manifold pipelines, and interprets results. It transforms raw sensor data from any domain into a geometric characterization of system state — without domain-specific knowledge.

## Core Insight

Systems lose dimensional coherence before failure. Effective dimensionality — measured through recursive eigendecomposition at multiple scales — predicts remaining useful life. The same mathematical framework that detects coherence loss before industrial failure also captures phase transitions in dynamical systems.

## Architecture

```
pmtvs              ← 244 signal analysis functions, 21 Rust-accelerated (pip install pmtvs)
 ↑      ↑
 |      |
Prime  Manifold    ← Prime orchestrates, Manifold computes
 |      ↑
 └──────┘          ← Prime calls Manifold via HTTP
```

Prime is the brain. Manifold is the muscle. pmtvs is the math.

## Quick Start

```bash
# Install
git clone https://github.com/rudder-framework/prime.git
cd prime
pip install -e .

# Run the full pipeline on a dataset
python -m prime ~/domains/rossler/train

# Or run stages individually
python -m prime.entry_points.stage_01_validate observations.parquet -o validated.parquet
python -m prime.entry_points.stage_02_typology observations.parquet -o typology_raw.parquet
python -m prime.entry_points.stage_03_classify typology_raw.parquet -o typology.parquet
python -m prime.entry_points.stage_04_manifest typology.parquet -o manifest.yaml
```

## Pipeline

```
Raw Data (CSV, Excel, Parquet)
     │
     ▼
1. INGEST          → observations.parquet (canonical schema)
2. VALIDATE        → sequential signal_0, no nulls, no constants
3. TYPOLOGY        → 27 raw measures per signal (hurst, entropy, spectral, etc.)
4. CLASSIFY        → 10 classification dimensions per signal
5. MANIFEST        → engine selection + window config per signal
6. COMPUTE         → Manifold produces geometry parquets
7. ANALYZE         → SQL layers on DuckDB (regime detection, drift, canaries)
8. EXPLORE         → Browser-based DuckDB-WASM explorer
```

## Canonical Schema

Every dataset, regardless of domain, is normalized to:

**observations.parquet:**

| Column | Type | Description |
|--------|------|-------------|
| signal_0 | Float64 | Ordering axis values (time, depth, cycles, pressure — user's choice) |
| signal_id | String | Signal identifier (sensor_2, temperature) |
| value | Float64 | The measurement |
| cohort | String | Cohort/unit/engine identifier (empty string if none). Optional. |

**signals.parquet** (written alongside observations.parquet):

| Column | Type | Description |
|--------|------|-------------|
| signal_id | String | Signal identifier (matches signal_id in observations) |
| unit | String | Unit string ("psi", "°F", "m/s", "rpm") — nullable |
| description | String | Human-readable description — nullable |
| source_name | String | Original column name from raw data before renaming |

`signal_0` is sorted ascending per signal. Domain-specific column mapping happens at ingest and nowhere else.

## Signal Classification

Prime characterizes each signal across 10 orthogonal dimensions before any computation begins:

1. **Continuity** — CONSTANT, EVENT, DISCRETE, CONTINUOUS
2. **Stationarity** — STATIONARY, TREND_STATIONARY, DIFFERENCE_STATIONARY, NON_STATIONARY
3. **Temporal pattern** — PERIODIC, QUASI_PERIODIC, CHAOTIC, RANDOM, STATIONARY
4. **Complexity** — LOW, MODERATE, HIGH
5. **Memory** — SHORT, MODERATE, LONG, ANTI_PERSISTENT
6. **Distribution** — CONSTANT, HEAVY_TAILED, LIGHT_TAILED, SKEWED_RIGHT, SKEWED_LEFT, GAUSSIAN
7. **Amplitude** — CONSTANT, IMPULSIVE, SMOOTH, NOISY, MIXED
8. **Spectral** — NARROWBAND, BROADBAND, ONE_OVER_F, HARMONIC
9. **Volatility** — CONSTANT, STABLE, MODERATE, HIGH, EXTREME
10. **Determinism** — DETERMINISTIC, STOCHASTIC, MIXED

This typology drives engine selection: each signal gets exactly the compute engines appropriate for its character. CONSTANT signals skip all engines. PERIODIC signals get harmonic analysis. CHAOTIC signals get Lyapunov exponents. No manual configuration required.

## Two-Scale Geometry

The framework applies identical mathematical machinery at two scales:

**Cohort Level (Scale 1):** Cross-signal eigendecomposition within each entity. 14 sensors on one engine → how are they moving together? Dimensional collapse here means the engine is losing degrees of freedom — degradation is forcing signals to correlate.

**System Level (Scale 2):** Cross-cohort eigendecomposition across entities. 100 engines in a fleet → how are their geometric signatures clustering? Divergence here means subpopulations are forming — some engines are on different trajectories.

Same eigendecomposition engine. Same vector enrichment (Laplacian, Fourier, Hilbert). Same dynamics (FTLE, Lyapunov, velocity, topology, thermodynamics). Applied recursively.

## Cross-Domain Validation

The framework is domain-agnostic. The same pipeline processes:

- **Turbofan engines** (C-MAPSS FD001–FD004) — degradation to failure
- **Battery cells** — capacity fade over charge cycles
- **Rössler attractors** — chaotic dynamical systems
- **Pendulum systems** — multi-entity coupled dynamics
- **Hydraulic systems** — condition monitoring across pump cohorts

No domain-specific code touches the analysis. The ingest layer maps raw formats to the canonical schema. Everything downstream sees only `cohort`, `signal_id`, `signal_0`, `value`.

## SQL Analysis

All interpretation runs as SQL against Manifold's parquet outputs via DuckDB:

```bash
# Run all analysis layers
duckdb < prime/sql/run_all.sql

# Run specific reports
duckdb < prime/sql/reports/03_drift_detection.sql
```

SQL layers compute: effective dimensionality trajectories, velocity/acceleration/jerk of dimensional collapse, regime classification, baseline departure scoring, canary signal sequencing, brittleness indices, and natural parameterization rankings.

## Explorer

Prime includes a browser-based explorer using DuckDB-WASM:

```bash
python -m prime.entry_points.stage_09_explore results/
# Opens http://localhost:8000 with interactive SQL console
```

Upload parquet files, run SQL queries, visualize geometry and dynamics interactively.

## Dependencies

- **Python** ≥ 3.10
- **pmtvs** — Rust-accelerated primitives (`pip install pmtvs`)
- **polars** — DataFrame operations
- **DuckDB** — SQL analysis engine
- **scipy** — Statistical tests

## Citation

If you use the Rudder Framework in your research, please cite:

```bibtex
@software{rudder_framework,
  author = {Rudder, Jason and Rudder, Avery},
  title = {Rudder Framework: Universal Signal Analysis via Recursive Eigendecomposition},
  url = {https://github.com/rudder-framework/prime},
  year = {2025--2026}
}
```

## License

See [LICENSE.md](LICENSE.md) for details.
