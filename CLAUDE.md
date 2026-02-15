# CLAUDE.md — Prime

## What is Prime

Prime is the orchestrator. It classifies signals, generates manifests, calls Manifold for computation, and analyzes results. Users run Prime. Prime runs everything else.

```
User → prime ~/domains/cmapss/FD_004/train → results
```

Prime does NOT compute signal trajectories, eigendecompositions, or state geometry. That's Manifold. Prime decides WHAT to compute and interprets the results AFTER computation.

## Architecture — Three repos, clean dependency tree

```
primitives             ← Rust+Python math functions (leaf dependency)
     ↑        ↑
     |        |
  Prime    Manifold    ← Prime orchestrates, Manifold computes
     |        ↑
     └────────┘        ← Prime calls manifold.run()
```

- **primitives** — `from primitives import hurst_exponent`. Pure functions. numpy in, scalar out. Handles Rust/Python toggle via `USE_RUST` env var.
- **manifold** — `from manifold import run`. Compute engine. Receives observations.parquet + manifest.yaml, writes output parquets. Never run directly.
- **prime** — The brain. Ingest, typology, classification, manifest, orchestration, SQL analysis, explorer.

## CLI commands

| Command | Entry point | Description |
|---------|-------------|-------------|
| `prime` | `prime.cli:main` | Interpret Manifold parquet outputs via DuckDB |
| `prime-explorer` | `prime.explorer.cli:main` | Launch DuckDB-WASM browser explorer |
| `prime-config` | `prime.ingest.data_reader:main` | Data reader / config generator |
| `prime-serve` | `prime.core.api:main` | FastAPI server for Prime API |

## Pipeline flow

```
prime ~/domains/FD_004/train

  1. INGEST       raw domain files → observations.parquet
  2. VALIDATE     observations → validated observations (I sequential, no nulls)
  3. TYPOLOGY     observations → typology_raw.parquet (30 measures per signal)
                  Uses primitives: hurst, perm_entropy, sample_entropy, lyapunov_rosenstein,
                  skewness, kurtosis, crest_factor, adf_test, kpss_test
  4. CLASSIFY     typology_raw → typology.parquet (10 classification dimensions)
                  Pure decision trees. No external dependencies.
  5. MANIFEST     typology → manifest.yaml (engine selection per signal)
                  Pure rules. No external dependencies.
  6. COMPUTE      observations + manifest → output/*.parquet
                  Calls manifold.run(). Prime does NOT do this computation.
  7. ANALYZE      output parquets → SQL layers + reports (DuckDB)
  8. EXPLORE      static HTML explorer (DuckDB-WASM)
```

Steps 1-5 and 7-8 are Prime. Step 6 is Manifold.

## Canonical schema

**observations.parquet** — the ONE file Prime creates from raw data:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| cohort | String | Optional | Grouping key (engine_1, pump_A) |
| signal_id | String | Yes | Signal identifier |
| I | UInt32 | Yes | Sequential index per signal (0, 1, 2, 3...) |
| value | Float64 | Yes | The measurement |

I is ALWAYS sequential integers starting at 0. Never timestamps. Never floats. Column mapping from raw formats happens at ingest, nowhere else.

## Directory structure

```
prime/
├── core/
│   ├── pipeline.py              # Main orchestrator: observations → results
│   ├── manifold_client.py       # Calls manifold.run() — thin wrapper
│   ├── api.py                   # FastAPI server
│   ├── server.py                # Server runner
│   ├── validation.py            # Input validation
│   └── data_reader.py           # CSV/parquet/Excel reader
│
├── ingest/
│   ├── typology_raw.py          # 30 raw measures per signal (uses primitives)
│   ├── validate_observations.py # Validates & repairs I sequencing
│   ├── transform.py             # Raw data → observations.parquet
│   ├── data_reader.py           # CSV/parquet/Excel reader
│   ├── paths.py                 # Path resolution
│   ├── streaming.py             # Streaming ingest
│   └── schema/                  # MANIFOLD_SCHEMA.yaml
│
├── typology/
│   ├── discrete_sparse.py       # PR5: discrete/sparse detection (runs FIRST)
│   ├── level2_corrections.py    # PR4: continuous classification
│   ├── constant_detection.py    # CV-based CONSTANT detection
│   ├── classification_stability.py
│   ├── control_detection.py
│   ├── window_factor.py
│   └── tests/                   # Inline test suites
│
├── config/
│   ├── typology_config.py       # All classification thresholds (no magic numbers)
│   ├── discrete_sparse_config.py
│   ├── stability_config.py
│   ├── domains.py               # Domain mappings
│   └── recommender.py           # Engine recommendation config
│
├── manifest/
│   ├── generator.py             # v2.6 manifest: engine gating, intervention mode
│   ├── characteristic_time.py   # Data-driven window from ACF, frequency, period
│   ├── system_window.py         # Multi-scale representation
│   ├── domain_clock.py          # Domain-specific clock detection
│   ├── window_recommender.py    # Window size recommendations
│   └── tests/                   # Inline test suites
│
├── services/
│   ├── physics_interpreter.py   # Symplectic structure loss detection
│   ├── dynamics_interpreter.py  # Lyapunov, basin stability, regime transitions
│   ├── state_analyzer.py        # State threshold analysis
│   ├── fingerprint_service.py   # Failure fingerprint matching
│   ├── job_manager.py           # Job tracking
│   ├── tuning_service.py        # Parameter tuning
│   └── concierge.py             # Guided workflow service
│
├── engines/
│   ├── classification_engine.py # Per-signal engine selection
│   ├── diagnostic_report.py     # Diagnostic outputs
│   ├── signal_geometry.py       # Signal geometry analysis
│   ├── stability_engine.py      # Stability checks
│   └── ...                      # 12 engine modules total
│
├── analysis/
│   ├── window_optimization.py   # Baseline window analysis
│   ├── thermodynamics.py        # Thermodynamic signatures
│   └── twenty_twenty.py         # 20/20 hindsight analysis
│
├── cohorts/
│   ├── baseline.py              # Baseline cohort identification
│   ├── detection.py             # Anomaly detection
│   └── discovery.py             # System structure discovery
│
├── early_warning/
│   ├── failure_fingerprint_detector.py
│   └── ml_predictor.py          # ML-based TTF prediction
│
├── shared/
│   ├── config_schema.py         # Pydantic models
│   ├── engine_registry.py       # Engine registry
│   ├── physics_constants.py     # Physical constants
│   └── window_config.py         # Window configuration
│
├── streaming/
│   ├── analyzers.py             # Real-time data processors
│   ├── websocket_server.py      # WebSocket API
│   ├── data_sources.py          # Streaming data sources
│   └── cli.py                   # Streaming CLI
│
├── inspection/
│   ├── capability_detector.py   # Check Manifold capabilities
│   ├── file_inspector.py        # Result file inspection
│   └── results_validator.py     # Output validation
│
├── state/
│   └── classification.py        # Signal classification state tracking
│
├── ml/
│   ├── lasso.py                 # Lasso feature selection
│   └── entry_points/            # ML pipeline: train, predict, evaluate, ablation
│
├── io/
│   └── readme_writer.py         # Markdown report generation
│
├── utils/
│   └── index_detection.py       # Auto-detect I column in raw data
│
├── sql/
│   ├── layers/                  # 36 core SQL layers (run in order)
│   ├── reports/                 # 21 analysis reports
│   ├── stages/                  # 6-stage execution plan
│   ├── views/                   # 5 reusable views
│   ├── ml/                      # ML feature engineering
│   └── docs/                    # Auto-generated SQL documentation
│
├── explorer/
│   ├── cli.py                   # Explorer CLI
│   ├── server.py                # WebSocket data server
│   └── static/                  # DuckDB-WASM browser UI (atlas, explorer, flow_viz, wizard)
│
└── entry_points/                # CLI entry points per stage
    ├── stage_01_validate.py → stage_13_train.py
    └── csv_to_atlas.py         # One-command pipeline

scripts/                         # Top-level (NOT under prime/)
├── process_all_domains.py       # Batch domain processing
└── regenerate_manifests.py      # Batch manifest regeneration
```

## Typology — 10 classification dimensions

Typology is the signal classification system. It computes 30 raw statistical measures per signal and classifies across 10 dimensions.

### Raw measures (typology_raw.py)

```
adf_pvalue, kpss_pvalue, variance_ratio, acf_half_life, hurst,
perm_entropy, sample_entropy, spectral_flatness, spectral_slope,
harmonic_noise_ratio, spectral_peak_snr, dominant_frequency,
is_first_bin_peak, turning_point_ratio, lyapunov_proxy,
determinism_score, arch_pvalue, rolling_var_std, kurtosis,
skewness, crest_factor, unique_ratio, is_integer, is_constant,
sparsity, signal_std, signal_mean, derivative_sparsity,
zero_run_ratio, n_samples
```

Four of these are expensive (hurst, perm_entropy, sample_entropy, lyapunov_rosenstein) and come from primitives with Rust acceleration. Five more (skewness, kurtosis, crest_factor, adf_test, kpss_test) also come from primitives submodules.

### 10 classification dimensions

1. **Continuity**: CONSTANT, EVENT, DISCRETE, CONTINUOUS
2. **Stationarity**: STATIONARY, TREND_STATIONARY, DIFFERENCE_STATIONARY, NON_STATIONARY
3. **Temporal pattern**: PERIODIC, QUASI_PERIODIC, CHAOTIC, RANDOM, STATIONARY
4. **Complexity**: LOW, MODERATE, HIGH
5. **Memory**: SHORT, MODERATE, LONG, ANTI_PERSISTENT
6. **Distribution**: CONSTANT, HEAVY_TAILED, LIGHT_TAILED, SKEWED_RIGHT, SKEWED_LEFT, GAUSSIAN
7. **Amplitude**: CONSTANT, IMPULSIVE, SMOOTH, NOISY, MIXED
8. **Spectral**: NARROWBAND, BROADBAND, ONE_OVER_F, HARMONIC
9. **Volatility**: CONSTANT, STABLE, MODERATE, HIGH, EXTREME
10. **Determinism**: DETERMINISTIC, STOCHASTIC, MIXED

### Classification order

1. Discrete/sparse classification runs FIRST (PR5: `discrete_sparse.py`)
2. Continuous classification runs only for non-discrete signals (PR4: `level2_corrections.py`)
3. All thresholds are in config files, no magic numbers in code

## Manifest — v2.6

The manifest tells Manifold exactly which engines to run per signal. Prime generates it from typology. Manifold executes exactly what's specified.

Key concepts:
- `system.window` — common window for multi-signal alignment
- `engine_windows` — global minimum window sizes for FFT/long-range engines
- `engine_window_overrides` — per-signal overrides when signal window < engine minimum
- `skip_signals` — CONSTANT signals skip all engines
- `pair_engines` — asymmetric pairwise engines (granger, transfer_entropy)
- `symmetric_pair_engines` — symmetric pairwise engines (correlation, mutual_info, cointegration)
- `intervention.enabled` — v2.6: intervention mode for fault injection / event response datasets
- `intervention.event_index` — sample index where intervention occurs

## SQL layers

All SQL runs on DuckDB against the parquet files Manifold produces. Prime does NOT compute — it queries.

```bash
# Run all layers
duckdb < prime/sql/run_all.sql

# Run specific report
duckdb < prime/sql/reports/03_drift_detection.sql
```

SQL layers are numbered and run in order. Reports are independent.

## How Prime calls Manifold

```python
from manifold import run

result = run(
    observations_path="observations.parquet",
    manifest_path="manifest.yaml",
    output_dir="output/",
    verbose=True,
)
```

That's it. One function call. Prime doesn't know about Manifold's stages, workers, or internals. It hands over two files and gets parquets back.

## How Prime uses primitives

```python
from primitives import hurst_exponent, permutation_entropy, sample_entropy, lyapunov_rosenstein
from primitives.individual.statistics import skewness, kurtosis, crest_factor
from primitives.stat_tests.stationarity_tests import adf_test, kpss_test

# One call per signal, full signal in, one number out
h = hurst_exponent(signal_values)
```

9 functions total from primitives. The 4 top-level imports are expensive (Rust-accelerated). The 5 submodule imports wrap scipy/statsmodels for consistency. Prime calls primitives once per signal for typology. Manifold calls primitives thousands of times per pipeline run (once per window per signal). Same functions, different scale.

## Env vars

| Variable | Controls | Default |
|----------|----------|---------|
| `USE_RUST` | Rust vs Python for primitives | `1` (Rust) |
| `PRIME_WORKERS` | Parallel signals in typology | `1` (single-threaded) |
| `PRIME_OUTPUT_DIR` | Default output directory | `data` |
| `MANIFOLD_WORKERS` | Parallel cohorts (Manifold's concern) | `0` (auto) |

## Rules

1. **Prime classifies. Manifold computes.** Never put computation in Prime. Never put classification in Manifold.
2. **Primitives is pure math.** numpy in, number out. No file I/O, no config, no domain knowledge.
3. **observations.parquet is the contract.** Everything downstream depends on this schema. I is sequential. Always.
4. **Manifest is the spec.** Manifold executes exactly what the manifest says. No interpretation, no overrides.
5. **SQL layers do not compute.** They query parquets that Manifold already wrote. Read-only.
6. **Thresholds live in config files.** No magic numbers in classification code. All thresholds in `config/typology_config.py` and `config/discrete_sparse_config.py`.
7. **Domain knowledge lives in ingest only.** CMAPSS column mappings, IMS file structure, battery CSV formats — all in ingest/transform. Nothing downstream knows what domain it's processing.

## Naming

- The GitHub org is `rudder-framework`. That's fine.
- The packages are `prime`, `manifold`, `primitives`. No branding in code.
- No "rudder" in imports, class names, function names, variable names, print statements, docstrings, SQL comments, or API routes.
- No "PRISM" anywhere. PRISM was the old name for Manifold. It's gone.
- "rudder" in LICENSE.md (copyright holder) and pyproject.toml (author) is fine — that's a person's name.

## Do NOT

- Do not add computation to Prime. If it's math, it goes in primitives or Manifold.
- Do not add classification to Manifold. If it's a decision, it goes in Prime.
- Do not run Manifold directly. Users run Prime. Prime calls Manifold.
- Do not put domain-specific code outside of ingest. The pipeline is domain-agnostic.
- Do not use `rudder` or `PRISM` in new code. The packages are `prime`, `manifold`, `primitives`.
- Do not guess file paths in Manifold. Prime tells Manifold exactly where files are.
- Do not put thresholds in code. They go in config files.
- Do not modify the observations schema. It's cohort, signal_id, I, value. That's it.