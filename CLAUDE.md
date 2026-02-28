# CLAUDE.md — Prime

## What is Prime

Prime is the orchestrator of the Rudder Framework. It classifies signals, generates manifests, calls Manifold for computation, and analyzes results. Users run Prime. Prime runs everything else.

```
User → prime ~/domains/cmapss/FD_004/train → results
```

Prime does NOT compute signal trajectories, eigendecompositions, or state geometry. That's Manifold. Prime decides WHAT to compute and interprets the results AFTER computation.

## ⛔ MANDATORY RULES — READ BEFORE EVERY ACTION

### Rule 0: SEARCH BEFORE YOU CREATE

Before writing ANY new code, search the repo for existing implementations.

```bash
find . -name "*.py" | xargs grep -l "function_name"
grep -r "def compute" prime/
```

If you think something doesn't exist, ASK before creating it.

### Rule 1: USE EXISTING CODE

If a function, engine, or pattern exists in the repo, USE IT. Do not recreate.

### Rule 2: NO ROGUE FILE CREATION

All new files go inside the existing repo structure with approval. Never `/tmp/`, never `~/`, never standalone scripts.

### Rule 3: SHOW YOUR WORK BEFORE CHANGES

Before modifying any file, show the existing file, the pattern you're following, and get explicit approval before creating NEW files.

### Rule 4: PRIME CLASSIFIES. MANIFOLD COMPUTES. VECTOR EXTRACTS. GEOMETRY MEASURES.

- Do NOT create computation logic in Prime (eigendecompositions, FTLE, Lyapunov, etc.)
- Do NOT create classification logic in Manifold (signal types, regime labels, etc.)
- Do NOT create new engines in Prime — windowed feature extraction lives in `packages/vector/`
- Do NOT create signal geometry or eigenvalue dynamics outside `packages/geometry/`
- Prime's only computation is typology (27 raw measures per signal via primitives)

### Rule 5: DO NOT GLOB FRAMEWORK FILES

Ingest must NEVER treat framework files as raw data. These stems are reserved and must be excluded from any file discovery:

```python
# In prime/ingest/upload.py
_framework_stems = {
    'observations', 'typology', 'typology_raw', 'validated', 'signals', 'ground_truth',
    # Battery-domain supplementary files (not raw sensor data):
    'charge', 'impedance', 'conditions',
}
```

If `observations.parquet` already exists, skip ingest entirely. Use `--force-ingest` to override.

### Rule 6: MANIFEST PATHS MUST BE ABSOLUTE

All paths in `manifest.yaml` must be absolute. Prime resolves paths at manifest write time via `resolve_manifest_paths()`. Manifold validates paths at startup via `validate_manifest_paths()`. No relative paths. No guessing.

## Architecture — Repos and packages

```
primitives (pmtvs)     ← Rust+Python math functions (leaf dependency, on PyPI)
     ↑        ↑
     |        |
  Prime    Manifold    ← Prime orchestrates, Manifold computes
  / | \       ↑
 /  |  \      |
typology vector geometry  ← Local packages under packages/ (editable installs)
     |        ↑
     └────────┘        ← Prime calls Manifold via direct Python import (orchestration package)
```

- **primitives (pmtvs)** — `from pmtvs import hurst_exponent`. Pure functions. numpy in, scalar out. Rust-accelerated functions. Published on PyPI. Two repos under `pmtvs` GitHub user: `pmtvs-core` (private) and `pmtvs-pip` (public/PyPI).
- **orchestration** — Pipeline sequencer. Runs compute stages in correct order: typology → vector → eigendecomp → geometry → dynamics → pairwise → velocity → etc. Lives at `packages/orchestration/`, installed editable. Prime calls `orchestration.run()` via `prime/core/manifold_client.py`.
- **vector** — Windowed feature extraction. 44 engines, 179 output keys, three scales (signal, cohort, system). Lives at `packages/vector/`, installed editable. `from vector.signal import compute_signal`. Engines are pure glue — import from pmtvs, call, namespace, return.
- **geometry** — Signal geometry and eigenvalue dynamics. Per-signal position relative to eigenstructure (distance, coherence, contribution, residual). Eigenvalue trajectory derivatives (velocity, acceleration, jerk, curvature). Collapse detection. Lives at `packages/geometry/`, installed editable. `from geometry import compute_signal_geometry, compute_eigenvalue_dynamics, detect_collapse`.
- **typology** — Signal classification and window sizing. Lives at `packages/typology/`, installed editable. `from typology import from_observations`.
- **eigendecomp** — Eigenvalue structure of signal ensembles at cohort scale. Bootstrap confidence intervals, continuity tracking. Lives at `packages/eigendecomp/`.
- **dynamics** — Finite-Time Lyapunov Exponents (FTLE) per signal. Trajectory sensitivity over finite windows. Lives at `packages/dynamics/`.
- **pairwise** — Cross-signal relationships (correlation, distance, coherence, coloading) at each window. Lives at `packages/pairwise/`.
- **velocity** — System state velocity vectors from observation-level pivoted data. Lives at `packages/velocity/`.
- **baseline** — Reference region discovery and segment analysis. Lives at `packages/baseline/`.
- **breaks** — Regime change detection (CUSUM, structural breaks). Lives at `packages/breaks/`.
- **divergence** — Directional information flow (Granger causality, transfer entropy). Lives at `packages/divergence/`.
- **fleet** — Cross-cohort analysis: cohort centroids as signals through eigendecomp/pairwise/velocity. Lives at `packages/fleet/`.
- **ridge** — FTLE + velocity field convergence. Urgency = rate of approach to regime boundary. Lives at `packages/ridge/`.
- **stability** — Per-signal rolling stability (Hilbert envelope, instantaneous amplitude). Lives at `packages/stability/`.
- **thermodynamics** — Statistical mechanics analogs from eigenvalue spectra (entropy, temperature, free energy). Lives at `packages/thermodynamics/`.
- **topology** — Topological Data Analysis via persistent homology (Betti numbers, persistence diagrams). Lives at `packages/topology/`.
- **prime** — The brain. Ingest, classification, manifest, orchestration, SQL analysis, explorer. Uses all packages above.

## Two-Scale Recursive Architecture

The same mathematical machinery operates at every scale. The `vector` package (`packages/vector/`) provides the three-scale extraction:

```
Signal Level:      vector.signal.compute_signal()
  44 engines on windowed observations → signal_vector rows
  (per-signal features: kurtosis, entropy, hurst, spectral, etc.)

Cohort Level (Scale 1):      vector.cohort.compute_cohort()
  Pivot signal vectors → matrix, centroid + dispersion → cohort_vector
  Cross-signal eigendecomp → cohort_geometry.parquet
  Dynamics applied         → cohort FTLE, Lyapunov, velocity, topology, thermo

System Level (Scale 2, only when n_cohorts > 1):      vector.system.compute_system()
  Cohort matrix → system centroid + fleet dispersion → system_vector
  Cross-cohort eigendecomp → system_geometry.parquet
  Dynamics applied         → system FTLE, Lyapunov, velocity, topology, thermo
```

The recursive unit: **Vector → Geometry → Vector (enriched)**

Vector characterizes the system. Dynamics characterizes how the system changes.

When there's only one cohort, system-level computation is skipped entirely (controlled by `system.mode` in manifest: auto/force/skip).

## Pipeline flow

```
prime ~/domains/FD_004/train

  1. INGEST       raw domain files → observations.parquet + signals.parquet
                  Skipped if observations.parquet already exists (use --force-ingest to override)
                  Framework files (ground_truth, typology, signals) are NEVER ingested as raw data
  2. VALIDATE     observations → validated observations (signal_0 sequential, no nulls)
  3. TYPOLOGY     observations → typology_raw.parquet (27 measures per signal)
                  Uses primitives: hurst, perm_entropy, sample_entropy, lyapunov_rosenstein
  4. CLASSIFY     typology_raw → typology.parquet (10 classification dimensions)
                  Two-stage: discrete/sparse (PR5) FIRST, then continuous (PR4)
  5. MANIFEST     typology → output_{axis}/manifest.yaml (engine selection per signal)
                  All paths resolved to absolute at write time
                  ordering_signal recorded in manifest
  6. COMPUTE      observations + manifest → output_{axis}/system/*.parquet
                  Submits to Manifold. Prime does NOT do this computation.
  7. ANALYZE      output parquets → SQL layers + reports (DuckDB)
  8. EXPLORE      static HTML explorer (DuckDB-WASM)
```

Steps 1-5 and 7-8 are Prime. Step 6 is Manifold.

## Canonical schema

**observations.parquet** — the ONE file Prime creates from raw data:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| signal_0 | Float64 | Yes | Ordering axis values (time, depth, cycles, pressure — user's choice) |
| signal_id | String | Yes | Signal identifier |
| value | Float64 | Yes | The measurement |
| cohort | String | Optional | Cohort/unit/engine identifier (empty string if none) |

signal_0 is sorted ascending per signal. Column mapping from raw formats happens at ingest, nowhere else.

**signals.parquet** — signal metadata sidecar, written at ingest alongside observations.parquet:

| Column | Type | Description |
|--------|------|-------------|
| signal_id | String | Signal identifier (matches signal_id in observations) |
| unit | String | Unit string ("psi", "°F", "m/s", "rpm") — nullable |
| description | String | Human-readable description — nullable |
| source_name | String | Original column name from raw data before renaming |

One row per unique signal_id. Always exists after ingest, even if units are unknown.

### signal_0 Principle

signal_0 is the ordering axis. Prime puts whatever the user chose as the ordering axis into signal_0. Default is time. User can select any signal via `--order-by`. Manifold never knows or cares what signal_0 represents. Typology characterizes ALL signals identically.

Float64 is correct — preserves real spacing between observations. If signal_0 is depth (100.3, 100.7, 101.2), forcing to integers loses physics. Gap between measurements matters for derivatives and rates.

## Multi-Axis Output Directories

Each ordering axis gets its own output directory. Nothing is overwritten when re-ordering.

```
domains/FD004/
├── observations.parquet              # raw data, immutable after ingest
├── signals.parquet                   # signal metadata, immutable after ingest
├── typology.parquet                  # signal classification (intrinsic, ordering-independent)
├── typology_raw.parquet              # raw typology measures
├── ground_truth.parquet              # if present (CMAPSS RUL, etc.)
│
├── output_cycles/                    # Manifold run ordered by "cycles"
│   ├── manifest.yaml                 # manifest that produced this run
│   └── system/                       # Manifold output parquets
│       ├── state_geometry.parquet
│       ├── velocity_field.parquet
│       └── geometry_dynamics.parquet
│
├── output_s_7/                       # Re-run ordered by HPC outlet pressure
│   ├── manifest.yaml
│   └── system/
│       └── ...
```

Rules:
- Directory name: `output_{signal_id}/` where signal_id matches observations.parquet
- manifest.yaml lives INSIDE the output directory (each ordering gets its own manifest)
- Typology does NOT re-run on reorder (signal characteristics are intrinsic)
- Manifest DOES re-run (windowing depends on ordering signal's range)
- observations.parquet is NEVER modified after ingest
- No bare `output/` directory. Every output directory is named.
- `--order-by` CLI flag selects the ordering signal. Default: ingest ordering.

```bash
# Default ordering
prime ~/domains/FD004/train --run-manifold

# Explicit ordering
prime ~/domains/FD004/train --run-manifold --order-by s_7

# List existing runs
ls ~/domains/FD004/output_*/
```

## Adaptive Baseline Discovery

`prime/shared/baseline.py` — `find_stable_baseline()` discovers the most stable region in any 1D time series by sliding a window and scoring by inverse variance. No assumption about which fraction is "healthy."

Used by:
- `physics_interpreter.py` — velocity threshold derivation
- Canary system — departure detection baseline
- Any threshold-based analysis

There are NO hardcoded "first 20%" baseline assumptions. The stable region is wherever stability exists in the data. Finding it IS the analysis.

`prime/cohorts/baseline.py` is a DIFFERENT function — works on multi-column polars DataFrames of geometry metrics. Different scope, different inputs. Both exist, both are needed.

## Ingest Safety

1. **Framework file exclusion**: `_find_raw_file()` skips `observations`, `typology`, `typology_raw`, `validated`, `signals`, `ground_truth` stems. `upload.py` applies the same exclusion.
2. **Skip when exists**: If `observations.parquet` exists, ingest is skipped entirely. Use `--force-ingest` to override.
3. **Never silently destroy data**: If overwrite would reduce row count by >50%, refuse without `--force-ingest`.

## Directory structure

```
packages/
├── typology/                          # Signal classification & window sizing
│   ├── pyproject.toml                 # rudder-typology (pip install -e)
│   └── src/typology/
│       ├── classify.py                # 10-dimension classification
│       ├── config.py                  # All thresholds
│       ├── observe.py                 # 27 raw measures (pure numpy)
│       └── window.py                  # Window sizing from signal length/measures
│
├── vector/                            # Windowed feature extraction
│   ├── pyproject.toml                 # rudder-vector (pip install -e)
│   └── src/vector/
│       ├── registry.py                # YAML-driven engine discovery, lazy loading
│       ├── signal.py                  # compute_signal() — windowed engines per signal
│       ├── cohort.py                  # compute_cohort() — centroid + dispersion across signals
│       ├── system.py                  # compute_system() — centroid across cohorts
│       ├── engines/                   # 44 engine .py files (bare compute(y) → dict)
│       └── engine_configs/            # 44 YAML declarations (window reqs, output keys)
│
├── geometry/                          # Signal geometry & eigenvalue dynamics
│   ├── pyproject.toml                 # rudder-geometry (pip install -e)
│   └── src/geometry/
│       ├── signal.py                  # Per-signal geometry relative to eigenstructure
│       ├── dynamics.py                # Eigenvalue trajectory derivatives (vel, accel, jerk, curvature)
│       └── collapse.py                # Dimensional collapse detection from velocity series

prime/
├── pipeline.py                  # Main orchestrator: observations → results
├── __main__.py                  # CLI entry point (prime command)
├── cli.py                       # Query CLI (prime query)
│
├── core/
│   ├── manifold_client.py       # Orchestration wrapper (direct Python import)
│   ├── api.py                   # FastAPI server
│   ├── validation.py            # Input validation
│   └── data_reader.py           # CSV/parquet/Excel reader
│
├── ingest/
│   ├── typology_raw.py          # Raw measures per signal (uses primitives)
│   ├── validate_observations.py # Validates & repairs signal_0 sequencing
│   ├── from_raw.py              # Raw data → observations.parquet
│   ├── transform.py             # Data transformations
│   ├── signal_metadata.py       # Write signals.parquet (units, descriptions)
│   ├── data_reader.py           # CSV/parquet/Excel reader
│   ├── paths.py                 # Path resolution
│   ├── upload.py                # Upload with framework file exclusion
│   └── schema/                  # MANIFOLD_SCHEMA.yaml
│
├── shared/
│   └── baseline.py              # Adaptive baseline discovery (1D numpy arrays)
│
├── typology/
│   ├── discrete_sparse.py       # PR5: discrete/sparse detection (runs FIRST)
│   ├── level2_corrections.py    # PR4: continuous classification
│   ├── constant_detection.py    # CV-based CONSTANT detection
│   ├── classification_stability.py
│   ├── overlap_zones.py         # Classification overlap handling
│   ├── window_factor.py
│   └── tests/                   # Inline test suites
│
├── config/
│   ├── typology_config.py       # All classification thresholds (no magic numbers)
│   ├── discrete_sparse_config.py
│   ├── domains.py               # Domain mappings
│   └── recommender.py           # Engine recommendation config
│
├── manifest/
│   └── generator.py             # v2.6 manifest: engine gating, path resolution
│
├── services/
│   ├── physics_interpreter.py   # Physics analysis with adaptive baselines + dimensionless energy
│   ├── concierge.py             # LLM-powered data validation and Q&A
│   ├── dynamics_interpreter.py  # Interpret Manifold dynamics outputs
│   ├── fingerprint_service.py   # Failure fingerprint detection
│   ├── job_manager.py           # Pipeline job tracking
│   ├── state_analyzer.py        # System state analysis
│   └── tuning_service.py        # Tuning result analysis
│
├── engines/
│   ├── diagnostic_report.py     # Diagnostic assessment engine
│   ├── classification_engine.py # Signal classification engine
│   ├── signal_geometry.py       # Geometry computation helpers
│   ├── stability_engine.py      # Stability assessment
│   ├── stationarity_engine.py   # Stationarity assessment
│   ├── structure_engine.py      # Structure detection
│   ├── tipping_engine.py        # Tipping point detection
│   ├── trajectory_monitor.py    # Trajectory monitoring
│   └── typology_engine.py       # Typology computation engine
│
├── early_warning/
│   ├── failure_fingerprint_detector.py  # Failure pattern detection
│   └── ml_predictor.py                  # ML-based prediction
│
├── cohorts/
│   ├── baseline.py              # Multi-column geometry baseline (polars DataFrames)
│   ├── discovery.py             # Identify coupled/decoupled units
│   └── detection.py             # Cohort structure detection
│
├── entry_points/
│   ├── stage_01_validate.py     # CLI: validate observations
│   ├── stage_02_typology.py     # CLI: compute typology measures
│   ├── stage_03_classify.py     # CLI: apply classification
│   ├── stage_04_manifest.py     # CLI: generate manifest
│   ├── stage_05_diagnostic.py   # CLI: diagnostic assessment
│   ├── stage_06_interpret.py    # CLI: interpret Manifold outputs
│   ├── stage_07_predict.py      # CLI: predict RUL, health, anomalies
│   ├── stage_08_alert.py        # CLI: early warning / failure fingerprints
│   ├── stage_09_explore.py      # CLI: Manifold visualization
│   ├── stage_10_inspect.py      # CLI: file inspection
│   ├── stage_11_fetch.py        # CLI: read/profile raw data
│   ├── stage_12_stream.py       # CLI: real-time streaming
│   ├── stage_13_train.py        # CLI: train ML models on Manifold features
│   └── csv_to_atlas.py          # One-shot: raw file → full pipeline
│
├── sql/
│   ├── layers/                  # 33 numbered SQL layers (run in order)
│   ├── reports/                 # Independent SQL reports (25+ files)
│   ├── views/                   # Reusable SQL views
│   ├── stages/                  # Stage-specific SQL
│   ├── typology/                # SQL-based typology pipeline (7 SQL files + runner)
│   ├── docs/                    # SQL layer documentation
│   └── runner.py                # Python SQL runner (supports output_{axis}/ layout)
│
├── explorer/
│   ├── server.py                # Explorer server runner
│   └── static/
│       └── index.html           # DuckDB-WASM browser explorer
│
├── generators/
│   └── rossler.py               # Rössler attractor synthetic dataset generator
│
├── inspection/
│   ├── file_inspector.py        # File format detection and profiling
│   ├── capability_detector.py   # Dataset capability detection
│   └── results_validator.py     # Output validation
│
├── utils/
│   └── index_detection.py       # Index column detection heuristics
│
├── ml/
│   └── entry_points/
│       └── ablation.py          # Feature importance ablation
│
├── parameterization/
│   └── compile.py               # Multi-run comparison across orderings
│
├── streaming/
│   └── ...                      # Real-time WebSocket streaming (stage_12)
│
└── io/
    ├── __init__.py
    └── readme_writer.py         # Auto-generate output READMEs
```

## Typology — 10 Classification Dimensions

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

Classification order:
1. Discrete/sparse runs FIRST (PR5: `discrete_sparse.py`)
2. Continuous classification runs only for non-discrete signals (PR4: `level2_corrections.py`)
3. All thresholds in config files, no magic numbers in code

## Manifest — v2.6

The manifest tells Manifold exactly which engines to run per signal. Prime generates it from typology. Manifold executes exactly what's specified.

Key fields:
- `ordering_signal` — which signal_id was used as the ordering axis
- `paths.observations` — absolute path to observations file
- `paths.output_dir` — absolute path to output directory
- `system.window` — common window for multi-signal alignment
- `system.mode` — auto/force/skip for system-level computation
- `engine_windows` — global minimum window sizes for FFT/long-range engines
- `engine_window_overrides` — per-signal overrides when signal window < engine minimum
- `skip_signals` — CONSTANT signals skip all engines
- `pair_engines` — asymmetric pairwise (granger, transfer_entropy)
- `symmetric_pair_engines` — symmetric pairwise (correlation, mutual_info, cointegration)
- `intervention.enabled` — fault injection / event response datasets
- `intervention.event_index` — sample index where intervention occurs

All paths in the manifest are absolute. Resolved at write time by `resolve_manifest_paths()`. Validated at read time by Manifold's `validate_manifest_paths()`.

## SQL Layers

All SQL runs on DuckDB against the parquet files Manifold produces. Prime does NOT compute — it queries.

```bash
prime query ~/domains/FD004/          # Preferred — uses Python runner, supports output_{axis}/
python -m prime.sql.runner ~/domains/FD004/   # Alternative
```

`sql/reports/00_run_all.sql` is DEPRECATED (hardcoded legacy paths). Use the Python runner instead.

SQL layers are numbered and run in order. Reports are independent.

Manifold produces numbers. SQL produces answers: regime classification, baseline scoring, departure detection, canary sequencing, signal ranking, drift detection, brittleness scores.

## How Prime calls Manifold

Prime uses direct Python imports via the orchestration package. There is no HTTP microservice layer.

```python
from prime.core.manifold_client import run_manifold

result = run_manifold(
    observations_path="observations.parquet",
    manifest_path="output_time/manifest.yaml",
    output_dir="output_time/",
)
```

`manifold_client.py` is a thin wrapper around `orchestration.run()`. One import, one call. Prime doesn't know about stages, workers, or internals.

## How Prime uses primitives

```python
from pmtvs import hurst_exponent, permutation_entropy, sample_entropy, lyapunov_rosenstein

h = hurst_exponent(signal_values)  # one call per signal, full signal in, one number out
```

Prime calls primitives once per signal for typology. Manifold calls primitives thousands of times per pipeline run (once per window per signal).

## How vector works

`packages/vector/` — 44 engines, each a `.py` + `.yaml` pair. YAML declares window requirements and output keys. Python provides `compute(y) → dict`. All output keys namespaced `{engine}_{key}` — zero collisions (179 unique keys).

```python
from vector.signal import compute_signal
from vector.cohort import compute_cohort, pivot_to_matrix
from vector.system import compute_system
from vector.registry import get_registry

# Signal level — windowed features for one signal
rows = compute_signal('sensor_2', values, window_size=256, stride=64)

# Cohort level — centroid + dispersion across signals at one window
reg = get_registry()
features = [k for name in reg.engine_names for k in reg.get_outputs(name)]
matrix = pivot_to_matrix(signal_rows, features)
cohort_row = compute_cohort(matrix, 'unit_001', window_index=0, feature_names=features)

# System level — centroid across cohorts (same math, one level up)
system_row = compute_system(cohort_matrix, window_index=0, feature_names=features)
```

Key design:
- **No BaseEngine class** — engines are bare `compute(y) → dict` functions
- **Registry does lazy loading** — engines only imported when first called, YAML discovery at init
- **Engines are pure glue** — import from pmtvs, call, namespace, return. Zero inline math. If pmtvs import fails, return NaN dict.
- **Three scales, same pattern** — signal (windowed engines), cohort (centroid across signals), system (centroid across cohorts)

## How geometry works

`packages/geometry/` — connects eigendecomp (the shape) to prediction (the trajectory).

```python
from geometry import compute_signal_geometry, compute_eigenvalue_dynamics, detect_collapse

# Per-signal geometry at one window
rows = compute_signal_geometry(
    signal_matrix,       # (n_signals, n_features)
    signal_ids,          # ['s_1', 's_2', ...]
    centroid,            # (n_features,) from cohort centroid
    principal_components=pcs,  # (n_pcs, n_features) from eigendecomp
    window_index=0,
)
# → list of dicts with: distance, coherence, contribution, residual, signal_magnitude, pc_projections

# Eigenvalue dynamics across windows
dynamics = compute_eigenvalue_dynamics(
    eigendecomp_results,  # list of dicts with effective_dim, eigenvalues, total_variance
    smooth_window=3,
)
# → list of dicts with: effective_dim_velocity, acceleration, jerk, curvature, per-eigenvalue velocity

# Collapse detection
collapse = detect_collapse(
    dynamics[i]['effective_dim_velocity'],  # or full velocity array
    threshold_velocity=-0.05,
    min_run_length=3,
)
# → dict with: collapse_detected, collapse_onset_idx, collapse_onset_fraction, max_run_length
```

Three modules:
- **signal.py** — where is each signal relative to the eigenstructure? Distance to centroid, coherence to PC1, contribution (projection), residual (orthogonal). Handles NaN signals, batch processing across windows.
- **dynamics.py** — how is the geometry changing? Central-difference derivatives of effective_dim and eigenvalues. Velocity, acceleration, jerk, curvature. Optional smoothing.
- **collapse.py** — is the system losing degrees of freedom? Finds sustained runs of negative velocity in effective_dim. Returns onset index and fraction. Does NOT classify or interpret — Prime's SQL does that.

## Physics Interpreter

`prime/services/physics_interpreter.py` uses adaptive baselines and dimensionally correct energy proxies.

- **Energy proxy**: `(y/σ_y)² + (dy/σ_dy)²` — both terms normalized to dimensionless before combining. Never `y² + dy²` (incompatible units).
- **Velocity thresholds**: Derived from discovered stable baseline via `find_stable_baseline()`, not hardcoded. `vel_threshold = baseline_mean + 3σ`.
- **Coherence thresholds**: Unchanged at 0.5 — already dimensionless (ratio between 0 and 1).
- **Risk score weights**: Unchanged — relative scoring, unit-independent.

## Test Datasets

Test datasets live OUTSIDE the repo at `~/domains/testing/`:

```
~/domains/testing/
├── rossler/                    # 3 signals, 1 cohort, continuous chaotic dynamics
├── logistic_ensemble/          # 1 signal, 5 cohorts, discrete maps at different chaos levels
└── coupled_oscillators/        # 4 signals, 2 cohorts, known causal coupling for validation
```

Each includes `observations.parquet`, `signals.parquet`, and `ground_truth.parquet` (where applicable).

## Env vars

| Variable | Controls | Default |
|----------|----------|---------|
| `PRIME_TYPOLOGY` | Typology backend: `sql`, `python`, or `compare` | `sql` |
| `PRIME_OUTPUT_DIR` | Default output directory | `data` |

## Rules

1. **Prime classifies. Manifold computes.** Never put computation in Prime. Never put classification in Manifold. Signal geometry lives in `packages/geometry/`.
2. **Primitives is pure math.** numpy in, number out. No file I/O, no config, no domain knowledge.
3. **observations.parquet is the contract.** Everything downstream depends on this schema. signal_0, signal_id, value, cohort.
4. **Manifest is the spec.** Manifold executes exactly what the manifest says. No interpretation, no overrides. All paths absolute.
5. **SQL layers do not compute.** They query parquets that Manifold already wrote. Read-only.
6. **Thresholds live in config files.** No magic numbers in classification code. All in `config/typology_config.py` and `config/discrete_sparse_config.py`.
7. **Domain knowledge lives in ingest only.** CMAPSS column mappings, IMS file structure, battery CSV formats — all in `ingest/transform`. Nothing downstream knows what domain it's processing.
8. **signal_0 is just a column name.** Manifold never interprets it. Typology treats it like any other signal.
9. **Same function at every scale.** `vector.cohort.compute_cohort()` and `vector.system.compute_system()` use identical math. If you write scale-specific logic, you're doing it wrong.
10. **Baselines are discovered, not assumed.** No hardcoded "first 20%" anywhere. Use `find_stable_baseline()` from `prime/shared/baseline.py`.
11. **Ingest never destroys data.** If observations.parquet exists, ingest is skipped. Framework files are never globbed as raw data. Use `--force-ingest` to override.
12. **One output directory per ordering axis.** `output_{signal_id}/`. Nothing is overwritten on reorder.

## Naming

- GitHub org: `rudder-framework`
- Packages: `prime`, `manifold`, `primitives` (pmtvs on PyPI), `vector` (rudder-vector, local), `geometry` (rudder-geometry, local), `typology` (rudder-typology, local)
- pmtvs GitHub: `pmtvs` user, repos `pmtvs-core` (private) and `pmtvs-pip` (public)
- No branding in code — no "rudder" in imports, class names, function names, variables, docstrings, SQL, or API routes
- No "PRISM" anywhere (old name for Manifold, retired)
- "rudder" in LICENSE.md (copyright holder) and pyproject.toml (author) is fine

## Do NOT

- Add computation to Prime. If it's math, it goes in primitives, vector, geometry, or Manifold.
- Create engines outside `packages/vector/engines/`. All windowed feature extraction lives there.
- Create signal geometry or eigenvalue dynamics outside `packages/geometry/`.
- Add classification to Manifold. If it's a decision, it goes in Prime.
- Run Manifold directly. Users run Prime. Prime calls Manifold.
- Put domain-specific code outside of ingest. The pipeline is domain-agnostic.
- Use `rudder` or `PRISM` in new code.
- Guess file paths in Manifold. Prime tells Manifold exactly where files are. All manifest paths absolute.
- Put thresholds in code. They go in config files.
- Modify the observations schema. It's signal_0, signal_id, value, cohort.
- Create files in `/tmp/` or `~/`. All work inside repo structure.
- Add patent, trademark, or commercial licensing language anywhere.
- Hardcode "first 20%" or any fixed fraction as a baseline. Use `find_stable_baseline()`.
- Glob `*.parquet` without excluding framework files. See Rule 5.
- Create a bare `output/` directory. Always `output_{signal_id}/`.
- Put relative paths in the manifest. Always absolute.

## Experiment Lifecycle — Tagging & Vault

After any Prime pipeline run that produces output worth keeping:

### 1. Git Tag

Format: `{domain}-{dataset}-{description}`

Examples:
- `cmapss-fd004-regime-normalization`
- `cmapss-fd004-prime-rerun`
- `femto-train-first-run`
- `femto-test-first-run`

```bash
git add .
git commit -m "descriptive message"
git tag {tag-name}
git push --tags
```

### 2. Vault Save (Immutable Archive)

Location: `~/tools/vault.py`

```bash
python ~/tools/vault.py save \
  --tag {same-tag-as-git} \
  --files {key-output-files} \
  --dirs {output-directories}
```

Example after a Prime run on FD004:
```bash
python ~/tools/vault.py save \
  --tag cmapss-fd004-regime-normalization \
  --dirs ~/domains/cmapss/FD_004/Train/output_time/ml/
```

### When to Tag + Vault

- After any Prime pipeline run that completes successfully
- After adding new pipeline stages (regime detection, normalization)
- After FEMTO or other cross-domain runs
- Before and after architectural changes

### Vault is Immutable

Once saved, vault entries cannot be overwritten. Git tags can be moved.
The vault is the audit trail. Never skip the vault step.
