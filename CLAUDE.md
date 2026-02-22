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

### Rule 4: PRIME CLASSIFIES. MANIFOLD COMPUTES. VECTOR EXTRACTS.

- Do NOT create computation logic in Prime (eigendecompositions, FTLE, Lyapunov, etc.)
- Do NOT create classification logic in Manifold (signal types, regime labels, etc.)
- Do NOT create new engines in Prime — windowed feature extraction lives in `packages/vector/`
- Prime's only computation is typology (27 raw measures per signal via primitives)

### Rule 5: DO NOT GLOB FRAMEWORK FILES

Ingest must NEVER treat framework files as raw data. These stems are reserved and must be excluded from any file discovery:

```python
FRAMEWORK_STEMS = {'observations', 'typology', 'typology_raw', 'validated', 'signals', 'ground_truth'}
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
  /    \      ↑
 /      \     |
typology vector        ← Local packages under packages/ (editable installs)
     |        ↑
     └────────┘        ← Prime calls Manifold via HTTP
```

- **primitives (pmtvs)** — `from pmtvs import hurst_exponent`. Pure functions. numpy in, scalar out. Rust-accelerated functions. Published on PyPI. Two repos under `pmtvs` GitHub user: `pmtvs-core` (private) and `pmtvs-pip` (public/PyPI).
- **vector** — Windowed feature extraction. 44 engines, 179 output keys, three scales (signal, cohort, system). Lives at `packages/vector/`, installed editable. `from vector.signal import compute_signal`. Engines import from pmtvs with inline fallbacks.
- **typology** — Signal classification and window sizing. Lives at `packages/typology/`, installed editable. `from typology import from_observations`.
- **manifold** — Compute engine. 29 pipeline stages in 5 groups. Receives observations.parquet + manifest.yaml, writes output parquets into `output_{axis}/system/`. Never run directly by users.
- **prime** — The brain. Ingest, classification, manifest, orchestration, SQL analysis, explorer. Uses typology and vector packages.

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

prime/
├── core/
│   ├── pipeline.py              # Main orchestrator: observations → results
│   ├── manifold_client.py       # HTTP client for Manifold API (httpx)
│   ├── api.py                   # FastAPI server
│   ├── server.py                # Server runner
│   ├── validation.py            # Input validation
│   └── data_reader.py           # CSV/parquet/Excel reader
│
├── ingest/
│   ├── typology_raw.py          # 27 raw measures per signal (uses primitives)
│   ├── validate_observations.py # Validates & repairs signal_0 sequencing
│   ├── transform.py             # Raw data → observations.parquet
│   ├── signal_metadata.py       # Write signals.parquet (units, descriptions)
│   ├── data_reader.py           # CSV/parquet/Excel reader
│   ├── paths.py                 # Path resolution
│   ├── streaming.py             # Streaming ingest
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
│   ├── generator.py             # v2.6 manifest: engine gating, path resolution
│   ├── characteristic_time.py   # Data-driven window from ACF, frequency, period
│   ├── system_window.py         # Multi-scale representation
│   └── parameterization.py      # Natural parameterization / axis selection
│
├── services/
│   └── physics_interpreter.py   # Physics analysis with adaptive baselines + dimensionless energy
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
│   ├── layers/                  # Numbered SQL layers (run in order)
│   │   ├── 00_observations.sql
│   │   ├── 02_signal_vector.sql
│   │   ├── 05_manifold_derived.sql
│   │   └── typology_v2.sql
│   ├── reports/                 # Independent SQL reports
│   │   └── 00_run_all.sql       # DEPRECATED — use `prime query` or python -m prime.sql.runner
│   └── runner.py                # Python SQL runner (supports output_{axis}/ layout)
│
├── analysis/
│   ├── window_optimization.py         # Option A: raw eigendecomp grid search
│   ├── window_optimization_manifold.py # Option B: full Manifold grid search
│   ├── twenty_twenty.py               # 20/20 predictive validation
│   └── canary_detection.py            # Early warning signal detection
│
├── explorer/
│   └── static/
│       └── index.html           # DuckDB-WASM browser explorer
│
├── ml/
│   └── entry_points/
│       └── ablation.py          # Feature importance ablation
│
├── parameterization/
│   └── compile.py               # Multi-run comparison across orderings
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

```python
from prime.core.manifold_client import ManifoldClient

client = ManifoldClient()  # defaults to MANIFOLD_URL=http://localhost:8100
client.health()

job = client.submit_manifest(
    manifest_path="output_time/manifest.yaml",
    observations_path="observations.parquet",
)
status = client.get_job_status(job["job_id"])
client.fetch_all_outputs(job["job_id"], output_dir="output_time/")
```

HTTP only. No Manifold imports. No shared code.

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
- **Engines import from pmtvs** with inline fallbacks if unavailable
- **Three scales, same pattern** — signal (windowed engines), cohort (centroid across signals), system (centroid across cohorts)

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
| `USE_RUST` | Rust vs Python for primitives | `1` (Rust) |
| `PRIME_WORKERS` | Parallel signals in typology | `1` (single-threaded) |
| `MANIFOLD_URL` | Manifold HTTP endpoint | `http://localhost:8100` |
| `PRIME_URL` | Prime callback URL for Manifold | `http://localhost:8000` |
| `MANIFOLD_WORKERS` | Parallel cohorts (Manifold's concern) | `0` (auto) |

## Rules

1. **Prime classifies. Manifold computes.** Never put computation in Prime. Never put classification in Manifold.
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
- Packages: `prime`, `manifold`, `primitives` (pmtvs on PyPI), `vector` (rudder-vector, local), `typology` (rudder-typology, local)
- pmtvs GitHub: `pmtvs` user, repos `pmtvs-core` (private) and `pmtvs-pip` (public)
- No branding in code — no "rudder" in imports, class names, function names, variables, docstrings, SQL, or API routes
- No "PRISM" anywhere (old name for Manifold, retired)
- No "ORTHON" in code (old name, retired — the classification architecture it described is now just "Prime's SQL layer")
- "rudder" in LICENSE.md (copyright holder) and pyproject.toml (author) is fine

## Do NOT

- Add computation to Prime. If it's math, it goes in primitives, vector, or Manifold.
- Create engines outside `packages/vector/engines/`. All windowed feature extraction lives there.
- Add classification to Manifold. If it's a decision, it goes in Prime.
- Run Manifold directly. Users run Prime. Prime calls Manifold.
- Put domain-specific code outside of ingest. The pipeline is domain-agnostic.
- Use `rudder`, `PRISM`, or `ORTHON` in new code.
- Guess file paths in Manifold. Prime tells Manifold exactly where files are. All manifest paths absolute.
- Put thresholds in code. They go in config files.
- Modify the observations schema. It's signal_0, signal_id, value, cohort.
- Create files in `/tmp/` or `~/`. All work inside repo structure.
- Add patent, trademark, or commercial licensing language anywhere.
- Hardcode "first 20%" or any fixed fraction as a baseline. Use `find_stable_baseline()`.
- Glob `*.parquet` without excluding framework files. See Rule 5.
- Create a bare `output/` directory. Always `output_{signal_id}/`.
- Put relative paths in the manifest. Always absolute.
