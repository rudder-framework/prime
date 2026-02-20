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

### Rule 4: PRIME CLASSIFIES. MANIFOLD COMPUTES.

- Do NOT create computation logic in Prime (eigendecompositions, FTLE, Lyapunov, etc.)
- Do NOT create classification logic in Manifold (signal types, regime labels, etc.)
- Prime's only computation is typology (27 raw measures per signal via primitives)

## Architecture — Three repos, clean dependency tree

```
primitives (pmtvs)     ← Rust+Python math functions (leaf dependency, on PyPI)
     ↑        ↑
     |        |
  Prime    Manifold    ← Prime orchestrates, Manifold computes
     |        ↑
     └────────┘        ← Prime calls Manifold via HTTP
```

- **primitives (pmtvs)** — `from primitives import hurst_exponent`. Pure functions. numpy in, scalar out. 281 Rust-accelerated functions. Handles Rust/Python toggle via `USE_RUST` env var.
- **manifold** — Compute engine. 29 pipeline stages in 5 groups. Receives observations.parquet + manifest.yaml, writes output parquets. Never run directly by users.
- **prime** — The brain. Ingest, typology, classification, manifest, orchestration, SQL analysis, explorer.

## Two-Scale Recursive Architecture

The same mathematical machinery operates at every scale:

```
Signal Level:
  Engines on raw observations → signal_vector.parquet
  (per-signal features: kurtosis, entropy, hurst, spectral, etc.)

Cohort Level (Scale 1):
  Cross-signal eigendecomp → cohort_geometry.parquet
  Centroid + transforms    → cohort_vector.parquet
  Dynamics applied         → cohort FTLE, Lyapunov, velocity, topology, thermo

System Level (Scale 2, only when n_cohorts > 1):
  Cross-cohort eigendecomp → system_geometry.parquet
  Centroid + transforms    → system_vector.parquet  (SAME function as cohort_vector)
  Dynamics applied         → system FTLE, Lyapunov, velocity, topology, thermo
```

The recursive unit: **Vector → Geometry → Vector (enriched)**

Vector characterizes the system. Dynamics characterizes how the system changes.

When there's only one cohort, system-level computation is skipped entirely (controlled by `system.mode` in manifest: auto/force/skip).

## Pipeline flow

```
prime ~/domains/FD_004/train

  1. INGEST       raw domain files → observations.parquet
  2. VALIDATE     observations → validated observations (I sequential, no nulls)
  3. TYPOLOGY     observations → typology_raw.parquet (27 measures per signal)
                  Uses primitives: hurst, perm_entropy, sample_entropy, lyapunov_rosenstein
  4. CLASSIFY     typology_raw → typology.parquet (10 classification dimensions)
                  Two-stage: discrete/sparse (PR5) FIRST, then continuous (PR4)
  5. MANIFEST     typology → manifest.yaml (engine selection per signal)
  6. COMPUTE      observations + manifest → output/*.parquet
                  Submits to Manifold via HTTP. Prime does NOT do this computation.
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

### signal_0 Principle

signal_0 is just the index column name. Prime puts whatever the user chose as the ordering axis into signal_0. Default is time. User can select any signal. Manifold never knows or cares what signal_0 represents. Typology characterizes ALL signals identically.

## Directory structure

```
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
│   └── parameterization.py      # Natural parameterization / axis selection
│
├── cohorts/
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
│   └── run_all.sql
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
- `system.window` — common window for multi-signal alignment
- `system.mode` — auto/force/skip for system-level computation
- `engine_windows` — global minimum window sizes for FFT/long-range engines
- `engine_window_overrides` — per-signal overrides when signal window < engine minimum
- `skip_signals` — CONSTANT signals skip all engines
- `pair_engines` — asymmetric pairwise (granger, transfer_entropy)
- `symmetric_pair_engines` — symmetric pairwise (correlation, mutual_info, cointegration)
- `intervention.enabled` — fault injection / event response datasets
- `intervention.event_index` — sample index where intervention occurs

## SQL Layers

All SQL runs on DuckDB against the parquet files Manifold produces. Prime does NOT compute — it queries.

```bash
duckdb < prime/sql/run_all.sql         # Run all layers
duckdb < prime/sql/reports/03_drift_detection.sql  # Specific report
```

SQL layers are numbered and run in order. Reports are independent.

Manifold produces numbers. SQL produces answers: regime classification, baseline scoring, departure detection, canary sequencing, signal ranking, drift detection, brittleness scores.

## How Prime calls Manifold

```python
from prime.core.manifold_client import ManifoldClient

client = ManifoldClient()  # defaults to MANIFOLD_URL=http://localhost:8100
client.health()

job = client.submit_manifest(
    manifest_path="manifest.yaml",
    observations_path="observations.parquet",
)
status = client.get_job_status(job["job_id"])
client.fetch_all_outputs(job["job_id"], output_dir="output/")
```

HTTP only. No Manifold imports. No shared code.

## How Prime uses primitives

```python
from primitives import hurst_exponent, permutation_entropy, sample_entropy, lyapunov_rosenstein

h = hurst_exponent(signal_values)  # one call per signal, full signal in, one number out
```

Prime calls primitives once per signal for typology. Manifold calls primitives thousands of times per pipeline run (once per window per signal).

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
3. **observations.parquet is the contract.** Everything downstream depends on this schema. I is sequential. Always.
4. **Manifest is the spec.** Manifold executes exactly what the manifest says. No interpretation, no overrides.
5. **SQL layers do not compute.** They query parquets that Manifold already wrote. Read-only.
6. **Thresholds live in config files.** No magic numbers in classification code. All in `config/typology_config.py` and `config/discrete_sparse_config.py`.
7. **Domain knowledge lives in ingest only.** CMAPSS column mappings, IMS file structure, battery CSV formats — all in `ingest/transform`. Nothing downstream knows what domain it's processing.
8. **signal_0 is just a column name.** Manifold never interprets it. Typology treats it like any other signal.
9. **Same function at every scale.** cohort_vector and system_vector use identical computation. If you write scale-specific logic, you're doing it wrong.

## Naming

- GitHub org: `rudder-framework`
- Packages: `prime`, `manifold`, `primitives` (pmtvs on PyPI)
- No branding in code — no "rudder" in imports, class names, function names, variables, docstrings, SQL, or API routes
- No "PRISM" anywhere (old name for Manifold, retired)
- No "ORTHON" in code (old name, retired — the classification architecture it described is now just "Prime's SQL layer")
- "rudder" in LICENSE.md (copyright holder) and pyproject.toml (author) is fine

## Do NOT

- Add computation to Prime. If it's math, it goes in primitives or Manifold.
- Add classification to Manifold. If it's a decision, it goes in Prime.
- Run Manifold directly. Users run Prime. Prime calls Manifold.
- Put domain-specific code outside of ingest. The pipeline is domain-agnostic.
- Use `rudder`, `PRISM`, or `ORTHON` in new code.
- Guess file paths in Manifold. Prime tells Manifold exactly where files are.
- Put thresholds in code. They go in config files.
- Modify the observations schema. It's cohort, signal_id, I, value.
- Create files in `/tmp/` or `~/`. All work inside repo structure.
- Add patent, trademark, or commercial licensing language anywhere.
