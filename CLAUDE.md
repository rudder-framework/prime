# ORTHON - AI Instructions

## Architecture

```
ORTHON = Brain (orchestration, typology, classification, interpretation)
PRISM  = Muscle (pure computation, no decisions, no classification)

PRISM computes numbers. ORTHON classifies.

ORTHON creates: observations.parquet + typology.parquet
PRISM reads: observations.parquet + typology.parquet
PRISM runs: signal_vector → state_vector → geometry → dynamics
ORTHON runs: classification SQL views on PRISM outputs

Typology is the ONLY statistical analysis in ORTHON.
ORTHON classifies signals; PRISM computes features.
New architecture (v2.1): Typology-guided, scale-invariant, eigenvalue-based
```

## The One Rule

```
observations.parquet and manifest.yaml ALWAYS go to:
$PRISM_DATA_DIR (default: ~/prism/data/)

NO EXCEPTIONS. No subdirectories. No domain folders.
```

## PRISM Format (observations.parquet) - v2.1

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| cohort | String | Optional | Which cohort/unit (engine_1, pump_A, bearing_3) - groups related signals |
| signal_id | String | Required | Which signal (temp, pressure, return) |
| I | UInt32 | Required | Observation index within unit+signal |
| value | Float64 | Required | The measurement |

**Note:** `cohort` replaces legacy `unit_id`/`entity_id`. Each cohort is a group of related signals (e.g., all sensors on one engine).

---

## Observations Validation

**CRITICAL:** Before running PRISM, ALWAYS validate observations.parquet.

### The I Column

I is the canonical index. NOT a timestamp. Must be sequential per signal_id.

```
CORRECT:
signal_id | I | value
----------|---|------
temp      | 0 | 45.2
temp      | 1 | 45.4
temp      | 2 | 45.6
pressure  | 0 | 101.3
pressure  | 1 | 101.5

WRONG (timestamps):
signal_id | I          | value
----------|------------|------
temp      | 1596760568 | 45.2   <- Unix timestamp, NOT sequential
temp      | 1596760569 | 45.4

WRONG (duplicates):
signal_id | I | value
----------|---|------
temp      | 1 | 45.2
temp      | 1 | 45.4   <- Duplicate I for same signal
```

### Auto-Repair

When user drops a file, ALWAYS run validation:
```python
from orthon.ingest.validate_observations import validate_and_save, ValidationStatus

# This will auto-repair common issues:
# - Timestamps in I -> regenerated as sequential
# - Column name aliases -> renamed to standard
# - Null signal_ids -> rows removed
# - Non-numeric values -> cast to Float64

result = validate_and_save("user_upload.parquet", "observations.parquet")

if result.status == ValidationStatus.FAILED:
    print(f"Cannot process file: {result.issues}")
```

### Common Issues & Fixes

| Issue | Detection | Auto-Fix |
|-------|-----------|----------|
| I contains timestamps | `I.max() > n_rows * 10` | Sort by I, regenerate as 0,1,2... |
| Duplicate (signal_id, I) | Group count > 1 | Sort and regenerate I |
| Missing I column | Column not present | Create from row order |
| Column named 'timestamp' | Alias detection | Rename to 'I' |
| Column named 'y' | Alias detection | Rename to 'value' |
| Null signal_id | Null count > 0 | Remove rows |

### Validation CLI
```bash
# Validate only (no changes)
python -m orthon.ingest.validate_observations --check data/observations.parquet

# Validate and repair (overwrites)
python -m orthon.ingest.validate_observations data/observations.parquet

# Validate, repair, save to new file
python -m orthon.ingest.validate_observations input.parquet output.parquet
```

---

## Classification SQL (orthon/sql/classification.sql)

**PRISM computes numbers. ORTHON classifies.**

All classification logic lives in ORTHON, not PRISM. PRISM outputs raw metrics (Lyapunov, effective_dim, etc.). ORTHON applies thresholds and labels.

### Lyapunov-Based Trajectory Classification

The gold standard for chaos detection is the Lyapunov exponent (λ), which measures sensitive dependence on initial conditions.

| λ Range | Classification | Meaning |
|---------|---------------|---------|
| λ > 0.1 | `chaotic` | Exponential divergence of nearby trajectories |
| λ > 0.01 | `quasi_periodic` | Edge of chaos, weak sensitivity |
| λ > -0.01 | `oscillating` | Limit cycle behavior |
| λ > -0.1 | `converging` | Damped oscillation |
| λ < -0.1 | `stable` | Fixed point attractor |

**DO NOT** use coefficient of variation (CV) for chaos detection. CV measures variability, not sensitive dependence.

### Classification Views

| View | Purpose |
|------|---------|
| `v_trajectory_type` | Lyapunov-based: chaotic/quasi_periodic/oscillating/converging/stable |
| `v_stability_class` | stable/marginally_stable/unstable with numeric score |
| `v_collapse_status` | collapsing/expanding/stable/drifting + lifecycle stage |
| `v_signal_classification` | Signal morphology, periodicity, tails, memory |
| `v_anomaly_severity` | critical/severe/moderate/mild/normal based on z-score |
| `v_coupling_strength` | strongly/moderately/weakly/uncoupled based on correlation |
| `v_system_health` | Unified health_score (0-1) and risk_level |
| `v_health_summary` | Aggregated status per unit |

### Collapse Detection

Collapse = sustained loss of degrees of freedom (negative effective_dim velocity).

| effective_dim_velocity | Status |
|------------------------|--------|
| < -0.1 | `collapsing` |
| > 0.1 | `expanding` |
| abs < 0.01 | `stable` |
| else | `drifting` |

### Lyapunov Data Requirements

For reliable Lyapunov estimation (Rosenstein algorithm):
- Minimum: ~1000 points (marginal significance)
- Recommended: 10^(d+1) points for embedding dimension d
- For d=4: need 100,000 points for high confidence

When Lyapunov unavailable, fallback to derivative-based classification.

---

## ORTHON Structure

```
~/orthon/
├── CLAUDE.md
├── orthon/
│   ├── __init__.py
│   │
│   ├── config/                    # Configuration
│   │   ├── manifest.py            # ENGINES list (53), Pydantic models
│   │   ├── domains.py             # Physics domains (7)
│   │   └── recommender.py         # Config recommendations
│   │
│   ├── ingest/                    # Data ingestion
│   │   ├── paths.py               # FIXED output paths
│   │   ├── streaming.py           # Universal streaming ingestor
│   │   ├── manifest_generator.py  # **Generates manifest from typology**
│   │   ├── typology_raw.py        # **Computes raw typology measures**
│   │   ├── schema_enforcer.py     # Schema validation (v2.1)
│   │   └── validate_observations.py # Validates & repairs observations
│   │
│   ├── corrections/               # Classification corrections (PR3)
│   │   ├── level1_corrections.py  # Window, stride corrections
│   │   ├── level2_corrections.py  # Temporal/spectral corrections
│   │   ├── manifest_corrections.py# Manifest stride fixes
│   │   └── tests/                 # 46 correction tests
│   │
│   ├── prediction/                # Predictive models
│   │   ├── health.py              # Health prediction
│   │   ├── anomaly.py             # Anomaly detection
│   │   ├── rul.py                 # RUL prediction
│   │   └── models/                # Pretrained models
│   │
│   ├── sql/                       # SQL classification
│   │   ├── typology.sql           # Typology classification views
│   │   └── classification.sql     # Lyapunov-based trajectory views
│   │
│   ├── intake/                    # UI file handling
│   │   ├── upload.py              # File upload
│   │   ├── validate.py            # Validation
│   │   └── transformer.py         # Transform to PRISM format
│   │
│   ├── analysis/                  # Analysis tools
│   │   ├── baseline_discovery.py  # Baseline modes
│   │   └── cohort_detection.py    # Cohort detection
│   │
│   ├── services/                  # Core services
│   │   ├── manifest_builder.py    # Build manifests
│   │   ├── job_manager.py         # Job management
│   │   ├── compute_pipeline.py    # Pipeline orchestration
│   │   ├── physics_interpreter.py # Physics interpretation
│   │   ├── dynamics_interpreter.py# Dynamics interpretation
│   │   ├── state_analyzer.py      # State analysis
│   │   ├── fingerprint_service.py # Fingerprinting
│   │   ├── tuning_service.py      # Parameter tuning
│   │   └── concierge.py           # AI concierge
│   │
│   ├── shared/                    # Shared utilities
│   │   ├── config_schema.py       # Config schemas
│   │   ├── engine_registry.py     # Engine registry
│   │   ├── physics_constants.py   # Physics constants
│   │   └── window_config.py       # Window configuration
│   │
│   ├── utils/                     # Utilities
│   │   └── index_detection.py     # Index column detection
│   │
│   ├── backend/                   # Backend connectors
│   │   ├── bridge.py              # PRISM bridge
│   │   └── fallback.py            # Fallback handlers
│   │
│   ├── inspection/                # Data inspection
│   │   ├── file_inspector.py
│   │   ├── capability_detector.py
│   │   └── results_validator.py
│   │
│   ├── explorer/                  # Data explorer
│   │   ├── loader.py
│   │   ├── renderer.py
│   │   ├── models.py
│   │   └── cli.py
│   │
│   ├── ml/                        # ML features
│   │   ├── discovery.py
│   │   ├── feature_export.py
│   │   └── create_features_parquet.py
│   │
│   ├── db/                        # Database
│   │   ├── connection.py
│   │   └── schema.py
│   │
│   ├── views/                     # UI views
│   │   └── views.py
│   │
│   ├── api.py                     # FastAPI endpoints
│   ├── server.py                  # Server
│   ├── cli.py                     # CLI
│   ├── pipeline.py                # **Pipeline orchestration**
│   ├── validation.py              # **Validation utilities**
│   ├── cohort_discovery.py        # **Cohort discovery**
│   ├── window_recommender.py      # **Window size recommendation**
│   ├── concierge.py               # AI concierge (main)
│   ├── prism_client.py            # PRISM HTTP client
│   └── data_reader.py             # Data reading utilities
│
├── domains/                       # Domain templates
├── data/                          # Benchmark data
├── fetchers/                      # Data fetchers
├── sql_reports/                   # SQL report templates
└── ml/                            # ML experiments
```

---

## Key Files

| File | Purpose |
|------|---------|
| `orthon/ingest/typology_raw.py` | **Computes raw typology measures** (27 metrics) |
| `orthon/ingest/manifest_generator.py` | **Creates v2.1 manifest** with nested cohorts |
| `orthon/corrections/level2_corrections.py` | **Classification corrections** (6 periodicity gates) |
| `orthon/sql/classification.sql` | Lyapunov-based classification views (on PRISM outputs) |
| `orthon/ingest/validate_observations.py` | Validates & repairs observations.parquet |
| `orthon/ingest/schema_enforcer.py` | Enforces v2.1 schema |
| `orthon/ingest/paths.py` | Fixed output paths (NO EXCEPTIONS) |
| `orthon/config/manifest.py` | ENGINES list, Pydantic models |
| `orthon/config/domains.py` | 7 physics domains |
| `orthon/pipeline.py` | Orchestrates observations → typology → manifest |
| `orthon/analysis/baseline_discovery.py` | Baseline modes |
| `orthon/prism_client.py` | HTTP client for PRISM |

---

## Typology (ORTHON's Responsibility)

**Typology is ORTHON's signal classification system.** It computes 27 statistical measures per signal and classifies across 10 dimensions. PRISM then uses these classifications for engine selection.

### typology_raw.parquet Schema (27 raw measures)

| Measure | Description |
|---------|-------------|
| dominant_frequency | FFT peak frequency |
| spectral_peak_snr | Peak SNR in dB |
| spectral_flatness | Spectrum flatness (0=peaked, 1=flat) |
| spectral_slope | Power law slope |
| acf_half_life | ACF decay to 0.5 |
| turning_point_ratio | Fraction of local extrema |
| hurst | Hurst exponent |
| perm_entropy | Permutation entropy |
| sample_entropy | Sample entropy |
| lyapunov_proxy | Sensitivity proxy |
| ... | (17 more measures) |

### typology.parquet Schema (10 classification dimensions)

| Dimension | Values |
|-----------|--------|
| temporal_pattern | PERIODIC, TRENDING, RANDOM, CHAOTIC, QUASI_PERIODIC, STATIONARY, CONSTANT, BINARY, DISCRETE, IMPULSIVE, EVENT |
| spectral | HARMONIC, NARROWBAND, BROADBAND, RED_NOISE, BLUE_NOISE, NONE, SWITCHING, QUANTIZED, SPARSE |
| stationarity | STATIONARY, NON_STATIONARY |
| memory | SHORT_MEMORY, LONG_MEMORY |
| complexity | LOW, MEDIUM, HIGH |
| continuity | CONTINUOUS, DISCRETE |
| determinism | DETERMINISTIC, STOCHASTIC |
| distribution | GAUSSIAN, LIGHT_TAILED, HEAVY_TAILED |
| amplitude | SMOOTH, BURSTY, MIXED |
| volatility | HOMOSCEDASTIC, VOLATILITY_CLUSTERING |

### Two-Stage Classification (PR5 → PR4)

**Stage 1: Discrete/Sparse Detection (PR5)**
Runs FIRST - catches non-continuous signals before continuous classification.

| Type | Detection | Spectral |
|------|-----------|----------|
| CONSTANT | signal_std ≈ 0 OR unique_ratio < 0.001 | NONE |
| BINARY | exactly 2 unique values | SWITCHING |
| DISCRETE | is_integer AND unique_ratio < 5% | QUANTIZED |
| IMPULSIVE | kurtosis > 20 AND crest_factor > 10 | BROADBAND |
| EVENT | sparsity > 80% AND kurtosis > 10 | SPARSE |

**Stage 2: Continuous Classification (PR4)**
If not discrete/sparse, apply continuous decision tree.

```
1. bounded_deterministic? → skip TRENDING (smooth chaos)
   hurst > 0.95 AND perm_entropy < 0.5 AND variance_ratio < 3.0
2. segment_trend? → TRENDING (oscillating trends like battery)
   monotonic segment means AND change > 20% AND hurst > 0.60
3. hurst >= 0.99? → TRENDING
4. spectral_override? → PERIODIC (noisy periodic)
   SNR > 30 dB AND flatness < 0.1
5. is_genuine_periodic? → PERIODIC (all 6 gates pass)
6. spectral_flatness > 0.9 AND perm_entropy > 0.99? → RANDOM
7. lyapunov > 0.5 AND perm_entropy > 0.95? → CHAOTIC
8. turning_point_ratio < 0.7? → QUASI_PERIODIC
9. default → STATIONARY
```

### Workflow
```
1. ORTHON computes typology_raw.parquet (29 measures per signal)
2. ORTHON applies discrete/sparse classification (PR5)
3. ORTHON applies continuous classification if needed (PR4)
4. ORTHON creates typology.parquet (10 classification dimensions)
5. ORTHON generates manifest.yaml (engine selection per signal)
6. PRISM reads manifest and executes engines
7. ORTHON classifies PRISM outputs (Lyapunov → trajectory type)
```

### Config Files
- `orthon/config/typology_config.py` - PR4 continuous classification thresholds
- `orthon/config/discrete_sparse_config.py` - PR5 discrete/sparse thresholds

---

## Unified Manifest System (v2.1)

ORTHON decides. PRISM executes.

### Workflow
```
1. ORTHON creates typology.parquet from observations
2. ORTHON generates manifest.yaml with per-signal engine selection
3. PRISM receives manifest and executes EXACTLY what's specified
```

### Manifest Generator
```bash
# Generate manifest from typology
python -m orthon.ingest.manifest_generator data/typology.parquet data/manifest.yaml
```

### Manifest v2.1 Structure (Nested by Cohort)
```yaml
version: "2.1"
job:
  id: prism_20260131_123456
  name: C-MAPSS Analysis
cohorts:
  engine_1:
    sensor_02:
      is_constant: false
      signal_type: SMOOTH
      periodicity: PERIODIC
      engines:
        - kurtosis
        - harmonics_ratio
    sensor_03:
      is_constant: true
      engines: []
  engine_2:
    sensor_02:
      is_constant: false
      signal_type: NOISY
      engines:
        - entropy
        - sample_entropy
skip_signals:
  - constant_signal
engines_required:
  signal: [kurtosis, skewness, entropy, ...]
  rolling: [rolling_kurtosis, rolling_entropy, ...]
```

**Note:** Unique time series = `(cohort, signal_id)`. Nested structure prevents key collisions.

---

## Engine Selection (Typology-Guided)

New architecture: ORTHON selects engines based on signal typology using **inclusive philosophy**.

### Philosophy: "If it's a maybe, run it."

Rather than removing engines that "might not apply", we run everything that could provide useful information. The only exception is CONSTANT signals (zero variance) where no engine produces meaningful results.

### Base Engines (always included, except CONSTANT)
```python
BASE_ENGINES = [
    # Distribution
    'crest_factor', 'kurtosis', 'skewness',
    # Spectral
    'spectral',
    # Complexity/Information (discriminators)
    'sample_entropy', 'perm_entropy',
    # Memory
    'hurst', 'acf_decay',
]
```

### Engine Gating by Temporal Pattern (PR6 Manifest Generator v2.2)

| Type | Engines Added | Remove |
|------|---------------|--------|
| **TRENDING** | hurst, rate_of_change, trend_r2, detrend_std, cusum, sample_entropy, acf_decay, variance_growth | none |
| **PERIODIC** | harmonics, thd, frequency_bands, fundamental_freq, phase_coherence, hurst, snr | none |
| **CHAOTIC** | lyapunov, correlation_dimension, recurrence_rate, determinism, harmonics, sample_entropy, perm_entropy, embedding_dim | none |
| **RANDOM** | spectral_entropy, band_power, hurst, frequency_bands, sample_entropy, perm_entropy, acf_decay | none |
| **QUASI_PERIODIC** | frequency_bands, harmonics, hurst, rate_of_change, sample_entropy | none |
| **STATIONARY** | hurst, frequency_bands, spectral_entropy, acf_decay, variance_ratio, adf_stat | none |
| **CONSTANT** | (none) | **`['*']` — removes all** |
| **BINARY** | transition_count, duty_cycle, mean_time_between, switching_frequency | none |
| **DISCRETE** | level_histogram, transition_matrix, dwell_times, level_count, entropy | none |
| **IMPULSIVE** | peak_detection, inter_arrival, peak_amplitude_dist, hurst, envelope, rise_time | none |
| **EVENT** | event_rate, inter_event_time, event_amplitude | none |
| **STEP** | changepoint_detection, level_means, regime_duration, hurst | none |
| **INTERMITTENT** | burst_detection, activity_ratio, silence_distribution, hurst | none |

### Window Parameters by Type

| Type | Window | Stride | Derivative Depth |
|------|--------|--------|------------------|
| TRENDING | 128 | 32 (75% overlap) | 2 |
| IMPULSIVE | 64 | 16 (75% overlap) | 1 |
| CONSTANT/BINARY/DISCRETE/EVENT | n_samples | n_samples | 0 |
| (default) | 128 | 64 (50% overlap) | 1 |

### Key Discriminator Engines

| Engine | Discriminates Between |
|--------|----------------------|
| sample_entropy | TRENDING (low) vs RANDOM (high) |
| perm_entropy | CHAOTIC (medium-high) vs PERIODIC (low) |
| acf_decay | TRENDING (slow/no decay) vs STATIONARY (exponential) |
| hurst | TRENDING (>0.85) vs STATIONARY (~0.5) vs anti-persistent (<0.5) |
| variance_ratio | Homoscedastic vs heteroscedastic |

### Deprecated (absolute values)
rms, peak, mean, std, rolling_rms, rolling_mean, rolling_std, envelope

### Legacy Mode
For backward compatibility: `python -m prism --legacy manifest.yaml` runs all 53 engines

---

## ORTHON Outputs

### Typology (ORTHON's only computation)
- `typology.parquet` - Signal characterization (smooth, noisy, periodic, etc.)

PRISM expects typology.parquet to exist. ORTHON creates it.

---

## PRISM Outputs (14 files)

### Pipeline Stage Outputs
- `signal_vector.parquet` - Per-signal scale-invariant features with I column
- `state_vector.parquet` - Centroids (position in feature space) - NO eigenvalues

### Geometry Layer
- `state_geometry.parquet` - Eigenvalues, effective_dim (SHAPE lives here)
- `signal_geometry.parquet` - Signal-to-centroid distances
- `signal_pairwise.parquet` - Pairwise signal relationships

### Geometry Dynamics (Differential Geometry)
- `geometry_dynamics.parquet` - Derivatives: velocity, acceleration, jerk, curvature
- `signal_dynamics.parquet` - Per-signal trajectory analysis
- `pairwise_dynamics.parquet` - Pairwise trajectory analysis
- PRISM computes derivatives. ORTHON classifies trajectories.

### Dynamics Layer
- `dynamics.parquet` - RQA, attractor metrics
- `information_flow.parquet` - Transfer entropy, Granger
- `lyapunov.parquet` - Lyapunov exponents per signal

### SQL Layer (no classification)
- `zscore.parquet` - Normalized metrics
- `statistics.parquet` - Summary statistics
- `correlation.parquet` - Correlation matrix

**Note:** regime_assignment removed - classification belongs in ORTHON, not PRISM.

---

## Baseline Modes (orthon/analysis/baseline_discovery.py)

| Mode | Use Case |
|------|----------|
| first_n_percent | Industrial (pump, bearing) - known healthy start |
| stable_discovery | Markets, bioreactor - unknown healthy state |
| last_n_percent | Post-maintenance scenarios |
| reference_period | Known-good time window |
| rolling | Gradual drift systems |

---

## Physics Domains (orthon/config/domains.py)

| Domain | Equations |
|--------|-----------|
| chemical | Pipe flow, thermodynamics |
| electrical | Transfer functions, causality |
| mechanical | Energy, momentum, Lagrangian |
| fluids | Vorticity, turbulence, Reynolds |
| thermal | Heat flux, enthalpy |
| signals | Statistics, entropy, events |
| dynamical | Chaos, recurrence, phase space |

---

## Domain Data Location

Raw domain data: `~/Domains/` (or `$ORTHON_DOMAINS_DIR`)

```
Domains/
├── battery/           # NASA battery degradation
├── calce/             # Battery calendar aging
├── cmapss/            # C-MAPSS turbofan
├── cwru/              # CWRU bearing fault
├── electrochemistry/  # Electrochemical data
├── FF/                # Fama-French financial
├── hydraulic/         # Hydraulic condition monitoring
├── industrial/        # SKAB, MetroPT
├── rossler/           # Chaotic system
└── secom/             # Semiconductor manufacturing
```

---

## Commands

```bash
# Full ORTHON pipeline: observations → typology → manifest
python -m orthon.pipeline data/observations.parquet data/

# Or run stages individually:
# 1. Compute raw typology measures (27 metrics per signal)
python -m orthon.ingest.typology_raw data/observations.parquet data/typology_raw.parquet

# 2. Apply classification corrections
python -c "from orthon.corrections.level2_corrections import apply_corrections; ..."

# 3. Generate manifest from typology
python -m orthon.ingest.manifest_generator data/typology.parquet data/manifest.yaml

# Validate observations
python -m orthon.ingest.validate_observations data/observations.parquet

# PRISM computes (requires typology.parquet + manifest.yaml from ORTHON)
cd ~/prism
./venv/bin/python -m prism data/cmapss                    # Full pipeline
./venv/bin/python -m prism signal-vector-temporal data/cmapss  # Individual stages
./venv/bin/python -m prism state-vector data/cmapss
./venv/bin/python -m prism geometry data/cmapss
./venv/bin/python -m prism geometry-dynamics data/cmapss
./venv/bin/python -m prism lyapunov data/cmapss
./venv/bin/python -m prism dynamics data/cmapss
./venv/bin/python -m prism sql data/cmapss
```

---

## Rules

1. **PRISM computes, ORTHON classifies** - all classification logic in ORTHON
2. **Typology is ORTHON's only computation** - PRISM reads typology.parquet
3. New architecture (v2): Typology-guided, scale-invariant engines
4. Insufficient data → return NaN, never skip
5. No domain-specific logic in PRISM
6. No RAM management in ORTHON (PRISM handles this)
7. Output paths are FIXED - never change them
8. Scale-invariant features only (no rms, peak, mean, std)
9. Use Lyapunov for chaos detection, NOT coefficient of variation
10. cohort is a grouping key for related signals, NOT a compute key (group by I or signal_id)

## Do NOT

- Put classification logic in PRISM (it goes in ORTHON)
- Let PRISM create typology (ORTHON creates it)
- Write to subdirectories of $PRISM_DATA_DIR
- Add RAM management to ORTHON
- Use absolute value features (rms, peak, mean, std) - deprecated
- Skip geometry dynamics when analyzing state evolution
- Use CV for chaos detection (use Lyapunov)
- Include cohort in groupby for computations (it's a grouping key, not a compute key)
