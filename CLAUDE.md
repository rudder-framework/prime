# ORTHON - AI Instructions

Orthon Engines is a domain-agnostic dynamical systems analysis platform.
- **orthon-engines/orthon** — dynamical systems analysis interpreter
- **orthon-engines/engines** — dynamical systems computation engines

## Architecture

```
Orthon  = Brain (orchestration, typology, classification, interpretation)
Engines = Muscle (pure computation, no decisions, no classification)

Engines computes numbers. Orthon classifies.

Orthon creates: observations.parquet + typology.parquet + manifest.yaml
Engines reads: observations.parquet + typology.parquet + manifest.yaml
Engines runs: signal_vector → state_vector → geometry → dynamics
Orthon runs: classification SQL views on Engines outputs

Typology is the ONLY statistical analysis in Orthon.
Orthon classifies signals; Engines computes features.
Current architecture (v2.5): Typology-guided, scale-invariant, multi-scale, per-engine window spec
```

## The One Rule

```
observations.parquet and manifest.yaml ALWAYS go to:
$ENGINES_DATA_DIR (default: ~/engines/data/)

NO EXCEPTIONS. No subdirectories. No domain folders.
```

## Engines Format (observations.parquet) - v2.5

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| cohort | String | Optional | Which cohort/unit (engine_1, pump_A, bearing_3) - groups related signals |
| signal_id | String | Required | Which signal (temp, pressure, return) |
| I | UInt32 | Required | Observation index within unit+signal |
| value | Float64 | Required | The measurement |

**Note:** `cohort` replaces legacy `unit_id`/`entity_id`. Each cohort is a group of related signals (e.g., all sensors on one engine).

---

## Observations Validation

**CRITICAL:** Before running Engines, ALWAYS validate observations.parquet.

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

## Classification SQL (orthon/sql/layers/classification.sql)

**Engines computes numbers. ORTHON classifies.**

All classification logic lives in ORTHON, not Engines. Engines outputs raw metrics (Lyapunov, effective_dim, etc.). ORTHON applies thresholds and labels.

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
├── CLAUDE.md                      # This file - AI instructions
├── README.md                      # Project overview
├── orthon/
│   ├── __init__.py
│   │
│   ├── entry_points/              # Pipeline stages (13 total)
│   │   ├── stage_01_validate      # Validation (remove constants, duplicates)
│   │   ├── stage_02_typology      # Compute raw typology measures (27 metrics)
│   │   ├── stage_03_classify      # Apply classification (discrete/sparse → continuous)
│   │   ├── stage_04_manifest      # Generate manifest for Engines
│   │   ├── stage_05_diagnostic    # Run diagnostic assessment
│   │   ├── stage_06_interpret     # Interpret Engines outputs (dynamics + physics)
│   │   ├── stage_07_predict       # Predict RUL, health, anomalies
│   │   ├── stage_08_alert         # Early warning / failure fingerprints
│   │   ├── stage_09_explore       # Manifold visualization
│   │   ├── stage_10_inspect       # File inspection / capability detection
│   │   ├── stage_11_fetch         # Read, profile, validate raw data
│   │   ├── stage_12_stream        # Real-time streaming analysis
│   │   └── stage_13_train         # Train ML models on Engines features
│   │
│   ├── config/                    # Configuration
│   │   ├── typology_config.py     # Classification thresholds
│   │   └── domains.py             # Physics domains (7)
│   │
│   ├── typology/                  # Signal classification
│   │   ├── level2_corrections.py  # Config-driven continuous classification
│   │   ├── discrete_sparse.py     # Discrete/sparse detection (PR5)
│   │   ├── constant_detection.py  # CV-based constant detection (PR8)
│   │   └── tests/
│   │
│   ├── manifest/                  # Manifest generation
│   │   ├── generator.py           # v2.5 manifest: engine gating, per-engine window spec
│   │   └── tests/
│   │
│   ├── window/                    # Window/stride computation
│   │   ├── characteristic_time.py # Data-driven window from ACF, frequency, etc.
│   │   ├── system_window.py       # Multi-scale representation (spectral vs trajectory)
│   │   ├── manifest_generator.py  # Alternative manifest generator (uses system_window)
│   │   └── tests/
│   │
│   ├── ingest/                    # Data ingestion
│   │   ├── typology_raw.py        # Computes 27 raw measures per signal
│   │   ├── validate_observations.py # Validates & repairs observations
│   │   ├── data_reader.py         # Read CSV/parquet/TSV, profile data
│   │   ├── validation.py          # SignalValidator, ValidationConfig
│   │   ├── schema_enforcer.py     # Schema validation
│   │   └── paths.py               # FIXED output paths
│   │
│   ├── services/                  # Interpreters & orchestration
│   │   ├── physics_interpreter.py # Symplectic structure loss detection
│   │   ├── dynamics_interpreter.py # Lyapunov, basin stability, regime transitions
│   │   ├── state_analyzer.py      # State velocity/acceleration anomaly detection
│   │   ├── fingerprint_service.py # Healthy/deviation/failure fingerprint matching
│   │   ├── tuning_service.py      # AI-guided threshold optimization
│   │   ├── concierge.py           # Natural language interface to ORTHON
│   │   └── job_manager.py         # Job lifecycle management
│   │
│   ├── prediction/                # Predictive models
│   │   ├── rul.py                 # Remaining useful life prediction
│   │   ├── health.py              # Health scoring
│   │   ├── anomaly.py             # Anomaly detection (zscore, IF, LOF)
│   │   └── cli.py                 # Prediction CLI
│   │
│   ├── early_warning/             # Early failure detection
│   │   ├── ml_predictor.py        # ML-based failure prediction
│   │   └── failure_fingerprint_detector.py # Heuristic fingerprint detection
│   │
│   ├── engines/                   # ORTHON diagnostic engines
│   │   ├── diagnostic_report.py   # Full diagnostic pipeline
│   │   ├── typology_engine.py     # Level 0: Typology
│   │   ├── stationarity_engine.py # Level 1: Stationarity
│   │   ├── classification_engine.py # Level 2: Classification
│   │   ├── signal_geometry.py     # Geometry analysis
│   │   ├── stability_engine.py    # Stability assessment
│   │   ├── tipping_engine.py      # Tipping point detection
│   │   └── spin_glass.py          # Spin glass model
│   │
│   ├── sql/                       # SQL classification & analysis
│   │   ├── trajectory_intelligence.py # Cross-system trajectory learning
│   │   ├── layers/                # Analysis pipeline (37 SQL files)
│   │   ├── views/                 # Dashboard views
│   │   ├── stages/                # Reporting stages
│   │   ├── reports/               # Deep analysis reports
│   │   ├── ml/                    # ML feature SQL
│   │   └── docs/                  # SQL documentation
│   │
│   ├── explorer/                  # Manifold visualization
│   │   ├── cli.py                 # CLI and server
│   │   ├── loader.py              # ManifoldLoader
│   │   └── renderer.py            # 2D/3D rendering
│   │
│   ├── inspection/                # Data inspection
│   │   ├── file_inspector.py      # File profiling
│   │   ├── capability_detector.py # Capability detection
│   │   └── results_validator.py   # Output validation
│   │
│   ├── streaming/                 # Real-time analysis
│   │   ├── cli.py                 # Dashboard, analyze, demo modes
│   │   ├── analyzers.py           # RealTimeAnalyzer
│   │   └── websocket_server.py    # Live dashboard server
│   │
│   ├── ml/                        # ML training pipeline
│   │   └── entry_points/          # train, predict, features, ablation, benchmark
│   │
│   ├── analysis/                  # Analysis tools
│   │   └── baseline_discovery.py  # Baseline modes
│   │
│   ├── core/                      # Core API & pipeline
│   │   ├── api.py                 # FastAPI endpoints
│   │   ├── pipeline.py            # Pipeline orchestration
│   │   ├── engines_client.py        # Engines HTTP client
│   │   ├── data_reader.py         # Re-export shim → ingest.data_reader
│   │   └── validation.py          # Re-export shim → ingest.validation
│   │
│   ├── cohorts/                   # Cohort discovery
│   ├── shared/                    # Shared constants (physics, etc.)
│   ├── state/                     # State management
│   └── utils/                     # Utility functions
│
├── _legacy/                       # Archived code (not in git)
├── docs/                          # Documentation & reports
├── tests/                         # Tests
└── scripts/                       # Utility scripts
    ├── process_all_domains.py     # Batch domain processing
    ├── regenerate_manifests.py    # Regenerate all manifests to v2.5
    └── test_pipeline.py           # Pipeline test runner
```

---

## Entry Points

All entry points are thin orchestrators — they call modules, not compute.

```bash
# Pre-Engines pipeline (stages 01-04)
python -m orthon.entry_points.stage_01_validate observations.parquet -o validated.parquet
python -m orthon.entry_points.stage_02_typology observations.parquet -o typology_raw.parquet
python -m orthon.entry_points.stage_03_classify typology_raw.parquet -o typology.parquet
python -m orthon.entry_points.stage_04_manifest typology.parquet -o manifest.yaml

# Diagnostic
python -m orthon.entry_points.stage_05_diagnostic observations.parquet -o report.txt

# Post-Engines interpretation
python -m orthon.entry_points.stage_06_interpret /path/to/engines/output --mode both
python -m orthon.entry_points.stage_07_predict /path/to/engines/output --mode health
python -m orthon.entry_points.stage_08_alert observations.parquet --mode predict

# Tools
python -m orthon.entry_points.stage_09_explore /path/to/engines/output -o manifold.png
python -m orthon.entry_points.stage_10_inspect data.parquet --mode inspect
python -m orthon.entry_points.stage_11_fetch raw_data.csv -o observations.parquet
python -m orthon.entry_points.stage_12_stream dashboard --source turbofan
python -m orthon.entry_points.stage_13_train --model xgboost
```

Programmatic usage:
```python
from orthon.entry_points import validate, compute_typology, classify, generate_manifest
from orthon.entry_points import interpret, predict, alert, explore, inspect, fetch
```

---

## Key Files

| File | Purpose |
|------|---------|
| `orthon/ingest/typology_raw.py` | Computes 27 raw typology measures per signal |
| `orthon/typology/discrete_sparse.py` | Discrete/sparse detection (runs FIRST) |
| `orthon/typology/level2_corrections.py` | Config-driven continuous classification |
| `orthon/typology/constant_detection.py` | CV-based CONSTANT detection |
| `orthon/manifest/generator.py` | **PRIMARY** - v2.5 manifest with engine gating, per-engine window spec |
| `orthon/window/characteristic_time.py` | Data-driven window from ACF, frequency, period |
| `orthon/window/system_window.py` | Multi-scale representation (spectral vs trajectory) |
| `orthon/config/typology_config.py` | All classification thresholds (no magic numbers) |
| `orthon/sql/layers/classification.sql` | Lyapunov-based trajectory classification (on Engines outputs) |
| `orthon/ingest/validate_observations.py` | Validates & repairs observations.parquet |
| `orthon/core/pipeline.py` | Orchestrates observations → typology → manifest |
| `orthon/services/physics_interpreter.py` | Symplectic structure loss detection |
| `orthon/services/dynamics_interpreter.py` | Lyapunov, basin stability, regime transitions |
| `scripts/process_all_domains.py` | Batch domain processing pipeline |
| `scripts/regenerate_manifests.py` | Batch regenerate all domain manifests to v2.5 |

---

## Typology (ORTHON's Responsibility)

**Typology is ORTHON's signal classification system.** It computes 27 statistical measures per signal and classifies across 10 dimensions. Engines then uses these classifications for engine selection.

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
| temporal_pattern | PERIODIC, TRENDING, DRIFTING, RANDOM, CHAOTIC, QUASI_PERIODIC, STATIONARY, CONSTANT, BINARY, DISCRETE, IMPULSIVE, EVENT |
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
4. is_drifting? → DRIFTING (non-stationary persistent drift buried in noise)
   hurst 0.85-0.99 AND perm_entropy >= 0.90 AND variance_ratio >= 0.2 AND NON_STATIONARY
5. spectral_override? → PERIODIC (noisy periodic)
   SNR > 30 dB AND flatness < 0.1
6. is_genuine_periodic? → PERIODIC (all 6 gates pass)
7. spectral_flatness > 0.9 AND perm_entropy > 0.99? → RANDOM
8. lyapunov > 0.5 AND perm_entropy > 0.95? → CHAOTIC
9. turning_point_ratio < 0.7? → QUASI_PERIODIC
10. default → STATIONARY
```

### Workflow
```
1. ORTHON computes typology_raw.parquet (27 measures per signal)
2. ORTHON applies discrete/sparse classification (PR5)
3. ORTHON applies continuous classification if needed (PR4)
4. ORTHON creates typology.parquet (10 classification dimensions)
5. ORTHON generates manifest.yaml (engine selection per signal)
6. Engines reads manifest and executes engines
7. ORTHON classifies Engines outputs (Lyapunov → trajectory type)
```

### Config Files
- `orthon/config/typology_config.py` - PR4 continuous classification thresholds
- `orthon/config/discrete_sparse_config.py` - PR5 discrete/sparse thresholds

---

## Unified Manifest System (v2.5)

ORTHON decides. Engines executes.

### Workflow
```
1. ORTHON computes typology_raw.parquet (27 measures per signal)
2. ORTHON applies discrete/sparse classification (PR5)
3. ORTHON applies continuous classification (PR4)
4. ORTHON generates manifest.yaml with per-signal engine selection + system_window
5. Engines receives manifest and executes EXACTLY what's specified
```

### Manifest v2.5 Structure
```yaml
version: '2.5'
job_id: orthon-20260202-143052
created_at: '2026-02-02T14:30:52'
generator: orthon.manifest_generator v2.5 (per-engine window spec)

paths:
  observations: observations.parquet
  typology: typology.parquet
  output_dir: output/

system:
  window: 128              # Common window for state_vector/geometry alignment
  stride: 64               # Common stride (median of all signal strides)
  note: Common window for state_vector/geometry alignment

engine_windows:            # Per-engine minimum window requirements
  spectral: 64
  harmonics: 64
  fundamental_freq: 64
  thd: 64
  frequency_bands: 64
  spectral_entropy: 64
  band_power: 64
  sample_entropy: 64
  hurst: 128
  note: Minimum window sizes for FFT-based and long-range engines

summary:
  total_signals: 24
  total_cohorts: 1
  active_signals: 22
  constant_signals: 2
  signal_engines: [crest_factor, entropy, harmonics, hurst, ...]
  n_signal_engines: 15

params:
  default_window: 128
  default_stride: 64
  min_samples: 64

cohorts:
  engine_1:
    sensor_02:
      engines: [kurtosis, harmonics, hurst, spectral, ...]
      rolling_engines: []
      window_size: 32            # Signal-specific window
      window_method: period
      window_confidence: high
      stride: 16
      derivative_depth: 1
      eigenvalue_budget: 5
      engine_window_overrides:   # Per-signal overrides when window < engine min
        spectral: 64
        harmonics: 64
        hurst: 128
      typology:
        temporal_pattern: PERIODIC
        spectral: NARROWBAND
      visualizations: [waterfall, phase_portrait, spectral_density]
      output_hints:
        spectral:
          output_mode: per_bin

skip_signals:
  - engine_1/constant_signal   # CONSTANT signals skip all engines

pair_engines: [granger, transfer_entropy]
symmetric_pair_engines: [cointegration, correlation, mutual_info]
```

**Key concepts:**
- `engine_windows`: Global minimum window sizes for engines that require more samples (FFT, long-range)
- `engine_window_overrides`: Per-signal overrides when signal's window_size < engine minimum (Engines uses expanded window)
- `system.window`: Common window for multi-signal alignment (EEG paradigm)
- `window_method`: How window was determined (period, acf_half_life, long_memory, frequency, default)
- `window_confidence`: high/medium/low based on data quality

---

## Engine Selection (Typology-Guided)

ORTHON selects engines based on signal typology using **inclusive philosophy**.

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

### Engine Gating by Temporal Pattern (Manifest Generator v2.5)

| Type | Engines Added | Remove |
|------|---------------|--------|
| **TRENDING** | hurst, rate_of_change, trend_r2, detrend_std, cusum, sample_entropy, acf_decay, variance_growth | none |
| **DRIFTING** | hurst, rate_of_change, trend_r2, detrend_std, cusum, sample_entropy, acf_decay, variance_growth | none |
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

### Per-Engine Minimum Windows (v2.5)

Some engines require minimum sample counts to produce meaningful results. When a signal's window is smaller than an engine's requirement, Engines uses an expanded window for that engine.

| Engine | Min Window | Reason |
|--------|------------|--------|
| spectral, harmonics, fundamental_freq, thd | 64 | FFT requires sufficient samples |
| frequency_bands, spectral_entropy, band_power | 64 | FFT-based |
| sample_entropy | 64 | Embedding dimension requirements |
| hurst | 128 | R/S analysis needs longer series |
| crest_factor, kurtosis, skewness, acf_decay | 32 | Work fine with small windows |

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
For backward compatibility: `python -m engines --legacy manifest.yaml` runs all 53 engines

---

## ORTHON Outputs

### Typology (ORTHON's only computation)
- `typology.parquet` - Signal characterization (smooth, noisy, periodic, etc.)

Engines expects typology.parquet to exist. ORTHON creates it.

---

## Engines Outputs (14 files)

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
- Engines computes derivatives. ORTHON classifies trajectories.

### Dynamics Layer
- `dynamics.parquet` - RQA, attractor metrics
- `information_flow.parquet` - Transfer entropy, Granger
- `lyapunov.parquet` - Lyapunov exponents per signal

### SQL Layer (no classification)
- `zscore.parquet` - Normalized metrics
- `statistics.parquet` - Summary statistics
- `correlation.parquet` - Correlation matrix

**Note:** regime_assignment removed - classification belongs in ORTHON, not Engines.

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
# Full ORTHON pipeline: observations → typology → manifest v2.5
python -m orthon.pipeline data/observations.parquet data/

# Or run via entry points (preferred):
python -m orthon.entry_points.stage_01_validate observations.parquet -o validated.parquet
python -m orthon.entry_points.stage_02_typology observations.parquet -o typology_raw.parquet
python -m orthon.entry_points.stage_03_classify typology_raw.parquet -o typology.parquet
python -m orthon.entry_points.stage_04_manifest typology.parquet -o manifest.yaml

# Validate observations
python -m orthon.ingest.validate_observations data/observations.parquet

# Batch process all domains
cd ~/orthon && ./venv/bin/python scripts/process_all_domains.py

# Regenerate ALL manifests in ~/Domains to v2.5
cd ~/orthon && ./venv/bin/python scripts/regenerate_manifests.py

# Engines computes (requires observations.parquet + typology.parquet + manifest.yaml)
cd ~/engines
./venv/bin/python -m engines data/cmapss                    # Full pipeline
./venv/bin/python -m engines signal-vector-temporal data/cmapss  # Individual stages
./venv/bin/python -m engines state-vector data/cmapss
./venv/bin/python -m engines geometry data/cmapss
./venv/bin/python -m engines geometry-dynamics data/cmapss
./venv/bin/python -m engines lyapunov data/cmapss
./venv/bin/python -m engines dynamics data/cmapss
./venv/bin/python -m engines sql data/cmapss
```

---

## Rules

1. **Engines computes, ORTHON classifies** - all classification logic in ORTHON
2. **Typology is ORTHON's only computation** - Engines reads typology.parquet
3. New architecture (v2.5): Typology-guided, scale-invariant engines
4. Insufficient data → return NaN, never skip
5. No domain-specific logic in Engines
6. No RAM management in ORTHON (Engines handles this)
7. Output paths are FIXED - never change them
8. Scale-invariant features only (no rms, peak, mean, std)
9. Use Lyapunov for chaos detection, NOT coefficient of variation
10. cohort is a grouping key for related signals, NOT a compute key (group by I or signal_id)

## Do NOT

- Put classification logic in Engines (it goes in ORTHON)
- Let Engines create typology (ORTHON creates it)
- Write to subdirectories of $ENGINES_DATA_DIR
- Add RAM management to ORTHON
- Use absolute value features (rms, peak, mean, std) - deprecated
- Skip geometry dynamics when analyzing state evolution
- Use CV for chaos detection (use Lyapunov)
- Include cohort in groupby for computations (it's a grouping key, not a compute key)
