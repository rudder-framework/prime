# ORTHON - AI Instructions

## Architecture

```
ORTHON = Brain (orchestration, manifest, interpretation)
PRISM  = Muscle (pure computation, no decisions)

ORTHON creates observations.parquet → PRISM data directory
PRISM runs: typology → signal_vector → state_vector → geometry → dynamics

New architecture (v2): Typology-guided, scale-invariant, eigenvalue-based
Legacy mode available: python -m prism --legacy manifest.yaml
```

## The One Rule

```
observations.parquet and manifest.yaml ALWAYS go to:
/Users/jasonrudder/prism/data/

NO EXCEPTIONS. No subdirectories. No domain folders.
```

## PRISM Format (observations.parquet)

| Column | Type | Description |
|--------|------|-------------|
| entity_id | String | Which entity (pump, bearing, industry) |
| I | UInt32 | Observation index within entity |
| signal_id | String | Which signal (temp, pressure, return) |
| value | Float64 | The measurement |

**If data is not in this format, ORTHON transforms it first.**

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
│   │   └── manifest_generator.py  # AI auto-generates manifest
│   │
│   ├── intake/                    # UI file handling
│   │   ├── upload.py              # File upload
│   │   ├── validate.py            # Validation
│   │   └── transformer.py         # Transform to PRISM format
│   │
│   ├── analysis/                  # Analysis tools
│   │   └── baseline_discovery.py  # Baseline modes
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
| `orthon/ingest/paths.py` | Fixed output paths (NO EXCEPTIONS) |
| `orthon/config/manifest.py` | ENGINES list (53), Pydantic models |
| `orthon/config/domains.py` | 7 physics domains |
| `orthon/analysis/baseline_discovery.py` | Baseline modes |
| `orthon/services/manifest_builder.py` | Build PRISM manifests |
| `orthon/prism_client.py` | HTTP client for PRISM |

---

## ENGINES List (orthon/config/manifest.py)

53 engines specified for PRISM to run:

### Tier 1: Basic Statistics (10)
mean, std, rms, peak, crest_factor, shape_factor, impulse_factor, margin_factor, skewness, kurtosis

### Tier 2: Distribution (5)
histogram, percentiles, iqr, mad, coefficient_of_variation

### Tier 3: Information Theory (6)
entropy_shannon, entropy_sample, entropy_permutation, entropy_spectral, mutual_information, transfer_entropy

### Tier 4: Spectral (11)
fft, psd, spectral_centroid, spectral_spread, spectral_rolloff, spectral_flatness, spectral_slope, spectral_entropy, spectral_peaks, harmonic_ratio, bandwidth

### Tier 5: Dynamics (10)
lyapunov, correlation_dimension, hurst_exponent, dfa, recurrence_rate, determinism, laminarity, trapping_time, divergence, attractor_dimension

### Tier 6: Topology (5)
betti_0, betti_1, persistence_entropy, persistence_landscape, wasserstein_distance

### Tier 7: Relationships (6)
cross_correlation, coherence, phase_coupling, granger_causality, cointegration, dtw_distance

---

## PRISM Outputs (New Architecture v2)

### Pipeline Stage Outputs
- `typology.parquet` - Signal characterization (smooth, noisy, periodic, etc.)
- `signal_vector.parquet` - Per-signal scale-invariant features
- `signal_vector_temporal.parquet` - Features with I column (for dynamics)
- `state_vector.parquet` - System state via eigenvalues (SVD)

### Geometry Layer
- `state_geometry.parquet` - Per-engine eigenvalues over time
- `signal_geometry.parquet` - Signal-to-state relationships
- `signal_pairwise.parquet` - Pairwise signal relationships

### Geometry Dynamics (Differential Geometry)
- `geometry_dynamics.parquet` - Derivatives: velocity, acceleration, jerk, curvature
- `signal_dynamics.parquet` - Per-signal trajectory analysis
- Trajectory classification: STABLE, CONVERGING, DIVERGING, OSCILLATING, CHAOTIC, COLLAPSING, EXPANDING
- Collapse detection: sustained loss of effective dimension

### Dynamics Layer
- `dynamics.parquet` - Lyapunov, RQA, Hurst
- `information_flow.parquet` - Transfer entropy, Granger

### SQL Reconciliation
- `zscore.parquet` - Normalized metrics
- `statistics.parquet` - Summary statistics
- `correlation.parquet` - Correlation matrix
- `regime_assignment.parquet` - State labels

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

Raw domain data: `/Users/jasonrudder/domains/`

```
domains/
├── bearing/           # FEMTO, IMS
├── turbomachinery/    # C-MAPSS
├── industrial/        # SKAB, MetroPT
├── finance/           # Fama-French
└── misc/              # Docs, scripts
```

---

## Commands

```bash
# ORTHON generates manifest
python -m orthon.ingest.manifest_generator /path/to/raw/data

# ORTHON ingests data
python -m orthon.ingest.streaming manifest.yaml

# PRISM computes (new architecture v2)
cd ~/prism
./venv/bin/python -m prism data/cmapss              # Full pipeline
./venv/bin/python -m prism typology data/cmapss     # Individual stages
./venv/bin/python -m prism signal-vector data/cmapss
./venv/bin/python -m prism state-vector data/cmapss
./venv/bin/python -m prism geometry data/cmapss
./venv/bin/python -m prism dynamics data/cmapss

# Legacy 53-engine mode (if needed)
./venv/bin/python -m prism --legacy data/manifest.yaml
```

---

## Rules

1. New architecture (v2): Typology-guided, scale-invariant engines
2. Legacy mode: `--legacy` flag runs all 53 engines
3. Insufficient data → return NaN, never skip
4. No domain-specific logic in PRISM
5. No RAM management in ORTHON (PRISM handles this)
6. Output paths are FIXED - never change them
7. Scale-invariant features only (no rms, peak, mean, std)

## Do NOT

- Write to subdirectories of /Users/jasonrudder/prism/data/
- Add RAM management to ORTHON
- Make ORTHON compute anything
- Use absolute value features (rms, peak, mean, std) - deprecated
- Skip geometry dynamics when analyzing state evolution
