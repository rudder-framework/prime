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

## PRISM Format (observations.parquet) - v2.0.0

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| unit_id | String | Optional | Which unit (pump, bearing, industry) - blank is fine |
| signal_id | String | Required | Which signal (temp, pressure, return) |
| I | UInt32 | Required | Observation index within unit+signal |
| value | Float64 | Required | The measurement |

**Note:** `unit_id` replaces legacy `entity_id`. ORTHON transforms data to this format.

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
| `orthon/ingest/validate_observations.py` | **NEW** Validates & repairs observations.parquet |
| `orthon/manifest_generator.py` | **NEW** Creates v2 manifest from typology |
| `orthon/engine_rules.yaml` | **NEW** Engine selection rules |
| `orthon/sql/typology.sql` | **NEW** Signal classification SQL |
| `orthon/ingest/paths.py` | Fixed output paths (NO EXCEPTIONS) |
| `orthon/config/manifest.py` | ENGINES list (53), Pydantic models |
| `orthon/config/domains.py` | 7 physics domains |
| `orthon/analysis/baseline_discovery.py` | Baseline modes |
| `orthon/services/manifest_builder.py` | Build PRISM manifests |
| `orthon/prism_client.py` | HTTP client for PRISM |

---

## Unified Manifest System (v2.0)

ORTHON decides. PRISM executes.

### Workflow
```
1. ORTHON runs typology.sql on observations.parquet
2. ORTHON generates manifest.yaml with per-signal engine selection
3. PRISM receives manifest and executes EXACTLY what's specified
```

### Manifest Generator
```bash
# Generate manifest from typology
python -m orthon.manifest_generator data/typology.parquet data/manifest.yaml
```

### Manifest v2.0 Structure
```yaml
version: "2.0"
job:
  id: prism_20260131_123456
  name: C-MAPSS Analysis
signals:
  sensor_02:
    is_constant: false
    signal_type: SMOOTH
    periodicity: PERIODIC
    engines:
      - kurtosis
      - harmonics_ratio
      - rolling_entropy
  constant_signal:
    is_constant: true
    engines: []
skip_signals:
  - constant_signal
engines_required:
  signal: [kurtosis, skewness, entropy, ...]
  rolling: [rolling_kurtosis, rolling_entropy, ...]
```

---

## Engine Selection (Typology-Guided)

New architecture: ORTHON selects engines based on signal typology.

### Core Engines (always run)
kurtosis, skewness, crest_factor

### By Signal Type
| Type | Engines |
|------|---------|
| SMOOTH | rolling_kurtosis, rolling_entropy, rolling_crest_factor |
| NOISY | entropy, sample_entropy |
| IMPULSIVE | crest_factor, peak_ratio |
| MIXED | entropy, crest_factor, sample_entropy |

### By Periodicity
| Type | Engines |
|------|---------|
| PERIODIC | harmonics_ratio, band_ratios, spectral_entropy, thd |
| QUASI_PERIODIC | band_ratios, spectral_entropy |
| APERIODIC | entropy, hurst, sample_entropy |

### By Tail Behavior
| Type | Engines |
|------|---------|
| HEAVY_TAILS | kurtosis, crest_factor |
| LIGHT_TAILS | entropy, sample_entropy |

### Deprecated (absolute values)
rms, peak, mean, std, rolling_rms, rolling_mean, rolling_std, envelope

### Legacy Mode
For backward compatibility: `python -m prism --legacy manifest.yaml` runs all 53 engines

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
