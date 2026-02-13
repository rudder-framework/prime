# Rudder Framework

A domain-agnostic dynamical systems analysis framework for signal classification, manifest generation, and interpretation.

```
CSV/Parquet → Framework classifies → Engines computes → Framework interprets
```

---

## What It Does

1. **Validates observations** — repairs timestamps, detects column aliases, removes constants
2. **Classifies signals** — 27 statistical measures across 10 dimensions (temporal pattern, spectral, stationarity, memory, complexity, continuity, determinism, distribution, amplitude, volatility)
3. **Generates manifests** — tells engines which computations to run per signal, with data-driven window sizing
4. **Interprets engine outputs** — Lyapunov-based trajectory classification, collapse detection, health scoring
5. **Explores results** — browser-based DuckDB explorer with flow visualization

---

## Installation

```bash
pip install rudder-framework
```

---

## Quick Start

```bash
# Full pre-engines pipeline: observations → typology → manifest
rudder validate observations.parquet -o validated.parquet
rudder typology observations.parquet -o typology_raw.parquet
rudder classify typology_raw.parquet -o typology.parquet
rudder manifest typology.parquet -o manifest.yaml

# Then run engines
engines run observations.parquet --manifest manifest.yaml

# Post-pipeline interpretation
rudder interpret /path/to/engines/output --mode both
rudder predict /path/to/engines/output --mode health

# Interactive explorer
rudder-explorer ~/Domains --port 8080
```

### Or Use Entry Points

```bash
python -m framework.entry_points.stage_01_validate observations.parquet -o validated.parquet
python -m framework.entry_points.stage_02_typology observations.parquet -o typology_raw.parquet
python -m framework.entry_points.stage_03_classify typology_raw.parquet -o typology.parquet
python -m framework.entry_points.stage_04_manifest typology.parquet -o manifest.yaml
```

---

## Two-Stage Signal Classification

### Stage 1: Discrete/Sparse Detection

Catches non-continuous signals before continuous analysis:

| Type | Detection | Example |
|------|-----------|---------|
| CONSTANT | std ~ 0 or unique_ratio < 0.001 | Sensor stuck at fixed value |
| BINARY | Exactly 2 unique values | On/off valve, relay state |
| DISCRETE | Integer values, unique_ratio < 5% | Step counter, gear position |
| IMPULSIVE | kurtosis > 20, crest_factor > 10 | Impact events, spikes |
| EVENT | sparsity > 80%, kurtosis > 10 | Rare alarm triggers |

### Stage 2: Continuous Classification

If not discrete/sparse, applies decision tree:

| Type | Key Indicators |
|------|---------------|
| TRENDING | hurst >= 0.99, or segment trend with change > 20% |
| DRIFTING | hurst 0.85-0.99, high perm_entropy, non-stationary |
| PERIODIC | 6-gate test: dominant frequency, SNR, spectral flatness, ACF regularity |
| CHAOTIC | lyapunov > 0.5, perm_entropy > 0.95 |
| RANDOM | spectral_flatness > 0.9, perm_entropy > 0.99 |
| QUASI_PERIODIC | turning_point_ratio < 0.7 |
| STATIONARY | Default (none of the above) |

---

## Manifest Generation (v2.6)

The framework generates engine manifests with:

- **Per-signal engine selection** based on typology classification
- **Data-driven window sizing** from ACF half-life, seasonal period, or dominant frequency
- **Per-engine minimum windows** (FFT engines need 64 samples, Hurst needs 128)
- **Inclusive philosophy**: "If it's a maybe, run it" — only CONSTANT removes all engines
- **Atlas section** for system-level engines (velocity field, FTLE rolling, ridge proximity)

```yaml
version: '2.6'
system:
  window: 128
  stride: 64
cohorts:
  engine_1:
    temperature:
      engines: [kurtosis, spectral, hurst, sample_entropy, ...]
      window_size: 64
      window_method: period
      typology:
        temporal_pattern: PERIODIC
atlas:
  geometry_dynamics: { enabled: true }
  velocity_field: { enabled: true, smooth: savgol }
  ridge_proximity: { enabled: true }
```

---

## Explorer

Browser-based tool for querying engine outputs:

```bash
rudder-explorer ~/Domains --port 8080
```

Available at:
- `http://localhost:8080/` — SQL query interface (DuckDB-WASM)
- `http://localhost:8080/explorer.html` — Pipeline data browser
- `http://localhost:8080/flow` — Flow visualization (eigenvector-projected trajectory with urgency coloring)
- `http://localhost:8080/atlas` — Dynamical atlas scenarios

---

## Architecture

```
framework/
├── entry_points/              Pipeline stages (13 total)
│   ├── stage_01_validate      Validation
│   ├── stage_02_typology      27 raw typology measures
│   ├── stage_03_classify      Two-stage classification
│   ├── stage_04_manifest      Manifest generation
│   ├── stage_05_diagnostic    Diagnostic assessment
│   ├── stage_06_interpret     Interpret engine outputs
│   ├── stage_07_predict       RUL, health, anomaly prediction
│   └── stage_08-13            Alert, explore, inspect, fetch, stream, train
│
├── typology/                  Signal classification
│   ├── level2_corrections.py  Continuous classification (decision tree)
│   ├── discrete_sparse.py     Discrete/sparse detection
│   └── constant_detection.py  CV-based constant detection
│
├── manifest/                  Manifest generation
│   └── generator.py           v2.6 manifest with atlas section
│
├── ingest/                    Data ingestion
│   ├── typology_raw.py        27 raw measures per signal
│   └── validate_observations.py  Validate & repair observations
│
├── services/                  Interpretation
│   ├── physics_interpreter.py Symplectic structure loss
│   ├── dynamics_interpreter.py Lyapunov, basin stability
│   └── fingerprint_service.py Failure pattern matching
│
├── explorer/                  Browser-based visualization
│   ├── server.py              HTTP server
│   └── static/                HTML + DuckDB-WASM + flow viz
│
└── sql/                       Classification SQL views
    └── layers/                37 SQL files for Lyapunov classification,
                               collapse detection, health scoring
```

---

## Key Outputs

| File | Producer | Purpose |
|------|----------|---------|
| `typology_raw.parquet` | Framework | 27 statistical measures per signal |
| `typology.parquet` | Framework | 10-dimension signal classification |
| `manifest.yaml` | Framework | Engine/window configuration for engines |
| `signal_vector.parquet` | Engines | Per-signal features per window |
| `state_geometry.parquet` | Engines | Eigenvalues, effective dimension, eigenvectors |
| `ftle.parquet` | Engines | Lyapunov exponents per signal |
| `velocity_field.parquet` | Engines | State-space speed, curvature |
| `ridge_proximity.parquet` | Engines | Urgency classes (nominal/warning/elevated/critical) |

---

## Documentation

See [CLAUDE.md](CLAUDE.md) for complete technical documentation:
- Typology system (27 measures, 10 classification dimensions)
- Manifest structure v2.6 (atlas section)
- Classification SQL views (Lyapunov, collapse, health)
- Engine selection rules per temporal pattern
- Engine output schemas

---

## Citation

If you use Rudder Framework in your research, please cite:

```bibtex
@software{rudder_framework,
  author = {Rudder, Jason},
  title = {Rudder Framework: Domain-Agnostic Dynamical Systems Analysis},
  year = {2026},
  url = {https://github.com/rudder-research/framework}
}
```

---

## License

MIT
