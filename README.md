# ORTHON

**Signal Classification & Diagnostic Interpreter for PRISM**

ORTHON is the brain; PRISM is the muscle. ORTHON classifies signals and interprets results. PRISM computes features.

---

## What ORTHON Does

1. **Classifies signals** - Computes 27 statistical measures, classifies across 10 dimensions using two-stage classification:
   - **Stage 1 (PR5)**: Discrete/sparse detection (CONSTANT, BINARY, DISCRETE, IMPULSIVE, EVENT)
   - **Stage 2 (PR4)**: Continuous classification (TRENDING, PERIODIC, CHAOTIC, RANDOM, etc.)
   - **PR8**: Robust CONSTANT detection using coefficient of variation
2. **Generates manifests** - Tells PRISM which engines to run per signal using **inclusive philosophy** ("If it's a maybe, run it")
3. **Multi-scale representation** - PR9/PR10: Data-driven window/stride, spectral vs trajectory based on characteristic_time
4. **Interprets results** - Applies Lyapunov-based trajectory classification to PRISM outputs

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline: observations → typology → manifest
python -m orthon.pipeline data/observations.parquet data/

# Or run stages individually:
python -m orthon.ingest.typology_raw data/observations.parquet data/typology_raw.parquet
python -m orthon.ingest.manifest_generator data/typology.parquet data/manifest.yaml

# Validate observations
python -m orthon.ingest.validate_observations data/observations.parquet
```

---

## Schema (v2.4)

```
observations.parquet
├── cohort     (str)     # Optional: grouping key (engine_1, pump_A)
├── signal_id  (str)     # Required: signal name (temp, pressure, sensor_01)
├── I          (UInt32)  # Required: sequential index per (cohort, signal_id)
└── value      (Float64) # Required: measurement
```

**Unique time series = `(cohort, signal_id)`**

---

## Architecture

```
ORTHON = Brain (classification, interpretation, orchestration)
PRISM  = Muscle (pure computation, no decisions)

observations.parquet  →  ORTHON  →  typology.parquet + manifest.yaml
                              ↓
                           PRISM
                              ↓
                    ORTHON interprets outputs
```

---

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed technical documentation:
- Typology system (27 measures, 10 classification dimensions)
- Manifest structure v2.4 (system_window, multi-scale representation)
- Classification SQL views
- Engine selection rules (inclusive philosophy)
- Lyapunov-based trajectory classification

---

## Engine Gating (Inclusive Philosophy)

> "If it's a maybe, run it." — Only CONSTANT signals remove all engines.

| Temporal Pattern | Key Engines Added |
|------------------|-------------------|
| TRENDING | hurst, rate_of_change, trend_r2, cusum, sample_entropy, acf_decay |
| PERIODIC | harmonics, thd, frequency_bands, phase_coherence, snr |
| CHAOTIC | lyapunov, correlation_dimension, recurrence_rate, perm_entropy |
| RANDOM | spectral_entropy, band_power, sample_entropy, acf_decay |
| CONSTANT | **removes all** (no information to extract) |
| BINARY | transition_count, duty_cycle, switching_frequency |
| DISCRETE | level_histogram, transition_matrix, dwell_times |
| IMPULSIVE | peak_detection, inter_arrival, envelope, rise_time |

---

## Key Components

| Component | Purpose |
|-----------|---------|
| `orthon/ingest/typology_raw.py` | Computes 27 statistical measures per signal |
| `orthon/typology/` | Signal classification (PR4/PR5/PR8) |
| `orthon/manifest/generator.py` | Creates v2.2 manifest with inclusive engine gating |
| `orthon/window/manifest_generator.py` | Creates v2.4 manifest with system_window |
| `orthon/window/characteristic_time.py` | Data-driven window from characteristic_time |
| `orthon/config/typology_config.py` | All classification thresholds |

---

## License

MIT
