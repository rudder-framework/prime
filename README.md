# ORTHON

**Signal Classification & Diagnostic Interpreter for PRISM**

ORTHON is the brain; PRISM is the muscle. ORTHON classifies signals and interprets results. PRISM computes features.

---

## What ORTHON Does

1. **Classifies signals** - Computes 27 statistical measures, classifies across 10 dimensions
2. **Generates manifests** - Tells PRISM which engines to run per signal
3. **Interprets results** - Applies Lyapunov-based trajectory classification to PRISM outputs
4. **Validates data** - Ensures observations.parquet conforms to schema v2.1

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

## Schema (v2.1)

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
- Manifest structure (v2.1 nested cohorts)
- Classification SQL views
- Engine selection rules
- Lyapunov-based trajectory classification

---

## Key Components

| Component | Purpose |
|-----------|---------|
| `orthon/ingest/typology_raw.py` | Computes 27 statistical measures |
| `orthon/corrections/level2_corrections.py` | 6-gate periodicity validation |
| `orthon/ingest/manifest_generator.py` | Creates v2.1 manifest |
| `orthon/sql/classification.sql` | Lyapunov trajectory classification |

---

## License

MIT
