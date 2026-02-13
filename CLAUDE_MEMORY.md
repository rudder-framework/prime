# Claude AI Memory - PRISM/Rudder Framework Architecture

**Last Updated:** 2026-02-05
**Session Summary:** Refactored runners to entry_points with ordered naming, identified missing runners

---

## Architecture Principle

```
PRISM = Muscle (pure computation, no decisions, no classification)
Rudder Framework = Brain (orchestration, typology, classification, interpretation)

PRISM computes numbers. Rudder Framework classifies.
```

---

## What We Just Did

### 1. Refactored Entry Points (Completed)

Moved stage runners to `entry_points/` with ordered naming. Entry points are **pure orchestration** - they call engines for computation, no embedded calculations.

### 2. PRISM Entry Points Created

Location: `/Users/jasonrudder/prism/prism/entry_points/`

| Stage | File | Output | Calls Engine |
|-------|------|--------|--------------|
| 01 | `stage_01_signal_vector.py` | `signal_vector.parquet` | `engines/signal/*` via registry |
| 02 | `stage_02_state_vector.py` | `state_vector.parquet` | `engines/state/centroid.py` |
| 03 | `stage_03_state_geometry.py` | `state_geometry.parquet` | `engines/state/eigendecomp.py` |
| 04 | `stage_04_cohorts.py` | `cohorts.parquet` | Pure aggregation |

**Old stages deprecated:** `prism/pipeline/stages/` marked with DEPRECATED.md

### 3. Rudder Framework Entry Points Created

Location: `/Users/jasonrudder/framework/framework/entry_points/`

| Stage | File | Output | Calls Module |
|-------|------|--------|--------------|
| 01 | `stage_01_validate.py` | `observations_validated.parquet` | `core/validation.py` |
| 02 | `stage_02_typology.py` | `typology_raw.parquet` | `ingest/typology_raw.py` |
| 03 | `stage_03_classify.py` | `typology.parquet` | `typology/discrete_sparse.py`, `typology/level2_corrections.py` |
| 04 | `stage_04_manifest.py` | `manifest.yaml` | `manifest/generator.py` |
| 05 | `stage_05_diagnostic.py` | `diagnostic_report.txt` | `engines/*` |

### 4. Rudder Framework Engines Implemented (Previous Session)

Location: `/Users/jasonrudder/framework/framework/engines/`

| Engine | File | Purpose |
|--------|------|---------|
| Level 0 | `typology_engine.py` | System classification (ACCUMULATION/DEGRADATION/CONSERVATION) |
| Level 1 | `stationarity_engine.py` | KPSS/ADF stationarity tests |
| Level 2 | `classification_engine.py` | Signal behavior classification |
| Geometry | `signal_geometry.py` | Eigenstructure (eff_dim, alignment) |
| Mass | `mass_engine.py` | Total variance/energy |
| Structure | `structure_engine.py` | Structure = Geometry × Mass |
| Stability | `stability_engine.py` | Lyapunov, CSD detection |
| Tipping | `tipping_engine.py` | B-tipping vs R-tipping (Granger causality) |
| Spin Glass | `spin_glass.py` | Parisi framework phases |
| Report | `diagnostic_report.py` | Unified diagnostic combining all engines |

---

## WHERE WE LEFT OFF

### Missing PRISM Runners (Need to Create)

One runner per parquet file. These are missing:

| Stage | Output File | Engine Source |
|-------|-------------|---------------|
| 05 | `signal_geometry.parquet` | `engines/signal_geometry.py` |
| 06 | `signal_pairwise.parquet` | `engines/signal_pairwise.py` |
| 07 | `geometry_dynamics.parquet` | `engines/geometry_dynamics.py` |
| 08 | `lyapunov.parquet` | `engines/parallel/dynamics_runner.py` |
| 09 | `dynamics.parquet` | `engines/parallel/dynamics_runner.py` |
| 10 | `information_flow.parquet` | `engines/parallel/information_flow_runner.py` |
| 11 | `topology.parquet` | `engines/parallel/topology_runner.py` |
| 12 | `zscore.parquet` | `engines/sql/` |
| 13 | `statistics.parquet` | `engines/sql/` |
| 14 | `correlation.parquet` | `engines/sql/` |

### Task: Create these runners following the pattern:

```python
"""
Stage XX: <Name> Entry Point
============================

Pure orchestration - calls <engine> for computation.
Stages: <input> → <output.parquet>
"""

# Import engine
from prism.engines.<path> import compute

def run(...):
    # 1. Load input
    # 2. Call engine
    # 3. Write output
    pass
```

---

## Key Files Reference

### PRISM
- Entry points: `/Users/jasonrudder/prism/prism/entry_points/`
- Engines: `/Users/jasonrudder/prism/prism/engines/`
- Signal engines: `/Users/jasonrudder/prism/prism/engines/signal/`
- State engines: `/Users/jasonrudder/prism/prism/engines/state/`
- Parallel runners: `/Users/jasonrudder/prism/prism/engines/parallel/`
- Deprecated stages: `/Users/jasonrudder/prism/prism/pipeline/stages/` (DO NOT USE)

### Rudder Framework
- Entry points: `/Users/jasonrudder/framework/framework/entry_points/`
- Engines: `/Users/jasonrudder/framework/framework/engines/`
- Manifest generator: `/Users/jasonrudder/framework/framework/manifest/generator.py`
- Typology: `/Users/jasonrudder/framework/framework/typology/`
- Core validation: `/Users/jasonrudder/framework/framework/core/validation.py`

---

## Pipeline Flow

```
Rudder Framework Pipeline:
observations.parquet
    → stage_01_validate → observations_validated.parquet
    → stage_02_typology → typology_raw.parquet
    → stage_03_classify → typology.parquet
    → stage_04_manifest → manifest.yaml

PRISM Pipeline:
observations.parquet + typology.parquet + manifest.yaml
    → stage_01_signal_vector → signal_vector.parquet
    → stage_02_state_vector → state_vector.parquet
    → stage_03_state_geometry → state_geometry.parquet
    → stage_04_cohorts → cohorts.parquet
    → [MISSING: stages 05-14 for remaining parquet files]

Back to Rudder Framework:
PRISM outputs → stage_05_diagnostic → diagnostic_report.txt
```

---

## Key Concepts

- **Structure = Geometry × Mass** - Both can fail independently
- **B-tipping** (geometry→mass): CSD provides early warning
- **R-tipping** (mass→geometry): NO early warning
- **Spin Glass phases**: Paramagnetic (healthy), Ferromagnetic (trending), Spin Glass (fragile), Mixed (critical)
- **effective_dim**: Participation ratio - 63% importance in RUL prediction

---

## Commands

```bash
# Rudder Framework pipeline
python -m framework.entry_points.stage_01_validate observations.parquet -o validated.parquet
python -m framework.entry_points.stage_02_typology observations.parquet -o typology_raw.parquet
python -m framework.entry_points.stage_03_classify typology_raw.parquet -o typology.parquet
python -m framework.entry_points.stage_04_manifest typology.parquet -o manifest.yaml

# PRISM pipeline
python -m prism.entry_points.signal_vector manifest.yaml
python -m prism.entry_points.stage_02_state_vector signal_vector.parquet typology.parquet
python -m prism.entry_points.stage_03_state_geometry signal_vector.parquet state_vector.parquet
python -m prism.entry_points.stage_04_cohorts state_vector.parquet state_geometry.parquet cohorts.parquet
```
