# ORTHON - AI Instructions

## What ORTHON Is

ORTHON = Brain. Orchestration, manifest generation, interpretation.
PRISM = Muscle. Pure computation, no decisions.

ORTHON does NOT compute. PRISM computes.

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

## Architecture

```
orthon/
├── config/
│   ├── manifest.py      # Single source of truth (53 engines)
│   ├── domains.py       # Physics domains (7 domains)
│   └── recommender.py   # Config recommendations
│
├── ingest/
│   ├── paths.py             # Fixed output paths (NO EXCEPTIONS)
│   ├── streaming.py         # Universal streaming ingestor
│   └── manifest_generator.py # AI auto-generates manifest
│
├── intake/                   # UI file handling only
│   ├── upload.py
│   ├── validate.py
│   └── transformer.py
│
├── analysis/
│   └── baseline_discovery.py # Baseline modes (first_n%, stable_discovery)
│
└── services/
    └── manifest_builder.py
```

## Key Files

### orthon/ingest/paths.py
```python
OUTPUT_DIR = Path("/Users/jasonrudder/prism/data")
OBSERVATIONS_PATH = OUTPUT_DIR / "observations.parquet"
MANIFEST_PATH = OUTPUT_DIR / "manifest.yaml"
```

### orthon/config/manifest.py
- 53 ENGINES (ALL enabled, no exceptions)
- Pydantic models: Manifest, PRISMConfig, BaselineConfig
- Factory: create_manifest(), generate_full_manifest()

### orthon/analysis/baseline_discovery.py
- BaselineMode: first_n_percent, stable_discovery, rolling, etc.
- discover_stable_baseline() for markets/unknown systems
- get_baseline() universal getter

## Baseline Modes

| Mode | Use Case |
|------|----------|
| first_n_percent | Industrial (pump, bearing) - known healthy start |
| stable_discovery | Markets, bioreactor - unknown healthy state |
| rolling | Gradual drift systems |

## Domain Data Location

Raw domain data goes to: `/Users/jasonrudder/domains/`

```
domains/
├── bearing/       # FEMTO, IMS
├── turbomachinery/ # C-MAPSS
├── industrial/    # SKAB, MetroPT
├── finance/       # Fama-French
└── misc/          # Docs, scripts
```

## PRISM Outputs (12 parquet files)

PRISM writes to: `/Users/jasonrudder/prism/data/` (same as input)

### Geometry
- primitives.parquet
- primitives_pairs.parquet
- geometry.parquet
- topology.parquet
- manifold.parquet

### Dynamics
- dynamics.parquet
- information_flow.parquet
- observations_enriched.parquet

### Energy
- physics.parquet

### SQL
- zscore.parquet
- statistics.parquet
- correlation.parquet
- regime_assignment.parquet

## Commands

```bash
# ORTHON generates manifest
python -m orthon.ingest.manifest_generator /path/to/raw/data

# ORTHON ingests data
python -m orthon.ingest.streaming manifest.yaml

# PRISM computes (in prism repo)
python -m prism manifest.yaml
```

## Rules

1. ALL 53 engines run. Always. No exceptions.
2. Insufficient data → return NaN, never skip
3. No domain-specific logic in compute
4. No parallel/RAM management in ORTHON (PRISM handles this)
5. Output paths are FIXED - never change them

## Do NOT

- Skip engines based on domain
- Gate metrics by observation count
- Write to subdirectories of /Users/jasonrudder/prism/data/
- Add RAM management to ORTHON
- Make ORTHON compute anything
