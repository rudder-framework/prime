# Framework + PRISM Architecture

## The Flow

```
Framework                           PRISM                            Framework
(instruct)                       (calculate)                      (interpret)
    │                                │                                │
    │  config.json                   │                                │
    │  ┌─────────────────────┐       │                                │
    │  │ discipline: reaction│       │                                │
    │  │ signals: [...]      │       │                                │
    │  │ constants: {...}    │       │                                │
    │  └─────────────────────┘       │                                │
    │            +                   │                                │
    │  observations.parquet          │                                │
    │                                │                                │
    ├───────────────────────────────►│                                │
    │                                │                                │
    │                    Read config │                                │
    │                    See: discipline="reaction"                   │
    │                    Map to engines:                              │
    │                      • conversion                               │
    │                      • rate_constant                            │
    │                      • arrhenius_fit                            │
    │                      • ...                                      │
    │                                │                                │
    │                    Run engines │                                │
    │                    Write parquets                               │
    │                                │                                │
    │◄───────────────────────────────┤                                │
    │                                │                                │
    │  7 parquets returned:          │                                │
    │  • vector.parquet              │                                │
    │  • geometry.parquet            │                                │
    │  • dynamics.parquet            │                                │
    │  • state.parquet               │                                │
    │  • physics.parquet ◄── discipline results                       │
    │  • fields.parquet              │                                │
    │  • systems.parquet             │                                │
    │                                                                 │
    ├────────────────────────────────────────────────────────────────►│
    │                                                                 │
    │                                              SQL queries on parquets
    │                                              Format for display
    │                                              Tables, plots, export
    │                                                                 │
    │◄────────────────────────────────────────────────────────────────┤
    │                                                                 │
    ▼                                                                 ▼
  USER                                                             THESIS
```

## Responsibilities

### Framework (Instruct)
- User uploads data
- User selects discipline from dropdown
- Framework creates config.json with `discipline: "reaction"`
- Framework transforms data to observations.parquet
- Framework sends both to PRISM

### PRISM (Calculate)
- Reads config.json
- Sees `discipline: "reaction"`
- Looks up engine mapping: reaction → [conversion, rate_constant, arrhenius_fit, ...]
- Runs core engines (always)
- Runs discipline engines (mapped from config)
- Writes raw numbers to 7 parquets
- Returns parquets to Framework
- **No guessing. No interpretation. Just math.**

### Framework (Interpret)
- Receives parquets from PRISM
- Queries via SQL (DuckDB/Polars)
- Knows discipline context from original config
- Formats results for display
- Creates tables, plots, thesis sections
- Exports publication-ready figures

## Engine Mapping in PRISM

```python
# prism/capability.py

DISCIPLINE_ENGINES = {
    "reaction": [
        "conversion",
        "reaction_rate",
        "arrhenius_fit",
        "residence_time",
        "cstr_rate_constant",
        "yield",
        "selectivity",
        "damkohler",
    ],
    "transport": [
        "reynolds",
        "nusselt",
        "prandtl",
        "schmidt",
        "peclet",
    ],
    "balances": [
        "material_balance",
        "energy_balance",
        "component_balance",
        "extent_of_reaction",
    ],
    # ... etc
}

def get_engines_for_config(config: dict) -> list[str]:
    """Map discipline to engines. No guessing."""
    engines = CORE_ENGINES.copy()  # Always run core

    discipline = config.get("discipline")
    if discipline and discipline in DISCIPLINE_ENGINES:
        engines.extend(DISCIPLINE_ENGINES[discipline])

    return engines
```

## SQL Interpretation in Framework

```python
# framework/display/interpreter.py

import duckdb

def interpret_reaction_results(parquet_dir: str, config: dict):
    """SQL queries to extract and format reaction results."""

    con = duckdb.connect()

    # Get rate constants per entity (temperature)
    results = con.execute(f"""
        SELECT
            entity_id,
            AVG(conversion) as conversion,
            AVG(rate_constant) as k,
            ANY_VALUE(temperature) as T
        FROM '{parquet_dir}/physics.parquet'
        GROUP BY entity_id
        ORDER BY T
    """).fetchdf()

    # Get Arrhenius parameters (aggregated)
    arrhenius = con.execute(f"""
        SELECT
            activation_energy,
            pre_exponential,
            r_squared
        FROM '{parquet_dir}/physics.parquet'
        WHERE activation_energy IS NOT NULL
        LIMIT 1
    """).fetchone()

    return {
        "results_table": results,
        "Ea_J_mol": arrhenius[0],
        "A": arrhenius[1],
        "r_squared": arrhenius[2],
    }
```

## Config.json — The Contract

```json
{
  "discipline": "reaction",

  "signals": [...],
  "global_constants": {...},

  "window": {"size": 5, "stride": 1},
  "baseline": {"fraction": 0.1},
  "regime": {"n_regimes": 3},
  "state": {"n_basins": 2}
}
```

- `discipline` tells PRISM which engines to run
- PRISM doesn't interpret, just maps and calculates
- Framework created this config, Framework interprets the results

## No Guessing

| Component | Guesses? | Why |
|-----------|----------|-----|
| Framework (instruct) | No | User explicitly selects discipline |
| PRISM | No | Reads discipline from config, maps to engines |
| Framework (interpret) | No | Knows discipline from config it created |

The discipline flows through the entire pipeline explicitly. No inference. No heuristics. No magic.
