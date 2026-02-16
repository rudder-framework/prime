# Streaming Pipeline

Processes large historical datasets (months/years of sensor data) through Prime + Manifold in manageable partitions with checkpoint/resume.

This is **separate from** `prime/streaming/` (real-time WebSocket analysis). This pipeline processes *historical* data that's too large to fit in memory at once.

## Directory Structure

```
streaming/
├── run.py              # CLI entry point
├── config/
│   ├── lumo.yaml       # LUMO dataset config
│   └── template.yaml   # Annotated template for new datasets
├── converters/
│   ├── base_converter.py    # Abstract base class
│   └── mat_converter.py     # MATLAB .mat files (vectorized)
├── workers/
│   ├── ingest_worker.py     # Raw files → observations.parquet
│   ├── compute_worker.py    # Prime stages 2-6 per partition
│   └── coordinator.py       # Orchestrator with checkpoint/resume
└── state/
    └── .gitkeep             # Runtime state lives here
```

## How to Run

```bash
# Full pipeline (ingest + compute)
python streaming/run.py streaming/config/lumo.yaml

# Ingest only (raw → partitioned parquet)
python streaming/run.py streaming/config/lumo.yaml --ingest-only

# Compute only (partitions must be ingested)
python streaming/run.py streaming/config/lumo.yaml --compute-only

# Skip Manifold (typology + classification only)
python streaming/run.py streaming/config/lumo.yaml --skip-manifold

# Check progress
python streaming/run.py streaming/config/lumo.yaml --status

# Reset state (start over)
python streaming/run.py streaming/config/lumo.yaml --reset
```

## Pipeline Phases

### 1. Ingest

Raw files are grouped into **weekly partitions** by file modification time. Each partition gets its own `observations.parquet` using the buffer → flush → concat pattern for constant memory usage.

```
raw/2023-01-01.mat  ─┐
raw/2023-01-02.mat  ─┤ → partitions/2023-W01/observations.parquet
raw/2023-01-07.mat  ─┘

raw/2023-01-08.mat  ─┐
raw/2023-01-14.mat  ─┤ → partitions/2023-W02/observations.parquet
raw/2023-01-13.mat  ─┘
```

### 2. Bootstrap

The first partition runs full Prime typology + classification. Since sensor types don't change over time (configurable via `bootstrap.reuse_typology`), this typology is reused for all subsequent partitions.

### 3. Compute

Each partition runs Prime stages 2-6:
- **[2] Typology raw** — 31 statistical measures per signal (or bootstrap copy)
- **[3] Classification** — 10 dimensions, dual classification
- **[4] Manifest** — engine selection per signal
- **[5] Manifold** — compute engine outputs
- **[6] SQL analysis** — DuckDB queries on Manifold output

Partition boundary overlap: the compute worker prepends trailing samples from the previous partition (configurable via `compute.overlap_samples`) to avoid edge effects.

## Resume Behavior

State is persisted to `pipeline_state.json` after each partition completes. If the pipeline is interrupted:

- Re-running the same command picks up where it left off
- Already-ingested partitions are skipped
- Already-computed partitions are skipped
- Use `--reset` to start from scratch

## Config Format

See `streaming/config/template.yaml` for all options. Key settings:

| Key | Description | Default |
|-----|-------------|---------|
| `paths.raw_dir` | Raw file directory | required |
| `paths.output_dir` | Partition output directory | required |
| `paths.file_pattern` | Glob for raw files | `**/*.mat` |
| `converter.type` | Converter type | `mat` |
| `partitioning.strategy` | How to group files | `weekly` |
| `partitioning.files_per_flush` | Buffer size | `50` |
| `bootstrap.reuse_typology` | Copy typology from first partition | `true` |
| `compute.overlap_samples` | Boundary overlap | `2048` |
| `compute.skip_manifold` | Skip Manifold compute | `false` |

## Output Structure

```
~/domains/lumo/streaming/
├── pipeline_state.json
└── partitions/
    ├── 2023-W01/
    │   ├── observations.parquet
    │   ├── typology_raw.parquet
    │   ├── typology.parquet
    │   ├── manifest.yaml
    │   └── output/
    │       ├── state_geometry.parquet
    │       ├── eigendecomposition.parquet
    │       └── sql/
    │           └── *.md
    ├── 2023-W02/
    │   └── ...
    └── 2023-W48/
        └── ...
```

Each partition is a self-contained domain directory that can be queried independently with `prime query`.
