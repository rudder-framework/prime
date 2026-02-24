# Dead Code Audit — 2026-02-24

## Summary

| Category | Count |
|----------|-------|
| Dead files (never imported/called) | 27 |
| Dead functions (defined but never called) | 19 |
| Dead imports (imported but unused) | 128 |
| Orphan SQL files (not loaded by runner) | 18 |
| Dead SQL views (defined but never consumed) | 16 |
| Orphan parquets (written but never read) | 6 |
| Dead config keys/sections | 12 |
| Duplicate code pairs | 4 |

---

## 1. Dead Files (never imported/called)

### 1A. Confirmed dead Python files — zero importers, not wired into CLI

| File | Last Modified | Description | Reason |
|------|---------------|-------------|--------|
| `prime/config/stability_config.py` | 2026-02-13 | `StabilityConfig` / `TypologyConfig` dataclasses + YAML schema | Zero importers anywhere in repo |
| `prime/core/server.py` | 2026-02-16 | Standalone FastAPI server (LLM-powered suggestions, health, schema) | Superseded by `core/api.py`; not in `pyproject.toml` scripts |
| `prime/core/pipeline.py` | 2026-02-23 | `ObservationPipeline` class + `process_observations()` | Superseded by `prime/pipeline.py`; `core/__init__.py` re-exports it but nothing imports from `prime.core` |
| `prime/typology/control_detection.py` | 2026-02-16 | CONTROL vs RESPONSE signal classifier | Zero importers |
| `prime/typology/transition_detection.py` | 2026-02-20 | Canary detector for typology character shifts | Zero importers |
| `prime/typology/classification_stability.py` | 2026-02-15 | Classification consistency checker across windows | Only imported by its own test file |
| `prime/ml/lasso.py` | 2026-02-15 | L1-regularized feature selection (pure numpy) | Zero importers |
| `prime/ml/entry_points/evaluate_test.py` | 2026-02-16 | C-MAPSS test evaluation script | Zero importers; not in `pyproject.toml` scripts |
| `prime/ingest/manifest_generator_v2.py` | 2026-02-16 | Manifest Generator v2 (superseded) | Replaced by `prime/manifest/generator.py` (v2.7) |
| `prime/ingest/schema_enforcer.py` | 2026-02-20 | Schema validation/repair for observations.parquet | Zero importers; standalone `__main__` only |
| `prime/ingest/streaming.py` | 2026-02-20 | Streaming ingest module | Zero external importers (separate `prime/streaming/` exists) |
| `prime/ingest/data_confirmation.py` | 2026-02-20 | Data confirmation prompts | Only self-referencing docstring; zero importers |
| `prime/notebook/generator.py` | 2026-02-23 | Jupyter notebook generator from SQL reports | Zero importers; `notebook/__init__.py` is empty |
| `prime/sql/audit.py` | 2026-02-23 | SQL file pass/fail tester | Zero importers; standalone `__main__` only |
| `prime/entry_points/stage_00_regime_normalize.py` | 2026-02-21 | Regime normalization entry point | Not in `entry_points/__init__.py` or `pyproject.toml` |
| `packages/dynamics/src/dynamics/embedding_cache.py` | 2026-02-24 | FTLE embedding parameter cache | Zero importers; not in `dynamics/__init__.py` |
| `prime/shared/engine_registry.py` | 2026-02-16 | Engine specs, category-based selection | Re-exported by `shared/__init__.py` but zero external consumers |
| `prime/shared/window_config.py` | 2026-02-20 | `WindowConfig`, auto-detection, domain defaults | Re-exported by `shared/__init__.py` but zero external consumers |
| `prime/shared/config_schema.py` | 2026-02-15 | Pydantic models (`ManifoldConfig`, `SignalInfo`, etc.) | Only `DISCIPLINES` is imported externally; everything else dead |
| `prime/manifest/system_window.py` | 2026-02-13 | Multi-scale representation window computation | Zero importers; generator.py computes window inline |
| `prime/manifest/characteristic_time.py` | 2026-02-13 | Window from ACF, frequency, period | Only imported by its own test file |
| `prime/manifest/domain_clock.py` | 2026-02-22 | Auto-detect domain frequency | Only imported by `window_recommender.py` (also dead) |
| `prime/manifest/window_recommender.py` | 2026-02-16 | Window sizing from typology raw measures | Only imported by its own test file |

### 1B. Dead module cluster — `prime/analysis/` (entire directory)

| File | Last Modified | Description |
|------|---------------|-------------|
| `prime/analysis/__init__.py` | 2026-02-14 | Re-exports from all submodules |
| `prime/analysis/study.py` | 2026-02-14 | Study orchestrator |
| `prime/analysis/canary.py` | 2026-02-20 | Canary signal detection |
| `prime/analysis/twenty_twenty.py` | 2026-02-16 | 20/20 predictive validation |
| `prime/analysis/thermodynamics.py` | 2026-02-16 | Thermodynamic interpretation |
| `prime/analysis/window_optimization.py` | 2026-02-16 | Raw eigendecomp grid search |
| `prime/analysis/window_optimization_manifold.py` | 2026-02-16 | Full Manifold grid search |

**Reason**: `prime.analysis` is never imported by any file outside itself. No entry point, CLI command, or pipeline step references it. Each file has `__main__` blocks but nothing in the import graph reaches them.

### 1C. Dead static assets (explorer)

| File | Last Modified | Reason |
|------|---------------|--------|
| `prime/explorer/static/index.html` | 2026-02-15 | Not served by `server.py` (which serves `explorer.html`) |
| `prime/explorer/static/index_v2.html` | 2026-02-20 | Not served by `server.py` |
| `prime/explorer/static/wizard.html` | 2026-02-16 | Not served by `server.py` |
| `prime/explorer/static/queries-v2.js` | 2026-02-20 | Only loaded by `index_v2.html` (itself dead) |

### 1D. Dead config files

| File | Last Modified | Reason |
|------|---------------|--------|
| `prime/config/engine_rules.yaml` | 2026-02-15 | Zero references from any Python code |
| `prime/config/TYPOLOGY_CONFIG.yaml` | 2026-02-13 | Zero references from any Python code |

---

## 2. Dead Functions (defined but never called)

### 2A. Superseded batch wrappers

These functions wrap lower-level functions that ARE used, but the wrappers themselves are never called.

| File | Function | Reason |
|------|----------|--------|
| `prime/manifest/characteristic_time.py:236` | `add_window_columns()` | Pandas `apply()` wrapper around `compute_window_config()`; callers use the underlying function directly |
| `prime/manifest/system_window.py:208` | `compute_system_representation()` | Batch wrapper around `compute_system_window()` + `compute_signal_representation()`; never integrated |
| `prime/manifest/system_window.py:259` | `summarize_representations()` | Diagnostic function; zero callers |
| `prime/typology/constant_detection.py:273` | `patch_constant_detection()` | Pandas `apply()` wrapper around `classify_constant_from_row()`; callers use row-level function |
| `packages/vector/src/vector/cohort.py:127` | `compute_cohort_batch()` | Batch wrapper; orchestration loops over windows itself |
| `packages/vector/src/vector/signal.py:154` | `compute_signal_batch()` | Python fallback; superseded by Rust fast-path `pmtvs_vector.compute_signal_batch` |

### 2B. Planned but never integrated

| File | Function | Reason |
|------|----------|--------|
| `prime/cohorts/detection.py:876` | `classify_coupling_trajectory()` | Zero callers; planned cohort analysis feature |
| `prime/cohorts/detection.py:969` | `generate_cohort_report()` | Zero callers; convenience wrapper never wired in |
| `prime/engines/mass_engine.py:147` | `classify_mass_state()` | Module imported for `compute_mass_trajectory()` only; this function unused |
| `prime/engines/stationarity_engine.py:147` | `compute_stationarity_metrics()` | Module imported for `classify_stationarity()` only; this function unused |
| `prime/manifest/domain_clock.py:423` | `characterize_domain()` | Convenience wrapper; `get_domain_window_config()` preferred |
| `prime/streaming/data_sources.py:334` | `get_source_info()` | Zero callers in streaming system |

### 2C. Legacy/superseded

| File | Function | Reason |
|------|----------|--------|
| `prime/core/pipeline.py:36` | `_check_dependencies()` | Duplicate of same function in `prime/pipeline.py:11`; this copy never called |
| `packages/dynamics/src/dynamics/embedding_cache.py:309` | `create_cache_for_signal()` | Factory function; callers construct `EmbeddingCache` directly |
| `packages/dynamics/src/dynamics/embedding_cache.py:351` | `rolling_ftle_with_cache()` | Superseded by Rust fast-path in `ftle.py` |

### 2D. Unused utilities

| File | Function | Reason |
|------|----------|--------|
| `packages/orchestration/src/orchestration/checksums.py:70` | `compare_checksums()` | Comparison feature never automated; `generate_checksums()` IS used |
| `packages/orchestration/src/orchestration/checksums.py:119` | `print_comparison()` | Depends on dead `compare_checksums()` |
| `prime/ml/lasso.py:315` | `compute_mutual_info()` | Entire `lasso.py` module is dead |

### 2E. Dead config accessor functions

| File | Function | Reason |
|------|----------|--------|
| `prime/config/typology_config.py:342` | `get_engine_adjustments()` | Exported but never called; manifest generator uses its own `apply_engine_adjustments()` |
| `prime/config/typology_config.py:351` | `get_viz_adjustments()` | Exported but never called |
| `prime/config/discrete_sparse_config.py:143` | `get_discrete_threshold()` | Exported but never called; `discrete_sparse.py` accesses `DISCRETE_SPARSE_CONFIG` directly |

---

## 3. Dead Imports (imported but unused)

128 dead imports found. Grouped by severity.

### 3A. Heavy/expensive unused imports (import cost matters)

| File | Import | Unused Name(s) |
|------|--------|-----------------|
| `prime/analysis/thermodynamics.py:36` | `from scipy import stats` | `stats` |
| `prime/ingest/validation.py:30` | `import numpy as np` | `np` |
| `prime/early_warning/ml_predictor.py:18` | `from sklearn.model_selection import cross_val_score, StratifiedKFold` | `cross_val_score`, `StratifiedKFold` |
| `prime/ml/entry_points/benchmark.py:19` | `from sklearn.metrics import r2_score` | `r2_score` |
| `prime/entry_points/stage_04_manifest.py:13` | `import polars as pl` | `pl` |
| `prime/entry_points/stage_11_fetch.py:13` | `import polars as pl` | `pl` |

### 3B. Unused names from otherwise-used imports

| File | Import Line | Unused Name(s) |
|------|-------------|-----------------|
| `prime/core/api.py:23` | `from fastapi.responses import ...` | `JSONResponse` |
| `prime/core/api.py:26` | `from prime.core.data_reader import ...` | `DataProfile` |
| `prime/core/api.py:39` | `from prime.utils.index_detection import ...` | `IndexDetector` |
| `prime/core/api.py:41` | `from prime.services.state_analyzer import ...` | `StateThresholds` |
| `prime/core/api.py:1488` | `from prime.services.concierge import ...` | `ConciergeResponse` |
| `prime/core/api.py:2512` | `from prime.services.tuning_service import ...` | `get_tuning_service` |
| `prime/engines/diagnostic_report.py:23` | `from .typology_engine import ...` | `SystemType` |
| `prime/engines/diagnostic_report.py:28` | `from .structure_engine import ...` | `interpret_coupling` |
| `prime/engines/diagnostic_report.py:30` | `from .tipping_engine import ...` | `interpret_tipping_for_domain` |
| `prime/engines/diagnostic_report.py:31` | `from .spin_glass import ...` | `generate_spin_glass_report` |
| `prime/entry_points/stage_11_fetch.py:17` | `from prime.ingest.data_reader import ...` | `DataProfile` |
| `prime/entry_points/stage_11_fetch.py:18` | `from prime.ingest.validate_observations import ...` | `ValidationStatus` |

### 3C. Unused typing imports (74 instances)

The most common pattern. These are low-impact but noisy.

| File | Unused Type(s) |
|------|----------------|
| `prime/core/pipeline.py:29` | `Tuple` |
| `prime/core/server.py:219` | `tempfile` (stdlib) |
| `prime/services/job_manager.py:23,29,33` | `os`, `Callable`, `asyncio` |
| `prime/services/concierge.py:16` | `Any` |
| `prime/services/dynamics_interpreter.py:21` | `Tuple` |
| `prime/services/fingerprint_service.py:34` | `json` |
| `prime/services/tuning_service.py:29,32` | `field`, `json` |
| `prime/engines/signal_geometry.py:16` | `Optional` |
| `prime/engines/spin_glass.py:21` | `Dict` |
| `prime/engines/stationarity_engine.py:20,21` | `Tuple`, `warnings` |
| `prime/engines/tipping_engine.py:15` | `Optional`, `List` |
| `prime/engines/typology_engine.py:16` | `Optional`, `Tuple` |
| `prime/engines/diagnostic_report.py:20` | `List` |
| `prime/ingest/upload.py:9` | `List`, `Dict`, `Any` |
| `prime/ingest/validate_observations.py:26` | `Dict`, `Any` |
| `prime/cohorts/detection.py:34` | `defaultdict` |
| `prime/cohorts/discovery.py:26,28,29` | `Set`, `defaultdict`, `json` |
| `prime/config/recommender.py:10` | `List` |
| `prime/explorer/models.py:9,10` | `field`, `Optional` |
| `prime/inspection/results_validator.py:13` | `Dict`, `Any` |
| `prime/manifest/system_window.py:16` | `Tuple` |
| `prime/ml/entry_points/ablation.py:23` | `Tuple` |
| `prime/ml/entry_points/features.py:30` | `np` |
| `prime/ml/entry_points/predict.py:19` | `Optional` |
| `prime/early_warning/ml_predictor.py:10,11,14` | `field`, `Any`, `Path` |
| `prime/shared/physics_constants.py:16` | `field` |
| `prime/shared/window_config.py:22,23` | `Any`, `math` |
| `prime/streaming/analyzers.py:10,11` | `datetime`, `List` |
| `prime/streaming/cli.py:13` | `Path` |
| `prime/streaming/websocket_server.py:9,14` | `JSONResponse`, `Dict`, `Any` |
| `prime/utils/index_detection.py:15,24` | `re`, `pd` |
| `prime/entry_points/csv_to_atlas.py:36` | `sys` |
| `prime/entry_points/stage_03_classify.py:16` | `Optional` |
| `prime/entry_points/stage_05_diagnostic.py:22` | `np` |
| `prime/entry_points/stage_08_alert.py:16` | `List` |
| `prime/entry_points/stage_10_inspect.py:15` | `Optional` |
| `prime/entry_points/stage_13_train.py:16` | `Optional` |
| `prime/analysis/window_optimization.py:49` | `Optional` |
| `prime/analysis/window_optimization_manifold.py:47,54` | `sys`, `Tuple`, `Any` |
| `prime/sql/audit.py:5` | `traceback` |
| `packages/eigendecomp/src/eigendecomp/flatten.py:10` | `Optional` |
| `packages/breaks/src/breaks/detection.py:14` | `List` |
| `packages/baseline/src/baseline/reference.py:12` | `List` |
| `packages/baseline/src/baseline/segments.py:9` | `Optional` |
| `packages/divergence/src/divergence/divergence.py:9` | `Optional` |
| `packages/pairwise/src/pairwise/coloading.py:14` | `Optional`, `Tuple` |
| `packages/typology/src/typology/classify.py:26` | `Optional` |
| `packages/typology/src/typology/window.py:16` | `Optional` |
| `packages/fleet/src/fleet/analysis.py:9` | `Optional` |
| `packages/vector/src/vector/cohort.py:22` | `Optional` |

### 3D. Unused `import pytest` in test files (11 instances)

These import pytest but don't use fixtures or marks. Not harmful (pytest autodiscovery still works), but unnecessary.

| File |
|------|
| `prime/typology/tests/test_config_corrections.py` (also has unused `math`) |
| `prime/typology/tests/test_constant_detection.py` |
| `prime/typology/tests/test_discrete_sparse.py` |
| `prime/manifest/tests/test_characteristic_time.py` |
| `packages/breaks/tests/test_breaks.py` |
| `packages/topology/tests/test_topology.py` |
| `packages/dynamics/tests/test_dynamics.py` |
| `packages/ridge/tests/test_ridge.py` |
| `packages/baseline/tests/test_baseline.py` |
| `packages/thermodynamics/tests/test_thermodynamics.py` |
| `packages/divergence/tests/test_divergence.py` |

### 3E. Broken imports (modules that don't exist)

These are lazy imports inside `prime/core/api.py` function bodies. The server starts fine, but these routes crash at runtime:

| File:Line | Import | Status |
|-----------|--------|--------|
| `prime/core/api.py:~1031` | `from prime.services.manifest_builder import build_manifest_from_units` | Module does not exist |
| `prime/core/api.py:~2113` | `from prime.services.compute_pipeline import ComputePipeline, get_compute_pipeline` | Module does not exist |
| `prime/core/api.py:~2115` | `from prime.ingest.config_generator import generate_manifest` | Module does not exist |
| `prime/core/api.py:~2116` | `from prime.ingest.transform import IntakeTransformer` | Name does not exist in module |
| `prime/core/api.py:~2116` | `from prime.ingest.transform import prepare_for_manifold` | Name does not exist in module |

---

## 4. Orphan SQL (not loaded by runner)

The main SQL runner (`prime/sql/runner.py`) only scans two directories: `sql/layers/*.sql` and `sql/reports/*.sql`.

### 4A. SQL files in unscanned directories — confirmed dead

| SQL File | Directory | Status |
|----------|-----------|--------|
| `prime/sql/_legacy/00_run_all.sql` | `_legacy/` | DEPRECATED header; only scanned by dead `audit.py` |
| `prime/sql/_legacy/00_configuration_audit.sql` | `_legacy/` | Only scanned by dead `audit.py` |
| `prime/sql/_legacy/06_physics.sql` | `_legacy/` | Replaced by `layers/00_physics_compat.sql` |
| `prime/sql/_legacy/25_sensitivity_analysis.sql` | `_legacy/` | Only scanned by dead `audit.py` |
| `prime/sql/_legacy/30_dynamics_stability.sql` | `_legacy/` | Only scanned by dead `audit.py` |
| `prime/sql/_legacy/31_regime_transitions.sql` | `_legacy/` | Only scanned by dead `audit.py` |
| `prime/sql/_legacy/32_basin_stability.sql` | `_legacy/` | Only scanned by dead `audit.py` |
| `prime/sql/_legacy/33_birth_certificate.sql` | `_legacy/` | Only scanned by dead `audit.py` |
| `prime/sql/_legacy/40_topology_departure.sql` | `_legacy/` | Only scanned by dead `audit.py` |
| `prime/sql/_legacy/50_information_departure.sql` | `_legacy/` | Only scanned by dead `audit.py` |
| `prime/sql/ml/11_ml_features.sql` | `ml/` | Not scanned by runner; zero Python references |
| `prime/sql/ml/26_ml_feature_export.sql` | `ml/` | Not scanned by runner; uses `.print` directives (CLI-only) |
| `scripts/export_classification.sql` | `scripts/` | Standalone DuckDB CLI script; zero Python references |

### 4B. SQL files referenced by broken paths

| SQL File | Referenced by | Problem |
|----------|---------------|---------|
| `prime/sql/layers/00_observations.sql` | `api.py:1822` | `_get_sql_path()` resolves to `prime/sql/00_observations.sql` (wrong — missing `layers/`) |
| `prime/sql/views/01_classification_units.sql` | `api.py:1858` | Wrong path resolution (missing `views/`) |
| `prime/sql/views/02_work_orders.sql` | `api.py:1891` | Wrong path resolution (missing `views/`) |
| `prime/sql/views/04_visualization.sql` | `api.py:2002` | Wrong path resolution (missing `views/`) |
| `prime/sql/views/05_summaries.sql` | `api.py:2003` | Wrong path resolution (missing `views/`) |
| `prime/sql/reports/60_ground_truth.sql` | `tuning_service.py:162` | Wrong path resolution (missing `reports/`) |
| `prime/sql/reports/61_lead_time_analysis.sql` | `tuning_service.py:163` | Wrong path resolution (missing `reports/`) |
| `prime/sql/reports/62_fault_signatures.sql` | `tuning_service.py:164` | Wrong path resolution (missing `reports/`) |
| `prime/sql/reports/63_threshold_optimization.sql` | `tuning_service.py:165` | Wrong path resolution (missing `reports/`) |

Additionally, `api.py` references `03_load_manifold_results.sql` which **does not exist** at all.

Also, `prime/core/server.py:525` resolves `sql_dir` to `prime/core/sql/` which does not exist — the `list_sql_docs` and `generate_sql_docs` endpoints are completely broken.

### 4C. SQL `stages/` and `views/` — limited scope

| Directory | File Count | Actually Used By |
|-----------|------------|------------------|
| `prime/sql/stages/` | 5 files | Only `explorer/server.py` (not main pipeline) |
| `prime/sql/views/` | 5 files | Only `cli.py` (not main pipeline) |

These are not dead per se, but only reachable through narrow, non-pipeline paths.

---

## 5. Dead SQL Views (defined but never consumed downstream)

All in `prime/sql/layers/05_manifold_derived.sql`:

| View | Status |
|------|--------|
| `v_geometry_dynamics` | Only used by `v_ml_features` (itself dead) |
| `v_topology` | Only used by `v_ml_features` (itself dead) |
| `v_statistics` | Zero downstream consumers |
| `v_feature_stats_ref` | Zero downstream consumers |
| `v_correlation` | Zero downstream consumers |
| `v_break_sequence` | Zero downstream consumers |
| `v_segment_comparison` | Zero downstream consumers |
| `v_info_flow_delta` | Zero downstream consumers |
| `v_velocity_field` | Zero downstream consumers |
| `v_cohort_topology` | Zero downstream consumers |
| `v_cohort_velocity_field` | Zero downstream consumers |
| `v_cohort_fingerprint` | Zero downstream consumers |
| `v_canary_signals` | Zero downstream consumers |
| `v_ml_features` | Zero downstream consumers (shadowed by `ml/11_ml_features.sql` which is also dead) |

In `prime/sql/layers/00_config.sql`:

| View/Table | Status |
|------------|--------|
| `v_data_sufficiency` | Zero downstream consumers |
| `v_lyapunov_reliability` | Zero downstream consumers |

---

## 6. Orphan Parquets (written but never read)

### 6A. SQL typology outputs — written but never loaded by runner

| Parquet | Produced By | Read By |
|---------|-------------|---------|
| `signal_windows.parquet` | `sql/typology/04_signal_windows.sql` | Nothing |
| `signal_correlations.parquet` | `sql/typology/05_signal_correlations.sql` | Nothing |
| `pairwise_windowed.parquet` | `sql/typology/05_signal_correlations.sql` | Nothing |
| `coupling_progression.parquet` | `sql/typology/05_signal_correlations.sql` | Nothing |

**Evidence**: `runner.py` line 174-176 only loads `signal_statistics`, `signal_derivatives`, `signal_temporal`, `signal_primitives`. These four parquets are never registered as DuckDB views.

### 6B. Schema-listed parquets — listed in MANIFOLD_SCHEMA.yaml but never consumed

| Parquet | Schema Line | SQL References |
|---------|-------------|----------------|
| `primitives_pairs.parquet` | ~line 175 | Zero SQL references anywhere |
| `observations_enriched.parquet` | ~line 185 | Zero SQL references anywhere |

### 6C. Stale documentation for removed parquets

7 orphan markdown files in `prime/sql/docs/` reference SQL files that no longer exist:
`03_load_manifold_results.md`, `05_geometry.md`, `06_dynamics.md`, `07_causality.md`, `09_physics.md`, `10_manifold.md`, `04_typology.md`

---

## 7. Dead Configuration

### 7A. Dead config sections in `prime/config/typology_config.py`

| Section | Reason |
|---------|--------|
| `TYPOLOGY_CONFIG['windowing']` (lines 229-234) | Never accessed via `get_threshold('windowing.*')` or direct key access |
| `TYPOLOGY_CONFIG['stationarity']` (lines 213-224) | Never accessed by any code |

### 7B. Dead config sections in `packages/typology/src/typology/config.py`

| Section | Reason |
|---------|--------|
| `CONFIG['stationarity']` (lines 131-141) | Never accessed by classify.py, window.py, or observe.py |
| `CONFIG['caps']` (lines 195-199) | Never accessed; ftle/sampen/rqa min/max caps unused |
| `CONFIG['engines']` (lines 204-229) | Never accessed within typology package (Prime has its own copy that IS used) |
| `CONFIG['artifacts']` (lines 17-21) | Never accessed within typology package (Prime's copy IS used) |

### 7C. Dead domain convenience exports in `prime/config/domains.py`

| Export | Reason |
|--------|--------|
| `turbofan` (line 563) | Never imported by any Python code |
| `bearings` (line 564) | Never imported by any Python code |
| `chemical` (line 565) | Never imported by any Python code |
| `hydraulic` (line 566) | Never imported by any Python code |
| `INPUT_DEFINITIONS` | Only used within `domains.py` itself; exported but zero external importers |
| `CAPABILITY_REQUIREMENTS` | Only used within `domains.py` itself; exported but zero external importers |
| `Capability` enum | Only used within `domains.py` itself; exported but zero external importers |

### 7D. Manifest keys written by Prime but never read by Manifold/orchestration

| Key | Status |
|-----|--------|
| `pair_engines` | Written by generator.py; orchestration never reads it |
| `symmetric_pair_engines` | Written by generator.py; orchestration never reads it |
| `engine_windows` | Written by generator.py; orchestration never reads it |
| `intervention` | Written by generator.py; orchestration never reads it |
| `atlas` | Written by generator.py; orchestration never reads it |
| `parameterization` | Metadata only; orchestration never reads it |

Note: `system`, `engine_gates`, `skip_signals`, and `cohorts` ARE read by orchestration.

---

## 8. Orphan Directories

### 8A. Fully orphaned directories (all contents dead)

| Directory | Files | Reason |
|-----------|-------|--------|
| `prime/notebook/` | `__init__.py` (empty), `generator.py` | Zero external importers |
| `prime/sql/_legacy/` | 10 `.sql` files | Deprecated; not scanned by runner |
| `prime/sql/ml/` | 2 `.sql` files | Not scanned by runner |
| `prime/sql/docs/` | 14 `.md` files | 7 of 14 reference non-existent SQL files |

### 8B. `__pycache__` committed to git

**None.** The repo is clean — `__pycache__` only exists in the working tree.

---

## 9. Duplicates

### 9A. Major duplicate — CohortDiscovery x2

| File A | File B |
|--------|--------|
| `prime/cohorts/discovery.py` (669 lines) | `prime/cohorts/detection.py` (1040+ lines) |

Both define `CohortDiscovery`, `CohortResult`, and `process_observations()`. `detection.py` is the v2 with extra methods (`detect_constants`, `detect_cohorts`, `classify_coupling_trajectory`). `discovery.py` is the one actually used by the pipeline (imported by `cohorts/__init__.py`). The extra methods in `detection.py` are exported but never called.

### 9B. Superseded manifest generator

| File A | File B |
|--------|--------|
| `prime/ingest/manifest_generator_v2.py` | `prime/manifest/generator.py` (v2.7, active) |

Both generate `manifest.yaml` from typology. File A is completely dead. File B is the active version.

### 9C. Four unused window-sizing modules

All four compute window sizes but none are used by the active manifest pipeline:

| File | Approach | Used? |
|------|----------|-------|
| `prime/manifest/characteristic_time.py` | ACF, frequency, period | Only by test |
| `prime/manifest/domain_clock.py` | Fastest signal frequency | Only by `window_recommender.py` (also dead) |
| `prime/manifest/window_recommender.py` | Typology raw measures | Only by test |
| `prime/manifest/system_window.py` | Multi-scale representation | Zero importers |
| `prime/manifest/generator.py` (inline) | Median of per-signal windows | **Active** |

### 9D. Backward-compatibility shims (intentional, low priority)

| Shim | Real Module |
|------|-------------|
| `prime/core/data_reader.py` (4 lines) | `prime/ingest/data_reader.py` |
| `prime/core/validation.py` (9 lines) | `prime/ingest/validation.py` |

These are re-export shims. Not broken, but could be removed if all callers update imports.

---

## 10. Recommendations

### Tier 1: Safe to delete immediately (zero risk)

1. **`prime/sql/_legacy/`** — 10 deprecated SQL files. Explicitly marked deprecated.
2. **`prime/sql/ml/`** — 2 SQL files never loaded by any runner.
3. **`prime/ingest/manifest_generator_v2.py`** — Fully superseded by `prime/manifest/generator.py`.
4. **`prime/config/stability_config.py`** — Zero importers.
5. **`prime/config/engine_rules.yaml`** — Zero references.
6. **`prime/config/TYPOLOGY_CONFIG.yaml`** — Zero references.
7. **`prime/typology/control_detection.py`** — Zero importers.
8. **`prime/typology/transition_detection.py`** — Zero importers.
9. **`prime/notebook/`** — Entire directory, zero importers.
10. **`prime/sql/docs/`** — 7 orphan markdown files for deleted SQL files.
11. **`packages/dynamics/src/dynamics/embedding_cache.py`** — Zero importers.
12. **`prime/shared/engine_registry.py`** — Re-exported but zero consumers.
13. **`prime/shared/window_config.py`** — Re-exported but zero consumers.
14. **Dead views in `05_manifold_derived.sql`** — 14 views with zero downstream consumers.
15. **128 dead imports** — Safe mechanical cleanup.

### Tier 2: Delete after review (investigate first)

1. **`prime/analysis/`** — Entire module cluster (7 files). Ambitious features (20/20, canary, thermo, window optimization) that were never wired in. Confirm no one is running these as standalone scripts.
2. **`prime/core/server.py`** — Appears to duplicate `api.py`. Confirm it's not used as an alternative server.
3. **`prime/core/pipeline.py`** — Confirm `prime.core.ObservationPipeline` is truly unused (it's re-exported from `core/__init__.py`).
4. **`prime/manifest/characteristic_time.py`**, **`domain_clock.py`**, **`window_recommender.py`**, **`system_window.py`** — Four window-sizing modules, none in the active pipeline. May be planned features.
5. **`prime/cohorts/detection.py`** — Major duplicate of `discovery.py` with extra methods. Merge or delete.
6. **4 orphan typology parquets** — `signal_windows`, `signal_correlations`, `pairwise_windowed`, `coupling_progression`. Confirm these aren't read by external tools.

### Tier 3: Fix broken references (bugs, not dead code)

1. **`api.py` `_get_sql_path()`** — Resolves to `prime/sql/{file}` instead of `prime/sql/layers/{file}` or `prime/sql/views/{file}`. All SQL-related API endpoints are silently broken.
2. **`tuning_service.py` SQL paths** — Same bug; resolves to wrong directory for ground truth reports.
3. **`server.py:525`** — `sql_dir` points to `prime/core/sql/` which doesn't exist.
4. **`api.py` references `03_load_manifold_results.sql`** — File doesn't exist at all.
5. **5 broken lazy imports in `api.py`** — Reference modules/names that don't exist (`manifest_builder`, `compute_pipeline`, `config_generator`, `IntakeTransformer`, `prepare_for_manifold`).
6. **Manifest keys** — `pair_engines`, `symmetric_pair_engines`, `engine_windows`, `intervention` are written but never consumed by orchestration. Either implement or stop writing them.

---

## Reproducibility

Key grep commands used to generate this report:

```bash
# Check if a function is called anywhere
grep -rn "function_name" prime/ packages/ --include="*.py" --include="*.sql"

# Check if a file is imported
grep -rn "from prime.module import" prime/ --include="*.py"
grep -rn "import prime.module" prime/ --include="*.py"

# Check if a SQL file is referenced by Python
grep -rn "filename.sql" prime/ --include="*.py"

# Check if a parquet is consumed by SQL
grep -rn "parquet_stem" prime/sql/ --include="*.sql"

# Check if a config key is accessed
grep -rn "CONFIG_NAME\[" prime/ --include="*.py"
grep -rn "get_threshold('section" prime/ --include="*.py"

# Check SQL runner directories
grep -n "glob\|layers\|reports" prime/sql/runner.py

# Check pyproject.toml entry points
grep -A50 "\[project.scripts\]" pyproject.toml
```
