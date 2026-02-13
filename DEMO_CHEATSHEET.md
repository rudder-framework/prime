# Rudder Framework DEMO CHEATSHEET

Copy-paste commands for running the full Rudder Framework + Engines pipeline.
Replace `____` with your actual file paths.

---

## 0. SETUP

```bash
# Activate virtual environments
source ~/framework/venv/bin/activate    # Rudder Framework
source ~/engines/venv/bin/activate   # Engines (separate terminal)

# Verify installations
python -c "import framework; print('framework OK')"
python -c "import engines; print('engines OK')"
```

---

## 1. FULL PIPELINE (NUMBERED STEPS)

### The Fast Way (Engines does everything)

```bash
# One command: CSV in, dynamical atlas out
engines run ____ --atlas

# Example with ferrocyanide CV data
engines run ~/Domains/electrochemistry/ferrocyanide_cv/observations.parquet --atlas
```

### The Full Way (Rudder Framework classifies, Engines computes)

```bash
# Step 1: Validate observations
python -m framework.entry_points.stage_01_validate ____ -o validated.parquet

# Step 2: Compute 27 raw typology measures per signal
python -m framework.entry_points.stage_02_typology ____ -o typology_raw.parquet

# Step 3: Classify signals (discrete/sparse first, then continuous)
python -m framework.entry_points.stage_03_classify typology_raw.parquet -o typology.parquet

# Step 4: Generate manifest (engine selection + windows per signal)
python -m framework.entry_points.stage_04_manifest typology.parquet -o manifest.yaml --observations ____

# Step 5: Run Engines core pipeline (15 stages)
engines run ____ --manifest manifest.yaml

# Step 6: Run Engines atlas pipeline (velocity fields, FTLE, urgency)
engines atlas ____

# Step 7: Interpret dynamics (Lyapunov, stability, regime transitions)
python -m framework.entry_points.stage_06_interpret ____/output --mode both

# Step 8: Predict health / RUL / anomalies
python -m framework.entry_points.stage_07_predict ____/output --mode health

# Step 9: Early warning alerts
python -m framework.entry_points.stage_08_alert ____ --mode predict

# Step 10: Explore results in browser
engines explore ____/output --port 8080
```

### Example: Ferrocyanide CV End-to-End

```bash
DATA=~/Domains/electrochemistry/ferrocyanide_cv

python -m framework.entry_points.stage_01_validate $DATA/observations.parquet -o $DATA/validated.parquet
python -m framework.entry_points.stage_02_typology $DATA/observations.parquet -o $DATA/typology_raw.parquet
python -m framework.entry_points.stage_03_classify $DATA/typology_raw.parquet -o $DATA/typology.parquet
python -m framework.entry_points.stage_04_manifest $DATA/typology.parquet -o $DATA/manifest.yaml --observations $DATA/observations.parquet
engines run $DATA --atlas
python -m framework.entry_points.stage_06_interpret $DATA/output --mode both
engines explore $DATA/output
```

---

## 2. Rudder Framework ENTRY POINTS

### Pre-Engines (classification)

| Stage | Command | What It Does | Output |
|-------|---------|-------------|--------|
| 01 | `python -m framework.entry_points.stage_01_validate ____ -o validated.parquet` | Remove constants, duplicates, repair timestamps | validated.parquet |
| 02 | `python -m framework.entry_points.stage_02_typology ____ -o typology_raw.parquet` | Compute 27 statistical measures per signal | typology_raw.parquet |
| 03 | `python -m framework.entry_points.stage_03_classify typology_raw.parquet -o typology.parquet` | Two-stage classification (10 dimensions) | typology.parquet |
| 04 | `python -m framework.entry_points.stage_04_manifest typology.parquet -o manifest.yaml` | Engine selection + window sizing per signal | manifest.yaml |

### Post-Engines (interpretation)

| Stage | Command | What It Does | Output |
|-------|---------|-------------|--------|
| 05 | `python -m framework.entry_points.stage_05_diagnostic ____ -o report.txt` | Full diagnostic without Engines (standalone) | report.txt |
| 06 | `python -m framework.entry_points.stage_06_interpret ____/output --mode both` | Lyapunov, stability, regime transitions | stdout/JSON |
| 07 | `python -m framework.entry_points.stage_07_predict ____/output --mode health` | RUL, health scoring, anomaly detection | stdout/JSON |
| 08 | `python -m framework.entry_points.stage_08_alert ____ --mode predict` | Early warning / failure fingerprints | stdout/JSON |

### Tools

| Stage | Command | What It Does | Output |
|-------|---------|-------------|--------|
| 09 | `python -m framework.entry_points.stage_09_explore ____/output -o manifold.png` | Manifold visualization (2D/3D) | PNG/PDF/SVG |
| 10 | `python -m framework.entry_points.stage_10_inspect ____ --mode inspect` | Profile any parquet file | stdout |
| 11 | `python -m framework.entry_points.stage_11_fetch raw.csv -o observations.parquet` | Read CSV/Excel/TSV, auto-repair schema | observations.parquet |
| 12 | `python -m framework.entry_points.stage_12_stream dashboard --port 8080` | Real-time streaming analysis (WebSocket) | dashboard |
| 13 | `python -m framework.entry_points.stage_13_train --model xgboost` | Train ML model on Engines features | ml_model.pkl |
| -- | `python -m framework.entry_points.csv_to_atlas ____ -o output/` | One-command: CSV to full dynamical atlas | everything |

### Entry Point Args Reference

```bash
# Stage 01: Validate
python -m framework.entry_points.stage_01_validate <observations> [-o OUTPUT] [--permissive] [-q]

# Stage 02: Typology
python -m framework.entry_points.stage_02_typology <observations> [-o OUTPUT] [-q]

# Stage 03: Classify
python -m framework.entry_points.stage_03_classify <typology_raw> [-o OUTPUT] [-q]

# Stage 04: Manifest
python -m framework.entry_points.stage_04_manifest <typology> [-o OUTPUT] [--observations OBS] [--output-dir DIR] [-q]

# Stage 05: Diagnostic (standalone, no Engines needed)
python -m framework.entry_points.stage_05_diagnostic <observations> [-o OUTPUT] [--domain general] [--window-size 100] [-q]

# Stage 06: Interpret
python -m framework.entry_points.stage_06_interpret <engines_dir> [--unit UNIT] [--mode dynamics|physics|both] [-q]

# Stage 07: Predict
python -m framework.entry_points.stage_07_predict <engines_dir> [--mode rul|health|anomaly] [--unit UNIT] [--threshold 0.8] [--method zscore|isolation_forest|lof|combined] [-q]

# Stage 08: Alert
python -m framework.entry_points.stage_08_alert <observations> [--mode predict|fingerprint|train] [--physics-path FILE] [-q]

# Stage 09: Explore
python -m framework.entry_points.stage_09_explore <engines_dir> [-o FILE] [--2d] [--axes 0,1] [--no-velocity] [--no-force] [-q]

# Stage 10: Inspect
python -m framework.entry_points.stage_10_inspect <path> [--mode inspect|capabilities|validate] [-q]

# Stage 11: Fetch
python -m framework.entry_points.stage_11_fetch <input> [-o OUTPUT] [--entity-col COL] [--timestamp-col COL] [--no-validate] [-q]

# Stage 12: Stream
python -m framework.entry_points.stage_12_stream <dashboard|analyze|demo> [--source turbofan] [--port 8080] [--window-size 100] [-q]

# Stage 13: Train
python -m framework.entry_points.stage_13_train [--data-dir data] [--model xgboost|catboost|lightgbm|randomforest|gradientboosting] [--tune] [--cv 5] [--split 0.2] [-q]

# CSV to Atlas (all-in-one)
python -m framework.entry_points.csv_to_atlas <input> [-o DIR] [--signals COL1,COL2] [--cohort-col COL] [--index-col COL] [--skip-engines] [-q]
```

---

## 3. ENGINES CLI

### User-Facing Commands

```bash
# Run full pipeline on any input (CSV, parquet, or directory)
engines run <input> [--output DIR] [--atlas] [--manifest FILE] [--segments name:start:end] [-q]

# Inspect data before running
engines inspect <input>

# Explore results in browser
engines explore <output_dir> [--port 8080]

# Check pipeline prerequisites
engines validate <data_dir> [--force]

# Show pipeline completion status
engines status <data_dir>

# Run full atlas pipeline (stages 16-23)
engines atlas <data_dir> [--continue-on-error] [-q]
```

### Individual Atlas Stages

```bash
engines break-sequence <data_dir>    # Break propagation order (stage 16)
engines ftle-backward <data_dir>     # Backward FTLE / attracting structures (stage 17)
engines segment-comparison <data_dir> # Pre/post segment geometry deltas (stage 18)
engines info-flow-delta <data_dir>   # Per-segment Granger deltas (stage 19)
engines velocity-field <data_dir>    # State-space velocity field (stage 21)
engines ftle-rolling <data_dir>      # Rolling FTLE stability evolution (stage 22)
engines ridge-proximity <data_dir>   # Urgency = velocity toward FTLE ridge (stage 23)
```

### Engines Output Files

| Stage | File | What It Contains |
|-------|------|-----------------|
| 00 | `breaks.parquet` | Regime changes (steps + impulses) per signal |
| 01 | `signal_vector.parquet` | Per-signal features: kurtosis, spectral entropy, Hurst, ACF |
| 02 | `state_vector.parquet` | System centroid (position in feature space) |
| 03 | `state_geometry.parquet` | Eigenvalues, effective_dim, eigenvector loadings, bootstrap CI |
| 04 | `cohorts.parquet` | Cohort-level aggregates |
| 05 | `signal_geometry.parquet` | Per-signal distance/coherence to system state |
| 06 | `signal_pairwise.parquet` | Pairwise signal correlations (eigenvector-gated) |
| 07 | `geometry_dynamics.parquet` | Velocity, acceleration, jerk of eigenstructure |
| 08 | `ftle.parquet` | Finite-Time Lyapunov Exponents per signal |
| 09 | `dynamics.parquet` | Per-signal stability classification |
| 10 | `information_flow.parquet` | Granger causality between signal pairs |
| 11 | `topology.parquet` | Topological features of signal manifold |
| 12 | `zscore.parquet` | Z-score normalization of all metrics |
| 13 | `statistics.parquet` | Summary statistics per signal |
| 14 | `correlation.parquet` | Feature correlation matrix |
| 15 | `ftle_field.parquet` | Spatiotemporal FTLE field (atlas) |
| 16 | `break_sequence.parquet` | Break propagation order (atlas) |
| 17 | `ftle_backward.parquet` | Backward FTLE / attracting structures (atlas) |
| 18 | `segment_comparison.parquet` | Pre/post segment geometry deltas (atlas) |
| 19 | `info_flow_delta.parquet` | Causality changes across segments (atlas) |
| 21 | `velocity_field.parquet` | State-space speed, curvature (atlas) |
| 22 | `ftle_rolling.parquet` | Rolling FTLE stability evolution (atlas) |
| 23 | `ridge_proximity.parquet` | Urgency classes: nominal/warning/elevated/critical (atlas) |

---

## 4. SQL SCRIPTS

### Classification SQL (layers/)

Run any SQL file with DuckDB. Most require loading data first.

```bash
# Interactive session pattern
duckdb
.read /path/to/data/output/*.parquet
.read ~/framework/framework/sql/layers/classification.sql
SELECT * FROM v_trajectory_type LIMIT 20;
```

### Key Classification Views

| SQL File | Creates | Purpose |
|----------|---------|---------|
| `layers/classification.sql` | v_trajectory_type, v_stability_class, v_collapse_status, v_signal_classification, v_anomaly_severity, v_coupling_strength, v_system_health, v_health_summary | PRIMARY: Lyapunov-based classification on Engines outputs |
| `layers/00_config.sql` | config_thresholds, v_lyapunov_reliability | All interpretation thresholds (no magic numbers) |
| `layers/01_typology.sql` | v_signal_typology, v_prism_requests | Signal behavioral classification + Engines work orders |
| `layers/02_geometry.sql` | v_correlation_matrix, v_coupling_network, v_lead_lag | Pairwise signal coupling and causality |
| `layers/03_dynamics.sql` | v_regime_assignment, v_attractors, v_bifurcation_candidates | Regime detection, attractors, bifurcations |
| `layers/04_causality.sql` | v_causal_roles, v_root_cause_candidates, v_causal_graph | Root cause detection, influence mapping |

### Atlas Analytics SQL

| SQL File | Creates | Purpose |
|----------|---------|---------|
| `layers/atlas_velocity_field.sql` | v_motion_class | Motion type: stationary/constant/accelerating/decelerating |
| `layers/atlas_ftle.sql` | v_ftle_evolution | FTLE stability: chaotic/marginal/neutral/stable |
| `layers/atlas_breaks.sql` | v_break_cascade | Break propagation: initiator/follower/late |
| `layers/atlas_topology.sql` | v_network_class | Network: sparse/moderate/dense/fully_connected |
| `layers/atlas_analytics.sql` | (dashboard views) | Aggregated atlas summaries |

### Stability & Physics SQL

| SQL File | Creates | Purpose |
|----------|---------|---------|
| `layers/30_dynamics_stability.sql` | v_dynamics_summary, v_dynamics_alerts | Lyapunov + RQA stability (CHAOTIC/STABLE/MARGINAL) |
| `layers/31_regime_transitions.sql` | (regime views) | Operational mode changes |
| `layers/32_basin_stability.sql` | (basin views) | Basin of attraction resilience |
| `layers/40_topology_health.sql` | (health views) | Network health and structure |
| `layers/50_information_health.sql` | (info views) | Information flow and dissipation |

### Report SQL (reports/)

```bash
# Run ALL reports in sequence
duckdb < ~/framework/framework/sql/reports/00_run_all.sql
```

| SQL File | Purpose |
|----------|---------|
| `reports/01_baseline_geometry.sql` | Establish baseline from stable periods |
| `reports/02_stable_baseline.sql` | Identify known-good windows |
| `reports/03_drift_detection.sql` | Signal drift detection (first 20% vs last 20%) |
| `reports/04_signal_ranking.sql` | Rank signals by importance |
| `reports/05_periodicity.sql` | Dominant frequency and periodicity |
| `reports/06_regime_detection.sql` | Distinct operational states |
| `reports/07_correlation_changes.sql` | Coupling evolution over time |
| `reports/08_lead_lag.sql` | Lead-lag influence ordering |
| `reports/09_causality_influence.sql` | Root cause mapping |
| `reports/10_process_health.sql` | Composite health assessment |
| `reports/11_validation_thresholds.sql` | Optimal anomaly thresholds |
| `reports/23_baseline_deviation.sql` | Deviations from baseline |
| `reports/24_incident_summary.sql` | Failure incident patterns |
| `reports/33_birth_certificate.sql` | Healthy baseline fingerprint |
| `reports/60_ground_truth.sql` | Compare predictions to actuals |
| `reports/61_lead_time_analysis.sql` | Early warning timing |
| `reports/62_fault_signatures.sql` | Fault pattern recognition |
| `reports/63_threshold_optimization.sql` | False positive/negative tradeoff |

### Stage Report SQL (stages/)

```bash
# Check results at each pipeline stage
duckdb -c "
  CREATE VIEW typology AS SELECT * FROM read_parquet('____/typology.parquet');
  .read ~/framework/framework/sql/stages/01_typology.sql
"
```

| SQL File | Reads | Purpose |
|----------|-------|---------|
| `stages/01_typology.sql` | typology.parquet | Classification distribution |
| `stages/02_signal_vector.sql` | signal_vector.parquet | Signal feature statistics |
| `stages/03_state_vector.sql` | state_vector.parquet | State centroid evolution |
| `stages/04_geometry.sql` | state_geometry.parquet | Eigenvalues, effective_dim, collapse |
| `stages/05_dynamics.sql` | dynamics.parquet | Lyapunov, RQA, attractors |
| `stages/06_physics.sql` | physics.parquet | Energy, entropy, coherence |

### ML Feature SQL (ml/)

| SQL File | Purpose |
|----------|---------|
| `ml/11_ml_features.sql` | Pivot Engines outputs into ML-ready feature matrix |
| `ml/26_ml_feature_export.sql` | Early warning features + risk scores (0-100) |

```bash
# Export ML features
duckdb -c "
  .read ~/framework/framework/sql/ml/11_ml_features.sql
  COPY v_ml_features TO 'ml_features.parquet' (FORMAT PARQUET);
"
```

---

## 5. Rudder Framework-ML ENTRY POINTS

### ML Pipeline

| Command | What It Does | Output |
|---------|-------------|--------|
| `python -m framework.ml.entry_points.features --target RUL` | Build ML feature matrix from Engines outputs | ml_features.parquet |
| `python -m framework.ml.entry_points.train --model xgboost` | Train model (XGBoost/CatBoost/LightGBM/RF/GB) | ml_model.pkl |
| `python -m framework.ml.entry_points.predict --model ml_model.pkl` | Run inference on test data | ml_predictions.parquet |
| `python -m framework.ml.entry_points.ablation --target RUL` | Layer-by-layer ablation (Rudder Framework contribution) | stdout/JSON |
| `python -m framework.ml.entry_points.benchmark` | Rudder Framework features vs raw baseline | stdout |
| `python -m framework.ml.entry_points.baseline --train train.txt --test test.txt` | Raw-sensor-only baseline | stdout |

### ML Args Reference

```bash
# Features: build ML feature matrix
python -m framework.ml.entry_points.features --target RUL [--entity engine_id] [--testing] [--limit 10]
  # Reads: data/vector.parquet, data/geometry.parquet, data/state.parquet, data/observations.parquet
  # Writes: data/ml_features.parquet

# Train: train model
python -m framework.ml.entry_points.train [--model xgboost] [--tune] [--split 0.8] [--cv 5] [--testing]
  # Reads: data/ml_features.parquet
  # Writes: data/ml_model.pkl, data/ml_model.json, data/ml_results.parquet, data/ml_importance.parquet

# Predict: run inference
python -m framework.ml.entry_points.predict [--model data/ml_model.pkl] [--features data/test/ml_features.parquet] [--ground-truth RUL.txt] [--output predictions.parquet]
  # Reads: ml_model.pkl, ml_features.parquet
  # Writes: ml_predictions.parquet, ml_predictions.csv

# Ablation: layer-by-layer contribution
python -m framework.ml.entry_points.ablation --target RUL [--show-discovery] [--cohort-ablation] [--output results.json]

# Baseline: raw sensor only
python -m framework.ml.entry_points.baseline [--train train.txt] [--test test.txt] [--rul RUL.txt] [--cap-rul 125]
```

### LASSO Feature Selection (Python API)

```python
from framework.ml.lasso import compute, compute_mutual_info

# L1 feature selection
result = compute(X, y, feature_names=names)
print(f"Selected {result['n_features_selected']} of {result['n_features_in']} features")
print(f"Top features: {result['nonzero_names'][:10]}")

# Mutual information (catches nonlinear relationships)
mi = compute_mutual_info(X, y, feature_names=names)
```

### Prediction Modules (Python API)

```python
# RUL prediction
from framework.prediction.rul import RULPredictor
predictor = RULPredictor(model_type="random_forest")
predictor.fit(train_features, train_rul)
predictions = predictor.predict(test_features)

# Health scoring (0-100)
from framework.prediction.health import HealthScorer
scorer = HealthScorer(baseline_mode="first_10_percent")
scores = scorer.score(engines_output_dir)

# Anomaly detection
from framework.prediction.anomaly import AnomalyDetector, AnomalyMethod
detector = AnomalyDetector(method=AnomalyMethod.COMBINED, threshold=3.0)
anomalies = detector.detect(engines_output_dir)
```

---

## 6. QUICK INSPECTION QUERIES (DuckDB)

### Look at any parquet file

```bash
# Schema
duckdb -c "DESCRIBE SELECT * FROM '____'"

# First 10 rows
duckdb -c "SELECT * FROM '____' LIMIT 10"

# Row count
duckdb -c "SELECT COUNT(*) FROM '____'"

# Column stats
duckdb -c "SUMMARIZE SELECT * FROM '____'"
```

### Typology Results

```bash
# Classification distribution
duckdb -c "
  SELECT temporal_pattern, COUNT(*) as n
  FROM '____/typology.parquet'
  GROUP BY temporal_pattern ORDER BY n DESC
"

# Which signals are constant?
duckdb -c "
  SELECT signal_id, temporal_pattern, spectral
  FROM '____/typology.parquet'
  WHERE temporal_pattern = 'CONSTANT'
"

# Full classification per signal
duckdb -c "
  SELECT signal_id, temporal_pattern, spectral, stationarity, memory, complexity
  FROM '____/typology.parquet'
  ORDER BY signal_id
"
```

### Eigenvalue Structure

```bash
# Effective dimension over time
duckdb -c "
  SELECT I, effective_dim, eigenvalue_1, eigenvalue_2, eigenvalue_3
  FROM '____/output/state_geometry.parquet'
  WHERE engine = 'shape'
  ORDER BY I
"

# Dimensional collapse check
duckdb -c "
  SELECT
    MIN(effective_dim) as min_dim,
    AVG(effective_dim) as avg_dim,
    MAX(effective_dim) as max_dim,
    MIN(effective_dim) / NULLIF(MAX(effective_dim), 0) as collapse_ratio
  FROM '____/output/state_geometry.parquet'
  WHERE engine = 'shape'
"
```

### FTLE / Lyapunov

```bash
# Stability per signal
duckdb -c "
  SELECT signal_id, ftle, ftle_std, embedding_dim, is_deterministic,
    CASE
      WHEN ftle > 0.1 THEN 'CHAOTIC'
      WHEN ftle > 0.01 THEN 'MARGINAL'
      WHEN ftle > -0.01 THEN 'NEUTRAL'
      ELSE 'STABLE'
    END as stability
  FROM '____/output/ftle.parquet'
  ORDER BY ftle DESC
"
```

### Urgency (Atlas)

```bash
# Urgency class distribution
duckdb -c "
  SELECT urgency_class, COUNT(*) as n
  FROM '____/output/ridge_proximity.parquet'
  GROUP BY urgency_class ORDER BY n DESC
"

# Critical windows
duckdb -c "
  SELECT *
  FROM '____/output/ridge_proximity.parquet'
  WHERE urgency_class = 'critical'
  ORDER BY I
"
```

### Velocity Field (Atlas)

```bash
# Speed distribution
duckdb -c "
  SELECT
    MIN(speed) as min_speed,
    AVG(speed) as avg_speed,
    MAX(speed) as max_speed,
    AVG(curvature) as avg_curvature
  FROM '____/output/velocity_field.parquet'
"
```

### Signal Features

```bash
# Per-signal feature summary
duckdb -c "
  SELECT signal_id, engine, value
  FROM '____/output/signal_vector.parquet'
  WHERE engine IN ('kurtosis', 'hurst', 'sample_entropy', 'spectral_entropy')
  ORDER BY signal_id, engine
"
```

### Coupling / Pairwise

```bash
# Strongest signal pairs
duckdb -c "
  SELECT signal_a, signal_b, engine, value
  FROM '____/output/signal_pairwise.parquet'
  WHERE engine = 'correlation'
  ORDER BY ABS(value) DESC
  LIMIT 20
"
```

### Breaks / Regime Changes

```bash
# All detected breaks
duckdb -c "
  SELECT signal_id, I, break_type, magnitude
  FROM '____/output/breaks.parquet'
  ORDER BY I
"
```

---

## 7. DATA FORMAT REQUIREMENTS

### observations.parquet (Input to Engines)

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| cohort | String | Optional | Groups related signals (engine_1, pump_A) |
| signal_id | String | Required | Signal name (temp, pressure, vibration) |
| I | UInt32 | Required | Sequential index per signal (0, 1, 2, ...) |
| value | Float64 | Required | The measurement |

### Rules

- I must be sequential per signal_id (NOT timestamps)
- No duplicate (signal_id, I) pairs
- No null signal_id values
- cohort is optional grouping (not used in computation)

### Auto-Repair

Rudder Framework auto-fixes common issues:

| Issue | Fix |
|-------|-----|
| I contains timestamps | Sort by I, regenerate as 0,1,2... |
| Duplicate (signal_id, I) | Sort and regenerate I |
| Missing I column | Create from row order |
| Column named 'timestamp' | Rename to I |
| Column named 'y' | Rename to value |
| Null signal_id | Remove rows |

### Validate Before Running

```bash
# Check only (no changes)
python -m framework.ingest.validate_observations --check ____

# Validate and repair (overwrites)
python -m framework.ingest.validate_observations ____

# Validate, repair, save to new file
python -m framework.ingest.validate_observations input.parquet output.parquet
```

---

## 8. JEFF'S DATA SECTION

When you have an unfamiliar CSV/parquet and want to run the full pipeline:

```bash
DATA=____                    # Path to your data file
WORK=~/Domains/newdata       # Working directory
mkdir -p $WORK
```

### Step 1: Look at the data

```bash
duckdb -c "SELECT * FROM '$DATA' LIMIT 5"
```

### Step 2: Check columns and types

```bash
duckdb -c "DESCRIBE SELECT * FROM '$DATA'"
duckdb -c "SELECT COUNT(*) as rows FROM '$DATA'"
```

### Step 3: Check for signals

```bash
# Wide format? (columns = signals)
duckdb -c "SELECT column_name FROM information_schema.columns WHERE table_name = '$DATA'"

# Long format? (signal_id column)
duckdb -c "SELECT DISTINCT signal_id FROM '$DATA' LIMIT 20"

# How many observations per signal?
duckdb -c "SELECT signal_id, COUNT(*) as n FROM '$DATA' GROUP BY signal_id ORDER BY n DESC"
```

### Step 4: Quick Engines run (auto-detects format)

```bash
# This handles wide CSV, long CSV, or parquet automatically
engines inspect $DATA
engines run $DATA --output $WORK --atlas
```

### Step 5: If manual conversion needed

```bash
# Convert with Rudder Framework fetch (handles CSV, Excel, TSV)
python -m framework.entry_points.stage_11_fetch $DATA -o $WORK/observations.parquet

# Or convert manually in Python
python3 -c "
import polars as pl
df = pl.read_csv('$DATA')
print(df.columns)
print(df.head())
print(df.shape)
"
```

### Step 6: Run Rudder Framework classification (optional, for full control)

```bash
python -m framework.entry_points.stage_02_typology $WORK/observations.parquet -o $WORK/typology_raw.parquet
python -m framework.entry_points.stage_03_classify $WORK/typology_raw.parquet -o $WORK/typology.parquet
python -m framework.entry_points.stage_04_manifest $WORK/typology.parquet -o $WORK/manifest.yaml --observations $WORK/observations.parquet
```

### Step 7: Check typology results

```bash
duckdb -c "
  SELECT signal_id, temporal_pattern, spectral, stationarity, complexity
  FROM '$WORK/typology.parquet'
  ORDER BY signal_id
"
```

### Step 8: Run Engines with Rudder Framework manifest

```bash
engines run $WORK --manifest $WORK/manifest.yaml --atlas
```

### Step 9: Check results

```bash
# What files were produced?
ls -la $WORK/output/*.parquet

# Eigenvalue structure
duckdb -c "
  SELECT I, effective_dim, eigenvalue_1, eigenvalue_2
  FROM '$WORK/output/state_geometry.parquet'
  WHERE engine = 'shape'
  ORDER BY I LIMIT 20
"

# Signal stability
duckdb -c "
  SELECT signal_id, ftle,
    CASE WHEN ftle > 0.1 THEN 'CHAOTIC'
         WHEN ftle > 0.01 THEN 'MARGINAL'
         WHEN ftle > -0.01 THEN 'NEUTRAL'
         ELSE 'STABLE' END as stability
  FROM '$WORK/output/ftle.parquet'
  ORDER BY ftle DESC
"

# Urgency
duckdb -c "
  SELECT urgency_class, COUNT(*) as n
  FROM '$WORK/output/ridge_proximity.parquet'
  GROUP BY urgency_class
"
```

### Step 10: Interpret and explore

```bash
python -m framework.entry_points.stage_06_interpret $WORK/output --mode both
engines explore $WORK/output --port 8080
```

---

## 9. EXPLORER

```bash
# Launch browser-based explorer on any Engines output
python -m framework.explorer.server ~/Domains --port 8080
engines explore ____/output --port 8080

# Pages:
#   http://localhost:8080/          SQL query interface (DuckDB-WASM)
#   http://localhost:8080/explorer  Pipeline data browser
#   http://localhost:8080/flow      Flow visualization (eigenvector-projected trajectory)
#   http://localhost:8080/atlas     Dynamical atlas scenarios
```

---

## 10. AVAILABLE DEMO DATA

```bash
# TEP (Tennessee Eastman Process) — in repo
ls ~/framework/data/

# Domains — external
ls ~/Domains/
# battery/       — NASA battery degradation
# calce/         — Battery calendar aging
# cmapss/        — C-MAPSS turbofan
# cwru/          — CWRU bearing fault
# electrochemistry/  — Ferrocyanide CV + PEM Fuel Cell
# hydraulic/     — Hydraulic condition monitoring
# industrial/    — SKAB, MetroPT
# rossler/       — Chaotic system (Rossler attractor)

# Quick start with any domain
engines run ~/Domains/rossler --atlas
engines run ~/Domains/electrochemistry/ferrocyanide_cv --atlas
engines run ~/Domains/electrochemistry/pem_fuel_cell --atlas
```

---

## 11. CLASSIFICATION REFERENCE

### Lyapunov-Based Trajectory Classification

| FTLE Range | Classification | Meaning |
|------------|---------------|---------|
| > 0.1 | CHAOTIC | Exponential divergence |
| > 0.01 | QUASI_PERIODIC | Edge of chaos |
| > -0.01 | OSCILLATING | Limit cycle |
| > -0.1 | CONVERGING | Damped oscillation |
| < -0.1 | STABLE | Fixed point attractor |

### Signal Typology (10 dimensions)

| Dimension | Possible Values |
|-----------|----------------|
| temporal_pattern | PERIODIC, TRENDING, DRIFTING, RANDOM, CHAOTIC, QUASI_PERIODIC, STATIONARY, CONSTANT, BINARY, DISCRETE, IMPULSIVE, EVENT |
| spectral | HARMONIC, NARROWBAND, BROADBAND, RED_NOISE, BLUE_NOISE, NONE, SWITCHING, QUANTIZED, SPARSE |
| stationarity | STATIONARY, NON_STATIONARY |
| memory | SHORT_MEMORY, LONG_MEMORY |
| complexity | LOW, MEDIUM, HIGH |
| continuity | CONTINUOUS, DISCRETE |
| determinism | DETERMINISTIC, STOCHASTIC |
| distribution | GAUSSIAN, LIGHT_TAILED, HEAVY_TAILED |
| amplitude | SMOOTH, BURSTY, MIXED |
| volatility | HOMOSCEDASTIC, VOLATILITY_CLUSTERING |

### Urgency Classes

| Class | Meaning |
|-------|---------|
| nominal | Stable, not approaching any boundary |
| warning | Moving toward FTLE ridge (early warning) |
| elevated | Near ridge, moving away |
| critical | Near ridge, heading in |
