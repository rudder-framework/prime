-- ============================================================================
-- Rudder SQL REPORTS - RUN ALL
-- ============================================================================
--
-- Execute all SQL reports in sequence.
--
-- Schema: observations.parquet with columns [cohort, signal_id, I, value]
--         Plus Engines outputs: signal_vector, state_vector, state_geometry, etc.
--
-- Sequence:
--   01 - Baseline Geometry:     Establish geometric baseline from healthy period
--   02 - Stable Baseline:       Identify stable operating periods
--   03 - Drift Detection:       Detect signals deviating from baseline
--   04 - Signal Ranking:        Rank signals by importance/variability
--   05 - Periodicity:           Analyze periodic patterns and frequencies
--   06 - Regime Detection:      Identify operational regime changes
--   07 - Correlation Changes:   Track correlation structure changes over time
--   08 - Lead-Lag:              Identify leading/lagging signal relationships
--   09 - Causality Influence:   Causal analysis and influence mapping
--   10 - Process Health:        Overall process health scoring
--   11 - Validation Thresholds: Threshold optimization and validation
--   12 - FF Stable Periods:     Fama-French specific stable period analysis
--
-- Usage with DuckDB:
--   duckdb < 00_run_all.sql
--
-- Or run individual reports:
--   duckdb < 01_baseline_geometry.sql
--
-- ============================================================================

-- Load observations
CREATE OR REPLACE VIEW observations AS
SELECT * FROM read_parquet('observations.parquet');

-- Load Engines outputs (if available)
CREATE OR REPLACE VIEW signal_vector AS
SELECT * FROM read_parquet('output/signal_vector.parquet');

CREATE OR REPLACE VIEW state_vector AS
SELECT * FROM read_parquet('output/state_vector.parquet');

CREATE OR REPLACE VIEW state_geometry AS
SELECT * FROM read_parquet('output/state_geometry.parquet');

CREATE OR REPLACE VIEW geometry_dynamics AS
SELECT * FROM read_parquet('output/geometry_dynamics.parquet');

CREATE OR REPLACE VIEW typology AS
SELECT * FROM read_parquet('typology.parquet');

-- Run reports in sequence
.read 01_baseline_geometry.sql
.read 02_stable_baseline.sql
.read 03_drift_detection.sql
.read 04_signal_ranking.sql
.read 05_periodicity.sql
.read 06_regime_detection.sql
.read 07_correlation_changes.sql
.read 08_lead_lag.sql
.read 09_causality_influence.sql
.read 10_process_health.sql
.read 11_validation_thresholds.sql
-- .read 12_ff_stable_periods.sql  -- Domain-specific, run manually if needed
