-- ============================================================================
-- REPORTS - RUN ALL (DEPRECATED)
-- ============================================================================
--
-- DEPRECATED: This file assumes the legacy output/ directory layout.
-- Use the Python runner or query CLI instead:
--
--   prime query ~/domains/FD004/train                  (auto-discovers output_*/)
--   python -m prime.sql.runner ~/domains/FD004/train/output_time
--
-- The Python runner (prime/sql/runner.py) loads parquets via recursive glob
-- and works with the multi-axis output_*/ directory layout.
--
-- ============================================================================
--
-- Legacy sequence (assumes output/ subdirectory):
--   01 - Baseline Geometry      05 - Periodicity         09 - Causality Influence
--   02 - Stable Baseline        06 - Regime Detection    10 - System Departure
--   03 - Drift Detection        07 - Correlation Changes 11 - Validation Thresholds
--   04 - Signal Ranking         08 - Lead-Lag            12 - FF Stable Periods
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
.read 10_system_departure.sql
.read 11_validation_thresholds.sql
-- .read 12_ff_stable_periods.sql  -- Domain-specific, run manually if needed
