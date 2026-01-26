-- ============================================================================
-- ORTHON SQL Engine: run_all.sql
-- ============================================================================
-- Execute all SQL engines in order
-- Run from orthon/ directory: duckdb < sql/run_all.sql
-- ============================================================================

-- Load data
.read sql/00_load.sql

-- 01: Calculus (derivatives, curvature)
.read sql/01_calculus/001_first_derivative.sql
.read sql/01_calculus/002_second_derivative.sql
.read sql/01_calculus/003_curvature.sql

-- 02: Signal Classification
.read sql/02_signal_class/001_from_units.sql
.read sql/02_signal_class/002_from_data.sql
.read sql/02_signal_class/003_classify.sql

-- ============================================================================
-- VALIDATION
-- ============================================================================

SELECT '=== SIGNAL CLASSIFICATION ===' AS section;
SELECT signal_id, value_unit, signal_class, interpolation_valid, class_source, est_period
FROM v_signal_class
ORDER BY signal_id;

SELECT '=== CLASSIFICATION SUMMARY ===' AS section;
SELECT * FROM v_signal_class_summary;

SELECT '=== VALIDATION: Compare to Ground Truth ===' AS section;
-- Ground truth expectations:
-- sine_pure: periodic, sine_noisy: periodic, damped_oscillation: periodic
-- random_walk: analog, trending: analog, mean_reverting: analog
-- coupled_follower: analog, regime_break: analog
-- step_digital: digital, event_sparse: event
