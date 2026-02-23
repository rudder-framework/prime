-- ============================================================================
-- ENGINES: LOAD & VALIDATE
-- ============================================================================
-- Load observations parquet, create base view, validate schema.
-- This is the foundation for all downstream views.
-- ============================================================================

-- ============================================================================
-- CANONICAL SCHEMA - THE RULE
-- ============================================================================
-- observations.parquet MUST have columns:
--
--   cohort     : str   - Entity identifier
--   signal_id  : str   - Signal name
--   signal_0   : float - Index (time, cycle, depth, distance, sample)
--   value      : float - Value (the measurement)
--
-- Units live in signals.parquet (joined here when available).
-- signal_0 means signal_0. value means value. No aliases. No mapping.
-- Column mapping happens at INTAKE, not here.
-- Pipeline will FAIL if any required column is missing.
-- ============================================================================

-- ============================================================================
-- 001: CREATE BASE VIEW FROM OBSERVATIONS
-- ============================================================================

CREATE OR REPLACE VIEW v_base AS
SELECT
    o.cohort,
    o.signal_id,
    o.signal_0,
    o.value,
    -- Units live in signals.parquet, not observations.parquet
    s.unit AS value_unit,
    NULL AS index_dimension,
    NULL AS signal_class
FROM observations o
LEFT JOIN signals s ON o.signal_id = s.signal_id;


-- ============================================================================
-- 002: SCHEMA VALIDATION
-- ============================================================================
-- Verify required columns exist and have data

CREATE OR REPLACE VIEW v_schema_validation AS
SELECT
    COUNT(*) AS n_rows,
    COUNT(DISTINCT cohort) AS n_entities,
    COUNT(DISTINCT signal_id) AS n_signals,
    MIN(signal_0) AS I_min,
    MAX(signal_0) AS I_max,
    CASE
        WHEN COUNT(*) = 0 THEN 'ERROR: No data'
        WHEN COUNT(DISTINCT signal_id) = 0 THEN 'ERROR: No signals'
        ELSE 'WITHIN_BASELINE'
    END AS validation_status
FROM v_base;


-- ============================================================================
-- 003: SIGNAL INVENTORY
-- ============================================================================
-- One row per signal with basic stats

CREATE OR REPLACE VIEW v_signal_inventory AS
SELECT
    cohort,
    signal_id,
    COUNT(*) AS n_points,
    MIN(signal_0) AS I_min,
    MAX(signal_0) AS I_max,
    MIN(value) AS value_min,
    MAX(value) AS value_max,
    AVG(value) AS value_mean,
    value_unit,
    index_dimension
FROM v_base
GROUP BY cohort, signal_id, value_unit, index_dimension;


-- ============================================================================
-- 004: DATA QUALITY FLAGS
-- ============================================================================

CREATE OR REPLACE VIEW v_data_quality AS
SELECT
    signal_id,
    cohort,
    n_points,
    CASE WHEN n_points < 50 THEN TRUE ELSE FALSE END AS insufficient_data,
    CASE WHEN value_min = value_max THEN TRUE ELSE FALSE END AS constant_signal,
    CASE WHEN value_min IS NULL OR value_max IS NULL THEN TRUE ELSE FALSE END AS has_nulls
FROM v_signal_inventory;
