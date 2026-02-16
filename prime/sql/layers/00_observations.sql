-- ============================================================================
-- 00_observations.sql
-- ============================================================================
-- Load raw data and create observations.parquet
-- This is the ONLY parquet file Prime creates.
-- All other parquet files come from Engines.
--
-- CANONICAL SCHEMA (THE RULE):
--   cohort : str   - Entity identifier
--   signal_id : str   - Signal identifier
--   signal_0  : float - Index (time, cycle, depth, sample)
--   y         : float - Value (the measurement)
--   unit      : str   - Unit string
--
-- signal_0 means signal_0. y means y. No aliases after intake.
-- ============================================================================

-- Load from uploaded file (path injected at runtime)
CREATE OR REPLACE TABLE raw_upload AS
SELECT * FROM read_parquet('{input_path}');

-- Create standardized observations table with CANONICAL schema
CREATE OR REPLACE TABLE observations AS
SELECT
    COALESCE(cohort, 'default') AS cohort,
    COALESCE(signal_id, 'signal_' || ROW_NUMBER() OVER ()) AS signal_id,
    signal_0,
    y,
    COALESCE(unit, 'unknown') AS unit
FROM raw_upload;

-- Export to parquet (ONLY file Prime creates)
COPY observations TO '{output_path}/observations.parquet' (FORMAT PARQUET);

-- Basic stats for UI
CREATE OR REPLACE VIEW v_observations_summary AS
SELECT
    COUNT(DISTINCT cohort) AS n_entities,
    COUNT(DISTINCT signal_id) AS n_signals,
    COUNT(*) AS n_rows,
    MIN(signal_0) AS i_min,
    MAX(signal_0) AS i_max,
    COUNT(DISTINCT unit) AS n_units
FROM observations;

-- Verify
SELECT * FROM v_observations_summary;
