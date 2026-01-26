-- ============================================================================
-- ORTHON SQL: 00_observations.sql
-- ============================================================================
-- Load raw data and create observations.parquet
-- This is the ONLY parquet file ORTHON creates.
-- All other parquet files come from PRISM.
--
-- CANONICAL SCHEMA (THE RULE):
--   entity_id : str   - Entity identifier
--   signal_id : str   - Signal identifier
--   I         : float - Index (time, cycle, depth, sample)
--   y         : float - Value (the measurement)
--   unit      : str   - Unit string
--
-- I means I. y means y. No aliases after intake.
-- ============================================================================

-- Load from uploaded file (path injected at runtime)
CREATE OR REPLACE TABLE raw_upload AS
SELECT * FROM read_parquet('{input_path}');

-- Create standardized observations table with CANONICAL schema
CREATE OR REPLACE TABLE observations AS
SELECT
    COALESCE(entity_id, 'default') AS entity_id,
    COALESCE(signal_id, 'signal_' || ROW_NUMBER() OVER ()) AS signal_id,
    I,
    y,
    COALESCE(unit, 'unknown') AS unit
FROM raw_upload;

-- Export to parquet (ONLY file ORTHON creates)
COPY observations TO '{output_path}/observations.parquet' (FORMAT PARQUET);

-- Basic stats for UI
CREATE OR REPLACE VIEW v_observations_summary AS
SELECT
    COUNT(DISTINCT entity_id) AS n_entities,
    COUNT(DISTINCT signal_id) AS n_signals,
    COUNT(*) AS n_rows,
    MIN(I) AS i_min,
    MAX(I) AS i_max,
    COUNT(DISTINCT unit) AS n_units
FROM observations;

-- Verify
SELECT * FROM v_observations_summary;
