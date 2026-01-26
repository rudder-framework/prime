-- ============================================================================
-- ORTHON SQL Engine: 00_load.sql
-- ============================================================================
-- Load observations and create base views
-- ============================================================================

-- Load demo data
CREATE OR REPLACE TABLE observations AS
SELECT * FROM read_parquet('data/demo/demo_signals.parquet');

-- Base view: standardized column names
CREATE OR REPLACE VIEW v_base AS
SELECT
    signal_id,
    I,
    y,
    unit AS value_unit
FROM observations;

-- Signal metadata view
CREATE OR REPLACE VIEW v_signal_meta AS
SELECT
    signal_id,
    COUNT(*) AS n_points,
    MIN(I) AS i_min,
    MAX(I) AS i_max,
    MIN(y) AS y_min,
    MAX(y) AS y_max,
    AVG(y) AS y_mean,
    STDDEV(y) AS y_std,
    ANY_VALUE(value_unit) AS value_unit
FROM v_base
GROUP BY signal_id;

-- Verify load
SELECT
    'Loaded' AS status,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT signal_id) AS n_signals
FROM observations;
