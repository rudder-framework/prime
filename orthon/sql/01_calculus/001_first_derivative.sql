-- ============================================================================
-- ORTHON SQL Engine: 01_calculus/001_first_derivative.sql
-- ============================================================================
-- First derivative: dy/dI using central difference
-- Formula: dy = (y[i+1] - y[i-1]) / 2
-- ============================================================================

CREATE OR REPLACE VIEW v_dy AS
SELECT
    signal_id,
    I,
    y,
    (LEAD(y) OVER (PARTITION BY signal_id ORDER BY I) -
     LAG(y) OVER (PARTITION BY signal_id ORDER BY I)) / 2.0 AS dy
FROM v_base;
