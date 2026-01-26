-- ============================================================================
-- ORTHON SQL Engine: 01_calculus/002_second_derivative.sql
-- ============================================================================
-- Second derivative: d²y/dI²
-- Formula: d2y = y[i+1] - 2*y[i] + y[i-1]
-- ============================================================================

CREATE OR REPLACE VIEW v_d2y AS
SELECT
    signal_id,
    I,
    y,
    dy,
    LEAD(y) OVER (PARTITION BY signal_id ORDER BY I) -
    2*y +
    LAG(y) OVER (PARTITION BY signal_id ORDER BY I) AS d2y
FROM v_dy;
