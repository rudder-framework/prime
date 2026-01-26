-- ============================================================================
-- ORTHON SQL Engine: 01_calculus/003_curvature.sql
-- ============================================================================
-- Curvature: kappa = |d²y| / (1 + (dy)²)^(3/2)
-- ============================================================================

CREATE OR REPLACE VIEW v_curvature AS
SELECT
    signal_id,
    I,
    y,
    dy,
    d2y,
    CASE
        WHEN dy IS NOT NULL AND d2y IS NOT NULL
        THEN ABS(d2y) / POWER(1 + dy*dy, 1.5)
        ELSE NULL
    END AS kappa
FROM v_d2y;

-- Curvature statistics per signal
CREATE OR REPLACE VIEW v_curvature_stats AS
SELECT
    signal_id,
    AVG(kappa) AS kappa_mean,
    STDDEV(kappa) AS kappa_std,
    CASE
        WHEN AVG(kappa) > 1e-10
        THEN STDDEV(kappa) / AVG(kappa)
        ELSE NULL
    END AS kappa_cv
FROM v_curvature
WHERE kappa IS NOT NULL
GROUP BY signal_id;
