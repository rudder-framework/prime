-- ============================================================================
-- Geometry Stage Reports
-- ============================================================================
-- State geometry: eigenvalues, effective dimension, shape analysis
--
-- Input: state_geometry.parquet (from Engines)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Geometry Schema
-- ----------------------------------------------------------------------------
DESCRIBE SELECT * FROM state_geometry LIMIT 1;

-- ----------------------------------------------------------------------------
-- 2. Geometry Summary by Cohort
-- ----------------------------------------------------------------------------
SELECT
    cohort,
    COUNT(*) as n_windows,
    ROUND(AVG(effective_dim), 2) as avg_eff_dim,
    ROUND(MIN(effective_dim), 2) as min_eff_dim,
    ROUND(MAX(effective_dim), 2) as max_eff_dim,
    ROUND(STDDEV(effective_dim), 3) as std_eff_dim
FROM state_geometry
GROUP BY cohort
ORDER BY cohort;

-- ----------------------------------------------------------------------------
-- 3. Effective Dimension Trajectory
-- ----------------------------------------------------------------------------
-- Track how effective dimension evolves over time
SELECT
    cohort,
    signal_0_center,
    effective_dim,
    LAG(effective_dim) OVER (PARTITION BY cohort ORDER BY signal_0_center) as prev_eff_dim,
    effective_dim - LAG(effective_dim) OVER (PARTITION BY cohort ORDER BY signal_0_center) as eff_dim_change
FROM state_geometry
ORDER BY cohort, signal_0_center;

-- ----------------------------------------------------------------------------
-- 4. Dimensionality Collapse Detection
-- ----------------------------------------------------------------------------
-- Windows where effective dimension dropped significantly
SELECT
    cohort,
    signal_0_center,
    effective_dim,
    LAG(effective_dim) OVER (PARTITION BY cohort ORDER BY signal_0_center) as prev_eff_dim,
    effective_dim - LAG(effective_dim) OVER (PARTITION BY cohort ORDER BY signal_0_center) as change
FROM state_geometry
WHERE effective_dim < LAG(effective_dim) OVER (PARTITION BY cohort ORDER BY signal_0_center) * 0.8
ORDER BY change ASC
LIMIT 50;

-- ----------------------------------------------------------------------------
-- 5. Recent Geometry (last 20 windows)
-- ----------------------------------------------------------------------------
SELECT *
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY cohort ORDER BY signal_0_center DESC) as rn
    FROM state_geometry
) ranked
WHERE rn <= 20
ORDER BY cohort, signal_0_center DESC;

-- ----------------------------------------------------------------------------
-- 6. Eigenvalue Distribution (if available)
-- ----------------------------------------------------------------------------
-- Note: eigenvalues may be stored as array or separate columns
SELECT
    cohort,
    signal_0_center,
    effective_dim,
    eigenvalue_1,
    eigenvalue_2,
    eigenvalue_3
FROM state_geometry
WHERE eigenvalue_1 IS NOT NULL
ORDER BY cohort, signal_0_center
LIMIT 100;
