-- ============================================================================
-- ORTHON SQL: State Vector Stage Reports
-- ============================================================================
-- System state centroids computed by Engines
--
-- Input: state_vector.parquet (from Engines)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. State Vector Schema
-- ----------------------------------------------------------------------------
DESCRIBE SELECT * FROM state_vector LIMIT 1;

-- ----------------------------------------------------------------------------
-- 2. State Summary
-- ----------------------------------------------------------------------------
SELECT
    cohort,
    COUNT(*) as n_windows,
    MIN(I) as first_window,
    MAX(I) as last_window
FROM state_vector
GROUP BY cohort
ORDER BY cohort;

-- ----------------------------------------------------------------------------
-- 3. State Trajectory Overview
-- ----------------------------------------------------------------------------
-- First few state vectors per cohort
SELECT *
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY cohort ORDER BY I) as rn
    FROM state_vector
) ranked
WHERE rn <= 5
ORDER BY cohort, I;

-- ----------------------------------------------------------------------------
-- 4. Recent States (last 10 windows)
-- ----------------------------------------------------------------------------
SELECT *
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY cohort ORDER BY I DESC) as rn
    FROM state_vector
) ranked
WHERE rn <= 10
ORDER BY cohort, I DESC;
