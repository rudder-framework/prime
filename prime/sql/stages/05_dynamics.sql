-- ============================================================================
-- Dynamics Stage Reports
-- ============================================================================
-- Dynamical systems analysis: Lyapunov, RQA, attractor metrics
--
-- Input: dynamics.parquet, lyapunov.parquet (from Engines)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Dynamics Schema
-- ----------------------------------------------------------------------------
DESCRIBE SELECT * FROM dynamics LIMIT 1;

-- ----------------------------------------------------------------------------
-- 2. Dynamics Summary by Cohort
-- ----------------------------------------------------------------------------
SELECT
    cohort,
    COUNT(*) as n_windows,
    ROUND(AVG(lyapunov_max), 4) as avg_lyapunov,
    ROUND(AVG(determinism), 4) as avg_determinism,
    ROUND(AVG(recurrence_rate), 4) as avg_recurrence
FROM dynamics
GROUP BY cohort
ORDER BY cohort;

-- ----------------------------------------------------------------------------
-- 3. Chaos Detection (Lyapunov > 0)
-- ----------------------------------------------------------------------------
SELECT
    cohort,
    signal_0_center,
    lyapunov_max,
    determinism,
    recurrence_rate,
    CASE
        WHEN lyapunov_max > 0.1 THEN 'CHAOTIC'
        WHEN lyapunov_max > 0.01 THEN 'QUASI_PERIODIC'
        WHEN lyapunov_max > -0.01 THEN 'OSCILLATING'
        WHEN lyapunov_max > -0.1 THEN 'CONVERGING'
        ELSE 'STABLE'
    END as trajectory_type
FROM dynamics
WHERE lyapunov_max IS NOT NULL
ORDER BY lyapunov_max DESC
LIMIT 100;

-- ----------------------------------------------------------------------------
-- 4. Stability Analysis
-- ----------------------------------------------------------------------------
-- Combine Lyapunov and determinism for stability score
SELECT
    cohort,
    signal_0_center,
    lyapunov_max,
    determinism,
    (-1 * COALESCE(lyapunov_max, 0) + COALESCE(determinism, 0)) AS stability_score
FROM dynamics
ORDER BY stability_score DESC
LIMIT 100;

-- ----------------------------------------------------------------------------
-- 5. Recurrence Quantification Analysis (RQA)
-- ----------------------------------------------------------------------------
SELECT
    cohort,
    signal_0_center,
    recurrence_rate,
    determinism,
    laminarity,
    trapping_time,
    entropy_diagonal
FROM dynamics
WHERE recurrence_rate IS NOT NULL
ORDER BY cohort, signal_0_center;

-- ----------------------------------------------------------------------------
-- 6. Regime Transitions
-- ----------------------------------------------------------------------------
-- Detect windows where dynamics changed significantly
SELECT
    cohort,
    signal_0_center,
    lyapunov_max,
    LAG(lyapunov_max) OVER (PARTITION BY cohort ORDER BY signal_0_center) as prev_lyapunov,
    lyapunov_max - LAG(lyapunov_max) OVER (PARTITION BY cohort ORDER BY signal_0_center) as lyapunov_change
FROM dynamics
WHERE ABS(lyapunov_max - LAG(lyapunov_max) OVER (PARTITION BY cohort ORDER BY signal_0_center)) > 0.1
ORDER BY ABS(lyapunov_max - LAG(lyapunov_max) OVER (PARTITION BY cohort ORDER BY signal_0_center)) DESC
LIMIT 50;

-- ----------------------------------------------------------------------------
-- 7. Recent Dynamics (last 20 windows)
-- ----------------------------------------------------------------------------
SELECT *
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY cohort ORDER BY signal_0_center DESC) as rn
    FROM dynamics
) ranked
WHERE rn <= 20
ORDER BY cohort, signal_0_center DESC;
