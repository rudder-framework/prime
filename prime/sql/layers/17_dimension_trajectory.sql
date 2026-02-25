-- ============================================================================
-- LAYER 17: DIMENSION TRAJECTORY ANALYSIS
-- Geometric failure mode detection via effective_dim early/late comparison
--
-- Collapsing: variance concentrating into fewer dimensions (HPC degradation)
-- Expanding:  dimensionality increasing (Fan degradation, decorrelated modes)
-- Stable:     mixed or slower-progressing failure
--
-- No labels required. The geometry resolves failure mode differences alone.
-- ============================================================================

-- Per-cohort dimension trajectory
CREATE OR REPLACE VIEW v_dimension_trajectory AS
WITH cohort_lifecycle AS (
    SELECT
        cohort,
        MIN(signal_0_center) AS min_I,
        MAX(signal_0_center) AS max_I,
        COUNT(*) AS n_windows
    FROM state_geometry
    GROUP BY cohort
),
early_late AS (
    SELECT
        sg.cohort,
        cl.min_I,
        cl.max_I,
        cl.n_windows,
        -- Early life: first 20% of windows
        AVG(CASE WHEN sg.signal_0_center <= cl.min_I + (cl.max_I - cl.min_I) * 0.2
            THEN sg.effective_dim END) AS early_dim,
        -- Late life: last 20% of windows
        AVG(CASE WHEN sg.signal_0_center >= cl.max_I - (cl.max_I - cl.min_I) * 0.2
            THEN sg.effective_dim END) AS late_dim
    FROM state_geometry sg
    JOIN cohort_lifecycle cl ON sg.cohort = cl.cohort
    GROUP BY sg.cohort, cl.min_I, cl.max_I, cl.n_windows
)
SELECT
    cohort,
    n_windows,
    ROUND(early_dim, 3) AS early_dim,
    ROUND(late_dim, 3) AS late_dim,
    ROUND(late_dim - early_dim, 3) AS dim_delta,
    CASE
        WHEN late_dim - early_dim < -0.2 THEN 'COLLAPSING'
        WHEN late_dim - early_dim > 0.2 THEN 'EXPANDING'
        ELSE 'STABLE'
    END AS trajectory_type,
    RANK() OVER (ORDER BY ABS(late_dim - early_dim) DESC) AS change_rank
FROM early_late
WHERE early_dim IS NOT NULL AND late_dim IS NOT NULL
ORDER BY dim_delta ASC;


-- ============================================================================
-- EIGENVALUE_1 TRAJECTORY
-- Dominant eigenvalue early/late comparison
-- CONCENTRATING: dominant mode gaining share (eigenvalue_1 increasing)
-- DISPERSING: dominant mode losing share (eigenvalue_1 decreasing)
-- ============================================================================

CREATE OR REPLACE VIEW v_eigenvalue_1_trajectory AS
WITH cohort_lifecycle AS (
    SELECT
        cohort,
        MIN(signal_0_center) AS min_I,
        MAX(signal_0_center) AS max_I,
        COUNT(*) AS n_windows
    FROM geometry_dynamics
    WHERE eigenvalue_1 IS NOT NULL AND NOT isnan(eigenvalue_1)
    GROUP BY cohort
),
early_late AS (
    SELECT
        gd.cohort,
        cl.n_windows,
        AVG(CASE WHEN rn <= 0.2 * cl.n_windows THEN gd.eigenvalue_1 END) AS early_eigenvalue,
        AVG(CASE WHEN rn > 0.8 * cl.n_windows THEN gd.eigenvalue_1 END) AS late_eigenvalue
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY cohort ORDER BY signal_0_center) AS rn
        FROM geometry_dynamics
        WHERE eigenvalue_1 IS NOT NULL AND NOT isnan(eigenvalue_1)
    ) gd
    JOIN cohort_lifecycle cl ON gd.cohort = cl.cohort
    GROUP BY gd.cohort, cl.n_windows
)
SELECT
    cohort,
    n_windows,
    ROUND(early_eigenvalue, 3) AS early_eigenvalue,
    ROUND(late_eigenvalue, 3) AS late_eigenvalue,
    ROUND(late_eigenvalue - early_eigenvalue, 3) AS eigenvalue_delta,
    CASE
        WHEN early_eigenvalue IS NULL OR early_eigenvalue = 0 THEN 'INSUFFICIENT_DATA'
        WHEN (late_eigenvalue - early_eigenvalue) > 0.1 * ABS(early_eigenvalue) THEN 'CONCENTRATING'
        WHEN (late_eigenvalue - early_eigenvalue) < -0.1 * ABS(early_eigenvalue) THEN 'DISPERSING'
        ELSE 'STABLE'
    END AS eigenvalue_trajectory
FROM early_late
WHERE early_eigenvalue IS NOT NULL AND late_eigenvalue IS NOT NULL
ORDER BY ABS(late_eigenvalue - early_eigenvalue) DESC;


-- ============================================================================
-- CONDITION NUMBER TRAJECTORY
-- CONCENTRATING: condition number shrinking (more well-conditioned)
-- ILL_CONDITIONING: condition number growing (numerical instability increasing)
-- ============================================================================

CREATE OR REPLACE VIEW v_condition_number_trajectory AS
WITH cohort_lifecycle AS (
    SELECT
        cohort,
        MIN(signal_0_center) AS min_I,
        MAX(signal_0_center) AS max_I,
        COUNT(*) AS n_windows
    FROM geometry_dynamics
    WHERE condition_number IS NOT NULL AND NOT isnan(condition_number)
    GROUP BY cohort
),
early_late AS (
    SELECT
        gd.cohort,
        cl.n_windows,
        AVG(CASE WHEN rn <= 0.2 * cl.n_windows THEN gd.condition_number END) AS early_cond,
        AVG(CASE WHEN rn > 0.8 * cl.n_windows THEN gd.condition_number END) AS late_cond
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY cohort ORDER BY signal_0_center) AS rn
        FROM geometry_dynamics
        WHERE condition_number IS NOT NULL AND NOT isnan(condition_number)
    ) gd
    JOIN cohort_lifecycle cl ON gd.cohort = cl.cohort
    GROUP BY gd.cohort, cl.n_windows
)
SELECT
    cohort,
    n_windows,
    ROUND(early_cond, 3) AS early_cond,
    ROUND(late_cond, 3) AS late_cond,
    ROUND(late_cond - early_cond, 3) AS cond_delta,
    CASE
        WHEN early_cond IS NULL OR early_cond = 0 THEN 'INSUFFICIENT_DATA'
        WHEN (late_cond - early_cond) > 0.1 * ABS(early_cond) THEN 'ILL_CONDITIONING'
        WHEN (late_cond - early_cond) < -0.1 * ABS(early_cond) THEN 'CONCENTRATING'
        ELSE 'STABLE'
    END AS condition_trajectory
FROM early_late
WHERE early_cond IS NOT NULL AND late_cond IS NOT NULL
ORDER BY ABS(late_cond - early_cond) DESC;
