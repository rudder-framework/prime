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
