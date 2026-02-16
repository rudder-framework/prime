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

-- Per-cohort dimension trajectory (shape engine — sensitive to failure geometry)
CREATE OR REPLACE VIEW v_dimension_trajectory AS
WITH cohort_lifecycle AS (
    SELECT
        cohort,
        MIN(signal_0_center) AS min_I,
        MAX(signal_0_center) AS max_I,
        COUNT(*) AS n_windows
    FROM state_geometry
    WHERE engine = 'shape'
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
    WHERE sg.engine = 'shape'
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

-- Fleet dimension trajectory summary
SELECT
    trajectory_type,
    COUNT(*) AS n_engines,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct,
    ROUND(AVG(dim_delta), 3) AS avg_delta,
    ROUND(MIN(dim_delta), 3) AS min_delta,
    ROUND(MAX(dim_delta), 3) AS max_delta,
    ROUND(AVG(early_dim), 3) AS avg_early_dim,
    ROUND(AVG(late_dim), 3) AS avg_late_dim
FROM v_dimension_trajectory
GROUP BY trajectory_type
ORDER BY avg_delta ASC;

-- Per-cohort trajectory detail
SELECT
    cohort,
    n_windows,
    early_dim,
    late_dim,
    dim_delta,
    trajectory_type,
    change_rank
FROM v_dimension_trajectory
ORDER BY dim_delta ASC;

-- Cross-engine effective_dim comparison (shape vs complexity vs spectral)
-- High divergence = engines "see" different things = mixed failure
WITH cohort_lifecycle AS (
    SELECT
        cohort,
        MIN(signal_0_center) AS min_I,
        MAX(signal_0_center) AS max_I
    FROM state_geometry
    WHERE engine = 'shape'
    GROUP BY cohort
),
engine_dims AS (
    SELECT
        sg.cohort,
        sg.signal_0_center,
        MAX(CASE WHEN sg.engine = 'complexity' THEN sg.effective_dim END) AS cx_dim,
        MAX(CASE WHEN sg.engine = 'shape' THEN sg.effective_dim END) AS sh_dim,
        MAX(CASE WHEN sg.engine = 'spectral' THEN sg.effective_dim END) AS sp_dim
    FROM state_geometry sg
    GROUP BY sg.cohort, sg.signal_0_center
),
late_only AS (
    SELECT ed.*
    FROM engine_dims ed
    JOIN cohort_lifecycle cl ON ed.cohort = cl.cohort
    WHERE ed.signal_0_center >= cl.max_I - (cl.max_I - cl.min_I) * 0.2
)
SELECT
    lo.cohort,
    ROUND(AVG(lo.cx_dim), 3) AS late_cx_dim,
    ROUND(AVG(lo.sh_dim), 3) AS late_sh_dim,
    ROUND(AVG(lo.sp_dim), 3) AS late_sp_dim,
    ROUND(AVG(ABS(lo.cx_dim - lo.sh_dim)), 3) AS cx_sh_gap,
    ROUND(AVG(ABS(lo.cx_dim - lo.sp_dim)), 3) AS cx_sp_gap,
    ROUND(AVG(ABS(lo.sh_dim - lo.sp_dim)), 3) AS sh_sp_gap,
    CASE
        WHEN AVG(ABS(lo.cx_dim - lo.sh_dim)) > 0.3 THEN 'HIGH_DIVERGENCE'
        WHEN AVG(ABS(lo.cx_dim - lo.sh_dim)) > 0.15 THEN 'MODERATE_DIVERGENCE'
        ELSE 'LOW_DIVERGENCE'
    END AS engine_agreement,
    dt.trajectory_type
FROM late_only lo
JOIN v_dimension_trajectory dt ON lo.cohort = dt.cohort
GROUP BY lo.cohort, dt.trajectory_type
ORDER BY cx_sh_gap DESC;

-- Do collapsing engines have different canaries than expanding ones?
SELECT
    dt.trajectory_type,
    cs.signal_id,
    COUNT(*) AS times_canary,
    ROUND(AVG(cs.first_departure_I), 1) AS avg_onset
FROM v_dimension_trajectory dt
JOIN v_canary_sequence cs ON dt.cohort = cs.cohort
WHERE cs.canary_rank = 1
GROUP BY dt.trajectory_type, cs.signal_id
ORDER BY dt.trajectory_type, times_canary DESC;

-- Do collapsing engines have different brittleness profiles?
WITH brittleness_summary AS (
    SELECT
        cohort,
        AVG(brittleness_score) AS avg_brittleness,
        AVG(condition_number) AS avg_condition_number,
        AVG(eigenvalue_gap) AS avg_eigenvalue_gap,
        AVG(temperature) AS avg_temperature
    FROM v_brittleness
    GROUP BY cohort
)
SELECT
    dt.trajectory_type,
    COUNT(DISTINCT dt.cohort) AS n_engines,
    ROUND(AVG(b.avg_brittleness), 4) AS avg_brittleness,
    ROUND(AVG(b.avg_condition_number), 2) AS avg_cond_number,
    ROUND(AVG(b.avg_eigenvalue_gap), 3) AS avg_eig_gap,
    ROUND(AVG(b.avg_temperature), 3) AS avg_temperature
FROM v_dimension_trajectory dt
LEFT JOIN brittleness_summary b ON dt.cohort = b.cohort
GROUP BY dt.trajectory_type
ORDER BY avg_brittleness DESC NULLS LAST;
