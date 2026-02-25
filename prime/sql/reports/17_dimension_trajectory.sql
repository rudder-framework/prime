-- ============================================================================
-- 17_dimension_trajectory.sql
-- ============================================================================
-- Dimension trajectory analysis report
-- Depends on: v_dimension_trajectory (layer 17), v_eigenvalue_1_trajectory (layer 17),
--             v_condition_number_trajectory (layer 17),
--             v_canary_sequence (layer 13), v_brittleness (layer 12)
-- ============================================================================

.print ''
.print '============================================================================'
.print '                     DIMENSION TRAJECTORY ANALYSIS                         '
.print '============================================================================'

-- Fleet dimension trajectory summary
SELECT
    trajectory_type,
    COUNT(*) AS n_cohorts,
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

-- Do collapsing cohorts have different canaries than expanding ones?
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

-- Do collapsing cohorts have different brittleness profiles?
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
    COUNT(DISTINCT dt.cohort) AS n_cohorts,
    ROUND(AVG(b.avg_brittleness), 4) AS avg_brittleness,
    ROUND(AVG(b.avg_condition_number), 2) AS avg_cond_number,
    ROUND(AVG(b.avg_eigenvalue_gap), 3) AS avg_eig_gap,
    ROUND(AVG(b.avg_temperature), 3) AS avg_temperature
FROM v_dimension_trajectory dt
LEFT JOIN brittleness_summary b ON dt.cohort = b.cohort
GROUP BY dt.trajectory_type
ORDER BY avg_brittleness DESC NULLS LAST;


.print ''
.print '============================================================================'
.print '                     EIGENVALUE_1 TRAJECTORY                               '
.print '============================================================================'

-- Fleet eigenvalue_1 trajectory summary
SELECT
    eigenvalue_trajectory,
    COUNT(*) AS n_cohorts,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct,
    ROUND(AVG(eigenvalue_delta), 3) AS avg_delta,
    ROUND(AVG(early_eigenvalue), 3) AS avg_early,
    ROUND(AVG(late_eigenvalue), 3) AS avg_late
FROM v_eigenvalue_1_trajectory
GROUP BY eigenvalue_trajectory
ORDER BY avg_delta DESC;

-- Per-cohort eigenvalue_1 trajectory detail
SELECT
    cohort,
    n_windows,
    early_eigenvalue,
    late_eigenvalue,
    eigenvalue_delta,
    eigenvalue_trajectory
FROM v_eigenvalue_1_trajectory
ORDER BY eigenvalue_delta DESC;


.print ''
.print '============================================================================'
.print '                     CONDITION NUMBER TRAJECTORY                           '
.print '============================================================================'

-- Fleet condition number trajectory summary
SELECT
    condition_trajectory,
    COUNT(*) AS n_cohorts,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct,
    ROUND(AVG(cond_delta), 3) AS avg_delta,
    ROUND(AVG(early_cond), 3) AS avg_early,
    ROUND(AVG(late_cond), 3) AS avg_late
FROM v_condition_number_trajectory
GROUP BY condition_trajectory
ORDER BY avg_delta DESC;

-- Per-cohort condition number trajectory detail
SELECT
    cohort,
    n_windows,
    early_cond,
    late_cond,
    cond_delta,
    condition_trajectory
FROM v_condition_number_trajectory
ORDER BY cond_delta DESC;
