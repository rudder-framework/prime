-- ============================================================================
-- REPORT 22: EXECUTIVE SUMMARY
-- ============================================================================
-- One-page overview: what does this dataset look like, what happened, and
-- what should the operator check first?
--
-- Pulls from all upstream reports to produce a unified status dashboard.
-- ============================================================================


-- ============================================================================
-- SECTION 1: DATASET OVERVIEW
-- ============================================================================

WITH signal_counts AS (
    SELECT
        COUNT(DISTINCT signal_id) AS n_signals,
        COUNT(DISTINCT cohort) AS n_cohorts,
        MIN(signal_0) AS signal_0_min,
        MAX(signal_0) AS signal_0_max,
        COUNT(*) AS n_observations
    FROM observations
),
geometry_counts AS (
    SELECT
        COUNT(*) AS n_geometry_windows,
        COUNT(DISTINCT cohort) AS n_geometry_cohorts
    FROM state_geometry
)
SELECT
    sc.n_signals,
    sc.n_cohorts,
    sc.n_observations,
    ROUND(sc.signal_0_min, 1) AS signal_0_min,
    ROUND(sc.signal_0_max, 1) AS signal_0_max,
    ROUND(sc.signal_0_max - sc.signal_0_min, 1) AS signal_0_range,
    gc.n_geometry_windows
FROM signal_counts sc
CROSS JOIN geometry_counts gc;


-- ============================================================================
-- SECTION 2: SYSTEM HEALTH SCORECARD
-- One row per cohort with key health indicators
-- ============================================================================

WITH geometry_health AS (
    SELECT
        cohort,
        ROUND(AVG(effective_dim), 2) AS avg_eff_dim,
        ROUND(AVG(condition_number), 2) AS avg_cond_num,
        ROUND(AVG(total_variance), 2) AS avg_energy,
        COUNT(*) AS n_windows
    FROM state_geometry
    GROUP BY cohort
),
brittleness_health AS (
    SELECT
        cohort,
        ROUND(AVG(brittleness_score), 4) AS avg_brittleness
    FROM v_brittleness
    GROUP BY cohort
),
deviation_health AS (
    SELECT
        cohort,
        departure_assessment,
        current_severity,
        ROUND(pct_abnormal, 1) AS pct_abnormal,
        most_common_deviation_source
    FROM v_deviation_entity_summary
),
coupling_health AS (
    SELECT
        cohort,
        ROUND(AVG(ABS(correlation)), 3) AS avg_abs_corr
    FROM signal_pairwise
    WHERE correlation IS NOT NULL AND NOT isnan(correlation)
    GROUP BY cohort
),
dim_trajectory_health AS (
    SELECT cohort, trajectory_type, dim_delta
    FROM v_dimension_trajectory
),
thermo_health AS (
    SELECT
        cohort,
        ROUND(temperature, 4) AS temperature,
        ROUND(entropy, 4) AS entropy
    FROM thermodynamics
)
SELECT
    g.cohort,
    g.n_windows,
    g.avg_eff_dim,
    g.avg_energy,
    g.avg_cond_num,
    b.avg_brittleness,
    CASE
        WHEN b.avg_brittleness IS NULL THEN 'NO_DATA'
        WHEN b.avg_brittleness > 100 THEN 'CRITICAL'
        WHEN b.avg_brittleness > 30  THEN 'ELEVATED'
        ELSE 'HEALTHY'
    END AS brittleness_level,
    d.departure_assessment,
    d.current_severity,
    d.pct_abnormal AS pct_abnormal,
    c.avg_abs_corr AS coupling,
    dt.trajectory_type AS dim_trajectory,
    th.temperature,
    th.entropy
FROM geometry_health g
LEFT JOIN brittleness_health b ON g.cohort = b.cohort
LEFT JOIN deviation_health d ON g.cohort = d.cohort
LEFT JOIN coupling_health c ON g.cohort = c.cohort
LEFT JOIN dim_trajectory_health dt ON g.cohort = dt.cohort
LEFT JOIN thermo_health th ON g.cohort = th.cohort
ORDER BY d.pct_abnormal DESC NULLS LAST;


-- ============================================================================
-- SECTION 3: TOP ALERTS (what to check first)
-- ============================================================================

-- Priority alerts from across the report suite
WITH alerts AS (
    -- Brittleness alerts
    SELECT
        cohort,
        'BRITTLENESS' AS alert_type,
        'avg=' || ROUND(AVG(brittleness_score), 2) || ', max=' || ROUND(MAX(brittleness_score), 2) AS detail,
        CASE WHEN AVG(brittleness_score) > 100 THEN 1 ELSE 2 END AS priority
    FROM v_brittleness
    GROUP BY cohort
    HAVING AVG(brittleness_score) > 30

    UNION ALL

    -- Departure alerts
    SELECT
        cohort,
        'DEPARTURE' AS alert_type,
        departure_assessment || ' (' || ROUND(pct_abnormal, 1) || '% abnormal)' AS detail,
        CASE departure_assessment WHEN 'departed' THEN 1 WHEN 'unstable' THEN 2 ELSE 3 END AS priority
    FROM v_deviation_entity_summary
    WHERE departure_assessment IN ('departed', 'unstable')

    UNION ALL

    -- Dimension collapse/expansion alerts
    SELECT
        cohort,
        'DIM_' || trajectory_type AS alert_type,
        'delta=' || ROUND(dim_delta, 3) || ' (early=' || early_dim || ' → late=' || late_dim || ')' AS detail,
        CASE WHEN ABS(dim_delta) > 0.5 THEN 1 ELSE 2 END AS priority
    FROM v_dimension_trajectory
    WHERE trajectory_type != 'STABLE'

    UNION ALL

    -- Energy balance gaps
    SELECT
        cohort,
        'ENERGY_' || UPPER(energy_balance_status) AS alert_type,
        'gap=' || ROUND(energy_gap, 4) AS detail,
        CASE gap_severity WHEN 'significant_gap' THEN 1 WHEN 'moderate_gap' THEN 2 ELSE 3 END AS priority
    FROM v_energy_balance
    WHERE energy_balance_status != 'balanced'
)
SELECT
    cohort,
    alert_type,
    detail,
    CASE priority WHEN 1 THEN 'HIGH' WHEN 2 THEN 'MEDIUM' ELSE 'LOW' END AS priority
FROM alerts
ORDER BY priority, cohort
LIMIT 30;


-- ============================================================================
-- SECTION 4: SIGNAL DYNAMICS OVERVIEW
-- FTLE and Lyapunov at a glance
-- ============================================================================

SELECT
    f.signal_id,
    f.cohort,
    ROUND(f.ftle, 4) AS ftle,
    ROUND(l.lyapunov, 4) AS lyapunov,
    f.embedding_dim,
    CASE
        WHEN f.ftle IS NULL OR isnan(f.ftle) THEN 'INSUFFICIENT_DATA'
        WHEN f.ftle > 0.05 THEN 'CHAOTIC'
        WHEN f.ftle > 0.01 THEN 'WEAKLY_CHAOTIC'
        WHEN f.ftle > -0.01 THEN 'MARGINAL'
        ELSE 'STABLE'
    END AS dynamics_class
FROM ftle f
LEFT JOIN lyapunov l ON f.signal_id = l.signal_id AND f.cohort = l.cohort
ORDER BY f.ftle DESC;


-- ============================================================================
-- SECTION 5: CANARY SIGNALS — EARLIEST WARNING
-- Which signals departed first? These are the early warning indicators.
-- ============================================================================

SELECT
    cohort,
    signal_id,
    canary_rank,
    ROUND(first_departure_I, 1) AS onset_I,
    ROUND(propagation_delay, 1) AS delay_from_first,
    ROUND(departure_severity, 4) AS severity
FROM v_canary_sequence
WHERE canary_rank <= 3
ORDER BY cohort, canary_rank
LIMIT 20;
