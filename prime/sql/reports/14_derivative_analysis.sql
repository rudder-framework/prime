-- ============================================================================
-- REPORT 14: DERIVATIVE ANALYSIS
-- "How fast is each signal changing, and is that changing?"
-- Sources: signal_derivatives, signal_statistics, typology
--          (loaded as views by runner)
-- ============================================================================

-- ============================================================================
-- 1. Derivative Depth Profile: how many meaningful derivative levels
-- ============================================================================
SELECT
    d.signal_id,
    d.derivative_depth,
    ROUND(d.d1_snr, 2) AS d1_snr,
    ROUND(d.d2_snr, 2) AS d2_snr,
    ROUND(d.d3_snr, 2) AS d3_snr,
    ROUND(d.d4_snr, 2) AS d4_snr,
    CASE
        WHEN d.derivative_depth >= 4 THEN 'DEEP'
        WHEN d.derivative_depth = 3 THEN 'MODERATE'
        WHEN d.derivative_depth = 2 THEN 'SHALLOW'
        WHEN d.derivative_depth = 1 THEN 'MINIMAL'
        ELSE 'FLAT'
    END AS depth_visual
FROM signal_derivatives d
ORDER BY d.derivative_depth DESC, d.d1_snr DESC;

-- ============================================================================
-- 2. D1 Analysis: velocity — how fast is each signal changing?
-- ============================================================================
SELECT
    d.signal_id,
    ROUND(d.d1_mean, 6) AS d1_mean,
    ROUND(d.d1_std, 6) AS d1_std,
    ROUND(d.d1_snr, 2) AS d1_snr,
    ROUND(d.d1_late_to_early_ratio, 4) AS d1_late_to_early,
    CASE
        WHEN d.d1_late_to_early_ratio > 2.0 THEN 'ACCELERATING (late > 2x early)'
        WHEN d.d1_late_to_early_ratio > 1.3 THEN 'SPEEDING_UP'
        WHEN d.d1_late_to_early_ratio BETWEEN 0.7 AND 1.3 THEN 'STEADY_RATE'
        WHEN d.d1_late_to_early_ratio < 0.5 THEN 'DECELERATING (late < 0.5x early)'
        ELSE 'SLOWING_DOWN'
    END AS velocity_trend,
    t.continuity
FROM signal_derivatives d
LEFT JOIN typology t
    ON d.cohort = t.cohort AND d.signal_id = t.signal_id
ORDER BY d.d1_snr DESC;

-- ============================================================================
-- 3. D2 Analysis: acceleration — is the rate of change itself changing?
-- ============================================================================
SELECT
    d.signal_id,
    ROUND(d.d2_mean, 6) AS d2_mean,
    ROUND(d.d2_std, 6) AS d2_std,
    ROUND(d.d2_snr, 2) AS d2_snr,
    ROUND(d.d2_late_to_early_ratio, 4) AS d2_late_to_early,
    d.d2_onset_pct,
    d.d2_max_region,
    CASE
        WHEN d.d2_snr < 1.0 THEN 'D2_IS_NOISE (no curvature signal)'
        WHEN d.d2_snr BETWEEN 1.0 AND 3.0 THEN 'WEAK_CURVATURE'
        WHEN d.d2_snr BETWEEN 3.0 AND 10.0 THEN 'MODERATE_CURVATURE'
        WHEN d.d2_snr > 10.0 THEN 'STRONG_CURVATURE'
        ELSE 'UNKNOWN'
    END AS curvature_class
FROM signal_derivatives d
ORDER BY d.d2_snr DESC;

-- ============================================================================
-- 4. Onset Detection Detail: where does D2 become significant?
-- ============================================================================
SELECT
    d.signal_id,
    d.d2_onset_pct,
    d.d2_max_region,
    ROUND(d.d2_late_to_early_ratio, 4) AS d2_late_to_early,
    CASE
        WHEN d.d2_onset_pct IS NULL THEN 'STATIONARY (no onset — D2 uniform throughout)'
        WHEN d.d2_onset_pct < 0.2 THEN 'EARLY_ONSET — degradation active from near start'
        WHEN d.d2_onset_pct < 0.4 THEN 'EARLY_MID_ONSET — limited stable period'
        WHEN d.d2_onset_pct < 0.6 THEN 'MID_LIFE_ONSET — balanced stable/active'
        WHEN d.d2_onset_pct < 0.8 THEN 'LATE_ONSET — long stable period before change'
        ELSE 'VERY_LATE_ONSET — sudden end-of-life acceleration'
    END AS onset_interpretation,
    CASE
        WHEN d.d2_max_region = 'EARLY' THEN 'Most curvature in first half (settling?)'
        WHEN d.d2_max_region = 'LATE' THEN 'Most curvature in second half (degradation?)'
        WHEN d.d2_max_region = 'MIDDLE' THEN 'Most curvature mid-life (transition?)'
        ELSE 'Uniform curvature throughout'
    END AS region_interpretation
FROM signal_derivatives d
ORDER BY d.d2_onset_pct NULLS LAST;

-- ============================================================================
-- 5. Late-to-Early Ratios: is the signal behavior changing over time?
-- ============================================================================
SELECT
    d.signal_id,
    ROUND(d.d1_late_to_early_ratio, 4) AS d1_ratio,
    ROUND(d.d2_late_to_early_ratio, 4) AS d2_ratio,
    CASE
        WHEN d.d1_late_to_early_ratio BETWEEN 0.8 AND 1.2
         AND d.d2_late_to_early_ratio BETWEEN 0.8 AND 1.2
            THEN 'STATIONARY — both velocity and acceleration stable'
        WHEN d.d1_late_to_early_ratio > 1.5 AND d.d2_late_to_early_ratio > 1.5
            THEN 'RUNAWAY — accelerating degradation'
        WHEN d.d1_late_to_early_ratio > 1.5 AND d.d2_late_to_early_ratio < 0.8
            THEN 'PLATEAUING — velocity up but acceleration down'
        WHEN d.d1_late_to_early_ratio < 0.5
            THEN 'STABILIZING — velocity decreasing'
        ELSE 'MIXED'
    END AS trajectory_assessment
FROM signal_derivatives d
ORDER BY d.d1_late_to_early_ratio DESC;

-- ============================================================================
-- 6. SNR Hierarchy: which derivative levels carry real information?
-- ============================================================================
SELECT
    d.signal_id,
    ROUND(d.d1_snr, 2) AS d1_snr,
    ROUND(d.d2_snr, 2) AS d2_snr,
    ROUND(d.d3_snr, 2) AS d3_snr,
    ROUND(d.d4_snr, 2) AS d4_snr,
    -- Visual: bar chart of SNR per level
    REPEAT('|', LEAST(GREATEST(CAST(d.d1_snr AS INTEGER), 0), 20)) AS d1_bar,
    REPEAT('|', LEAST(GREATEST(CAST(d.d2_snr AS INTEGER), 0), 20)) AS d2_bar,
    REPEAT('|', LEAST(GREATEST(CAST(d.d3_snr AS INTEGER), 0), 20)) AS d3_bar,
    REPEAT('|', LEAST(GREATEST(CAST(d.d4_snr AS INTEGER), 0), 20)) AS d4_bar
FROM signal_derivatives d
ORDER BY d.derivative_depth DESC;

-- ============================================================================
-- 7. Cross-signal comparison: which signals have similar derivative structure?
-- ============================================================================
SELECT
    a.cohort,
    a.signal_id AS signal_a,
    b.signal_id AS signal_b,
    ABS(a.derivative_depth - b.derivative_depth) AS depth_diff,
    ABS(a.d1_snr - b.d1_snr) AS d1_snr_diff,
    ABS(a.d2_snr - b.d2_snr) AS d2_snr_diff,
    CASE
        WHEN a.derivative_depth = b.derivative_depth
         AND ABS(a.d1_snr - b.d1_snr) < 2.0
         AND ABS(a.d2_snr - b.d2_snr) < 2.0
            THEN 'SIMILAR_DYNAMICS'
        WHEN ABS(a.derivative_depth - b.derivative_depth) >= 3
            THEN 'VERY_DIFFERENT_DYNAMICS'
        ELSE 'MODERATE_DIFFERENCE'
    END AS dynamics_similarity
FROM signal_derivatives a
JOIN signal_derivatives b ON a.cohort = b.cohort AND a.signal_id < b.signal_id
ORDER BY depth_diff, d1_snr_diff;

-- ============================================================================
-- 8. Eigendecomp Derivative Summary: geometry acceleration per cohort
-- ============================================================================
SELECT
    cohort AS engine,
    -- effective_dim derivatives
    ROUND(AVG(effective_dim_velocity), 4) AS avg_dim_velocity,
    ROUND(AVG(effective_dim_acceleration), 4) AS avg_dim_accel,
    -- eigenvalue_1 derivatives
    ROUND(AVG(eigenvalue_1_velocity), 4) AS avg_eig1_velocity,
    ROUND(AVG(eigenvalue_1_acceleration), 4) AS avg_eig1_accel,
    -- condition_number derivatives
    ROUND(AVG(condition_number_velocity), 4) AS avg_cond_velocity,
    ROUND(AVG(condition_number_acceleration), 4) AS avg_cond_accel,
    -- Trajectory assessment
    CASE
        WHEN AVG(eigenvalue_1_acceleration) > 0 AND AVG(condition_number_acceleration) > 0
            THEN 'RUNAWAY'
        WHEN AVG(eigenvalue_1_acceleration) > 0
            THEN 'CONCENTRATING'
        WHEN AVG(condition_number_acceleration) > 0
            THEN 'ILL_CONDITIONING'
        ELSE 'STABLE_GEOMETRY'
    END AS geometry_trajectory
FROM geometry_dynamics
WHERE eigenvalue_1_velocity IS NOT NULL
  AND NOT isnan(eigenvalue_1_velocity)
GROUP BY cohort
ORDER BY ABS(AVG(eigenvalue_1_acceleration)) DESC
LIMIT 30;

-- ============================================================================
-- 9. Adaptive Windowing Recommendations: what the derivatives tell us
-- ============================================================================
SELECT
    d.signal_id,
    d.derivative_depth,
    d.d2_onset_pct,
    t.memory_class,
    CASE
        WHEN d.d2_onset_pct IS NULL AND d.derivative_depth >= 3
            THEN 'GLOBAL_CACHE — stationary dynamics, embed once'
        WHEN d.d2_onset_pct IS NOT NULL AND d.d2_onset_pct >= 0.4
            THEN 'SPLIT_CACHE — stable then active, embed twice'
        WHEN d.d2_onset_pct IS NOT NULL AND d.d2_onset_pct < 0.4
            THEN 'PERIODIC_CACHE — early onset, refresh embedding regularly'
        WHEN d.derivative_depth <= 1
            THEN 'MINIMAL_COMPUTE — simple signal, basic windowing sufficient'
        ELSE 'DEFAULT'
    END AS cache_recommendation,
    CASE
        WHEN d.derivative_depth >= 4 THEN 'Full FTLE + Lyapunov'
        WHEN d.derivative_depth = 3 THEN 'FTLE sufficient, skip Lyapunov spectrum'
        WHEN d.derivative_depth = 2 THEN 'Trend analysis + basic FTLE'
        WHEN d.derivative_depth <= 1 THEN 'Skip dynamics — use drift detection only'
        ELSE 'DEFAULT'
    END AS compute_recommendation
FROM signal_derivatives d
LEFT JOIN typology t
    ON d.cohort = t.cohort AND d.signal_id = t.signal_id
ORDER BY d.derivative_depth DESC;
