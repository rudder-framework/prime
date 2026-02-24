-- ============================================================================
-- REPORT 18: FTLE & LYAPUNOV SUMMARY
-- ============================================================================
-- Finite-Time Lyapunov Exponents measure sensitivity to initial conditions.
-- Positive FTLE = nearby trajectories diverge = chaotic / unstable dynamics.
-- Negative FTLE = trajectories converge = stable / dissipative dynamics.
--
-- Sources: ftle, ftle_backward, ftle_rolling, lyapunov, ridge_proximity
-- ============================================================================


-- ============================================================================
-- SECTION 1: GLOBAL FTLE PER SIGNAL
-- Full-signal FTLE (forward and backward) — one number per signal
-- ============================================================================

SELECT
    f.signal_id,
    f.cohort,
    ROUND(f.ftle, 4) AS ftle_forward,
    ROUND(b.ftle, 4) AS ftle_backward,
    f.embedding_dim,
    f.embedding_tau,
    f.n_samples,
    CASE
        WHEN f.ftle > 0.05 THEN 'CHAOTIC'
        WHEN f.ftle > 0.01 THEN 'WEAKLY_CHAOTIC'
        WHEN f.ftle > -0.01 THEN 'MARGINAL'
        ELSE 'STABLE'
    END AS dynamics_class,
    CASE
        WHEN f.ftle > 0 AND b.ftle < 0 THEN 'ATTRACTOR_WITH_SENSITIVITY'
        WHEN f.ftle > 0 AND b.ftle > 0 THEN 'FULLY_UNSTABLE'
        WHEN f.ftle < 0 AND b.ftle < 0 THEN 'STRONGLY_STABLE'
        ELSE 'MIXED'
    END AS stability_type
FROM ftle f
LEFT JOIN ftle_backward b ON f.signal_id = b.signal_id AND f.cohort = b.cohort
ORDER BY f.ftle DESC;


-- ============================================================================
-- SECTION 2: LYAPUNOV EXPONENTS
-- ============================================================================

SELECT
    signal_id,
    cohort,
    ROUND(lyapunov, 4) AS lyapunov_exponent,
    embedding_dim,
    embedding_tau,
    n_samples,
    ROUND(confidence, 3) AS confidence,
    CASE
        WHEN lyapunov > 0.05 THEN 'POSITIVE — chaotic'
        WHEN lyapunov > 0 THEN 'WEAKLY_POSITIVE — edge of chaos'
        WHEN lyapunov > -0.05 THEN 'NEAR_ZERO — marginal stability'
        ELSE 'NEGATIVE — stable'
    END AS interpretation
FROM lyapunov
ORDER BY lyapunov DESC;


-- ============================================================================
-- SECTION 3: ROLLING FTLE TRAJECTORY
-- How does FTLE evolve over time? Rising FTLE = system destabilizing.
-- ============================================================================

WITH ftle_trajectory AS (
    SELECT
        signal_id,
        cohort,
        signal_0_center,
        ftle,
        AVG(ftle) OVER (PARTITION BY signal_id, cohort
            ORDER BY signal_0_center
            ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING) AS ftle_smooth,
        ftle - LAG(ftle) OVER (PARTITION BY signal_id, cohort ORDER BY signal_0_center) AS ftle_delta
    FROM ftle_rolling
),
signal_summary AS (
    SELECT
        signal_id,
        cohort,
        ROUND(AVG(ftle), 4) AS mean_ftle,
        ROUND(MAX(ftle), 4) AS max_ftle,
        ROUND(MIN(ftle), 4) AS min_ftle,
        ROUND(STDDEV_POP(ftle), 4) AS ftle_std,
        ROUND(REGR_SLOPE(ftle, signal_0_center), 8) AS ftle_trend_slope,
        COUNT(*) AS n_windows
    FROM ftle_trajectory
    GROUP BY signal_id, cohort
)
SELECT
    signal_id,
    cohort,
    mean_ftle,
    max_ftle,
    min_ftle,
    ftle_std,
    ftle_trend_slope,
    n_windows,
    CASE
        WHEN ftle_trend_slope > 0.0001 THEN 'DESTABILIZING'
        WHEN ftle_trend_slope < -0.0001 THEN 'STABILIZING'
        ELSE 'STATIONARY'
    END AS ftle_trend
FROM signal_summary
ORDER BY ftle_trend_slope DESC;


-- ============================================================================
-- SECTION 4: RIDGE PROXIMITY — TIME TO FTLE RIDGE
-- How close is each signal to the FTLE ridge (divergence maximum)?
-- ============================================================================

WITH latest_proximity AS (
    SELECT DISTINCT ON (signal_id, cohort)
        signal_id,
        cohort,
        signal_0_center,
        ROUND(ftle_current, 4) AS ftle_current,
        ROUND(ftle_gradient, 6) AS ftle_gradient,
        ROUND(speed, 4) AS state_speed,
        ROUND(urgency, 6) AS urgency,
        ROUND(time_to_ridge, 1) AS time_to_ridge
    FROM ridge_proximity
    ORDER BY signal_id, cohort, signal_0_center DESC
)
SELECT
    signal_id,
    cohort,
    ftle_current,
    ftle_gradient,
    state_speed,
    urgency,
    time_to_ridge,
    CASE
        WHEN time_to_ridge IS NOT NULL AND time_to_ridge < 50 THEN 'IMMINENT'
        WHEN time_to_ridge IS NOT NULL AND time_to_ridge < 200 THEN 'APPROACHING'
        WHEN ftle_gradient > 0 THEN 'DIVERGING_SLOWLY'
        ELSE 'DISTANT'
    END AS ridge_status
FROM latest_proximity
ORDER BY time_to_ridge ASC NULLS LAST;


-- ============================================================================
-- SECTION 5: FTLE EARLY VS LATE (destabilization detection)
-- ============================================================================

WITH lifecycle AS (
    SELECT
        signal_id, cohort,
        MIN(signal_0_center) AS min_I,
        MAX(signal_0_center) AS max_I
    FROM ftle_rolling
    GROUP BY signal_id, cohort
),
early_late AS (
    SELECT
        r.signal_id,
        r.cohort,
        AVG(CASE WHEN r.signal_0_center <= lc.min_I + (lc.max_I - lc.min_I) * 0.3
            THEN r.ftle END) AS early_ftle,
        AVG(CASE WHEN r.signal_0_center >= lc.max_I - (lc.max_I - lc.min_I) * 0.3
            THEN r.ftle END) AS late_ftle
    FROM ftle_rolling r
    JOIN lifecycle lc ON r.signal_id = lc.signal_id AND r.cohort = lc.cohort
    GROUP BY r.signal_id, r.cohort
)
SELECT
    signal_id,
    cohort,
    ROUND(early_ftle, 4) AS early_ftle,
    ROUND(late_ftle, 4) AS late_ftle,
    ROUND(late_ftle - early_ftle, 4) AS ftle_shift,
    CASE
        WHEN late_ftle - early_ftle > 0.02 THEN 'DESTABILIZED'
        WHEN late_ftle - early_ftle < -0.02 THEN 'STABILIZED'
        ELSE 'UNCHANGED'
    END AS ftle_evolution
FROM early_late
WHERE early_ftle IS NOT NULL AND late_ftle IS NOT NULL
ORDER BY (late_ftle - early_ftle) DESC;
