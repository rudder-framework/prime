-- ============================================================================
-- Engines SIGNAL RANKING REPORTS
-- ============================================================================
--
-- Identifies which signals are most important, problematic, or informative.
-- Helps operators focus attention on the right sensors.
--
-- Usage: Run against observations and primitives tables
-- ============================================================================


-- ============================================================================
-- REPORT 1: SIGNAL IMPORTANCE RANKING
-- Ranks signals by information content and variability
-- ============================================================================

WITH
signal_stats AS (
    SELECT
        cohort,
        signal_id,
        COUNT(*) AS n_points,
        AVG(value) AS mean_val,
        STDDEV_POP(value) AS std_val,
        MIN(value) AS min_val,
        MAX(value) AS max_val,
        (MAX(value) - MIN(value)) / NULLIF(STDDEV_POP(value), 0) AS range_std_ratio
    FROM observations
    GROUP BY cohort, signal_id
),

entropy_proxy AS (
    -- Approximate entropy via histogram bin diversity
    SELECT
        cohort,
        signal_id,
        COUNT(DISTINCT FLOOR(10 * (value - MIN(value) OVER w) / NULLIF(MAX(value) OVER w - MIN(value) OVER w, 0))) AS n_bins_used
    FROM observations
    WINDOW w AS (PARTITION BY cohort, signal_id)
    GROUP BY cohort, signal_id, value
)

SELECT
    s.cohort,
    s.signal_id,
    s.n_points,
    ROUND(s.std_val / NULLIF(ABS(s.mean_val), 0), 3) AS coeff_variation,
    ROUND(s.range_std_ratio, 2) AS range_std_ratio,
    ROUND(100.0 * s.std_val / (SELECT MAX(std_val) FROM signal_stats WHERE cohort = s.cohort), 1) AS pct_max_variability,
    CASE
        WHEN s.std_val / NULLIF(ABS(s.mean_val), 0) > 0.5 THEN 'HIGH_INFO'
        WHEN s.std_val / NULLIF(ABS(s.mean_val), 0) > 0.1 THEN 'MEDIUM_INFO'
        ELSE 'LOW_INFO'
    END AS information_class,
    RANK() OVER (PARTITION BY s.cohort ORDER BY s.std_val DESC) AS variability_rank
FROM signal_stats s
ORDER BY s.cohort, variability_rank;


-- ============================================================================
-- REPORT 2: PROBLEM SIGNALS (needs attention)
-- ============================================================================

WITH
time_bounds AS (
    SELECT
        cohort,
        MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations
    GROUP BY cohort
),

baseline AS (
    SELECT
        o.cohort,
        o.signal_id,
        AVG(o.value) AS baseline_mean,
        STDDEV_POP(o.value) AS baseline_std
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort
    WHERE o.I <= t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

current_state AS (
    SELECT
        o.cohort,
        o.signal_id,
        AVG(o.value) AS current_mean,
        STDDEV_POP(o.value) AS current_std,
        COUNT(*) AS n_current
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort
    WHERE o.I > t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

problems AS (
    SELECT
        b.cohort,
        b.signal_id,
        ABS((c.current_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0)) AS drift_z,
        c.current_std / NULLIF(b.baseline_std, 0) AS volatility_ratio,
        CASE WHEN ABS((c.current_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0)) > 2 THEN 1 ELSE 0 END AS has_drift,
        CASE WHEN c.current_std / NULLIF(b.baseline_std, 0) > 1.5 THEN 1 ELSE 0 END AS has_vol_increase
    FROM baseline b
    JOIN current_state c USING (cohort, signal_id)
)

SELECT
    cohort,
    signal_id,
    ROUND(drift_z, 2) AS drift_sigma,
    ROUND(volatility_ratio, 2) AS vol_ratio,
    has_drift + has_vol_increase AS problem_score,
    CASE
        WHEN has_drift = 1 AND has_vol_increase = 1 THEN 'CRITICAL'
        WHEN has_drift = 1 THEN 'DRIFTING'
        WHEN has_vol_increase = 1 THEN 'UNSTABLE'
        ELSE 'OK'
    END AS problem_type,
    RANK() OVER (PARTITION BY cohort ORDER BY drift_z + volatility_ratio DESC) AS problem_rank
FROM problems
WHERE has_drift = 1 OR has_vol_increase = 1
ORDER BY cohort, problem_rank;


-- ============================================================================
-- REPORT 3: LEADING INDICATOR CANDIDATES
-- Signals that change before others (potential early warnings)
-- ============================================================================

WITH
windowed AS (
    SELECT
        cohort,
        signal_id,
        NTILE(10) OVER (PARTITION BY cohort, signal_id ORDER BY I) AS window_id,
        value
    FROM observations
),

window_stats AS (
    SELECT
        cohort,
        signal_id,
        window_id,
        AVG(value) AS window_mean,
        STDDEV_POP(value) AS window_std
    FROM windowed
    GROUP BY cohort, signal_id, window_id
),

first_movement AS (
    SELECT
        cohort,
        signal_id,
        MIN(CASE
            WHEN ABS(window_mean - LAG(window_mean) OVER w) / NULLIF(window_std, 0) > 0.5
            THEN window_id
        END) AS first_move_window
    FROM window_stats
    WINDOW w AS (PARTITION BY cohort, signal_id ORDER BY window_id)
    GROUP BY cohort, signal_id
)

SELECT
    cohort,
    signal_id,
    first_move_window,
    CASE
        WHEN first_move_window <= 2 THEN 'EARLY_MOVER'
        WHEN first_move_window <= 4 THEN 'MID_MOVER'
        WHEN first_move_window <= 6 THEN 'LATE_MOVER'
        ELSE 'STABLE'
    END AS movement_class,
    RANK() OVER (PARTITION BY cohort ORDER BY COALESCE(first_move_window, 99)) AS early_rank
FROM first_movement
WHERE first_move_window IS NOT NULL
ORDER BY cohort, early_rank;


-- ============================================================================
-- REPORT 4: REDUNDANT SIGNALS
-- Identifies highly correlated signals (potential redundancy)
-- ============================================================================

WITH
signal_pairs AS (
    SELECT
        a.cohort,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        CORR(a.value, b.value) AS correlation
    FROM observations a
    JOIN observations b
        ON a.cohort = b.cohort
        AND a.I = b.I
        AND a.signal_id < b.signal_id
    GROUP BY a.cohort, a.signal_id, b.signal_id
)

SELECT
    cohort,
    signal_a,
    signal_b,
    ROUND(correlation, 3) AS correlation,
    CASE
        WHEN ABS(correlation) > 0.95 THEN 'HIGHLY_REDUNDANT'
        WHEN ABS(correlation) > 0.85 THEN 'REDUNDANT'
        WHEN ABS(correlation) > 0.7 THEN 'CORRELATED'
        ELSE 'INDEPENDENT'
    END AS redundancy_class,
    CASE
        WHEN correlation > 0.95 THEN 'Consider removing one'
        WHEN correlation < -0.95 THEN 'Inverse relationship - keep both'
        ELSE 'OK'
    END AS recommendation
FROM signal_pairs
WHERE ABS(correlation) > 0.7
ORDER BY cohort, ABS(correlation) DESC;


-- ============================================================================
-- REPORT 5: SIGNAL HEALTH DASHBOARD
-- Comprehensive signal-by-signal health view
-- ============================================================================

WITH
time_bounds AS (
    SELECT cohort, MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations GROUP BY cohort
),

baseline AS (
    SELECT o.cohort, o.signal_id,
        AVG(o.value) AS mu, STDDEV_POP(o.value) AS sigma,
        MIN(o.value) AS min_val, MAX(o.value) AS max_val
    FROM observations o JOIN time_bounds t USING (cohort)
    WHERE o.I <= t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

current AS (
    SELECT o.cohort, o.signal_id,
        AVG(o.value) AS current_mean, STDDEV_POP(o.value) AS current_std,
        MIN(o.value) AS current_min, MAX(o.value) AS current_max
    FROM observations o JOIN time_bounds t USING (cohort)
    WHERE o.I > t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

health AS (
    SELECT
        b.cohort,
        b.signal_id,
        ROUND(b.mu, 4) AS baseline_mean,
        ROUND(c.current_mean, 4) AS current_mean,
        ROUND((c.current_mean - b.mu) / NULLIF(b.sigma, 0), 2) AS mean_shift_z,
        ROUND(c.current_std / NULLIF(b.sigma, 0), 2) AS std_ratio,
        ROUND(100 * (c.current_max - b.max_val) / NULLIF(b.max_val, 0), 1) AS max_change_pct,
        ROUND(100 * (c.current_min - b.min_val) / NULLIF(ABS(b.min_val), 0), 1) AS min_change_pct
    FROM baseline b
    JOIN current c USING (cohort, signal_id)
)

SELECT
    cohort,
    signal_id,
    baseline_mean,
    current_mean,
    mean_shift_z,
    std_ratio,
    -- Overall health score (lower is better)
    ROUND(ABS(mean_shift_z) + ABS(std_ratio - 1) * 2, 2) AS health_score,
    -- Traffic light
    CASE
        WHEN ABS(mean_shift_z) > 2 OR std_ratio > 1.5 THEN 'RED'
        WHEN ABS(mean_shift_z) > 1 OR std_ratio > 1.2 THEN 'YELLOW'
        ELSE 'GREEN'
    END AS status,
    -- Specific issues
    CASE
        WHEN mean_shift_z > 2 THEN 'HIGH'
        WHEN mean_shift_z < -2 THEN 'LOW'
        WHEN std_ratio > 1.5 THEN 'UNSTABLE'
        ELSE 'NORMAL'
    END AS issue
FROM health
ORDER BY cohort, ABS(mean_shift_z) + ABS(std_ratio - 1) DESC;
