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
-- REPORT 2: PROBLEM SIGNALS — ranked by volatility change and trajectory departure
-- No z-scores. Volatility ratio and slope change do the work.
-- ============================================================================

WITH
signal_halves AS (
    SELECT
        o.cohort,
        o.signal_id,
        o.I,
        o.value,
        CASE
            WHEN o.I < life.min_I + (life.max_I - life.min_I) * 0.5
            THEN 'early'
            ELSE 'late'
        END AS half
    FROM observations o
    JOIN (
        SELECT cohort, MIN(I) AS min_I, MAX(I) AS max_I
        FROM observations GROUP BY cohort
    ) life ON o.cohort = life.cohort
),
half_stats AS (
    SELECT
        cohort,
        signal_id,
        half,
        AVG(value) AS half_mean,
        STDDEV(value) AS half_std,
        REGR_SLOPE(value, I) AS half_slope
    FROM signal_halves
    GROUP BY cohort, signal_id, half
),
signal_trajectory AS (
    SELECT
        e.cohort,
        e.signal_id,
        e.half_mean AS early_mean,
        l.half_mean AS late_mean,
        e.half_std AS early_std,
        l.half_std AS late_std,
        l.half_std / NULLIF(e.half_std, 0) AS vol_ratio,
        e.half_slope AS early_slope,
        l.half_slope AS late_slope,
        l.half_slope - e.half_slope AS slope_change,
        ABS(l.half_slope - e.half_slope) / NULLIF(ABS(e.half_slope), 0) AS slope_change_pct
    FROM half_stats e
    JOIN half_stats l ON e.cohort = l.cohort
        AND e.signal_id = l.signal_id
        AND e.half = 'early' AND l.half = 'late'
)
SELECT
    cohort,
    signal_id,
    ROUND(vol_ratio, 2) AS vol_ratio,
    ROUND(early_slope, 6) AS early_slope,
    ROUND(late_slope, 6) AS late_slope,
    ROUND(slope_change_pct, 2) AS slope_change_pct,
    CASE
        WHEN vol_ratio > 1.5 AND ABS(slope_change_pct) > 2 THEN 'CRITICAL'
        WHEN ABS(slope_change_pct) > 2 THEN 'TRAJECTORY_CHANGED'
        WHEN vol_ratio > 1.5 THEN 'UNSTABLE'
        ELSE 'OK'
    END AS problem_type,
    RANK() OVER (
        PARTITION BY cohort
        ORDER BY vol_ratio * (1.0 + COALESCE(ABS(slope_change_pct), 0)) DESC NULLS LAST
    ) AS problem_rank
FROM signal_trajectory
WHERE vol_ratio > 1.3 OR ABS(slope_change_pct) > 1.0
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
-- Comprehensive signal-by-signal health view using trajectory metrics
-- ============================================================================

WITH
signal_halves AS (
    SELECT
        o.cohort,
        o.signal_id,
        o.I,
        o.value,
        CASE
            WHEN o.I < life.min_I + (life.max_I - life.min_I) * 0.5
            THEN 'early'
            ELSE 'late'
        END AS half
    FROM observations o
    JOIN (
        SELECT cohort, MIN(I) AS min_I, MAX(I) AS max_I
        FROM observations GROUP BY cohort
    ) life ON o.cohort = life.cohort
),
half_stats AS (
    SELECT
        cohort,
        signal_id,
        half,
        AVG(value) AS half_mean,
        STDDEV(value) AS half_std,
        REGR_SLOPE(value, I) AS half_slope
    FROM signal_halves
    GROUP BY cohort, signal_id, half
),
health AS (
    SELECT
        e.cohort,
        e.signal_id,
        ROUND(e.half_mean, 4) AS early_mean,
        ROUND(l.half_mean, 4) AS late_mean,
        ROUND(l.half_std / NULLIF(e.half_std, 0), 2) AS vol_ratio,
        ROUND(e.half_slope, 6) AS early_slope,
        ROUND(l.half_slope, 6) AS late_slope,
        ROUND(l.half_slope / NULLIF(e.half_slope, 0), 2) AS slope_ratio,
        CASE WHEN SIGN(e.half_slope) != SIGN(l.half_slope) THEN 1 ELSE 0 END AS slope_reversed
    FROM half_stats e
    JOIN half_stats l ON e.cohort = l.cohort
        AND e.signal_id = l.signal_id
        AND e.half = 'early' AND l.half = 'late'
)

SELECT
    cohort,
    signal_id,
    early_mean,
    late_mean,
    vol_ratio,
    early_slope,
    late_slope,
    slope_ratio,
    slope_reversed,
    -- Health score: combined trajectory departure
    ROUND(
        COALESCE(ABS(slope_ratio - 1.0), 0) + ABS(vol_ratio - 1.0) * 2 + slope_reversed * 3,
        2
    ) AS health_score,
    -- Traffic light based on trajectory
    CASE
        WHEN slope_reversed = 1 OR ABS(slope_ratio) > 3.0 OR vol_ratio > 1.5 THEN 'RED'
        WHEN ABS(slope_ratio) > 2.0 OR vol_ratio > 1.2 THEN 'YELLOW'
        ELSE 'GREEN'
    END AS status,
    -- Specific issues
    CASE
        WHEN slope_reversed = 1 THEN 'SLOPE_REVERSED'
        WHEN slope_ratio > 3.0 THEN 'ACCELERATING'
        WHEN slope_ratio < -1.0 THEN 'REVERSED'
        WHEN vol_ratio > 1.5 THEN 'UNSTABLE'
        ELSE 'NORMAL'
    END AS issue
FROM health
ORDER BY cohort, COALESCE(ABS(slope_ratio - 1.0), 0) + ABS(vol_ratio - 1.0) DESC;
