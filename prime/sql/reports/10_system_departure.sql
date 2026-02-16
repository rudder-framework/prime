-- ============================================================================
-- Engines SYSTEM DEPARTURE REPORTS
-- ============================================================================
--
-- Departure summaries per cohort.
-- Status based on trajectory metrics.
-- Uses trajectory metrics (slope_ratio, vol_ratio) instead of z-scores.
--
-- Usage: Run against observations and primitives tables
-- ============================================================================


-- ============================================================================
-- REPORT 1: OVERALL SYSTEM DEPARTURE SCORECARD
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

signal_departure AS (
    SELECT
        e.cohort,
        e.signal_id,
        l.half_std / NULLIF(e.half_std, 0) AS vol_ratio,
        l.half_slope / NULLIF(e.half_slope, 0) AS slope_ratio,
        CASE WHEN SIGN(e.half_slope) != SIGN(l.half_slope) THEN 1 ELSE 0 END AS slope_reversed,
        -- Trajectory status
        CASE
            WHEN SIGN(e.half_slope) != SIGN(l.half_slope) THEN 'DEPARTED'
            WHEN ABS(l.half_slope / NULLIF(e.half_slope, 0)) > 3.0 THEN 'DEPARTED'
            WHEN ABS(l.half_slope / NULLIF(e.half_slope, 0)) > 2.0 THEN 'SHIFTED'
            ELSE 'STABLE'
        END AS trajectory_status,
        CASE
            WHEN l.half_std / NULLIF(e.half_std, 0) > 1.5 THEN 'DEPARTED'
            WHEN l.half_std / NULLIF(e.half_std, 0) > 1.2 THEN 'SHIFTED'
            ELSE 'STABLE'
        END AS volatility_status
    FROM half_stats e
    JOIN half_stats l ON e.cohort = l.cohort
        AND e.signal_id = l.signal_id
        AND e.half = 'early' AND l.half = 'late'
)

SELECT
    cohort,
    COUNT(*) AS total_signals,

    -- Trajectory departure
    SUM(CASE WHEN trajectory_status = 'STABLE' THEN 1 ELSE 0 END) AS traj_stable,
    SUM(CASE WHEN trajectory_status = 'SHIFTED' THEN 1 ELSE 0 END) AS traj_shifted,
    SUM(CASE WHEN trajectory_status = 'DEPARTED' THEN 1 ELSE 0 END) AS traj_departed,

    -- Volatility departure
    SUM(CASE WHEN volatility_status = 'STABLE' THEN 1 ELSE 0 END) AS vol_stable,
    SUM(CASE WHEN volatility_status = 'SHIFTED' THEN 1 ELSE 0 END) AS vol_shifted,
    SUM(CASE WHEN volatility_status = 'DEPARTED' THEN 1 ELSE 0 END) AS vol_departed,

    -- Trajectory-specific counts
    SUM(slope_reversed) AS n_slope_reversed,
    SUM(CASE WHEN vol_ratio > 1.3 THEN 1 ELSE 0 END) AS n_vol_elevated,
    SUM(CASE WHEN ABS(slope_ratio) > 2 OR ABS(slope_ratio) < 0.5 THEN 1 ELSE 0 END) AS n_trajectory_changed,

    -- Overall status
    CASE
        WHEN SUM(slope_reversed) > 3
          OR SUM(CASE WHEN ABS(slope_ratio) > 2 OR ABS(slope_ratio) < 0.5 THEN 1 ELSE 0 END) > COUNT(*) * 0.25 THEN 'DEPARTED'
        WHEN SUM(CASE WHEN ABS(slope_ratio) > 2 OR ABS(slope_ratio) < 0.5 THEN 1 ELSE 0 END) > 0
          OR SUM(slope_reversed) BETWEEN 1 AND 3 THEN 'SHIFTED'
        ELSE 'STABLE'
    END AS overall_status,

    -- Summary metrics
    ROUND(AVG(vol_ratio), 2) AS avg_vol_ratio,
    ROUND(AVG(ABS(slope_ratio)), 2) AS avg_abs_slope_ratio

FROM signal_departure
GROUP BY cohort;


-- ============================================================================
-- REPORT 2: SIGNAL-LEVEL DEPARTURE MATRIX
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
        cohort, signal_id, half,
        AVG(value) AS half_mean,
        STDDEV(value) AS half_std,
        REGR_SLOPE(value, I) AS half_slope
    FROM signal_halves
    GROUP BY cohort, signal_id, half
)

SELECT
    e.cohort,
    e.signal_id,
    ROUND(e.half_mean, 4) AS early_mean,
    ROUND(l.half_mean, 4) AS late_mean,
    ROUND(e.half_slope, 6) AS early_slope,
    ROUND(l.half_slope, 6) AS late_slope,
    ROUND(l.half_slope / NULLIF(e.half_slope, 0), 2) AS slope_ratio,
    ROUND(100 * (l.half_std - e.half_std) / NULLIF(e.half_std, 0), 1) AS vol_change_pct,

    -- Departure status
    CASE
        WHEN SIGN(e.half_slope) != SIGN(l.half_slope) THEN 'DEPARTED'
        WHEN ABS(l.half_slope / NULLIF(e.half_slope, 0)) > 3.0 THEN 'DEPARTED'
        WHEN ABS(l.half_slope / NULLIF(e.half_slope, 0)) > 2.0 THEN 'SHIFTED'
        ELSE 'STABLE'
    END AS traj_light,

    CASE
        WHEN l.half_std / NULLIF(e.half_std, 0) > 1.5 THEN 'DEPARTED'
        WHEN l.half_std / NULLIF(e.half_std, 0) > 1.2 THEN 'SHIFTED'
        ELSE 'STABLE'
    END AS vol_light,

    -- Status flag
    CASE
        WHEN SIGN(e.half_slope) != SIGN(l.half_slope) THEN 'DEPARTED'
        WHEN ABS(l.half_slope / NULLIF(e.half_slope, 0)) > 2.0 THEN 'SHIFTED'
        ELSE 'WITHIN_BASELINE'
    END AS action

FROM half_stats e
JOIN half_stats l ON e.cohort = l.cohort
    AND e.signal_id = l.signal_id
    AND e.half = 'early' AND l.half = 'late'
ORDER BY
    CASE WHEN SIGN(e.half_slope) != SIGN(l.half_slope) THEN 0 ELSE 1 END,
    ABS(l.half_slope / NULLIF(e.half_slope, 0)) DESC NULLS LAST;


-- ============================================================================
-- REPORT 3: TREND DETECTION (monotonic movement)
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

window_means AS (
    SELECT cohort, signal_id, window_id, AVG(value) AS window_mean
    FROM windowed
    GROUP BY cohort, signal_id, window_id
),

trend_check AS (
    SELECT
        cohort,
        signal_id,
        SUM(CASE WHEN window_mean > LAG(window_mean) OVER w THEN 1 ELSE 0 END) AS up_moves,
        SUM(CASE WHEN window_mean < LAG(window_mean) OVER w THEN 1 ELSE 0 END) AS down_moves,
        COUNT(*) - 1 AS total_moves
    FROM window_means
    WINDOW w AS (PARTITION BY cohort, signal_id ORDER BY window_id)
    GROUP BY cohort, signal_id
)

SELECT
    cohort,
    signal_id,
    up_moves,
    down_moves,
    total_moves,
    CASE
        WHEN up_moves >= 0.8 * total_moves THEN 'TRENDING_UP'
        WHEN down_moves >= 0.8 * total_moves THEN 'TRENDING_DOWN'
        WHEN up_moves >= 0.6 * total_moves THEN 'DRIFTING_UP'
        WHEN down_moves >= 0.6 * total_moves THEN 'DRIFTING_DOWN'
        ELSE 'STABLE'
    END AS trend_status,
    CASE
        WHEN up_moves >= 0.8 * total_moves OR down_moves >= 0.8 * total_moves THEN 'DEPARTED'
        WHEN up_moves >= 0.6 * total_moves OR down_moves >= 0.6 * total_moves THEN 'SHIFTED'
        ELSE 'WITHIN_BASELINE'
    END AS action
FROM trend_check
WHERE up_moves >= 0.6 * total_moves OR down_moves >= 0.6 * total_moves
ORDER BY GREATEST(up_moves, down_moves) DESC;


-- ============================================================================
-- REPORT 4: DEVIATION SUMMARY
-- Counts trajectory departure events per signal using slope departure
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
        cohort, signal_id, half,
        AVG(value) AS half_mean,
        STDDEV(value) AS half_std,
        REGR_SLOPE(value, I) AS half_slope
    FROM signal_halves
    GROUP BY cohort, signal_id, half
),
deviation_check AS (
    SELECT
        e.signal_id,
        e.cohort,
        l.half_std / NULLIF(e.half_std, 0) AS vol_ratio,
        l.half_slope / NULLIF(e.half_slope, 0) AS slope_ratio,
        CASE WHEN SIGN(e.half_slope) != SIGN(l.half_slope) THEN 1 ELSE 0 END AS slope_reversed
    FROM half_stats e
    JOIN half_stats l ON e.cohort = l.cohort
        AND e.signal_id = l.signal_id
        AND e.half = 'early' AND l.half = 'late'
)

SELECT
    signal_id,
    COUNT(*) AS total_cohorts,
    SUM(CASE WHEN vol_ratio > 1.3 THEN 1 ELSE 0 END) AS cohorts_vol_elevated,
    SUM(CASE WHEN ABS(slope_ratio) > 2 OR ABS(slope_ratio) < 0.5 THEN 1 ELSE 0 END) AS cohorts_trajectory_changed,
    SUM(slope_reversed) AS cohorts_slope_reversed,
    ROUND(100.0 * SUM(CASE WHEN vol_ratio > 1.3 OR ABS(slope_ratio) > 2 OR slope_reversed = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_deviated,
    ROUND(MAX(vol_ratio), 2) AS max_vol_ratio,
    CASE
        WHEN 100.0 * SUM(CASE WHEN vol_ratio > 1.3 OR ABS(slope_ratio) > 2 OR slope_reversed = 1 THEN 1 ELSE 0 END) / COUNT(*) > 10 THEN 'HIGH_DEVIATION_RATE'
        WHEN 100.0 * SUM(CASE WHEN vol_ratio > 1.3 OR ABS(slope_ratio) > 2 OR slope_reversed = 1 THEN 1 ELSE 0 END) / COUNT(*) > 5 THEN 'ELEVATED_DEVIATION_RATE'
        ELSE 'WITHIN_BASELINE'
    END AS status
FROM deviation_check
GROUP BY signal_id
HAVING SUM(CASE WHEN vol_ratio > 1.3 OR ABS(slope_ratio) > 2 OR slope_reversed = 1 THEN 1 ELSE 0 END) > 0
ORDER BY pct_deviated DESC;


-- ============================================================================
-- REPORT 5: EXECUTIVE DASHBOARD (single row summary)
-- ============================================================================

WITH
signal_halves AS (
    SELECT
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
        SELECT MIN(I) AS min_I, MAX(I) AS max_I
        FROM observations
    ) life ON 1=1
),

half_stats AS (
    SELECT
        signal_id, half,
        AVG(value) AS half_mean,
        STDDEV(value) AS half_std,
        REGR_SLOPE(value, I) AS half_slope
    FROM signal_halves
    GROUP BY signal_id, half
),

departure_check AS (
    SELECT
        e.signal_id,
        l.half_std / NULLIF(e.half_std, 0) AS vol_ratio,
        l.half_slope / NULLIF(e.half_slope, 0) AS slope_ratio,
        CASE WHEN SIGN(e.half_slope) != SIGN(l.half_slope) THEN 1 ELSE 0 END AS slope_reversed
    FROM half_stats e
    JOIN half_stats l ON e.signal_id = l.signal_id
        AND e.half = 'early' AND l.half = 'late'
)

SELECT
    (SELECT COUNT(DISTINCT signal_id) FROM observations) AS total_signals,
    SUM(CASE WHEN ABS(slope_ratio) <= 1.5 AND vol_ratio <= 1.2 AND slope_reversed = 0 THEN 1 ELSE 0 END) AS signals_stable,
    SUM(CASE WHEN (ABS(slope_ratio) > 1.5 AND ABS(slope_ratio) <= 3.0) OR (vol_ratio > 1.2 AND vol_ratio <= 1.5) THEN 1 ELSE 0 END) AS signals_watch,
    SUM(CASE WHEN slope_reversed = 1 OR ABS(slope_ratio) > 3.0 OR vol_ratio > 1.5 THEN 1 ELSE 0 END) AS signals_alert,
    SUM(slope_reversed) AS n_slope_reversed,
    ROUND(AVG(vol_ratio), 2) AS avg_vol_ratio,
    ROUND(MAX(vol_ratio), 2) AS max_vol_ratio,
    CASE
        WHEN SUM(slope_reversed) > 3 OR SUM(CASE WHEN ABS(slope_ratio) > 3.0 THEN 1 ELSE 0 END) > 0 THEN 'DEPARTED'
        WHEN SUM(slope_reversed) > 0 OR SUM(CASE WHEN ABS(slope_ratio) > 2.0 THEN 1 ELSE 0 END) > 3 THEN 'SHIFTED'
        ELSE 'STABLE'
    END AS system_status,
    CURRENT_TIMESTAMP AS report_time
FROM departure_check;
