-- ============================================================================
-- PRISM REGIME DETECTION REPORTS
-- ============================================================================
--
-- Identifies distinct operating states/regimes in process data.
-- Detects when the system shifts between different modes of operation.
--
-- Usage: Run against observations table
-- ============================================================================


-- ============================================================================
-- REPORT 1: OPERATING REGIME IDENTIFICATION
-- Clusters time windows by signal statistics to find distinct regimes
-- ============================================================================

WITH
windowed AS (
    SELECT
        cohort,
        signal_id,
        NTILE(20) OVER (PARTITION BY cohort, signal_id ORDER BY I) AS window_id,
        I,
        value
    FROM observations
),

window_stats AS (
    SELECT
        cohort,
        window_id,
        AVG(value) AS overall_mean,
        STDDEV_POP(value) AS overall_std
    FROM windowed
    GROUP BY cohort, window_id
),

-- Detect regime changes via statistical shifts
regime_boundaries AS (
    SELECT
        cohort,
        window_id,
        overall_mean,
        overall_std,
        LAG(overall_mean) OVER w AS prev_mean,
        LAG(overall_std) OVER w AS prev_std,
        ABS(overall_mean - LAG(overall_mean) OVER w) / NULLIF(overall_std, 0) AS mean_change_z,
        overall_std / NULLIF(LAG(overall_std) OVER w, 0) AS std_ratio
    FROM window_stats
    WINDOW w AS (PARTITION BY cohort ORDER BY window_id)
)

SELECT
    cohort,
    window_id,
    ROUND(overall_mean, 4) AS regime_mean,
    ROUND(overall_std, 4) AS regime_std,
    ROUND(mean_change_z, 2) AS change_magnitude,
    CASE
        WHEN mean_change_z > 1.5 OR std_ratio > 1.5 OR std_ratio < 0.67 THEN 'REGIME_CHANGE'
        ELSE 'SAME_REGIME'
    END AS boundary_type,
    SUM(CASE WHEN mean_change_z > 1.5 OR std_ratio > 1.5 OR std_ratio < 0.67 THEN 1 ELSE 0 END)
        OVER (PARTITION BY cohort ORDER BY window_id) AS regime_id
FROM regime_boundaries
ORDER BY cohort, window_id;


-- ============================================================================
-- REPORT 2: REGIME SUMMARY
-- Characterizes each detected regime
-- ============================================================================

WITH
windowed AS (
    SELECT
        cohort,
        signal_id,
        NTILE(20) OVER (PARTITION BY cohort, signal_id ORDER BY I) AS window_id,
        value
    FROM observations
),

window_stats AS (
    SELECT cohort, window_id,
        AVG(value) AS overall_mean, STDDEV_POP(value) AS overall_std
    FROM windowed
    GROUP BY cohort, window_id
),

regime_boundaries AS (
    SELECT
        cohort, window_id, overall_mean, overall_std,
        ABS(overall_mean - LAG(overall_mean) OVER w) / NULLIF(overall_std, 0) AS mean_change_z,
        overall_std / NULLIF(LAG(overall_std) OVER w, 0) AS std_ratio
    FROM window_stats
    WINDOW w AS (PARTITION BY cohort ORDER BY window_id)
),

regime_windows AS (
    SELECT
        cohort, window_id, overall_mean, overall_std,
        SUM(CASE WHEN mean_change_z > 1.5 OR std_ratio > 1.5 OR std_ratio < 0.67 THEN 1 ELSE 0 END)
            OVER (PARTITION BY cohort ORDER BY window_id) AS regime_id
    FROM regime_boundaries
)

SELECT
    cohort,
    regime_id,
    MIN(window_id) AS start_window,
    MAX(window_id) AS end_window,
    MAX(window_id) - MIN(window_id) + 1 AS duration_windows,
    ROUND(AVG(overall_mean), 4) AS avg_level,
    ROUND(AVG(overall_std), 4) AS avg_volatility,
    CASE
        WHEN AVG(overall_std) > 1.5 * (SELECT AVG(overall_std) FROM regime_windows WHERE cohort = r.cohort) THEN 'HIGH_VARIABILITY'
        WHEN AVG(overall_std) < 0.5 * (SELECT AVG(overall_std) FROM regime_windows WHERE cohort = r.cohort) THEN 'LOW_VARIABILITY'
        ELSE 'NORMAL_VARIABILITY'
    END AS variability_class
FROM regime_windows r
GROUP BY cohort, regime_id
ORDER BY cohort, regime_id;


-- ============================================================================
-- REPORT 3: STEADY STATE DETECTION
-- Identifies periods of stable operation
-- ============================================================================

WITH
windowed AS (
    SELECT
        cohort,
        signal_id,
        NTILE(50) OVER (PARTITION BY cohort, signal_id ORDER BY I) AS window_id,
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

steady_check AS (
    SELECT
        cohort,
        window_id,
        -- Count signals that are steady (low derivative, low volatility)
        SUM(CASE
            WHEN ABS(window_mean - LAG(window_mean) OVER w) / NULLIF(window_std, 0) < 0.2
            THEN 1 ELSE 0
        END) AS n_steady_signals,
        COUNT(DISTINCT signal_id) AS n_total_signals
    FROM window_stats
    WINDOW w AS (PARTITION BY cohort, signal_id ORDER BY window_id)
    GROUP BY cohort, window_id
)

SELECT
    cohort,
    window_id,
    n_steady_signals,
    n_total_signals,
    ROUND(100.0 * n_steady_signals / n_total_signals, 1) AS pct_steady,
    CASE
        WHEN 100.0 * n_steady_signals / n_total_signals > 90 THEN 'STEADY_STATE'
        WHEN 100.0 * n_steady_signals / n_total_signals > 70 THEN 'QUASI_STEADY'
        WHEN 100.0 * n_steady_signals / n_total_signals > 50 THEN 'TRANSITION'
        ELSE 'DYNAMIC'
    END AS operating_mode
FROM steady_check
ORDER BY cohort, window_id;


-- ============================================================================
-- REPORT 4: TRANSIENT DETECTION
-- Identifies rapid change events (startups, shutdowns, disturbances)
-- ============================================================================

WITH
derivatives AS (
    SELECT
        cohort,
        signal_id,
        I,
        value,
        value - LAG(value) OVER w AS dy,
        I - LAG(I) OVER w AS dI
    FROM observations
    WINDOW w AS (PARTITION BY cohort, signal_id ORDER BY I)
),

rates AS (
    SELECT
        cohort,
        signal_id,
        I,
        dy / NULLIF(dI, 0) AS rate_of_change
    FROM derivatives
    WHERE dy IS NOT NULL
),

rate_stats AS (
    SELECT
        cohort,
        signal_id,
        AVG(ABS(rate_of_change)) AS avg_rate,
        STDDEV_POP(rate_of_change) AS rate_std
    FROM rates
    GROUP BY cohort, signal_id
),

transients AS (
    SELECT
        r.cohort,
        r.signal_id,
        r.I,
        r.rate_of_change,
        (r.rate_of_change - s.avg_rate) / NULLIF(s.rate_std, 0) AS rate_z
    FROM rates r
    JOIN rate_stats s USING (cohort, signal_id)
)

SELECT
    cohort,
    signal_id,
    I AS transient_time,
    ROUND(rate_of_change, 4) AS rate,
    ROUND(rate_z, 2) AS rate_z_score,
    CASE
        WHEN rate_z > 3 THEN 'RAPID_INCREASE'
        WHEN rate_z < -3 THEN 'RAPID_DECREASE'
        WHEN ABS(rate_z) > 2 THEN 'SIGNIFICANT_CHANGE'
        ELSE 'NORMAL'
    END AS transient_type
FROM transients
WHERE ABS(rate_z) > 2
ORDER BY cohort, I, ABS(rate_z) DESC;


-- ============================================================================
-- REPORT 5: MODE CLUSTERING
-- Groups similar operating states across all signals
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

window_profiles AS (
    SELECT
        cohort,
        window_id,
        -- Create a profile of normalized means across signals
        AVG(value) AS overall_mean,
        STDDEV_POP(value) AS overall_std,
        MAX(value) - MIN(value) AS range_val,
        COUNT(DISTINCT signal_id) AS n_signals
    FROM windowed
    GROUP BY cohort, window_id
),

normalized AS (
    SELECT
        cohort,
        window_id,
        overall_mean,
        overall_std,
        (overall_mean - AVG(overall_mean) OVER (PARTITION BY cohort)) /
            NULLIF(STDDEV(overall_mean) OVER (PARTITION BY cohort), 0) AS norm_mean,
        (overall_std - AVG(overall_std) OVER (PARTITION BY cohort)) /
            NULLIF(STDDEV(overall_std) OVER (PARTITION BY cohort), 0) AS norm_std
    FROM window_profiles
)

SELECT
    cohort,
    window_id,
    ROUND(overall_mean, 4) AS level,
    ROUND(overall_std, 4) AS volatility,
    -- Simple mode classification based on level and volatility
    CASE
        WHEN norm_mean > 1 AND norm_std > 1 THEN 'HIGH_ACTIVE'
        WHEN norm_mean > 1 AND norm_std < -1 THEN 'HIGH_STABLE'
        WHEN norm_mean < -1 AND norm_std > 1 THEN 'LOW_ACTIVE'
        WHEN norm_mean < -1 AND norm_std < -1 THEN 'LOW_STABLE'
        WHEN norm_std > 1 THEN 'TRANSITION'
        ELSE 'NOMINAL'
    END AS operating_mode,
    ROUND(norm_mean, 2) AS normalized_level,
    ROUND(norm_std, 2) AS normalized_volatility
FROM normalized
ORDER BY cohort, window_id;


-- ============================================================================
-- REPORT 6: REGIME CHANGE TIMELINE
-- Shows when regime changes occurred
-- ============================================================================

WITH
windowed AS (
    SELECT cohort, signal_id,
        NTILE(20) OVER (PARTITION BY cohort, signal_id ORDER BY I) AS window_id,
        MIN(I) OVER (PARTITION BY cohort, signal_id,
            NTILE(20) OVER (PARTITION BY cohort, signal_id ORDER BY I)) AS window_start,
        MAX(I) OVER (PARTITION BY cohort, signal_id,
            NTILE(20) OVER (PARTITION BY cohort, signal_id ORDER BY I)) AS window_end,
        value
    FROM observations
),

window_stats AS (
    SELECT cohort, window_id, MIN(window_start) AS t_start, MAX(window_end) AS t_end,
        AVG(value) AS overall_mean, STDDEV_POP(value) AS overall_std
    FROM windowed
    GROUP BY cohort, window_id
),

changes AS (
    SELECT
        cohort,
        window_id,
        t_start,
        t_end,
        overall_mean,
        overall_std,
        ABS(overall_mean - LAG(overall_mean) OVER w) / NULLIF(overall_std, 0) AS change_z,
        overall_std / NULLIF(LAG(overall_std) OVER w, 0) AS vol_ratio
    FROM window_stats
    WINDOW w AS (PARTITION BY cohort ORDER BY window_id)
)

SELECT
    cohort,
    window_id,
    t_start AS change_time,
    ROUND(change_z, 2) AS change_magnitude,
    ROUND(vol_ratio, 2) AS volatility_shift,
    CASE
        WHEN change_z > 2 THEN 'MAJOR_SHIFT'
        WHEN change_z > 1 THEN 'MODERATE_SHIFT'
        WHEN vol_ratio > 1.5 THEN 'VOLATILITY_JUMP'
        WHEN vol_ratio < 0.67 THEN 'STABILIZATION'
        ELSE 'MINOR'
    END AS change_type,
    CASE
        WHEN change_z > 2 OR vol_ratio > 1.5 THEN 'INVESTIGATE'
        WHEN change_z > 1 THEN 'MONITOR'
        ELSE 'OK'
    END AS action
FROM changes
WHERE change_z > 0.5 OR vol_ratio > 1.3 OR vol_ratio < 0.77
ORDER BY cohort, t_start;
