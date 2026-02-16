-- ============================================================================
-- Engines DRIFT DETECTION REPORTS
-- ============================================================================
--
-- Philosophy: Drift is a trajectory concept, not a distribution concept.
-- A signal drifting 0.01/cycle for 300 cycles has drifted 3.0 total but might
-- have z < 1 at any given point because the variance is large.
--
-- Uses slope_ratio, slope_reversal, and accumulated drift_magnitude
-- instead of z-scores and drift_sigma.
--
-- Usage: Run against observations table with columns [cohort, signal_id, I, value]
-- ============================================================================


-- ============================================================================
-- REPORT 1: SIGNALS DEVIATING FROM BASELINE TRAJECTORY
-- Compares baseline slope (first 20%) to late slope (last 20%)
-- Flags signals with significant trajectory change
-- ============================================================================

WITH
time_bounds AS (
    SELECT
        cohort,
        signal_id,
        MIN(I) AS i_min,
        MAX(I) AS i_max,
        MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end,
        MIN(I) + 0.8 * (MAX(I) - MIN(I)) AS late_start,
        COUNT(*) AS n_observations
    FROM observations
    GROUP BY cohort, signal_id
),

baseline AS (
    SELECT
        o.cohort,
        o.signal_id,
        AVG(o.value) AS baseline_mean,
        STDDEV_POP(o.value) AS baseline_std,
        REGR_SLOPE(o.value, o.I) AS baseline_slope,
        COUNT(*) AS baseline_n
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort AND o.signal_id = t.signal_id
    WHERE o.I <= t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

late_period AS (
    SELECT
        o.cohort,
        o.signal_id,
        AVG(o.value) AS late_mean,
        STDDEV_POP(o.value) AS late_std,
        REGR_SLOPE(o.value, o.I) AS late_slope,
        COUNT(*) AS late_n
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort AND o.signal_id = t.signal_id
    WHERE o.I >= t.late_start
    GROUP BY o.cohort, o.signal_id
)

SELECT
    b.cohort,
    b.signal_id,
    ROUND(b.baseline_slope, 6) AS baseline_slope,
    ROUND(l.late_slope, 6) AS late_slope,
    ROUND(l.late_slope / NULLIF(b.baseline_slope, 0), 2) AS slope_ratio,
    CASE WHEN SIGN(b.baseline_slope) != SIGN(l.late_slope) THEN 'YES' ELSE 'NO' END AS slope_reversed,
    ROUND(ABS(l.late_slope - b.baseline_slope) * t.n_observations, 4) AS drift_magnitude,
    ROUND(l.late_std / NULLIF(b.baseline_std, 0), 2) AS vol_ratio,
    ROUND(100.0 * (l.late_std - b.baseline_std) / NULLIF(b.baseline_std, 0), 1) AS std_change_pct,
    CASE
        WHEN SIGN(b.baseline_slope) != SIGN(l.late_slope) THEN 'REVERSED'
        WHEN ABS(l.late_slope / NULLIF(b.baseline_slope, 0)) > 3.0 THEN 'DRIFT'
        WHEN ABS(l.late_slope / NULLIF(b.baseline_slope, 0)) > 1.5 THEN 'WATCH'
        ELSE 'WITHIN_BASELINE'
    END AS drift_status
FROM baseline b
JOIN late_period l ON b.cohort = l.cohort AND b.signal_id = l.signal_id
JOIN time_bounds t ON b.cohort = t.cohort AND b.signal_id = t.signal_id
ORDER BY ABS(l.late_slope - b.baseline_slope) * t.n_observations DESC;


-- ============================================================================
-- REPORT 2: WHEN DID DRIFT START?
-- Rolling window analysis to pinpoint trajectory departure onset
-- ============================================================================

WITH
time_bounds AS (
    SELECT
        cohort,
        signal_id,
        MIN(I) AS i_min,
        MAX(I) AS i_max,
        MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations
    GROUP BY cohort, signal_id
),

baseline AS (
    SELECT
        o.cohort,
        o.signal_id,
        REGR_SLOPE(o.value, o.I) AS baseline_slope
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort AND o.signal_id = t.signal_id
    WHERE o.I <= t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

windowed AS (
    SELECT
        o.cohort,
        o.signal_id,
        NTILE(10) OVER (PARTITION BY o.cohort, o.signal_id ORDER BY o.I) AS window_id,
        o.value,
        o.I
    FROM observations o
),

window_stats AS (
    SELECT
        cohort,
        signal_id,
        window_id,
        AVG(value) AS window_mean,
        REGR_SLOPE(value, I) AS window_slope,
        MIN(I) AS window_start,
        MAX(I) AS window_end
    FROM windowed
    GROUP BY cohort, signal_id, window_id
)

SELECT
    w.cohort,
    w.signal_id,
    w.window_id,
    ROUND(w.window_start, 1) AS t_start,
    ROUND(w.window_end, 1) AS t_end,
    ROUND(b.baseline_slope, 6) AS baseline_slope,
    ROUND(w.window_slope, 6) AS window_slope,
    ROUND(w.window_slope / NULLIF(b.baseline_slope, 0), 2) AS slope_ratio,
    CASE
        WHEN SIGN(w.window_slope) != SIGN(b.baseline_slope) THEN 'REVERSED'
        WHEN ABS(w.window_slope / NULLIF(b.baseline_slope, 0)) > 2.0 THEN 'ACCELERATING'
        WHEN ABS(w.window_slope / NULLIF(b.baseline_slope, 0)) < 0.5 THEN 'DECELERATING'
        ELSE 'WITHIN_BASELINE'
    END AS status
FROM window_stats w
JOIN baseline b ON w.cohort = b.cohort AND w.signal_id = b.signal_id
WHERE w.window_id > 2  -- Skip baseline windows
  AND (SIGN(w.window_slope) != SIGN(b.baseline_slope)
       OR ABS(w.window_slope / NULLIF(b.baseline_slope, 0)) > 1.5
       OR ABS(w.window_slope / NULLIF(b.baseline_slope, 0)) < 0.5)
ORDER BY w.cohort, w.signal_id, w.window_id;


-- ============================================================================
-- REPORT 3: NON-REVERTING DRIFT
-- Identifies signals that drifted and stayed drifted (persistent trajectory change)
-- ============================================================================

WITH
time_bounds AS (
    SELECT
        cohort,
        signal_id,
        MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations
    GROUP BY cohort, signal_id
),

baseline AS (
    SELECT
        o.cohort,
        o.signal_id,
        REGR_SLOPE(o.value, o.I) AS baseline_slope
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort AND o.signal_id = t.signal_id
    WHERE o.I <= t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

windowed AS (
    SELECT
        o.cohort,
        o.signal_id,
        NTILE(5) OVER (PARTITION BY o.cohort, o.signal_id ORDER BY o.I) AS window_id,
        o.value,
        o.I
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort AND o.signal_id = t.signal_id
    WHERE o.I > t.baseline_end
),

window_stats AS (
    SELECT
        cohort,
        signal_id,
        window_id,
        REGR_SLOPE(value, I) AS window_slope,
        AVG(value) AS window_mean
    FROM windowed
    GROUP BY cohort, signal_id, window_id
),

drift_pattern AS (
    SELECT
        w.cohort,
        w.signal_id,
        b.baseline_slope,
        -- Count windows where slope is consistently in one direction
        SUM(CASE WHEN w.window_slope > 0 THEN 1 ELSE 0 END) AS windows_slope_positive,
        SUM(CASE WHEN w.window_slope < 0 THEN 1 ELSE 0 END) AS windows_slope_negative,
        COUNT(*) AS total_windows,
        AVG(w.window_slope) AS avg_post_slope,
        AVG(ABS(w.window_slope - b.baseline_slope)) AS avg_slope_deviation
    FROM window_stats w
    JOIN baseline b ON w.cohort = b.cohort AND w.signal_id = b.signal_id
    GROUP BY w.cohort, w.signal_id, b.baseline_slope
)

SELECT
    cohort,
    signal_id,
    ROUND(baseline_slope, 6) AS baseline_slope,
    ROUND(avg_post_slope, 6) AS avg_post_slope,
    ROUND(avg_post_slope / NULLIF(baseline_slope, 0), 2) AS slope_ratio,
    windows_slope_positive || '/' || total_windows AS windows_positive,
    windows_slope_negative || '/' || total_windows AS windows_negative,
    CASE
        WHEN windows_slope_positive = total_windows THEN 'PERSISTENT_INCREASE'
        WHEN windows_slope_negative = total_windows THEN 'PERSISTENT_DECREASE'
        WHEN windows_slope_positive >= 4 THEN 'MOSTLY_INCREASING'
        WHEN windows_slope_negative >= 4 THEN 'MOSTLY_DECREASING'
        ELSE 'OSCILLATING'
    END AS drift_pattern
FROM drift_pattern
WHERE windows_slope_positive >= 4 OR windows_slope_negative >= 4
ORDER BY ABS(avg_post_slope - baseline_slope) DESC;


-- ============================================================================
-- REPORT 4: VOLATILITY DRIFT
-- Identifies signals where variance is increasing or decreasing
-- ============================================================================

WITH
time_bounds AS (
    SELECT
        cohort, signal_id,
        MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end,
        MIN(I) + 0.8 * (MAX(I) - MIN(I)) AS late_start
    FROM observations
    GROUP BY cohort, signal_id
),

baseline AS (
    SELECT o.cohort, o.signal_id,
        VAR_POP(o.value) AS baseline_var,
        STDDEV_POP(o.value) AS baseline_std
    FROM observations o
    JOIN time_bounds t USING (cohort, signal_id)
    WHERE o.I <= t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

late_period AS (
    SELECT o.cohort, o.signal_id,
        VAR_POP(o.value) AS late_var,
        STDDEV_POP(o.value) AS late_std
    FROM observations o
    JOIN time_bounds t USING (cohort, signal_id)
    WHERE o.I >= t.late_start
    GROUP BY o.cohort, o.signal_id
)

SELECT
    b.cohort,
    b.signal_id,
    ROUND(b.baseline_std, 6) AS baseline_std,
    ROUND(l.late_std, 6) AS late_std,
    ROUND(100.0 * (l.late_std - b.baseline_std) / NULLIF(b.baseline_std, 0), 1) AS pct_change,
    ROUND(l.late_var / NULLIF(b.baseline_var, 0), 2) AS variance_ratio,
    CASE
        WHEN (l.late_var / NULLIF(b.baseline_var, 0)) > 1.5 THEN 'VOLATILITY_UP'
        WHEN (l.late_var / NULLIF(b.baseline_var, 0)) < 0.67 THEN 'VOLATILITY_DOWN'
        ELSE 'STABLE'
    END AS volatility_status
FROM baseline b
JOIN late_period l USING (cohort, signal_id)
ORDER BY ABS(100.0 * (l.late_std - b.baseline_std) / NULLIF(b.baseline_std, 0)) DESC;


-- ============================================================================
-- REPORT 5: SYSTEM-WIDE DRIFT SUMMARY
-- Aggregates drift direction using slope comparison
-- ============================================================================

WITH
time_bounds AS (
    SELECT signal_id,
        MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations
    GROUP BY signal_id
),

baseline AS (
    SELECT o.signal_id,
        REGR_SLOPE(o.value, o.I) AS baseline_slope
    FROM observations o
    JOIN time_bounds t USING (signal_id)
    WHERE o.I <= t.baseline_end
    GROUP BY o.signal_id
),

post_baseline AS (
    SELECT o.signal_id,
        REGR_SLOPE(o.value, o.I) AS post_slope
    FROM observations o
    JOIN time_bounds t USING (signal_id)
    WHERE o.I > t.baseline_end
    GROUP BY o.signal_id
),

drift_calc AS (
    SELECT
        b.signal_id,
        b.baseline_slope,
        p.post_slope,
        p.post_slope / NULLIF(b.baseline_slope, 0) AS slope_ratio,
        CASE
            WHEN SIGN(b.baseline_slope) != SIGN(p.post_slope) THEN 'REVERSED'
            WHEN p.post_slope / NULLIF(b.baseline_slope, 0) > 1.5 THEN 'ACCELERATING'
            WHEN p.post_slope / NULLIF(b.baseline_slope, 0) < 0.5 THEN 'DECELERATING'
            ELSE 'STABLE'
        END AS direction
    FROM baseline b
    JOIN post_baseline p USING (signal_id)
)

SELECT
    direction,
    COUNT(*) AS n_signals,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_total,
    ROUND(AVG(slope_ratio), 2) AS avg_slope_ratio,
    ROUND(MIN(slope_ratio), 2) AS min_slope_ratio,
    ROUND(MAX(slope_ratio), 2) AS max_slope_ratio
FROM drift_calc
GROUP BY direction
ORDER BY n_signals DESC;


-- ============================================================================
-- REPORT 6: BASELINE EXCEEDANCE EVENTS
-- Identifies when signals exceeded baseline envelope (mean +/- 2*std)
-- Uses envelope breach, not z-score
-- ============================================================================

WITH
time_bounds AS (
    SELECT signal_id,
        MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations
    GROUP BY signal_id
),

baseline AS (
    SELECT o.signal_id,
        AVG(o.value) AS mu,
        STDDEV_POP(o.value) AS sigma,
        AVG(o.value) + 2 * STDDEV_POP(o.value) AS upper_bound,
        AVG(o.value) - 2 * STDDEV_POP(o.value) AS lower_bound
    FROM observations o
    JOIN time_bounds t USING (signal_id)
    WHERE o.I <= t.baseline_end
    GROUP BY o.signal_id
),

exceedances AS (
    SELECT
        o.signal_id,
        o.I,
        o.value,
        CASE WHEN o.value > b.upper_bound THEN 'HIGH' ELSE 'LOW' END AS breach_direction
    FROM observations o
    JOIN baseline b USING (signal_id)
    JOIN time_bounds t USING (signal_id)
    WHERE o.I > t.baseline_end
      AND (o.value > b.upper_bound OR o.value < b.lower_bound)
)

SELECT
    signal_id,
    COUNT(*) AS n_exceedances,
    ROUND(MIN(I), 1) AS first_exceedance_t,
    ROUND(MAX(I), 1) AS last_exceedance_t,
    SUM(CASE WHEN breach_direction = 'HIGH' THEN 1 ELSE 0 END) AS n_high_exceedances,
    SUM(CASE WHEN breach_direction = 'LOW' THEN 1 ELSE 0 END) AS n_low_exceedances
FROM exceedances
GROUP BY signal_id
ORDER BY n_exceedances DESC;


-- ============================================================================
-- REPORT 7: COORDINATED DRIFT DETECTION
-- Identifies signals that started trajectory departure at the same time
-- ============================================================================

WITH
time_bounds AS (
    SELECT signal_id,
        MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations
    GROUP BY signal_id
),

baseline AS (
    SELECT o.signal_id,
        REGR_SLOPE(o.value, o.I) AS baseline_slope
    FROM observations o
    JOIN time_bounds t USING (signal_id)
    WHERE o.I <= t.baseline_end
    GROUP BY o.signal_id
),

windowed AS (
    SELECT o.signal_id,
        NTILE(10) OVER (PARTITION BY o.signal_id ORDER BY o.I) AS window_id,
        o.value,
        o.I
    FROM observations o
    JOIN time_bounds t USING (signal_id)
    WHERE o.I > t.baseline_end
),

window_slopes AS (
    SELECT
        w.signal_id,
        w.window_id,
        REGR_SLOPE(w.value, w.I) AS window_slope,
        b.baseline_slope
    FROM windowed w
    JOIN baseline b USING (signal_id)
    GROUP BY w.signal_id, w.window_id, b.baseline_slope
),

first_departure AS (
    SELECT signal_id,
        MIN(window_id) AS first_departure_window
    FROM window_slopes
    WHERE ABS(window_slope / NULLIF(baseline_slope, 0)) > 2.0
       OR SIGN(window_slope) != SIGN(baseline_slope)
    GROUP BY signal_id
)

SELECT
    first_departure_window,
    COUNT(*) AS n_signals,
    STRING_AGG(signal_id, ', ' ORDER BY signal_id) AS affected_signals
FROM first_departure
GROUP BY first_departure_window
ORDER BY first_departure_window;


-- ============================================================================
-- REPORT 8: DRIFT TIMELINE
-- Shows progression of trajectory departure over time for all signals
-- ============================================================================

WITH
time_bounds AS (
    SELECT signal_id,
        MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations
    GROUP BY signal_id
),

baseline AS (
    SELECT o.signal_id,
        REGR_SLOPE(o.value, o.I) AS baseline_slope
    FROM observations o
    JOIN time_bounds t USING (signal_id)
    WHERE o.I <= t.baseline_end
    GROUP BY o.signal_id
),

windowed AS (
    SELECT
        o.signal_id,
        NTILE(10) OVER (PARTITION BY o.signal_id ORDER BY o.I) AS window_id,
        o.value,
        o.I
    FROM observations o
),

window_stats AS (
    SELECT
        signal_id,
        window_id,
        MIN(I) AS t,
        REGR_SLOPE(value, I) AS window_slope
    FROM windowed
    GROUP BY signal_id, window_id
)

SELECT
    w.signal_id,
    w.window_id,
    ROUND(w.t, 1) AS t,
    ROUND(b.baseline_slope, 6) AS baseline_slope,
    ROUND(w.window_slope, 6) AS window_slope,
    ROUND(w.window_slope / NULLIF(b.baseline_slope, 0), 2) AS slope_ratio,
    REPEAT('█', CAST(GREATEST(0, LEAST(20, 10 + 5 * COALESCE(w.window_slope / NULLIF(b.baseline_slope, 0), 0))) AS INTEGER)) AS visual
FROM window_stats w
JOIN baseline b USING (signal_id)
WHERE b.baseline_slope IS NOT NULL AND b.baseline_slope != 0
ORDER BY w.signal_id, w.window_id;
