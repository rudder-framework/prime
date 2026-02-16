-- ============================================================================
-- Engines PERIODICITY DETECTION REPORTS
-- ============================================================================
--
-- Detects cyclic patterns, oscillations, and repetitive behavior in signals.
-- Helps identify mechanical vibrations, control loop cycling, or process rhythms.
--
-- Usage: Run against observations table
-- ============================================================================


-- ============================================================================
-- REPORT 1: ZERO CROSSING FREQUENCY
-- Estimates dominant frequency via zero-crossing rate
-- ============================================================================

WITH
centered AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        value - AVG(value) OVER (PARTITION BY cohort, signal_id) AS y_centered
    FROM observations
),

crossings AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        y_centered,
        LAG(y_centered) OVER w AS prev_y,
        CASE
            WHEN y_centered * LAG(y_centered) OVER w < 0 THEN 1
            ELSE 0
        END AS is_crossing
    FROM centered
    WINDOW w AS (PARTITION BY cohort, signal_id ORDER BY signal_0)
),

signal_range AS (
    SELECT
        cohort,
        signal_id,
        MIN(signal_0) AS i_min,
        MAX(signal_0) AS i_max,
        COUNT(*) AS n_points
    FROM observations
    GROUP BY cohort, signal_id
)

SELECT
    c.cohort,
    c.signal_id,
    SUM(c.is_crossing) AS n_crossings,
    r.n_points,
    r.i_max - r.i_min AS time_span,
    ROUND(1.0 * SUM(c.is_crossing) / (r.i_max - r.i_min), 4) AS crossing_rate,
    -- Approximate frequency (crossings / 2 = cycles)
    ROUND(0.5 * SUM(c.is_crossing) / (r.i_max - r.i_min), 4) AS approx_frequency,
    CASE
        WHEN 1.0 * SUM(c.is_crossing) / (r.i_max - r.i_min) > 0.5 THEN 'HIGH_FREQUENCY'
        WHEN 1.0 * SUM(c.is_crossing) / (r.i_max - r.i_min) > 0.1 THEN 'MEDIUM_FREQUENCY'
        WHEN 1.0 * SUM(c.is_crossing) / (r.i_max - r.i_min) > 0.01 THEN 'LOW_FREQUENCY'
        ELSE 'QUASI_STATIC'
    END AS frequency_class
FROM crossings c
JOIN signal_range r USING (cohort, signal_id)
GROUP BY c.cohort, c.signal_id, r.n_points, r.i_min, r.i_max
ORDER BY crossing_rate DESC;


-- ============================================================================
-- REPORT 2: OSCILLATION AMPLITUDE
-- Measures the strength of oscillatory behavior
-- ============================================================================

WITH
centered AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        value,
        value - AVG(value) OVER (PARTITION BY cohort, signal_id) AS y_centered,
        STDDEV_POP(value) OVER (PARTITION BY cohort, signal_id) AS y_std
    FROM observations
),

peaks AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        y_centered,
        y_std,
        CASE
            WHEN y_centered > LAG(y_centered) OVER w
             AND y_centered > LEAD(y_centered) OVER w THEN 'PEAK'
            WHEN y_centered < LAG(y_centered) OVER w
             AND y_centered < LEAD(y_centered) OVER w THEN 'TROUGH'
            ELSE NULL
        END AS extrema_type
    FROM centered
    WINDOW w AS (PARTITION BY cohort, signal_id ORDER BY signal_0)
),

extrema_stats AS (
    SELECT
        cohort,
        signal_id,
        COUNT(*) FILTER (WHERE extrema_type = 'PEAK') AS n_peaks,
        COUNT(*) FILTER (WHERE extrema_type = 'TROUGH') AS n_troughs,
        AVG(ABS(y_centered)) FILTER (WHERE extrema_type IS NOT NULL) AS avg_amplitude,
        MAX(ABS(y_centered)) FILTER (WHERE extrema_type IS NOT NULL) AS max_amplitude,
        MAX(y_std) AS signal_std
    FROM peaks
    GROUP BY cohort, signal_id
)

SELECT
    cohort,
    signal_id,
    n_peaks,
    n_troughs,
    ROUND(avg_amplitude, 4) AS avg_oscillation_amp,
    ROUND(max_amplitude, 4) AS max_oscillation_amp,
    ROUND(avg_amplitude / NULLIF(signal_std, 0), 2) AS normalized_amplitude,
    CASE
        WHEN avg_amplitude / NULLIF(signal_std, 0) > 1.5 THEN 'STRONG_OSCILLATION'
        WHEN avg_amplitude / NULLIF(signal_std, 0) > 0.8 THEN 'MODERATE_OSCILLATION'
        WHEN avg_amplitude / NULLIF(signal_std, 0) > 0.3 THEN 'WEAK_OSCILLATION'
        ELSE 'NO_OSCILLATION'
    END AS oscillation_class
FROM extrema_stats
ORDER BY cohort, avg_amplitude / NULLIF(signal_std, 0) DESC;


-- ============================================================================
-- REPORT 3: PERIOD ESTIMATION (peak-to-peak)
-- Estimates dominant period from peak spacing
-- ============================================================================

WITH
centered AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        value - AVG(value) OVER (PARTITION BY cohort, signal_id) AS y_centered
    FROM observations
),

peaks AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        ROW_NUMBER() OVER (PARTITION BY cohort, signal_id ORDER BY signal_0) AS peak_num
    FROM centered
    WHERE y_centered > LAG(y_centered) OVER (PARTITION BY cohort, signal_id ORDER BY signal_0)
      AND y_centered > LEAD(y_centered) OVER (PARTITION BY cohort, signal_id ORDER BY signal_0)
),

peak_intervals AS (
    SELECT
        cohort,
        signal_id,
        signal_0 - LAG(signal_0) OVER (PARTITION BY cohort, signal_id ORDER BY peak_num) AS interval
    FROM peaks
)

SELECT
    cohort,
    signal_id,
    COUNT(*) AS n_intervals,
    ROUND(AVG(interval), 4) AS avg_period,
    ROUND(STDDEV_POP(interval), 4) AS period_std,
    ROUND(MIN(interval), 4) AS min_period,
    ROUND(MAX(interval), 4) AS max_period,
    ROUND(STDDEV_POP(interval) / NULLIF(AVG(interval), 0), 3) AS period_variability,
    CASE
        WHEN STDDEV_POP(interval) / NULLIF(AVG(interval), 0) < 0.1 THEN 'HIGHLY_PERIODIC'
        WHEN STDDEV_POP(interval) / NULLIF(AVG(interval), 0) < 0.3 THEN 'QUASI_PERIODIC'
        WHEN STDDEV_POP(interval) / NULLIF(AVG(interval), 0) < 0.5 THEN 'IRREGULAR'
        ELSE 'APERIODIC'
    END AS periodicity_class
FROM peak_intervals
WHERE interval IS NOT NULL AND interval > 0
GROUP BY cohort, signal_id
HAVING COUNT(*) >= 3
ORDER BY cohort, period_variability;


-- ============================================================================
-- REPORT 4: AUTOCORRELATION PROXY
-- Estimates correlation with lagged self (without actual CORR on lag)
-- ============================================================================

WITH
lagged AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        value,
        LAG(value, 1) OVER w AS y_lag1,
        LAG(value, 5) OVER w AS y_lag5,
        LAG(value, 10) OVER w AS y_lag10
    FROM observations
    WINDOW w AS (PARTITION BY cohort, signal_id ORDER BY signal_0)
),

correlations AS (
    SELECT
        cohort,
        signal_id,
        CORR(value, y_lag1) AS acf_lag1,
        CORR(value, y_lag5) AS acf_lag5,
        CORR(value, y_lag10) AS acf_lag10
    FROM lagged
    WHERE y_lag10 IS NOT NULL
    GROUP BY cohort, signal_id
)

SELECT
    cohort,
    signal_id,
    ROUND(acf_lag1, 3) AS acf_1,
    ROUND(acf_lag5, 3) AS acf_5,
    ROUND(acf_lag10, 3) AS acf_10,
    CASE
        WHEN acf_lag1 > 0.95 THEN 'HIGHLY_PERSISTENT'
        WHEN acf_lag1 > 0.8 THEN 'PERSISTENT'
        WHEN acf_lag1 > 0.5 THEN 'MODERATE_MEMORY'
        WHEN acf_lag1 > 0 THEN 'LOW_MEMORY'
        ELSE 'ANTI_PERSISTENT'
    END AS memory_class,
    CASE
        WHEN acf_lag5 > 0.5 AND acf_lag10 > 0.3 THEN 'LONG_MEMORY'
        WHEN acf_lag5 > 0.3 THEN 'MEDIUM_MEMORY'
        ELSE 'SHORT_MEMORY'
    END AS memory_horizon,
    -- Oscillation hint: if lag-5 or lag-10 shows negative correlation
    CASE
        WHEN acf_lag5 < -0.3 OR acf_lag10 < -0.3 THEN 'OSCILLATING'
        ELSE 'NON_OSCILLATING'
    END AS oscillation_hint
FROM correlations
ORDER BY cohort, acf_lag1 DESC;


-- ============================================================================
-- REPORT 5: CYCLIC PATTERN CHANGES
-- Detects when oscillation behavior changes over time
-- ============================================================================

WITH
windowed AS (
    SELECT
        cohort,
        signal_id,
        NTILE(5) OVER (PARTITION BY cohort, signal_id ORDER BY signal_0) AS window_id,
        signal_0,
        value - AVG(value) OVER (PARTITION BY cohort, signal_id) AS y_centered
    FROM observations
),

window_crossings AS (
    SELECT
        cohort,
        signal_id,
        window_id,
        SUM(CASE
            WHEN y_centered * LAG(y_centered) OVER w < 0 THEN 1
            ELSE 0
        END) AS n_crossings,
        COUNT(*) AS n_points
    FROM windowed
    WINDOW w AS (PARTITION BY cohort, signal_id, window_id ORDER BY signal_0)
    GROUP BY cohort, signal_id, window_id
),

crossing_rate AS (
    SELECT
        cohort,
        signal_id,
        window_id,
        1.0 * n_crossings / n_points AS rate
    FROM window_crossings
)

SELECT
    cohort,
    signal_id,
    MAX(CASE WHEN window_id = 1 THEN ROUND(rate, 4) END) AS rate_w1,
    MAX(CASE WHEN window_id = 2 THEN ROUND(rate, 4) END) AS rate_w2,
    MAX(CASE WHEN window_id = 3 THEN ROUND(rate, 4) END) AS rate_w3,
    MAX(CASE WHEN window_id = 4 THEN ROUND(rate, 4) END) AS rate_w4,
    MAX(CASE WHEN window_id = 5 THEN ROUND(rate, 4) END) AS rate_w5,
    ROUND(MAX(rate) - MIN(rate), 4) AS rate_change,
    CASE
        WHEN MAX(rate) - MIN(rate) > 0.2 THEN 'FREQUENCY_SHIFT'
        WHEN MAX(rate) > 1.5 * MIN(rate) THEN 'FREQUENCY_DRIFT'
        ELSE 'STABLE_FREQUENCY'
    END AS frequency_stability
FROM crossing_rate
GROUP BY cohort, signal_id
HAVING MAX(rate) - MIN(rate) > 0.05
ORDER BY cohort, MAX(rate) - MIN(rate) DESC;


-- ============================================================================
-- REPORT 6: HUNTING/CYCLING DETECTION
-- Identifies control loop hunting (oscillation around setpoint)
-- ============================================================================

WITH
centered AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        value,
        AVG(value) OVER (PARTITION BY cohort, signal_id) AS mean_val,
        STDDEV_POP(value) OVER (PARTITION BY cohort, signal_id) AS std_val,
        value - AVG(value) OVER (PARTITION BY cohort, signal_id) AS deviation
    FROM observations
),

direction_changes AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        deviation,
        SIGN(deviation) AS direction,
        CASE
            WHEN SIGN(deviation) != LAG(SIGN(deviation)) OVER w THEN 1
            ELSE 0
        END AS is_reversal
    FROM centered
    WINDOW w AS (PARTITION BY cohort, signal_id ORDER BY signal_0)
),

hunting_stats AS (
    SELECT
        cohort,
        signal_id,
        SUM(is_reversal) AS n_reversals,
        COUNT(*) AS n_points,
        AVG(ABS(deviation)) AS avg_deviation
    FROM direction_changes
    GROUP BY cohort, signal_id
)

SELECT
    cohort,
    signal_id,
    n_reversals,
    n_points,
    ROUND(1.0 * n_reversals / n_points, 4) AS reversal_rate,
    ROUND(avg_deviation, 4) AS avg_excursion,
    CASE
        WHEN 1.0 * n_reversals / n_points > 0.3 THEN 'SEVERE_HUNTING'
        WHEN 1.0 * n_reversals / n_points > 0.2 THEN 'MODERATE_HUNTING'
        WHEN 1.0 * n_reversals / n_points > 0.1 THEN 'MILD_HUNTING'
        ELSE 'WITHIN_BASELINE'
    END AS hunting_severity,
    CASE
        WHEN 1.0 * n_reversals / n_points > 0.2 THEN 'Check control loop tuning'
        ELSE 'WITHIN_BASELINE'
    END AS recommendation
FROM hunting_stats
ORDER BY cohort, 1.0 * n_reversals / n_points DESC;
