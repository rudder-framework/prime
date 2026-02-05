-- ============================================================================
-- PRISM DRIFT DETECTION REPORTS
-- ============================================================================
--
-- Philosophy: Establish baseline from first 20% of samples, flag deviations
-- These reports identify signals that deviate and don't revert.
--
-- Usage: Run against observations table with columns [cohort, signal_id, I, value]
-- ============================================================================


-- ============================================================================
-- REPORT 1: SIGNALS DEVIATING FROM BASELINE
-- Compares baseline (first 20%) to late period (last 20%)
-- Flags signals with significant mean drift
-- ============================================================================

WITH
time_bounds AS (
    SELECT
        cohort,
        signal_id,
        MIN(I) AS i_min,
        MAX(I) AS i_max,
        MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end,
        MIN(I) + 0.8 * (MAX(I) - MIN(I)) AS late_start
    FROM observations
    GROUP BY cohort, signal_id
),

baseline AS (
    SELECT
        o.cohort,
        o.signal_id,
        AVG(o.value) AS baseline_mean,
        STDDEV_POP(o.value) AS baseline_std,
        MIN(o.value) AS baseline_min,
        MAX(o.value) AS baseline_max,
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
        COUNT(*) AS late_n
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort AND o.signal_id = t.signal_id
    WHERE o.I >= t.late_start
    GROUP BY o.cohort, o.signal_id
)

SELECT
    b.cohort,
    b.signal_id,
    ROUND(b.baseline_mean, 4) AS baseline_mean,
    ROUND(l.late_mean, 4) AS late_mean,
    ROUND(l.late_mean - b.baseline_mean, 4) AS mean_drift,
    ROUND((l.late_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0), 2) AS drift_sigma,
    ROUND(b.baseline_std, 6) AS baseline_std,
    ROUND(l.late_std, 6) AS late_std,
    ROUND(100.0 * (l.late_std - b.baseline_std) / NULLIF(b.baseline_std, 0), 1) AS std_change_pct,
    CASE
        WHEN ABS((l.late_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0)) > 2 THEN 'DRIFT'
        WHEN ABS((l.late_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0)) > 1 THEN 'WATCH'
        ELSE 'OK'
    END AS drift_status
FROM baseline b
JOIN late_period l ON b.cohort = l.cohort AND b.signal_id = l.signal_id
ORDER BY ABS((l.late_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0)) DESC;


-- ============================================================================
-- REPORT 2: WHEN DID DRIFT START?
-- Rolling window analysis to pinpoint drift onset
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
        AVG(o.value) AS baseline_mean,
        STDDEV_POP(o.value) AS baseline_std
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
        STDDEV_POP(value) AS window_std,
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
    ROUND(b.baseline_mean, 4) AS baseline,
    ROUND(w.window_mean, 4) AS window_mean,
    ROUND((w.window_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0), 2) AS deviation_sigma,
    CASE
        WHEN (w.window_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0) > 1 THEN 'HIGH'
        WHEN (w.window_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0) < -1 THEN 'LOW'
        ELSE 'OK'
    END AS status
FROM window_stats w
JOIN baseline b ON w.cohort = b.cohort AND w.signal_id = b.signal_id
WHERE ABS((w.window_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0)) > 0.5
  AND w.window_id > 2  -- Skip baseline windows
ORDER BY w.cohort, w.signal_id, w.window_id;


-- ============================================================================
-- REPORT 3: NON-REVERTING DRIFT
-- Identifies signals that drifted and stayed drifted (persistent shift)
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
        AVG(o.value) AS baseline_mean,
        STDDEV_POP(o.value) AS baseline_std
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
        o.value
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort AND o.signal_id = t.signal_id
    WHERE o.I > t.baseline_end
),

window_stats AS (
    SELECT
        cohort,
        signal_id,
        window_id,
        AVG(value) AS window_mean
    FROM windowed
    GROUP BY cohort, signal_id, window_id
),

drift_pattern AS (
    SELECT
        w.cohort,
        w.signal_id,
        b.baseline_mean,
        b.baseline_std,
        SUM(CASE WHEN w.window_mean > b.baseline_mean THEN 1 ELSE 0 END) AS windows_above,
        SUM(CASE WHEN w.window_mean < b.baseline_mean THEN 1 ELSE 0 END) AS windows_below,
        COUNT(*) AS total_windows,
        AVG(w.window_mean) AS avg_post_baseline,
        AVG(ABS(w.window_mean - b.baseline_mean)) AS avg_deviation
    FROM window_stats w
    JOIN baseline b ON w.cohort = b.cohort AND w.signal_id = b.signal_id
    GROUP BY w.cohort, w.signal_id, b.baseline_mean, b.baseline_std
)

SELECT
    cohort,
    signal_id,
    ROUND(baseline_mean, 4) AS baseline,
    ROUND(avg_post_baseline, 4) AS post_baseline_avg,
    ROUND((avg_post_baseline - baseline_mean) / NULLIF(baseline_std, 0), 2) AS drift_sigma,
    windows_above || '/' || total_windows AS windows_above,
    windows_below || '/' || total_windows AS windows_below,
    CASE
        WHEN windows_above = total_windows THEN 'PERSISTENT_HIGH'
        WHEN windows_below = total_windows THEN 'PERSISTENT_LOW'
        WHEN windows_above >= 4 THEN 'MOSTLY_HIGH'
        WHEN windows_below >= 4 THEN 'MOSTLY_LOW'
        ELSE 'OSCILLATING'
    END AS drift_pattern
FROM drift_pattern
WHERE windows_above >= 4 OR windows_below >= 4
ORDER BY ABS((avg_post_baseline - baseline_mean) / NULLIF(baseline_std, 0)) DESC;


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
-- Aggregates drift direction across all signals
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
        AVG(o.value) AS baseline_mean,
        STDDEV_POP(o.value) AS baseline_std
    FROM observations o
    JOIN time_bounds t USING (signal_id)
    WHERE o.I <= t.baseline_end
    GROUP BY o.signal_id
),

post_baseline AS (
    SELECT o.signal_id, AVG(o.value) AS post_mean
    FROM observations o
    JOIN time_bounds t USING (signal_id)
    WHERE o.I > t.baseline_end
    GROUP BY o.signal_id
),

drift_calc AS (
    SELECT
        b.signal_id,
        (p.post_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0) AS drift_sigma,
        CASE
            WHEN (p.post_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0) > 0.1 THEN 'UP'
            WHEN (p.post_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0) < -0.1 THEN 'DOWN'
            ELSE 'FLAT'
        END AS direction
    FROM baseline b
    JOIN post_baseline p USING (signal_id)
)

SELECT
    direction,
    COUNT(*) AS n_signals,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_total,
    ROUND(AVG(drift_sigma), 2) AS avg_drift_sigma,
    ROUND(MIN(drift_sigma), 2) AS min_drift_sigma,
    ROUND(MAX(drift_sigma), 2) AS max_drift_sigma
FROM drift_calc
GROUP BY direction
ORDER BY n_signals DESC;


-- ============================================================================
-- REPORT 6: BASELINE EXCEEDANCE EVENTS
-- Identifies when signals exceeded 2σ from baseline
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
        STDDEV_POP(o.value) AS sigma
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
        b.mu,
        b.sigma,
        (o.value - b.mu) / NULLIF(b.sigma, 0) AS z_score
    FROM observations o
    JOIN baseline b USING (signal_id)
    JOIN time_bounds t USING (signal_id)
    WHERE o.I > t.baseline_end
      AND ABS((o.value - b.mu) / NULLIF(b.sigma, 0)) > 2
)

SELECT
    signal_id,
    COUNT(*) AS n_exceedances,
    ROUND(MIN(I), 1) AS first_exceedance_t,
    ROUND(MAX(I), 1) AS last_exceedance_t,
    ROUND(MAX(ABS(z_score)), 2) AS max_z_score,
    SUM(CASE WHEN z_score > 0 THEN 1 ELSE 0 END) AS n_high_exceedances,
    SUM(CASE WHEN z_score < 0 THEN 1 ELSE 0 END) AS n_low_exceedances
FROM exceedances
GROUP BY signal_id
ORDER BY n_exceedances DESC;


-- ============================================================================
-- REPORT 7: COORDINATED DRIFT DETECTION
-- Identifies signals that started drifting at the same time
-- Useful for finding common cause / upstream disturbances
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
        STDDEV_POP(o.value) AS sigma
    FROM observations o
    JOIN time_bounds t USING (signal_id)
    WHERE o.I <= t.baseline_end
    GROUP BY o.signal_id
),

windowed AS (
    SELECT o.signal_id,
        NTILE(10) OVER (PARTITION BY o.signal_id ORDER BY o.I) AS window_id,
        o.value
    FROM observations o
    JOIN time_bounds t USING (signal_id)
    WHERE o.I > t.baseline_end
),

window_drift AS (
    SELECT
        w.signal_id,
        w.window_id,
        (AVG(w.value) - b.mu) / NULLIF(b.sigma, 0) AS drift_sigma
    FROM windowed w
    JOIN baseline b USING (signal_id)
    GROUP BY w.signal_id, w.window_id, b.mu, b.sigma
),

first_drift AS (
    SELECT signal_id,
        MIN(window_id) AS first_drift_window
    FROM window_drift
    WHERE ABS(drift_sigma) > 0.5
    GROUP BY signal_id
)

SELECT
    first_drift_window,
    COUNT(*) AS n_signals,
    STRING_AGG(signal_id, ', ' ORDER BY signal_id) AS affected_signals
FROM first_drift
GROUP BY first_drift_window
ORDER BY first_drift_window;


-- ============================================================================
-- REPORT 8: DRIFT TIMELINE
-- Shows progression of drift over time for flagged signals
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
        STDDEV_POP(o.value) AS sigma
    FROM observations o
    JOIN time_bounds t USING (signal_id)
    WHERE o.I <= t.baseline_end
    GROUP BY o.signal_id
),

windowed AS (
    SELECT
        o.signal_id,
        NTILE(10) OVER (PARTITION BY o.signal_id ORDER BY o.I) AS window_id,
        MIN(o.I) OVER (PARTITION BY o.signal_id, NTILE(10) OVER (PARTITION BY o.signal_id ORDER BY o.I)) AS window_start,
        o.value
    FROM observations o
),

window_stats AS (
    SELECT
        signal_id,
        window_id,
        MIN(window_start) AS t,
        AVG(value) AS window_mean
    FROM windowed
    GROUP BY signal_id, window_id
)

SELECT
    w.signal_id,
    w.window_id,
    ROUND(w.t, 1) AS t,
    ROUND(b.mu, 4) AS baseline,
    ROUND(w.window_mean, 4) AS current,
    ROUND((w.window_mean - b.mu) / NULLIF(b.sigma, 0), 2) AS z_score,
    REPEAT('█', CAST(GREATEST(0, LEAST(20, 10 + 5 * (w.window_mean - b.mu) / NULLIF(b.sigma, 0))) AS INTEGER)) AS visual
FROM window_stats w
JOIN baseline b USING (signal_id)
WHERE b.sigma > 0
ORDER BY w.signal_id, w.window_id;
