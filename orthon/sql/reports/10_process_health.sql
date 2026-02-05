-- ============================================================================
-- PRISM PROCESS HEALTH REPORTS
-- ============================================================================
--
-- Executive-level summaries of process health.
-- Traffic light status for plant managers.
--
-- Usage: Run against observations and primitives tables
-- ============================================================================


-- ============================================================================
-- REPORT 1: OVERALL PROCESS HEALTH SCORECARD
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
        STDDEV_POP(o.value) AS current_std
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort
    WHERE o.I > t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

signal_health AS (
    SELECT
        b.cohort,
        b.signal_id,
        (c.current_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0) AS mean_drift_sigma,
        c.current_std / NULLIF(b.baseline_std, 0) AS volatility_ratio,
        CASE
            WHEN ABS((c.current_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0)) > 2 THEN 'RED'
            WHEN ABS((c.current_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0)) > 1 THEN 'YELLOW'
            ELSE 'GREEN'
        END AS drift_status,
        CASE
            WHEN c.current_std / NULLIF(b.baseline_std, 0) > 1.5 THEN 'RED'
            WHEN c.current_std / NULLIF(b.baseline_std, 0) > 1.2 THEN 'YELLOW'
            ELSE 'GREEN'
        END AS volatility_status
    FROM baseline b
    JOIN current_state c ON b.cohort = c.cohort AND b.signal_id = c.signal_id
)

SELECT
    cohort,
    COUNT(*) AS total_signals,

    -- Drift health
    SUM(CASE WHEN drift_status = 'GREEN' THEN 1 ELSE 0 END) AS drift_green,
    SUM(CASE WHEN drift_status = 'YELLOW' THEN 1 ELSE 0 END) AS drift_yellow,
    SUM(CASE WHEN drift_status = 'RED' THEN 1 ELSE 0 END) AS drift_red,

    -- Volatility health
    SUM(CASE WHEN volatility_status = 'GREEN' THEN 1 ELSE 0 END) AS vol_green,
    SUM(CASE WHEN volatility_status = 'YELLOW' THEN 1 ELSE 0 END) AS vol_yellow,
    SUM(CASE WHEN volatility_status = 'RED' THEN 1 ELSE 0 END) AS vol_red,

    -- Overall status
    CASE
        WHEN SUM(CASE WHEN drift_status = 'RED' THEN 1 ELSE 0 END) > 0
          OR SUM(CASE WHEN volatility_status = 'RED' THEN 1 ELSE 0 END) > 0 THEN 'RED'
        WHEN SUM(CASE WHEN drift_status = 'YELLOW' THEN 1 ELSE 0 END) > 2
          OR SUM(CASE WHEN volatility_status = 'YELLOW' THEN 1 ELSE 0 END) > 2 THEN 'YELLOW'
        ELSE 'GREEN'
    END AS overall_status,

    -- Summary metrics
    ROUND(AVG(ABS(mean_drift_sigma)), 2) AS avg_drift_sigma,
    ROUND(MAX(ABS(mean_drift_sigma)), 2) AS max_drift_sigma,
    ROUND(AVG(volatility_ratio), 2) AS avg_volatility_ratio

FROM signal_health
GROUP BY cohort;


-- ============================================================================
-- REPORT 2: SIGNAL-LEVEL HEALTH MATRIX
-- ============================================================================

WITH
time_bounds AS (
    SELECT cohort, MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations GROUP BY cohort
),

baseline AS (
    SELECT o.cohort, o.signal_id,
        AVG(o.value) AS baseline_mean, STDDEV_POP(o.value) AS baseline_std
    FROM observations o JOIN time_bounds t USING (cohort)
    WHERE o.I <= t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

current_state AS (
    SELECT o.cohort, o.signal_id,
        AVG(o.value) AS current_mean, STDDEV_POP(o.value) AS current_std
    FROM observations o JOIN time_bounds t USING (cohort)
    WHERE o.I > t.baseline_end
    GROUP BY o.cohort, o.signal_id
)

SELECT
    b.cohort,
    b.signal_id,
    ROUND(b.baseline_mean, 4) AS baseline_mean,
    ROUND(c.current_mean, 4) AS current_mean,
    ROUND((c.current_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0), 2) AS drift_z,
    ROUND(100 * (c.current_std - b.baseline_std) / NULLIF(b.baseline_std, 0), 1) AS vol_change_pct,

    -- Traffic lights
    CASE
        WHEN ABS((c.current_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0)) > 2 THEN '🔴'
        WHEN ABS((c.current_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0)) > 1 THEN '🟡'
        ELSE '🟢'
    END AS drift_light,

    CASE
        WHEN c.current_std / NULLIF(b.baseline_std, 0) > 1.5 THEN '🔴'
        WHEN c.current_std / NULLIF(b.baseline_std, 0) > 1.2 THEN '🟡'
        ELSE '🟢'
    END AS vol_light,

    -- Action flag
    CASE
        WHEN ABS((c.current_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0)) > 2 THEN 'INVESTIGATE'
        WHEN ABS((c.current_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0)) > 1 THEN 'MONITOR'
        ELSE 'OK'
    END AS action

FROM baseline b
JOIN current_state c USING (cohort, signal_id)
ORDER BY ABS((c.current_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0)) DESC;


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
        WHEN up_moves >= 0.8 * total_moves OR down_moves >= 0.8 * total_moves THEN 'INVESTIGATE'
        WHEN up_moves >= 0.6 * total_moves OR down_moves >= 0.6 * total_moves THEN 'MONITOR'
        ELSE 'OK'
    END AS action
FROM trend_check
WHERE up_moves >= 0.6 * total_moves OR down_moves >= 0.6 * total_moves
ORDER BY GREATEST(up_moves, down_moves) DESC;


-- ============================================================================
-- REPORT 4: ANOMALY SUMMARY
-- Quick count of anomalous events per signal
-- ============================================================================

WITH
time_bounds AS (
    SELECT signal_id, MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations GROUP BY signal_id
),

baseline AS (
    SELECT o.signal_id, AVG(o.value) AS mu, STDDEV_POP(o.value) AS sigma
    FROM observations o JOIN time_bounds t USING (signal_id)
    WHERE o.I <= t.baseline_end
    GROUP BY o.signal_id
),

anomalies AS (
    SELECT
        o.signal_id,
        o.I,
        ABS((o.value - b.mu) / NULLIF(b.sigma, 0)) AS z_score
    FROM observations o
    JOIN baseline b USING (signal_id)
    JOIN time_bounds t USING (signal_id)
    WHERE o.I > t.baseline_end
)

SELECT
    signal_id,
    COUNT(*) AS total_points,
    SUM(CASE WHEN z_score > 2 THEN 1 ELSE 0 END) AS anomalies_2sigma,
    SUM(CASE WHEN z_score > 3 THEN 1 ELSE 0 END) AS anomalies_3sigma,
    ROUND(100.0 * SUM(CASE WHEN z_score > 2 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_anomalous,
    ROUND(MAX(z_score), 2) AS max_z_score,
    CASE
        WHEN 100.0 * SUM(CASE WHEN z_score > 2 THEN 1 ELSE 0 END) / COUNT(*) > 10 THEN 'HIGH_ANOMALY_RATE'
        WHEN 100.0 * SUM(CASE WHEN z_score > 2 THEN 1 ELSE 0 END) / COUNT(*) > 5 THEN 'ELEVATED_ANOMALY_RATE'
        ELSE 'NORMAL'
    END AS status
FROM anomalies
GROUP BY signal_id
HAVING SUM(CASE WHEN z_score > 2 THEN 1 ELSE 0 END) > 0
ORDER BY pct_anomalous DESC;


-- ============================================================================
-- REPORT 5: EXECUTIVE DASHBOARD (single row summary)
-- ============================================================================

WITH
time_bounds AS (
    SELECT MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end FROM observations
),

baseline AS (
    SELECT o.signal_id, AVG(o.value) AS mu, STDDEV_POP(o.value) AS sigma
    FROM observations o, time_bounds t
    WHERE o.I <= t.baseline_end
    GROUP BY o.signal_id
),

current_state AS (
    SELECT o.signal_id, AVG(o.value) AS current_mean, STDDEV_POP(o.value) AS current_std
    FROM observations o, time_bounds t
    WHERE o.I > t.baseline_end
    GROUP BY o.signal_id
),

health_check AS (
    SELECT
        b.signal_id,
        ABS((c.current_mean - b.mu) / NULLIF(b.sigma, 0)) AS drift_z,
        c.current_std / NULLIF(b.sigma, 0) AS vol_ratio
    FROM baseline b
    JOIN current_state c USING (signal_id)
)

SELECT
    (SELECT COUNT(DISTINCT signal_id) FROM observations) AS total_signals,
    SUM(CASE WHEN drift_z < 1 THEN 1 ELSE 0 END) AS signals_stable,
    SUM(CASE WHEN drift_z >= 1 AND drift_z < 2 THEN 1 ELSE 0 END) AS signals_watch,
    SUM(CASE WHEN drift_z >= 2 THEN 1 ELSE 0 END) AS signals_alert,
    ROUND(AVG(drift_z), 2) AS avg_drift,
    ROUND(MAX(drift_z), 2) AS max_drift,
    ROUND(AVG(vol_ratio), 2) AS avg_vol_ratio,
    CASE
        WHEN SUM(CASE WHEN drift_z >= 2 THEN 1 ELSE 0 END) > 0 THEN '🔴 ALERT'
        WHEN SUM(CASE WHEN drift_z >= 1 THEN 1 ELSE 0 END) > 3 THEN '🟡 WATCH'
        ELSE '🟢 NORMAL'
    END AS process_status,
    CURRENT_TIMESTAMP AS report_time
FROM health_check;
