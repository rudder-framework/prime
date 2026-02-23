-- ============================================================================
-- Engines LEAD-LAG ANALYSIS REPORTS
-- ============================================================================
--
-- Identifies which signals lead or lag others in time.
-- Critical for fault propagation analysis and root cause detection.
--
-- Usage: Run against observations table with columns [cohort, signal_id, signal_0, value]
-- ============================================================================


-- ============================================================================
-- REPORT 1: CROSS-CORRELATION AT MULTIPLE LAGS
-- For each signal pair, find the lag that maximizes correlation
-- ============================================================================

WITH
non_constant AS (
    SELECT signal_id
    FROM observations
    GROUP BY signal_id
    HAVING STDDEV_POP(value) > 0
),

-- Create lagged versions of signals
signal_lags AS (
    SELECT
        a.cohort,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        a.signal_0,
        a.value AS value_a,
        b.value AS value_b,
        LAG(a.value, 1) OVER wa AS value_a_lag1,
        LAG(a.value, 2) OVER wa AS value_a_lag2,
        LAG(a.value, 5) OVER wa AS value_a_lag5,
        LAG(a.value, 10) OVER wa AS value_a_lag10,
        LEAD(a.value, 1) OVER wa AS value_a_lead1,
        LEAD(a.value, 2) OVER wa AS value_a_lead2,
        LEAD(a.value, 5) OVER wa AS value_a_lead5,
        LEAD(a.value, 10) OVER wa AS value_a_lead10
    FROM observations a
    JOIN observations b
        ON a.cohort = b.cohort
        AND a.signal_0 = b.signal_0
        AND a.signal_id < b.signal_id
    WHERE a.signal_id IN (SELECT signal_id FROM non_constant)
      AND b.signal_id IN (SELECT signal_id FROM non_constant)
    WINDOW wa AS (PARTITION BY a.cohort, a.signal_id, b.signal_id ORDER BY a.signal_0)
),

-- Compute correlations at each lag
lag_correlations AS (
    SELECT
        cohort,
        signal_a,
        signal_b,
        CORR(value_a, value_b) AS corr_lag0,
        CORR(value_a_lag1, value_b) AS corr_a_leads_1,
        CORR(value_a_lag2, value_b) AS corr_a_leads_2,
        CORR(value_a_lag5, value_b) AS corr_a_leads_5,
        CORR(value_a_lag10, value_b) AS corr_a_leads_10,
        CORR(value_a_lead1, value_b) AS corr_b_leads_1,
        CORR(value_a_lead2, value_b) AS corr_b_leads_2,
        CORR(value_a_lead5, value_b) AS corr_b_leads_5,
        CORR(value_a_lead10, value_b) AS corr_b_leads_10
    FROM signal_lags
    WHERE value_a_lag10 IS NOT NULL AND value_a_lead10 IS NOT NULL
    GROUP BY cohort, signal_a, signal_b
)

SELECT
    lc.cohort,
    lc.signal_a,
    ua.unit AS unit_a,
    lc.signal_b,
    ub.unit AS unit_b,
    ROUND(lc.corr_lag0, 3) AS sync_corr,
    ROUND(lc.corr_a_leads_1, 3) AS a_leads_1,
    ROUND(lc.corr_a_leads_5, 3) AS a_leads_5,
    ROUND(lc.corr_b_leads_1, 3) AS b_leads_1,
    ROUND(lc.corr_b_leads_5, 3) AS b_leads_5,
    -- Determine leader
    CASE
        WHEN isnan(lc.corr_lag0) OR isnan(lc.corr_a_leads_5) OR isnan(lc.corr_b_leads_5) THEN 'UNDEFINED'
        WHEN ABS(lc.corr_a_leads_5) > ABS(lc.corr_lag0) + 0.05
         AND ABS(lc.corr_a_leads_5) > ABS(lc.corr_b_leads_5) + 0.05 THEN lc.signal_a || ' [' || ua.unit || '] LEADS'
        WHEN ABS(lc.corr_b_leads_5) > ABS(lc.corr_lag0) + 0.05
         AND ABS(lc.corr_b_leads_5) > ABS(lc.corr_a_leads_5) + 0.05 THEN lc.signal_b || ' [' || ub.unit || '] LEADS'
        WHEN ABS(lc.corr_lag0) > 0.5 THEN 'SYNCHRONOUS'
        ELSE 'INDEPENDENT'
    END AS lead_lag_relationship,
    -- Best correlation found
    GREATEST(ABS(lc.corr_lag0), ABS(lc.corr_a_leads_5), ABS(lc.corr_b_leads_5)) AS max_corr
FROM lag_correlations lc
LEFT JOIN (SELECT DISTINCT o.signal_id, s.unit FROM observations o LEFT JOIN signals s ON o.signal_id = s.signal_id) ua ON lc.signal_a = ua.signal_id
LEFT JOIN (SELECT DISTINCT o.signal_id, s.unit FROM observations o LEFT JOIN signals s ON o.signal_id = s.signal_id) ub ON lc.signal_b = ub.signal_id
WHERE lc.corr_lag0 IS NOT NULL AND NOT isnan(lc.corr_lag0)
  AND (ABS(lc.corr_lag0) > 0.2 OR ABS(lc.corr_a_leads_5) > 0.2 OR ABS(lc.corr_b_leads_5) > 0.2)
ORDER BY max_corr DESC;


-- ============================================================================
-- REPORT 2: FIRST MOVER DETECTION
-- Identifies which signals deviate from baseline first
-- ============================================================================

WITH
time_bounds AS (
    SELECT
        cohort,
        MIN(signal_0) AS i_min,
        MAX(signal_0) AS i_max,
        MIN(signal_0) + 0.2 * (MAX(signal_0) - MIN(signal_0)) AS baseline_end
    FROM observations
    GROUP BY cohort
),

baseline_stats AS (
    SELECT
        o.cohort,
        o.signal_id,
        AVG(o.value) AS baseline_mean,
        PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY o.value) AS baseline_p05,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY o.value) AS baseline_p95
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort
    WHERE o.signal_0 <= t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

-- Find exceedance for each point after baseline using percentile bounds
deviations AS (
    SELECT
        o.cohort,
        o.signal_id,
        o.signal_0,
        o.value,
        b.baseline_mean,
        b.baseline_p05,
        b.baseline_p95,
        (o.value < b.baseline_p05 OR o.value > b.baseline_p95) AS out_of_range
    FROM observations o
    JOIN baseline_stats b ON o.cohort = b.cohort AND o.signal_id = b.signal_id
    JOIN time_bounds t ON o.cohort = t.cohort
    WHERE o.signal_0 > t.baseline_end
),

first_deviation AS (
    SELECT
        cohort,
        signal_id,
        MIN(CASE WHEN out_of_range THEN signal_0 END) AS first_exceed_p95
    FROM deviations
    GROUP BY cohort, signal_id
)

SELECT
    fd.cohort,
    fd.signal_id,
    u.unit,
    ROUND(fd.first_exceed_p95, 4) AS first_oor_time,
    RANK() OVER (PARTITION BY fd.cohort ORDER BY fd.first_exceed_p95 NULLS LAST) AS response_order,
    CASE
        WHEN RANK() OVER (PARTITION BY fd.cohort ORDER BY fd.first_exceed_p95 NULLS LAST) <= 3 THEN 'FIRST_RESPONDER'
        WHEN RANK() OVER (PARTITION BY fd.cohort ORDER BY fd.first_exceed_p95 NULLS LAST) <= 10 THEN 'EARLY_RESPONDER'
        WHEN fd.first_exceed_p95 IS NOT NULL THEN 'LATE_RESPONDER'
        ELSE 'NO_DEVIATION'
    END AS response_class
FROM first_deviation fd
LEFT JOIN (SELECT DISTINCT o.signal_id, s.unit FROM observations o LEFT JOIN signals s ON o.signal_id = s.signal_id) u USING (signal_id)
ORDER BY fd.cohort, fd.first_exceed_p95 NULLS LAST;


-- ============================================================================
-- REPORT 3: PROPAGATION CHAIN DETECTION
-- Finds sequences: A deviates, then B deviates, then C deviates
-- ============================================================================

WITH
time_bounds AS (
    SELECT cohort, MIN(signal_0) + 0.2 * (MAX(signal_0) - MIN(signal_0)) AS baseline_end
    FROM observations GROUP BY cohort
),

baseline_stats AS (
    SELECT o.cohort, o.signal_id,
        AVG(o.value) AS baseline_mean, STDDEV_POP(o.value) AS baseline_std
    FROM observations o JOIN time_bounds t USING (cohort)
    WHERE o.signal_0 <= t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

first_deviation AS (
    SELECT
        o.cohort,
        o.signal_id,
        MIN(CASE WHEN ABS((o.value - b.baseline_mean) / NULLIF(b.baseline_std, 0)) > 2.0 THEN o.signal_0 END) AS first_dev_time
    FROM observations o
    JOIN baseline_stats b USING (cohort, signal_id)
    JOIN time_bounds t USING (cohort)
    WHERE o.signal_0 > t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

-- Find pairs where A deviated before B
propagation_pairs AS (
    SELECT
        a.cohort,
        a.signal_id AS source_signal,
        b.signal_id AS target_signal,
        a.first_dev_time AS source_dev_time,
        b.first_dev_time AS target_dev_time,
        b.first_dev_time - a.first_dev_time AS propagation_delay
    FROM first_deviation a
    JOIN first_deviation b
        ON a.cohort = b.cohort
        AND a.signal_id != b.signal_id
        AND a.first_dev_time IS NOT NULL
        AND b.first_dev_time IS NOT NULL
        AND b.first_dev_time > a.first_dev_time
        AND b.first_dev_time - a.first_dev_time < 5  -- Within 5 time units
)

SELECT
    cohort,
    source_signal,
    target_signal,
    ROUND(source_dev_time, 4) AS source_time,
    ROUND(target_dev_time, 4) AS target_time,
    ROUND(propagation_delay, 4) AS delay,
    CASE
        WHEN propagation_delay < 0.1 THEN 'IMMEDIATE'
        WHEN propagation_delay < 1.0 THEN 'FAST'
        WHEN propagation_delay < 2.0 THEN 'MODERATE'
        ELSE 'SLOW'
    END AS propagation_speed
FROM propagation_pairs
ORDER BY cohort, source_dev_time, propagation_delay;


-- ============================================================================
-- REPORT 4: FAULT PROPAGATION SUMMARY
-- Counts how many signals each signal "infects" after it deviates
-- ============================================================================

WITH
time_bounds AS (
    SELECT cohort, MIN(signal_0) + 0.2 * (MAX(signal_0) - MIN(signal_0)) AS baseline_end
    FROM observations GROUP BY cohort
),

baseline_stats AS (
    SELECT o.cohort, o.signal_id,
        AVG(o.value) AS baseline_mean, STDDEV_POP(o.value) AS baseline_std
    FROM observations o JOIN time_bounds t USING (cohort)
    WHERE o.signal_0 <= t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

first_deviation AS (
    SELECT
        o.cohort,
        o.signal_id,
        MIN(CASE WHEN ABS((o.value - b.baseline_mean) / NULLIF(b.baseline_std, 0)) > 2.0 THEN o.signal_0 END) AS first_dev_time
    FROM observations o
    JOIN baseline_stats b USING (cohort, signal_id)
    JOIN time_bounds t USING (cohort)
    WHERE o.signal_0 > t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

downstream_count AS (
    SELECT
        a.cohort,
        a.signal_id,
        a.first_dev_time,
        COUNT(b.signal_id) AS n_downstream
    FROM first_deviation a
    LEFT JOIN first_deviation b
        ON a.cohort = b.cohort
        AND a.signal_id != b.signal_id
        AND b.first_dev_time > a.first_dev_time
        AND b.first_dev_time - a.first_dev_time < 5
    WHERE a.first_dev_time IS NOT NULL
    GROUP BY a.cohort, a.signal_id, a.first_dev_time
)

SELECT
    cohort,
    signal_id,
    ROUND(first_dev_time, 4) AS deviation_time,
    n_downstream AS signals_affected_downstream,
    RANK() OVER (PARTITION BY cohort ORDER BY first_dev_time) AS deviation_order,
    CASE
        WHEN RANK() OVER (PARTITION BY cohort ORDER BY first_dev_time) = 1 THEN 'ROOT_CAUSE_CANDIDATE'
        WHEN RANK() OVER (PARTITION BY cohort ORDER BY first_dev_time) <= 3 THEN 'EARLY_INDICATOR'
        ELSE 'DOWNSTREAM'
    END AS role
FROM downstream_count
ORDER BY cohort, first_dev_time;


-- ============================================================================
-- REPORT 5: LEAD-LAG STABILITY OVER TIME
-- Checks if lead-lag relationships change during the observation period
-- ============================================================================

WITH
non_constant AS (
    SELECT signal_id
    FROM observations
    GROUP BY signal_id
    HAVING STDDEV_POP(value) > 0
),

windowed AS (
    SELECT
        cohort,
        signal_id,
        NTILE(5) OVER (PARTITION BY cohort, signal_id ORDER BY signal_0) AS window_id,
        signal_0,
        value
    FROM observations
    WHERE signal_id IN (SELECT signal_id FROM non_constant)
),

-- Create pairs within each window
window_pairs AS (
    SELECT
        a.cohort,
        a.window_id,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        a.signal_0,
        a.value AS value_a,
        b.value AS value_b,
        LAG(a.value, 2) OVER w AS value_a_lag2
    FROM windowed a
    JOIN windowed b
        ON a.cohort = b.cohort
        AND a.signal_0 = b.signal_0
        AND a.window_id = b.window_id
        AND a.signal_id < b.signal_id
    WINDOW w AS (PARTITION BY a.cohort, a.signal_id, b.signal_id, a.window_id ORDER BY a.signal_0)
),

window_lead_lag AS (
    SELECT
        cohort,
        window_id,
        signal_a,
        signal_b,
        CORR(value_a, value_b) AS sync_corr,
        CORR(value_a_lag2, value_b) AS a_leads_corr
    FROM window_pairs
    WHERE value_a_lag2 IS NOT NULL
    GROUP BY cohort, window_id, signal_a, signal_b
)

SELECT
    cohort,
    signal_a,
    signal_b,
    MAX(CASE WHEN window_id = 1 THEN ROUND(sync_corr, 3) END) AS w1_sync,
    MAX(CASE WHEN window_id = 3 THEN ROUND(sync_corr, 3) END) AS w3_sync,
    MAX(CASE WHEN window_id = 5 THEN ROUND(sync_corr, 3) END) AS w5_sync,
    MAX(CASE WHEN window_id = 1 THEN ROUND(a_leads_corr, 3) END) AS w1_lead,
    MAX(CASE WHEN window_id = 5 THEN ROUND(a_leads_corr, 3) END) AS w5_lead,
    -- Detect if relationship changed
    CASE
        WHEN MAX(CASE WHEN window_id = 5 THEN sync_corr END) IS NULL
          OR MAX(CASE WHEN window_id = 1 THEN sync_corr END) IS NULL THEN 'UNDEFINED'
        WHEN isnan(MAX(CASE WHEN window_id = 5 THEN sync_corr END))
          OR isnan(MAX(CASE WHEN window_id = 1 THEN sync_corr END)) THEN 'UNDEFINED'
        WHEN ABS(MAX(CASE WHEN window_id = 5 THEN sync_corr END) -
                 MAX(CASE WHEN window_id = 1 THEN sync_corr END)) > 0.3 THEN 'CORRELATION_SHIFT'
        WHEN ABS(MAX(CASE WHEN window_id = 5 THEN a_leads_corr END) -
                 MAX(CASE WHEN window_id = 1 THEN a_leads_corr END)) > 0.3 THEN 'LEAD_LAG_SHIFT'
        ELSE 'STABLE'
    END AS stability
FROM window_lead_lag
WHERE sync_corr IS NOT NULL AND NOT isnan(sync_corr)
GROUP BY cohort, signal_a, signal_b
HAVING ABS(MAX(sync_corr) - MIN(sync_corr)) > 0.2
ORDER BY cohort, ABS(MAX(sync_corr) - MIN(sync_corr)) DESC;


-- ============================================================================
-- REPORT 6: EXECUTIVE LEAD-LAG SUMMARY
-- One row per entity with key metrics
-- ============================================================================

WITH
time_bounds AS (
    SELECT cohort, MIN(signal_0) + 0.2 * (MAX(signal_0) - MIN(signal_0)) AS baseline_end
    FROM observations GROUP BY cohort
),

baseline_stats AS (
    SELECT o.cohort, o.signal_id,
        AVG(o.value) AS baseline_mean, STDDEV_POP(o.value) AS baseline_std
    FROM observations o JOIN time_bounds t USING (cohort)
    WHERE o.signal_0 <= t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

first_deviation AS (
    SELECT
        o.cohort,
        o.signal_id,
        MIN(CASE WHEN ABS((o.value - b.baseline_mean) / NULLIF(b.baseline_std, 0)) > 2.0 THEN o.signal_0 END) AS first_dev_time
    FROM observations o
    JOIN baseline_stats b USING (cohort, signal_id)
    JOIN time_bounds t USING (cohort)
    WHERE o.signal_0 > t.baseline_end
    GROUP BY o.cohort, o.signal_id
)

SELECT
    cohort,
    COUNT(DISTINCT signal_id) AS total_signals,
    COUNT(DISTINCT CASE WHEN first_dev_time IS NOT NULL THEN signal_id END) AS signals_deviated,
    MIN(first_dev_time) AS first_deviation_time,
    MAX(first_dev_time) AS last_deviation_time,
    MAX(first_dev_time) - MIN(first_dev_time) AS propagation_window,
    (SELECT signal_id FROM first_deviation f2
     WHERE f2.cohort = first_deviation.cohort
     ORDER BY first_dev_time LIMIT 1) AS first_mover,
    (SELECT u.unit FROM (SELECT DISTINCT o.signal_id, s.unit FROM observations o LEFT JOIN signals s ON o.signal_id = s.signal_id) u
     WHERE u.signal_id = (SELECT signal_id FROM first_deviation f2
     WHERE f2.cohort = first_deviation.cohort
     ORDER BY first_dev_time LIMIT 1)) AS first_mover_unit,
    CASE
        WHEN COUNT(DISTINCT CASE WHEN first_dev_time IS NOT NULL THEN signal_id END) = 0 THEN 'STABLE'
        WHEN MAX(first_dev_time) - MIN(first_dev_time) < 1 THEN 'RAPID_CASCADE'
        WHEN MAX(first_dev_time) - MIN(first_dev_time) < 5 THEN 'GRADUAL_PROPAGATION'
        ELSE 'SLOW_SPREAD'
    END AS propagation_pattern
FROM first_deviation
GROUP BY cohort;
