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
-- Create lagged versions of signals
signal_lags AS (
    SELECT
        a.cohort,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        a.signal_0,
        a.value AS y_a,
        b.value AS y_b,
        LAG(a.value, 1) OVER wa AS y_a_lag1,
        LAG(a.value, 2) OVER wa AS y_a_lag2,
        LAG(a.value, 5) OVER wa AS y_a_lag5,
        LAG(a.value, 10) OVER wa AS y_a_lag10,
        LEAD(a.value, 1) OVER wa AS y_a_lead1,
        LEAD(a.value, 2) OVER wa AS y_a_lead2,
        LEAD(a.value, 5) OVER wa AS y_a_lead5,
        LEAD(a.value, 10) OVER wa AS y_a_lead10
    FROM observations a
    JOIN observations b
        ON a.cohort = b.cohort
        AND a.signal_0 = b.signal_0
        AND a.signal_id < b.signal_id
    WINDOW wa AS (PARTITION BY a.cohort, a.signal_id, b.signal_id ORDER BY a.signal_0)
),

-- Compute correlations at each lag
lag_correlations AS (
    SELECT
        cohort,
        signal_a,
        signal_b,
        CORR(y_a, y_b) AS corr_lag0,
        CORR(y_a_lag1, y_b) AS corr_a_leads_1,
        CORR(y_a_lag2, y_b) AS corr_a_leads_2,
        CORR(y_a_lag5, y_b) AS corr_a_leads_5,
        CORR(y_a_lag10, y_b) AS corr_a_leads_10,
        CORR(y_a_lead1, y_b) AS corr_b_leads_1,
        CORR(y_a_lead2, y_b) AS corr_b_leads_2,
        CORR(y_a_lead5, y_b) AS corr_b_leads_5,
        CORR(y_a_lead10, y_b) AS corr_b_leads_10
    FROM signal_lags
    WHERE y_a_lag10 IS NOT NULL AND y_a_lead10 IS NOT NULL
    GROUP BY cohort, signal_a, signal_b
)

SELECT
    cohort,
    signal_a,
    signal_b,
    ROUND(corr_lag0, 3) AS sync_corr,
    ROUND(corr_a_leads_1, 3) AS a_leads_1,
    ROUND(corr_a_leads_5, 3) AS a_leads_5,
    ROUND(corr_b_leads_1, 3) AS b_leads_1,
    ROUND(corr_b_leads_5, 3) AS b_leads_5,
    -- Determine leader
    CASE
        WHEN ABS(corr_a_leads_5) > ABS(corr_lag0) + 0.05
         AND ABS(corr_a_leads_5) > ABS(corr_b_leads_5) + 0.05 THEN signal_a || ' LEADS'
        WHEN ABS(corr_b_leads_5) > ABS(corr_lag0) + 0.05
         AND ABS(corr_b_leads_5) > ABS(corr_a_leads_5) + 0.05 THEN signal_b || ' LEADS'
        WHEN ABS(corr_lag0) > 0.5 THEN 'SYNCHRONOUS'
        ELSE 'INDEPENDENT'
    END AS lead_lag_relationship,
    -- Best correlation found
    GREATEST(ABS(corr_lag0), ABS(corr_a_leads_5), ABS(corr_b_leads_5)) AS max_corr
FROM lag_correlations
WHERE ABS(corr_lag0) > 0.2 OR ABS(corr_a_leads_5) > 0.2 OR ABS(corr_b_leads_5) > 0.2
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
    cohort,
    signal_id,
    ROUND(first_exceed_p95, 4) AS first_oor_time,
    RANK() OVER (PARTITION BY cohort ORDER BY first_exceed_p95 NULLS LAST) AS response_order,
    CASE
        WHEN RANK() OVER (PARTITION BY cohort ORDER BY first_exceed_p95 NULLS LAST) <= 3 THEN 'FIRST_RESPONDER'
        WHEN RANK() OVER (PARTITION BY cohort ORDER BY first_exceed_p95 NULLS LAST) <= 10 THEN 'EARLY_RESPONDER'
        WHEN first_exceed_p95 IS NOT NULL THEN 'LATE_RESPONDER'
        ELSE 'NO_DEVIATION'
    END AS response_class
FROM first_deviation
ORDER BY cohort, first_exceed_p95 NULLS LAST;


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
windowed AS (
    SELECT
        cohort,
        signal_id,
        NTILE(5) OVER (PARTITION BY cohort, signal_id ORDER BY signal_0) AS window_id,
        signal_0,
        value
    FROM observations
),

-- Create pairs within each window
window_pairs AS (
    SELECT
        a.cohort,
        a.window_id,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        a.signal_0,
        a.value AS y_a,
        b.value AS y_b,
        LAG(a.value, 2) OVER w AS y_a_lag2
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
        CORR(y_a, y_b) AS sync_corr,
        CORR(y_a_lag2, y_b) AS a_leads_corr
    FROM window_pairs
    WHERE y_a_lag2 IS NOT NULL
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
        WHEN ABS(MAX(CASE WHEN window_id = 5 THEN sync_corr END) -
                 MAX(CASE WHEN window_id = 1 THEN sync_corr END)) > 0.3 THEN 'CORRELATION_SHIFT'
        WHEN ABS(MAX(CASE WHEN window_id = 5 THEN a_leads_corr END) -
                 MAX(CASE WHEN window_id = 1 THEN a_leads_corr END)) > 0.3 THEN 'LEAD_LAG_SHIFT'
        ELSE 'STABLE'
    END AS stability
FROM window_lead_lag
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
    CASE
        WHEN COUNT(DISTINCT CASE WHEN first_dev_time IS NOT NULL THEN signal_id END) = 0 THEN 'STABLE'
        WHEN MAX(first_dev_time) - MIN(first_dev_time) < 1 THEN 'RAPID_CASCADE'
        WHEN MAX(first_dev_time) - MIN(first_dev_time) < 5 THEN 'GRADUAL_PROPAGATION'
        ELSE 'SLOW_SPREAD'
    END AS propagation_pattern
FROM first_deviation
GROUP BY cohort;
