-- =============================================================================
-- LEAD TIME ANALYSIS
-- =============================================================================
-- Per-metric lead time computation for AI-guided tuning.
-- Answers: "How early did each metric detect each fault?"
--
-- Key insight: Different metrics may have different lead times.
-- Coherence might detect valve faults 47 samples early,
-- while entropy detects cavitation 71 samples early.
--
-- Uses percentile-based baseline range detection, not z-score thresholds.
-- =============================================================================

-- Drop existing views for clean reload
DROP VIEW IF EXISTS v_metric_aligned_to_fault;
DROP VIEW IF EXISTS v_metric_baseline;
DROP VIEW IF EXISTS v_metric_first_deviation;
DROP VIEW IF EXISTS v_metric_lead_times;
DROP VIEW IF EXISTS v_metric_performance;

-- =============================================================================
-- ALIGN METRICS TO FAULT TIMESTAMPS
-- =============================================================================

-- Normalize signal_0_center so that fault_start_I = 0 (negative = before fault)
CREATE VIEW v_metric_aligned_to_fault AS
SELECT
    p.cohort,
    p.signal_id,
    p.metric_name,
    p.signal_0_center,
    f.fault_start_I,
    f.label_name,
    p.signal_0_center - f.fault_start_I AS I_relative,  -- Negative = before fault
    p.metric_value
FROM (
    -- Pivot physics metrics to long format (raw values only, no z-scores)
    SELECT cohort, signal_id, signal_0_center,
           'coherence' AS metric_name, coherence AS metric_value
    FROM physics WHERE coherence IS NOT NULL
    UNION ALL
    SELECT cohort, signal_id, signal_0_center,
           'entropy' AS metric_name, entropy AS metric_value
    FROM physics WHERE entropy IS NOT NULL
    UNION ALL
    SELECT cohort, signal_id, signal_0_center,
           'lyapunov' AS metric_name, lyapunov AS metric_value
    FROM physics WHERE lyapunov IS NOT NULL
    UNION ALL
    SELECT cohort, signal_id, signal_0_center,
           'hurst' AS metric_name, hurst AS metric_value
    FROM physics WHERE hurst IS NOT NULL
    UNION ALL
    SELECT cohort, signal_id, signal_0_center,
           'energy_total' AS metric_name, energy_total AS metric_value
    FROM physics WHERE energy_total IS NOT NULL
    UNION ALL
    SELECT cohort, signal_id, signal_0_center,
           'dissipation_rate' AS metric_name, dissipation_rate AS metric_value
    FROM physics WHERE dissipation_rate IS NOT NULL
) p
JOIN v_fault_times f ON p.cohort = f.cohort
WHERE f.fault_start_I IS NOT NULL;

-- =============================================================================
-- ESTABLISH BASELINE (PRE-FAULT STABLE PERIOD)
-- =============================================================================

-- Baseline stats per entity per metric (percentile bounds, not mean/std)
-- Use I_relative between -500 and -100 as "stable" pre-fault period
CREATE VIEW v_metric_baseline AS
SELECT
    cohort,
    signal_id,
    metric_name,
    label_name,
    AVG(metric_value) AS baseline_mean,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY metric_value) AS baseline_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY metric_value) AS baseline_p95,
    MIN(metric_value) AS baseline_min,
    MAX(metric_value) AS baseline_max,
    COUNT(*) AS baseline_n
FROM v_metric_aligned_to_fault
WHERE I_relative BETWEEN -500 AND -100  -- Well before fault
GROUP BY cohort, signal_id, metric_name, label_name
HAVING COUNT(*) >= 10;  -- Need enough samples for reliable baseline

-- =============================================================================
-- FIND FIRST DEVIATION (LEAD TIME)
-- =============================================================================

-- First significant deviation from baseline for each metric
-- Uses baseline percentile range [p05, p95] instead of sigma thresholds
CREATE VIEW v_metric_first_deviation AS
SELECT
    a.cohort,
    a.signal_id,
    a.metric_name,
    a.label_name,
    b.baseline_mean,
    b.baseline_p05,
    b.baseline_p95,

    -- First out-of-range exceedance (value outside [p05, p95])
    MIN(a.I_relative) FILTER (WHERE
        (a.metric_value < b.baseline_p05 OR a.metric_value > b.baseline_p95) AND
        a.I_relative < 0
    ) AS first_p95_relative_I,

    -- First extreme exceedance (value beyond 2x range width from bounds)
    MIN(a.I_relative) FILTER (WHERE
        (b.baseline_p95 - b.baseline_p05) > 0 AND
        GREATEST(
            CASE WHEN a.metric_value > b.baseline_p95
                 THEN (a.metric_value - b.baseline_p95) / (b.baseline_p95 - b.baseline_p05)
                 WHEN a.metric_value < b.baseline_p05
                 THEN (b.baseline_p05 - a.metric_value) / (b.baseline_p95 - b.baseline_p05)
                 ELSE 0
            END, 0
        ) > 1.0 AND
        a.I_relative < 0
    ) AS first_extreme_relative_I,

    -- Max exceedance observed (range-normalized distance outside [p05, p95])
    MAX(
        CASE
            WHEN (b.baseline_p95 - b.baseline_p05) > 0
            THEN GREATEST(
                CASE WHEN a.metric_value > b.baseline_p95
                     THEN (a.metric_value - b.baseline_p95) / (b.baseline_p95 - b.baseline_p05)
                     WHEN a.metric_value < b.baseline_p05
                     THEN (b.baseline_p05 - a.metric_value) / (b.baseline_p95 - b.baseline_p05)
                     ELSE 0
                END, 0
            )
            ELSE 0
        END
    ) AS max_exceedance_observed

FROM v_metric_aligned_to_fault a
JOIN v_metric_baseline b ON
    a.cohort = b.cohort AND
    a.signal_id = b.signal_id AND
    a.metric_name = b.metric_name AND
    a.label_name = b.label_name
GROUP BY a.cohort, a.signal_id, a.metric_name, a.label_name, b.baseline_mean, b.baseline_p05, b.baseline_p95;

-- =============================================================================
-- LEAD TIME RESULTS
-- =============================================================================

-- Convert relative signal_0_center to lead time (positive = early detection)
CREATE VIEW v_metric_lead_times AS
SELECT
    cohort,
    signal_id,
    metric_name,
    label_name,
    baseline_mean,
    baseline_p05,
    baseline_p95,

    -- Lead times (negate relative signal_0_center to get positive lead time)
    -first_p95_relative_I AS lead_time_p95,
    -first_extreme_relative_I AS lead_time_extreme,

    max_exceedance_observed,

    -- Detection outcome at p95 baseline range
    CASE
        WHEN first_p95_relative_I IS NULL THEN 'NOT_DETECTED'
        WHEN first_p95_relative_I < 0 THEN 'EARLY_DETECTION'
        ELSE 'LATE_DETECTION'
    END AS outcome_p95

FROM v_metric_first_deviation;

-- =============================================================================
-- METRIC PERFORMANCE SUMMARY
-- =============================================================================

-- Aggregate: Which metric has best average lead time?
CREATE VIEW v_metric_performance AS
SELECT
    metric_name,
    label_name,
    COUNT(*) AS n_entities,

    -- Detection counts
    SUM(CASE WHEN outcome_p95 = 'EARLY_DETECTION' THEN 1 ELSE 0 END) AS n_early_detections,
    SUM(CASE WHEN outcome_p95 = 'NOT_DETECTED' THEN 1 ELSE 0 END) AS n_missed,

    -- Detection rate (%)
    ROUND(100.0 * SUM(CASE WHEN outcome_p95 = 'EARLY_DETECTION' THEN 1 ELSE 0 END) / COUNT(*), 1) AS detection_rate_pct,

    -- Lead time statistics (only for early detections)
    ROUND(AVG(lead_time_p95) FILTER (WHERE lead_time_p95 > 0), 1) AS avg_lead_time,
    ROUND(STDDEV(lead_time_p95) FILTER (WHERE lead_time_p95 > 0), 1) AS std_lead_time,
    MIN(lead_time_p95) FILTER (WHERE lead_time_p95 > 0) AS min_lead_time,
    MAX(lead_time_p95) FILTER (WHERE lead_time_p95 > 0) AS max_lead_time,

    -- Average max exceedance observed
    ROUND(AVG(max_exceedance_observed), 2) AS avg_max_exceedance

FROM v_metric_lead_times
GROUP BY metric_name, label_name
ORDER BY avg_lead_time DESC NULLS LAST;
