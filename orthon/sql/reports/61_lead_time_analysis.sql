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

-- Normalize I so that fault_start_I = 0 (negative = before fault)
CREATE VIEW v_metric_aligned_to_fault AS
SELECT
    p.entity_id,
    p.signal_id,
    p.metric_name,
    p.I,
    f.fault_start_I,
    f.label_name,
    p.I - f.fault_start_I AS I_relative,  -- Negative = before fault
    p.metric_value,
    p.z_score,
    p.percentile
FROM (
    -- Pivot physics metrics to long format
    SELECT entity_id, signal_id, I,
           'coherence' AS metric_name, coherence AS metric_value, z_coherence AS z_score, pct_coherence AS percentile
    FROM physics WHERE coherence IS NOT NULL
    UNION ALL
    SELECT entity_id, signal_id, I,
           'entropy' AS metric_name, entropy AS metric_value, z_entropy AS z_score, pct_entropy AS percentile
    FROM physics WHERE entropy IS NOT NULL
    UNION ALL
    SELECT entity_id, signal_id, I,
           'lyapunov' AS metric_name, lyapunov AS metric_value, z_lyapunov AS z_score, pct_lyapunov AS percentile
    FROM physics WHERE lyapunov IS NOT NULL
    UNION ALL
    SELECT entity_id, signal_id, I,
           'hurst' AS metric_name, hurst AS metric_value, z_hurst AS z_score, pct_hurst AS percentile
    FROM physics WHERE hurst IS NOT NULL
    UNION ALL
    SELECT entity_id, signal_id, I,
           'energy_total' AS metric_name, energy_total AS metric_value, NULL AS z_score, NULL AS percentile
    FROM physics WHERE energy_total IS NOT NULL
    UNION ALL
    SELECT entity_id, signal_id, I,
           'dissipation_rate' AS metric_name, dissipation_rate AS metric_value, NULL AS z_score, NULL AS percentile
    FROM physics WHERE dissipation_rate IS NOT NULL
) p
JOIN v_fault_times f ON p.entity_id = f.entity_id
WHERE f.fault_start_I IS NOT NULL;

-- =============================================================================
-- ESTABLISH BASELINE (PRE-FAULT STABLE PERIOD)
-- =============================================================================

-- Baseline stats per entity per metric
-- Use I_relative between -500 and -100 as "stable" pre-fault period
CREATE VIEW v_metric_baseline AS
SELECT
    entity_id,
    signal_id,
    metric_name,
    label_name,
    AVG(metric_value) AS baseline_mean,
    STDDEV(metric_value) AS baseline_std,
    MIN(metric_value) AS baseline_min,
    MAX(metric_value) AS baseline_max,
    COUNT(*) AS baseline_n
FROM v_metric_aligned_to_fault
WHERE I_relative BETWEEN -500 AND -100  -- Well before fault
GROUP BY entity_id, signal_id, metric_name, label_name
HAVING COUNT(*) >= 10;  -- Need enough samples for reliable baseline

-- =============================================================================
-- FIND FIRST DEVIATION (LEAD TIME)
-- =============================================================================

-- First significant deviation from baseline for each metric
CREATE VIEW v_metric_first_deviation AS
SELECT
    a.entity_id,
    a.signal_id,
    a.metric_name,
    a.label_name,
    b.baseline_mean,
    b.baseline_std,

    -- First 2-sigma deviation (using raw values)
    MIN(a.I_relative) FILTER (WHERE
        b.baseline_std > 0 AND
        ABS(a.metric_value - b.baseline_mean) / b.baseline_std > 2.0 AND
        a.I_relative < 0
    ) AS first_2sigma_relative_I,

    -- First 2.5-sigma deviation
    MIN(a.I_relative) FILTER (WHERE
        b.baseline_std > 0 AND
        ABS(a.metric_value - b.baseline_mean) / b.baseline_std > 2.5 AND
        a.I_relative < 0
    ) AS first_2_5sigma_relative_I,

    -- First 3-sigma deviation
    MIN(a.I_relative) FILTER (WHERE
        b.baseline_std > 0 AND
        ABS(a.metric_value - b.baseline_mean) / b.baseline_std > 3.0 AND
        a.I_relative < 0
    ) AS first_3sigma_relative_I,

    -- Max z-score observed
    MAX(ABS((a.metric_value - b.baseline_mean) / NULLIF(b.baseline_std, 0))) AS max_z_observed

FROM v_metric_aligned_to_fault a
JOIN v_metric_baseline b ON
    a.entity_id = b.entity_id AND
    a.signal_id = b.signal_id AND
    a.metric_name = b.metric_name AND
    a.label_name = b.label_name
GROUP BY a.entity_id, a.signal_id, a.metric_name, a.label_name, b.baseline_mean, b.baseline_std;

-- =============================================================================
-- LEAD TIME RESULTS
-- =============================================================================

-- Convert relative I to lead time (positive = early detection)
CREATE VIEW v_metric_lead_times AS
SELECT
    entity_id,
    signal_id,
    metric_name,
    label_name,
    baseline_mean,
    baseline_std,

    -- Lead times (negate relative I to get positive lead time)
    -first_2sigma_relative_I AS lead_time_2sigma,
    -first_2_5sigma_relative_I AS lead_time_2_5sigma,
    -first_3sigma_relative_I AS lead_time_3sigma,

    max_z_observed,

    -- Detection outcome at 2-sigma
    CASE
        WHEN first_2sigma_relative_I IS NULL THEN 'NOT_DETECTED'
        WHEN first_2sigma_relative_I < 0 THEN 'EARLY_DETECTION'
        ELSE 'LATE_DETECTION'
    END AS outcome_2sigma

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
    SUM(CASE WHEN outcome_2sigma = 'EARLY_DETECTION' THEN 1 ELSE 0 END) AS n_early_detections,
    SUM(CASE WHEN outcome_2sigma = 'NOT_DETECTED' THEN 1 ELSE 0 END) AS n_missed,

    -- Detection rate (%)
    ROUND(100.0 * SUM(CASE WHEN outcome_2sigma = 'EARLY_DETECTION' THEN 1 ELSE 0 END) / COUNT(*), 1) AS detection_rate_pct,

    -- Lead time statistics (only for early detections)
    ROUND(AVG(lead_time_2sigma) FILTER (WHERE lead_time_2sigma > 0), 1) AS avg_lead_time,
    ROUND(STDDEV(lead_time_2sigma) FILTER (WHERE lead_time_2sigma > 0), 1) AS std_lead_time,
    MIN(lead_time_2sigma) FILTER (WHERE lead_time_2sigma > 0) AS min_lead_time,
    MAX(lead_time_2sigma) FILTER (WHERE lead_time_2sigma > 0) AS max_lead_time,

    -- Average max z-score observed
    ROUND(AVG(max_z_observed), 2) AS avg_max_z

FROM v_metric_lead_times
GROUP BY metric_name, label_name
ORDER BY avg_lead_time DESC NULLS LAST;
