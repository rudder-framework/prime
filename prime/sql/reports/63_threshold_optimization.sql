-- =============================================================================
-- THRESHOLD OPTIMIZATION — Percentile-Based Candidates
-- =============================================================================
-- Test at every 5th percentile of observed deviation_scores.
-- The data defines the thresholds, not the analyst.
--
-- Output: ROC-like curve data for threshold selection
-- =============================================================================

-- Drop existing views for clean reload
DROP VIEW IF EXISTS v_threshold_candidates;
DROP VIEW IF EXISTS v_threshold_detection;
DROP VIEW IF EXISTS v_threshold_performance;
DROP VIEW IF EXISTS v_optimal_threshold;
DROP VIEW IF EXISTS v_threshold_curve;

-- =============================================================================
-- THRESHOLD CANDIDATES (data-driven from observed distribution)
-- =============================================================================

CREATE VIEW v_threshold_candidates AS
WITH deviation_percentiles AS (
    SELECT
        PERCENTILE_CONT(p / 100.0) WITHIN GROUP (ORDER BY deviation_score) AS deviation_threshold,
        p AS percentile_level
    FROM baseline_deviation
    CROSS JOIN generate_series(5, 95, 5) AS t(p)
    WHERE deviation_score IS NOT NULL
    GROUP BY p
)
SELECT DISTINCT
    ROUND(deviation_threshold, 4) AS deviation_threshold,
    percentile_level
FROM deviation_percentiles
WHERE deviation_threshold > 0
ORDER BY deviation_threshold;

-- =============================================================================
-- DETECTION AT EACH THRESHOLD
-- =============================================================================

-- For each threshold, compute detection time per entity
CREATE VIEW v_threshold_detection AS
WITH aligned_metrics AS (
    -- Get deviation_scores from baseline_deviation aligned to fault times
    SELECT
        bd.cohort,
        bd.signal_0_center,
        ft.fault_start_I,
        bd.signal_0_center - ft.fault_start_I AS I_relative,
        bd.deviation_score
    FROM baseline_deviation bd
    JOIN v_fault_times ft ON bd.cohort = ft.cohort
    WHERE bd.deviation_score IS NOT NULL
      AND ft.fault_start_I IS NOT NULL
)
SELECT
    tc.deviation_threshold,
    tc.percentile_level,
    am.cohort,
    am.fault_start_I,
    -- First detection at this threshold (before fault)
    MIN(am.signal_0_center) FILTER (WHERE am.deviation_score > tc.deviation_threshold AND am.I_relative < 0) AS first_detection_I,
    -- Lead time if detected before fault
    -(MIN(am.I_relative) FILTER (WHERE am.deviation_score > tc.deviation_threshold AND am.I_relative < 0)) AS lead_time,
    -- Max deviation observed
    MAX(am.deviation_score) AS max_deviation_observed
FROM aligned_metrics am
CROSS JOIN v_threshold_candidates tc
GROUP BY tc.deviation_threshold, tc.percentile_level, am.cohort, am.fault_start_I;

-- =============================================================================
-- PERFORMANCE AT EACH THRESHOLD (ranked)
-- =============================================================================

CREATE VIEW v_threshold_performance AS
SELECT
    deviation_threshold,
    percentile_level,
    COUNT(*) AS n_entities,

    -- True positives: detected before fault
    SUM(CASE WHEN lead_time > 0 THEN 1 ELSE 0 END) AS true_positives,

    -- False negatives: not detected or detected after fault
    SUM(CASE WHEN lead_time IS NULL OR lead_time <= 0 THEN 1 ELSE 0 END) AS false_negatives,

    -- Detection rate (sensitivity / recall)
    ROUND(100.0 * SUM(CASE WHEN lead_time > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS detection_rate_pct,

    -- Miss rate
    ROUND(100.0 * SUM(CASE WHEN lead_time IS NULL OR lead_time <= 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS miss_rate_pct,

    -- Lead time statistics (for successful detections)
    ROUND(AVG(lead_time) FILTER (WHERE lead_time > 0), 1) AS avg_lead_time,
    ROUND(STDDEV(lead_time) FILTER (WHERE lead_time > 0), 1) AS std_lead_time,
    MIN(lead_time) FILTER (WHERE lead_time > 0) AS min_lead_time,
    MAX(lead_time) FILTER (WHERE lead_time > 0) AS max_lead_time,

    -- Average max deviation (sanity check)
    ROUND(AVG(max_deviation_observed), 4) AS avg_max_deviation,

    -- Rank thresholds by detection rate
    RANK() OVER (ORDER BY SUM(CASE WHEN lead_time > 0 THEN 1 ELSE 0 END) DESC) AS detection_rate_rank,

    -- Rank thresholds by lead time (for those with good detection)
    RANK() OVER (ORDER BY AVG(lead_time) FILTER (WHERE lead_time > 0) DESC NULLS LAST) AS lead_time_rank

FROM v_threshold_detection
GROUP BY deviation_threshold, percentile_level
ORDER BY deviation_threshold;

-- =============================================================================
-- OPTIMAL THRESHOLD SELECTION (ranked)
-- =============================================================================

CREATE VIEW v_optimal_threshold AS
WITH ranked_thresholds AS (
    SELECT
        deviation_threshold,
        percentile_level,
        detection_rate_pct,
        avg_lead_time,
        miss_rate_pct,
        true_positives,
        false_negatives,
        n_entities,

        -- Adjusted score: detection rate minus penalty for misses
        detection_rate_pct - (miss_rate_pct * 0.5) AS adjusted_score,

        -- Combined score weighting lead time
        detection_rate_pct * COALESCE(avg_lead_time, 0) / 100.0 AS lead_time_weighted_score

    FROM v_threshold_performance
    WHERE n_entities > 0
)
SELECT
    'highest_detection_rate' AS criterion,
    deviation_threshold AS optimal_deviation,
    percentile_level,
    detection_rate_pct,
    avg_lead_time,
    miss_rate_pct
FROM ranked_thresholds
ORDER BY detection_rate_pct DESC, deviation_threshold ASC
LIMIT 1

UNION ALL

SELECT
    'best_balanced' AS criterion,
    deviation_threshold AS optimal_deviation,
    percentile_level,
    detection_rate_pct,
    avg_lead_time,
    miss_rate_pct
FROM ranked_thresholds
ORDER BY adjusted_score DESC, deviation_threshold ASC
LIMIT 1

UNION ALL

SELECT
    'longest_lead_time_80pct_detection' AS criterion,
    deviation_threshold AS optimal_deviation,
    percentile_level,
    detection_rate_pct,
    avg_lead_time,
    miss_rate_pct
FROM ranked_thresholds
WHERE detection_rate_pct >= 80
ORDER BY avg_lead_time DESC
LIMIT 1;

-- =============================================================================
-- THRESHOLD CURVE (FOR VISUALIZATION)
-- =============================================================================

CREATE VIEW v_threshold_curve AS
SELECT
    deviation_threshold,
    percentile_level,
    detection_rate_pct AS sensitivity,
    miss_rate_pct AS miss_rate,
    100.0 - miss_rate_pct AS specificity_proxy,
    avg_lead_time,
    true_positives,
    false_negatives,
    -- Cumulative metrics
    SUM(true_positives) OVER (ORDER BY deviation_threshold) AS cumulative_tp,
    SUM(false_negatives) OVER (ORDER BY deviation_threshold DESC) AS cumulative_fn
FROM v_threshold_performance
ORDER BY deviation_threshold;
