-- =============================================================================
-- THRESHOLD OPTIMIZATION
-- =============================================================================
-- Find the optimal z-threshold that maximizes early detection rate.
-- Tests multiple thresholds and computes detection metrics at each.
--
-- Output: ROC-like curve data for threshold selection
--
-- =============================================================================

-- Drop existing views for clean reload
DROP VIEW IF EXISTS v_threshold_candidates;
DROP VIEW IF EXISTS v_threshold_detection;
DROP VIEW IF EXISTS v_threshold_performance;
DROP VIEW IF EXISTS v_optimal_threshold;
DROP VIEW IF EXISTS v_threshold_curve;

-- =============================================================================
-- THRESHOLD CANDIDATES
-- =============================================================================

-- Generate candidate thresholds to test
CREATE VIEW v_threshold_candidates AS
SELECT unnest([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]) AS z_threshold;

-- =============================================================================
-- DETECTION AT EACH THRESHOLD
-- =============================================================================

-- For each threshold, compute detection time per entity
CREATE VIEW v_threshold_detection AS
WITH aligned_metrics AS (
    -- Get z-scores from baseline_deviation aligned to fault times
    SELECT
        bd.entity_id,
        bd.I,
        ft.fault_start_I,
        bd.I - ft.fault_start_I AS I_relative,
        bd.z_total
    FROM baseline_deviation bd
    JOIN v_fault_times ft ON bd.entity_id = ft.entity_id
    WHERE bd.z_total IS NOT NULL
      AND ft.fault_start_I IS NOT NULL
)
SELECT
    tc.z_threshold,
    am.entity_id,
    am.fault_start_I,
    -- First detection at this threshold (before fault)
    MIN(am.I) FILTER (WHERE ABS(am.z_total) > tc.z_threshold AND am.I_relative < 0) AS first_detection_I,
    -- Lead time if detected before fault
    -(MIN(am.I_relative) FILTER (WHERE ABS(am.z_total) > tc.z_threshold AND am.I_relative < 0)) AS lead_time,
    -- Max z observed
    MAX(ABS(am.z_total)) AS max_z_observed
FROM aligned_metrics am
CROSS JOIN v_threshold_candidates tc
GROUP BY tc.z_threshold, am.entity_id, am.fault_start_I;

-- =============================================================================
-- PERFORMANCE AT EACH THRESHOLD
-- =============================================================================

-- Aggregate detection performance at each threshold
CREATE VIEW v_threshold_performance AS
SELECT
    z_threshold,
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

    -- Average max z-score (sanity check)
    ROUND(AVG(max_z_observed), 2) AS avg_max_z

FROM v_threshold_detection
GROUP BY z_threshold
ORDER BY z_threshold;

-- =============================================================================
-- OPTIMAL THRESHOLD SELECTION
-- =============================================================================

-- Find the best threshold based on different criteria
CREATE VIEW v_optimal_threshold AS
WITH ranked_thresholds AS (
    SELECT
        z_threshold,
        detection_rate_pct,
        avg_lead_time,
        miss_rate_pct,
        true_positives,
        false_negatives,
        n_entities,

        -- F1-like score: 2 * (detection_rate * (1 - miss_rate/100)) / (detection_rate + (1 - miss_rate/100))
        -- Simplified: just use detection_rate - penalty for misses
        detection_rate_pct - (miss_rate_pct * 0.5) AS adjusted_score,

        -- Combined score weighting lead time
        detection_rate_pct * COALESCE(avg_lead_time, 0) / 100.0 AS lead_time_weighted_score

    FROM v_threshold_performance
    WHERE n_entities > 0
)
SELECT
    'highest_detection_rate' AS criterion,
    z_threshold AS optimal_z,
    detection_rate_pct,
    avg_lead_time,
    miss_rate_pct
FROM ranked_thresholds
ORDER BY detection_rate_pct DESC, z_threshold ASC
LIMIT 1

UNION ALL

SELECT
    'best_balanced' AS criterion,
    z_threshold AS optimal_z,
    detection_rate_pct,
    avg_lead_time,
    miss_rate_pct
FROM ranked_thresholds
ORDER BY adjusted_score DESC, z_threshold ASC
LIMIT 1

UNION ALL

SELECT
    'longest_lead_time_80pct_detection' AS criterion,
    z_threshold AS optimal_z,
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

-- Data for ROC-like threshold curve visualization
CREATE VIEW v_threshold_curve AS
SELECT
    z_threshold,
    detection_rate_pct AS sensitivity,
    miss_rate_pct AS miss_rate,
    100.0 - miss_rate_pct AS specificity_proxy,  -- Not true specificity without negative examples
    avg_lead_time,
    true_positives,
    false_negatives,
    -- Cumulative metrics
    SUM(true_positives) OVER (ORDER BY z_threshold) AS cumulative_tp,
    SUM(false_negatives) OVER (ORDER BY z_threshold DESC) AS cumulative_fn
FROM v_threshold_performance
ORDER BY z_threshold;
