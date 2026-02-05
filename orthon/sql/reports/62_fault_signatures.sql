-- =============================================================================
-- FAULT SIGNATURE LEARNING
-- =============================================================================
-- Learn which metrics detect which fault types best.
-- This is the "AI" part - discovering patterns in detection performance.
--
-- Key output: For each fault_type, which metric has:
--   1. Highest detection rate
--   2. Longest lead time
--   3. Best combined score
--
-- =============================================================================

-- Drop existing views for clean reload
DROP VIEW IF EXISTS v_fault_type_labels;
DROP VIEW IF EXISTS v_fault_type_signatures;
DROP VIEW IF EXISTS v_best_detectors;
DROP VIEW IF EXISTS v_fault_signature_matrix;
DROP VIEW IF EXISTS v_recommended_metrics;

-- =============================================================================
-- FAULT TYPE EXTRACTION
-- =============================================================================

-- Extract fault type from labels (if available as separate column)
-- Or use label_name as fault type proxy
CREATE VIEW v_fault_type_labels AS
SELECT DISTINCT
    entity_id,
    label_name,
    -- Try to extract fault type from label_value if it contains type info
    CASE
        WHEN LOWER(label_value) LIKE '%valve%' THEN 'valve'
        WHEN LOWER(label_value) LIKE '%cavitation%' THEN 'cavitation'
        WHEN LOWER(label_value) LIKE '%imbalance%' THEN 'imbalance'
        WHEN LOWER(label_value) LIKE '%bearing%' THEN 'bearing'
        WHEN LOWER(label_value) LIKE '%thermal%' THEN 'thermal'
        WHEN LOWER(label_value) LIKE '%leak%' THEN 'leak'
        WHEN LOWER(label_value) LIKE '%blockage%' THEN 'blockage'
        ELSE label_name  -- Fall back to label_name as fault type
    END AS fault_type
FROM labels
WHERE label_value IN ('1', 'True', 'true', 'anomaly', 'fault', 'changepoint')
   OR LOWER(label_value) NOT IN ('0', 'false', 'normal', 'healthy');

-- =============================================================================
-- FAULT TYPE SIGNATURES
-- =============================================================================

-- Join fault type to metric performance
CREATE VIEW v_fault_type_signatures AS
SELECT
    COALESCE(ftl.fault_type, lt.label_name) AS fault_type,
    lt.metric_name,
    COUNT(*) AS n_entities,

    -- Detection metrics
    ROUND(100.0 * SUM(CASE WHEN lt.outcome_2sigma = 'EARLY_DETECTION' THEN 1 ELSE 0 END) / COUNT(*), 1) AS detection_rate_pct,
    ROUND(AVG(lt.lead_time_2sigma) FILTER (WHERE lt.lead_time_2sigma > 0), 1) AS avg_lead_time,
    ROUND(STDDEV(lt.lead_time_2sigma) FILTER (WHERE lt.lead_time_2sigma > 0), 1) AS std_lead_time,

    -- Combined score: (detection_rate * avg_lead_time) / 100
    -- Higher is better: captures both reliability and earliness
    ROUND(
        (100.0 * SUM(CASE WHEN lt.outcome_2sigma = 'EARLY_DETECTION' THEN 1 ELSE 0 END) / COUNT(*)) *
        COALESCE(AVG(lt.lead_time_2sigma) FILTER (WHERE lt.lead_time_2sigma > 0), 0) / 100.0
    , 2) AS combined_score

FROM v_metric_lead_times lt
LEFT JOIN v_fault_type_labels ftl ON lt.entity_id = ftl.entity_id AND lt.label_name = ftl.label_name
GROUP BY COALESCE(ftl.fault_type, lt.label_name), lt.metric_name
HAVING COUNT(*) >= 3;  -- Need at least 3 samples for reliable signature

-- =============================================================================
-- BEST DETECTOR PER FAULT TYPE
-- =============================================================================

-- Find the single best metric for each fault type
CREATE VIEW v_best_detectors AS
WITH ranked_metrics AS (
    SELECT
        fault_type,
        metric_name,
        detection_rate_pct,
        avg_lead_time,
        combined_score,
        n_entities,
        ROW_NUMBER() OVER (
            PARTITION BY fault_type
            ORDER BY combined_score DESC NULLS LAST
        ) AS rank_combined,
        ROW_NUMBER() OVER (
            PARTITION BY fault_type
            ORDER BY detection_rate_pct DESC NULLS LAST
        ) AS rank_detection,
        ROW_NUMBER() OVER (
            PARTITION BY fault_type
            ORDER BY avg_lead_time DESC NULLS LAST
        ) AS rank_lead_time
    FROM v_fault_type_signatures
)
SELECT
    fault_type,
    metric_name AS best_metric,
    detection_rate_pct,
    avg_lead_time,
    combined_score,
    n_entities,
    -- Alternative rankings
    rank_detection,
    rank_lead_time
FROM ranked_metrics
WHERE rank_combined = 1
ORDER BY fault_type;

-- =============================================================================
-- SIGNATURE MATRIX (HEATMAP DATA)
-- =============================================================================

-- Matrix view: fault_type x metric_name → combined_score
-- For visualization as a heatmap
CREATE VIEW v_fault_signature_matrix AS
SELECT
    fault_type,
    metric_name,
    detection_rate_pct,
    avg_lead_time,
    combined_score,
    -- Normalize score to 0-1 for heatmap coloring
    combined_score / NULLIF(MAX(combined_score) OVER (PARTITION BY fault_type), 0) AS normalized_score
FROM v_fault_type_signatures
ORDER BY fault_type, combined_score DESC;

-- =============================================================================
-- RECOMMENDED METRICS (OUTPUT)
-- =============================================================================

-- Final recommendations for configuration
CREATE VIEW v_recommended_metrics AS
SELECT
    'Primary Metrics' AS category,
    STRING_AGG(DISTINCT best_metric, ', ') AS metrics,
    COUNT(DISTINCT fault_type) AS n_fault_types_covered
FROM v_best_detectors

UNION ALL

SELECT
    'High Detection Rate (>80%)' AS category,
    STRING_AGG(DISTINCT metric_name, ', ') AS metrics,
    COUNT(DISTINCT fault_type) AS n_fault_types_covered
FROM v_fault_type_signatures
WHERE detection_rate_pct >= 80

UNION ALL

SELECT
    'Long Lead Time (>50 samples)' AS category,
    STRING_AGG(DISTINCT metric_name, ', ') AS metrics,
    COUNT(DISTINCT fault_type) AS n_fault_types_covered
FROM v_fault_type_signatures
WHERE avg_lead_time >= 50;
