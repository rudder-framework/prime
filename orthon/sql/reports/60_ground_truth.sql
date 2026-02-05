-- =============================================================================
-- GROUND TRUTH INFRASTRUCTURE
-- =============================================================================
-- Load ground truth labels and align with PRISM detection results.
-- This enables validation of detection timing against actual fault timestamps.
--
-- Usage:
--   Run after loading physics.parquet and labels.parquet
--
-- Inputs:
--   - labels table (from labels.parquet)
--   - physics table (from PRISM physics.parquet)
--
-- =============================================================================

-- Drop existing views/tables for clean reload
DROP VIEW IF EXISTS v_fault_times;
DROP VIEW IF EXISTS v_label_summary;
DROP VIEW IF EXISTS v_first_deviation;
DROP VIEW IF EXISTS v_detection_vs_truth;
DROP VIEW IF EXISTS v_detection_outcome_summary;

-- =============================================================================
-- LABEL ANALYSIS
-- =============================================================================

-- Summary of available labels
CREATE VIEW v_label_summary AS
SELECT
    label_name,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT entity_id) AS n_entities,
    COUNT(DISTINCT label_value) AS n_unique_values,
    STRING_AGG(DISTINCT label_value, ', ') AS unique_values
FROM labels
GROUP BY label_name
ORDER BY label_name;

-- =============================================================================
-- FAULT TIMESTAMPS
-- =============================================================================

-- Extract fault start time per entity
-- A fault starts when label transitions to 1/'anomaly'/'fault'/etc.
CREATE VIEW v_fault_times AS
SELECT
    entity_id,
    label_name,
    -- First occurrence of fault indicator
    MIN(I) FILTER (WHERE
        label_value = '1' OR
        label_value = 'True' OR
        label_value = 'true' OR
        LOWER(label_value) = 'anomaly' OR
        LOWER(label_value) = 'fault' OR
        LOWER(label_value) = 'changepoint'
    ) AS fault_start_I,
    -- Last timestamp in data
    MAX(I) AS experiment_end_I,
    -- Count of fault samples
    COUNT(*) FILTER (WHERE
        label_value = '1' OR
        label_value = 'True' OR
        label_value = 'true' OR
        LOWER(label_value) = 'anomaly' OR
        LOWER(label_value) = 'fault' OR
        LOWER(label_value) = 'changepoint'
    ) AS n_fault_samples,
    -- Total samples
    COUNT(*) AS n_total_samples
FROM labels
GROUP BY entity_id, label_name;

-- =============================================================================
-- PRISM DETECTION TIMESTAMPS
-- =============================================================================

-- First deviation detected by PRISM per entity
-- Uses z_total (overall deviation) from baseline_deviation
CREATE VIEW v_first_deviation AS
WITH deviation_with_threshold AS (
    SELECT
        entity_id,
        I,
        z_total,
        -- Different threshold levels
        CASE WHEN ABS(z_total) > 2.0 THEN 1 ELSE 0 END AS exceeds_2sigma,
        CASE WHEN ABS(z_total) > 2.5 THEN 1 ELSE 0 END AS exceeds_2_5sigma,
        CASE WHEN ABS(z_total) > 3.0 THEN 1 ELSE 0 END AS exceeds_3sigma
    FROM baseline_deviation
    WHERE z_total IS NOT NULL
)
SELECT
    entity_id,
    MIN(I) FILTER (WHERE exceeds_2sigma = 1) AS first_2sigma_I,
    MIN(I) FILTER (WHERE exceeds_2_5sigma = 1) AS first_2_5sigma_I,
    MIN(I) FILTER (WHERE exceeds_3sigma = 1) AS first_3sigma_I,
    MAX(ABS(z_total)) AS max_z_total
FROM deviation_with_threshold
GROUP BY entity_id;

-- =============================================================================
-- DETECTION VS GROUND TRUTH ALIGNMENT
-- =============================================================================

-- Compare PRISM detection timing to actual fault timestamps
CREATE VIEW v_detection_vs_truth AS
SELECT
    f.entity_id,
    f.label_name,
    f.fault_start_I AS actual_fault_I,
    d.first_2sigma_I,
    d.first_2_5sigma_I,
    d.first_3sigma_I,
    d.max_z_total,

    -- Lead time at each threshold (positive = early detection, negative = late)
    f.fault_start_I - d.first_2sigma_I AS lead_time_2sigma,
    f.fault_start_I - d.first_2_5sigma_I AS lead_time_2_5sigma,
    f.fault_start_I - d.first_3sigma_I AS lead_time_3sigma,

    -- Detection outcome at 2-sigma threshold
    CASE
        WHEN d.first_2sigma_I IS NULL THEN 'MISSED'
        WHEN d.first_2sigma_I < f.fault_start_I THEN 'EARLY_DETECTION'
        WHEN d.first_2sigma_I = f.fault_start_I THEN 'ON_TIME'
        ELSE 'LATE_DETECTION'
    END AS outcome_2sigma,

    -- Detection outcome at 2.5-sigma threshold
    CASE
        WHEN d.first_2_5sigma_I IS NULL THEN 'MISSED'
        WHEN d.first_2_5sigma_I < f.fault_start_I THEN 'EARLY_DETECTION'
        WHEN d.first_2_5sigma_I = f.fault_start_I THEN 'ON_TIME'
        ELSE 'LATE_DETECTION'
    END AS outcome_2_5sigma,

    -- Detection outcome at 3-sigma threshold
    CASE
        WHEN d.first_3sigma_I IS NULL THEN 'MISSED'
        WHEN d.first_3sigma_I < f.fault_start_I THEN 'EARLY_DETECTION'
        WHEN d.first_3sigma_I = f.fault_start_I THEN 'ON_TIME'
        ELSE 'LATE_DETECTION'
    END AS outcome_3sigma

FROM v_fault_times f
LEFT JOIN v_first_deviation d ON f.entity_id = d.entity_id
WHERE f.fault_start_I IS NOT NULL;

-- =============================================================================
-- DETECTION OUTCOME SUMMARY
-- =============================================================================

-- Overall detection performance
CREATE VIEW v_detection_outcome_summary AS
SELECT
    label_name,
    COUNT(*) AS n_entities,

    -- 2-sigma performance
    SUM(CASE WHEN outcome_2sigma = 'EARLY_DETECTION' THEN 1 ELSE 0 END) AS early_2sigma,
    SUM(CASE WHEN outcome_2sigma = 'ON_TIME' THEN 1 ELSE 0 END) AS ontime_2sigma,
    SUM(CASE WHEN outcome_2sigma = 'LATE_DETECTION' THEN 1 ELSE 0 END) AS late_2sigma,
    SUM(CASE WHEN outcome_2sigma = 'MISSED' THEN 1 ELSE 0 END) AS missed_2sigma,
    ROUND(100.0 * SUM(CASE WHEN outcome_2sigma IN ('EARLY_DETECTION', 'ON_TIME') THEN 1 ELSE 0 END) / COUNT(*), 1) AS detection_rate_2sigma,

    -- 2.5-sigma performance
    SUM(CASE WHEN outcome_2_5sigma = 'EARLY_DETECTION' THEN 1 ELSE 0 END) AS early_2_5sigma,
    SUM(CASE WHEN outcome_2_5sigma = 'MISSED' THEN 1 ELSE 0 END) AS missed_2_5sigma,
    ROUND(100.0 * SUM(CASE WHEN outcome_2_5sigma IN ('EARLY_DETECTION', 'ON_TIME') THEN 1 ELSE 0 END) / COUNT(*), 1) AS detection_rate_2_5sigma,

    -- 3-sigma performance
    SUM(CASE WHEN outcome_3sigma = 'EARLY_DETECTION' THEN 1 ELSE 0 END) AS early_3sigma,
    SUM(CASE WHEN outcome_3sigma = 'MISSED' THEN 1 ELSE 0 END) AS missed_3sigma,
    ROUND(100.0 * SUM(CASE WHEN outcome_3sigma IN ('EARLY_DETECTION', 'ON_TIME') THEN 1 ELSE 0 END) / COUNT(*), 1) AS detection_rate_3sigma,

    -- Lead time stats (at 2-sigma)
    ROUND(AVG(lead_time_2sigma) FILTER (WHERE lead_time_2sigma > 0), 1) AS avg_lead_time_2sigma,
    ROUND(STDDEV(lead_time_2sigma) FILTER (WHERE lead_time_2sigma > 0), 1) AS std_lead_time_2sigma,
    MIN(lead_time_2sigma) FILTER (WHERE lead_time_2sigma > 0) AS min_lead_time_2sigma,
    MAX(lead_time_2sigma) FILTER (WHERE lead_time_2sigma > 0) AS max_lead_time_2sigma

FROM v_detection_vs_truth
GROUP BY label_name;
