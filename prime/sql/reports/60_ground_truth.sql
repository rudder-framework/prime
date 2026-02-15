-- =============================================================================
-- GROUND TRUTH INFRASTRUCTURE — Ranked Views
-- =============================================================================
-- Load ground truth labels and align with Engines detection results.
-- Replaces hardcoded sigma thresholds with percentile-ranked deviations.
--
-- Usage:
--   Run after loading physics.parquet and labels.parquet
--
-- Inputs:
--   - labels table (from labels.parquet)
--   - physics table (from Engines physics.parquet)
--
-- =============================================================================

-- Drop existing views/tables for clean reload
DROP VIEW IF EXISTS v_fault_times;
DROP VIEW IF EXISTS v_label_summary;
DROP VIEW IF EXISTS v_first_deviation;
DROP VIEW IF EXISTS v_deviation_ranked;
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
    COUNT(DISTINCT cohort) AS n_entities,
    COUNT(DISTINCT label_value) AS n_unique_values,
    STRING_AGG(DISTINCT label_value, ', ') AS unique_values
FROM labels
GROUP BY label_name
ORDER BY label_name;

-- =============================================================================
-- FAULT TIMESTAMPS
-- =============================================================================

-- Extract fault start time per entity
CREATE VIEW v_fault_times AS
SELECT
    cohort,
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
GROUP BY cohort, label_name;

-- =============================================================================
-- DEVIATION RANKED (replaces hardcoded sigma thresholds)
-- =============================================================================

-- Rank all deviations by magnitude — no hardcoded thresholds
CREATE VIEW v_deviation_ranked AS
SELECT
    cohort,
    I,
    deviation_score,

    -- Percentile of this deviation_score within the cohort's history
    PERCENT_RANK() OVER (
        PARTITION BY cohort
        ORDER BY deviation_score
    ) AS deviation_pctile,

    -- Rank within this cohort (most extreme first)
    RANK() OVER (
        PARTITION BY cohort
        ORDER BY deviation_score DESC
    ) AS deviation_rank,

    -- Fleet-wide rank at this timestep
    RANK() OVER (
        PARTITION BY I
        ORDER BY deviation_score DESC
    ) AS fleet_deviation_rank,

    -- Rank cohorts by when they first showed large deviation
    -- (using top-5% of their own distribution as "large")
    RANK() OVER (
        ORDER BY I ASC
    ) AS earliest_deviation_rank

FROM baseline_deviation
WHERE deviation_score IS NOT NULL;

-- =============================================================================
-- FIRST DEVIATION (percentile-based, replaces sigma thresholds)
-- =============================================================================

-- First time each cohort exceeded its own Nth percentile
CREATE VIEW v_first_deviation AS
SELECT
    cohort,
    MIN(I) FILTER (WHERE deviation_pctile > 0.90) AS first_p90_I,
    MIN(I) FILTER (WHERE deviation_pctile > 0.95) AS first_p95_I,
    MIN(I) FILTER (WHERE deviation_pctile > 0.99) AS first_p99_I,
    MAX(deviation_score) AS max_deviation_score
FROM v_deviation_ranked
GROUP BY cohort;

-- =============================================================================
-- DETECTION VS GROUND TRUTH ALIGNMENT
-- =============================================================================

-- Compare Engines detection timing to actual fault timestamps
CREATE VIEW v_detection_vs_truth AS
SELECT
    f.cohort,
    f.label_name,
    f.fault_start_I AS actual_fault_I,
    d.first_p90_I,
    d.first_p95_I,
    d.first_p99_I,
    d.max_deviation_score,

    -- Lead time at each percentile threshold (positive = early detection, negative = late)
    f.fault_start_I - d.first_p90_I AS lead_time_p90,
    f.fault_start_I - d.first_p95_I AS lead_time_p95,
    f.fault_start_I - d.first_p99_I AS lead_time_p99,

    -- Detection outcome at p90 threshold
    CASE
        WHEN d.first_p90_I IS NULL THEN 'MISSED'
        WHEN d.first_p90_I < f.fault_start_I THEN 'EARLY_DETECTION'
        WHEN d.first_p90_I = f.fault_start_I THEN 'ON_TIME'
        ELSE 'LATE_DETECTION'
    END AS outcome_p90,

    -- Detection outcome at p95 threshold
    CASE
        WHEN d.first_p95_I IS NULL THEN 'MISSED'
        WHEN d.first_p95_I < f.fault_start_I THEN 'EARLY_DETECTION'
        WHEN d.first_p95_I = f.fault_start_I THEN 'ON_TIME'
        ELSE 'LATE_DETECTION'
    END AS outcome_p95,

    -- Detection outcome at p99 threshold
    CASE
        WHEN d.first_p99_I IS NULL THEN 'MISSED'
        WHEN d.first_p99_I < f.fault_start_I THEN 'EARLY_DETECTION'
        WHEN d.first_p99_I = f.fault_start_I THEN 'ON_TIME'
        ELSE 'LATE_DETECTION'
    END AS outcome_p99

FROM v_fault_times f
LEFT JOIN v_first_deviation d ON f.cohort = d.cohort
WHERE f.fault_start_I IS NOT NULL;

-- =============================================================================
-- DETECTION OUTCOME SUMMARY
-- =============================================================================

-- Overall detection performance
CREATE VIEW v_detection_outcome_summary AS
SELECT
    label_name,
    COUNT(*) AS n_entities,

    -- p90 performance
    SUM(CASE WHEN outcome_p90 = 'EARLY_DETECTION' THEN 1 ELSE 0 END) AS early_p90,
    SUM(CASE WHEN outcome_p90 = 'ON_TIME' THEN 1 ELSE 0 END) AS ontime_p90,
    SUM(CASE WHEN outcome_p90 = 'LATE_DETECTION' THEN 1 ELSE 0 END) AS late_p90,
    SUM(CASE WHEN outcome_p90 = 'MISSED' THEN 1 ELSE 0 END) AS missed_p90,
    ROUND(100.0 * SUM(CASE WHEN outcome_p90 IN ('EARLY_DETECTION', 'ON_TIME') THEN 1 ELSE 0 END) / COUNT(*), 1) AS detection_rate_p90,

    -- p95 performance
    SUM(CASE WHEN outcome_p95 = 'EARLY_DETECTION' THEN 1 ELSE 0 END) AS early_p95,
    SUM(CASE WHEN outcome_p95 = 'MISSED' THEN 1 ELSE 0 END) AS missed_p95,
    ROUND(100.0 * SUM(CASE WHEN outcome_p95 IN ('EARLY_DETECTION', 'ON_TIME') THEN 1 ELSE 0 END) / COUNT(*), 1) AS detection_rate_p95,

    -- p99 performance
    SUM(CASE WHEN outcome_p99 = 'EARLY_DETECTION' THEN 1 ELSE 0 END) AS early_p99,
    SUM(CASE WHEN outcome_p99 = 'MISSED' THEN 1 ELSE 0 END) AS missed_p99,
    ROUND(100.0 * SUM(CASE WHEN outcome_p99 IN ('EARLY_DETECTION', 'ON_TIME') THEN 1 ELSE 0 END) / COUNT(*), 1) AS detection_rate_p99,

    -- Lead time stats (at p90)
    ROUND(AVG(lead_time_p90) FILTER (WHERE lead_time_p90 > 0), 1) AS avg_lead_time_p90,
    ROUND(STDDEV(lead_time_p90) FILTER (WHERE lead_time_p90 > 0), 1) AS std_lead_time_p90,
    MIN(lead_time_p90) FILTER (WHERE lead_time_p90 > 0) AS min_lead_time_p90,
    MAX(lead_time_p90) FILTER (WHERE lead_time_p90 > 0) AS max_lead_time_p90

FROM v_detection_vs_truth
GROUP BY label_name;
