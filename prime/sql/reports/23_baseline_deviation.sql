-- ============================================================================
-- Rudder SQL: 23_baseline_deviation.sql
-- ============================================================================
-- BASELINE & DEVIATION: Self-Referential Anomaly Detection
--
-- The Question: "Is this still normal?"
--
-- Phase 1 — Baseline:
--   Each entity. Each metric. Establish normal from first 10% of life.
--   Percentile bounds define the range. No Gaussian assumptions.
--
-- Phase 2 — Monitor:
--   Every new window. Compare to baseline.
--   Ratios and shifts — how has the trajectory changed?
--
-- Phase 3 — Flag:
--   Metric outside its own baseline range? Flag it.
--   Fleet percentile rank determines severity.
--
-- No z-scores. No sigma thresholds. No distribution assumptions.
-- ============================================================================

-- ============================================================================
-- CONFIGURATION
-- ============================================================================

CREATE OR REPLACE TABLE config_baseline AS
SELECT
    -- Baseline training window (first N% of data or first N points)
    0.10  AS baseline_pct,           -- Use first 10% of data for baseline
    100   AS baseline_min_points,    -- Minimum points for baseline

    -- Percentile bounds for baseline range
    0.05  AS percentile_low,         -- 5th percentile
    0.95  AS percentile_high,        -- 95th percentile

    -- Sensitivity: how many metrics must deviate to flag entity
    1     AS min_deviations_to_flag  -- Just one is enough
;


-- ============================================================================
-- PHASE 1: ESTABLISH BASELINES
-- ============================================================================
-- Each entity. Each metric. Define what "normal" looks like.

-- Determine baseline window per entity
CREATE OR REPLACE VIEW v_baseline_windows AS
SELECT
    cohort,
    MIN(I) AS I_min,
    MAX(I) AS I_max,
    COUNT(*) AS n_total,
    GREATEST(
        CAST(COUNT(*) * (SELECT baseline_pct FROM config_baseline) AS INTEGER),
        (SELECT baseline_min_points FROM config_baseline)
    ) AS n_baseline,
    MIN(I) + GREATEST(
        CAST(COUNT(*) * (SELECT baseline_pct FROM config_baseline) AS INTEGER),
        (SELECT baseline_min_points FROM config_baseline)
    ) AS baseline_end_I
FROM physics
GROUP BY cohort;


-- Baseline statistics per entity per metric
-- Includes percentile bounds for all metrics (for range-based flagging)
CREATE OR REPLACE TABLE baselines AS
WITH baseline_data AS (
    SELECT
        p.cohort,
        p.I,
        -- L4: Energy metrics
        p.energy_proxy,
        p.energy_velocity,
        p.dissipation_rate,
        -- L2: Coherence metrics
        p.coherence,
        p.coherence_velocity,
        p.effective_dim,
        p.eigenvalue_entropy,
        -- L1: State metrics
        p.state_distance,
        p.state_velocity,
        p.state_acceleration
    FROM physics p
    JOIN v_baseline_windows b ON p.cohort = b.cohort
    WHERE p.I <= b.baseline_end_I
)
SELECT
    cohort,

    -- Energy baseline
    AVG(energy_proxy) AS energy_proxy_mean,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY energy_proxy) AS energy_proxy_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY energy_proxy) AS energy_proxy_p95,

    AVG(energy_velocity) AS energy_velocity_mean,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY energy_velocity) AS energy_velocity_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY energy_velocity) AS energy_velocity_p95,

    AVG(dissipation_rate) AS dissipation_rate_mean,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY dissipation_rate) AS dissipation_rate_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY dissipation_rate) AS dissipation_rate_p95,

    -- Coherence baseline
    AVG(coherence) AS coherence_mean,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY coherence) AS coherence_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY coherence) AS coherence_p95,

    AVG(coherence_velocity) AS coherence_velocity_mean,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY coherence_velocity) AS coherence_velocity_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY coherence_velocity) AS coherence_velocity_p95,

    AVG(effective_dim) AS effective_dim_mean,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY effective_dim) AS effective_dim_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY effective_dim) AS effective_dim_p95,

    AVG(eigenvalue_entropy) AS eigenvalue_entropy_mean,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY eigenvalue_entropy) AS eigenvalue_entropy_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY eigenvalue_entropy) AS eigenvalue_entropy_p95,

    -- State baseline
    AVG(state_distance) AS state_distance_mean,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY state_distance) AS state_distance_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY state_distance) AS state_distance_p95,

    AVG(state_velocity) AS state_velocity_mean,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY state_velocity) AS state_velocity_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY state_velocity) AS state_velocity_p95,

    AVG(state_acceleration) AS state_acceleration_mean,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY state_acceleration) AS state_acceleration_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY state_acceleration) AS state_acceleration_p95,

    -- Baseline metadata
    COUNT(*) AS n_baseline_points,
    MIN(I) AS baseline_start,
    MAX(I) AS baseline_end

FROM baseline_data
GROUP BY cohort;


-- ============================================================================
-- PHASE 2: COMPUTE DEVIATIONS (ratio-based, no z-scores)
-- ============================================================================
-- Every point. Compare to baseline. How far outside normal range?

CREATE OR REPLACE VIEW v_deviation_scores AS
WITH raw_deviations AS (
    SELECT
        p.cohort,
        p.I,

        -- Signed deviations: ratios for unbounded, shifts for bounded
        p.energy_proxy / NULLIF(b.energy_proxy_mean, 0) AS energy_proxy_ratio,
        p.energy_velocity - b.energy_velocity_mean AS energy_velocity_shift,
        p.dissipation_rate / NULLIF(b.dissipation_rate_mean, 0) AS dissipation_rate_ratio,
        p.coherence - b.coherence_mean AS coherence_shift,
        p.coherence_velocity - b.coherence_velocity_mean AS coherence_velocity_shift,
        p.effective_dim - b.effective_dim_mean AS effective_dim_shift,
        p.eigenvalue_entropy - b.eigenvalue_entropy_mean AS eigenvalue_entropy_shift,
        p.state_distance / NULLIF(b.state_distance_mean, 0) AS state_distance_ratio,
        p.state_velocity - b.state_velocity_mean AS state_velocity_shift,
        p.state_acceleration - b.state_acceleration_mean AS state_acceleration_shift,

        -- Range exceedance: how far outside [p05, p95] is this value?
        -- 0 = within range, >0 = outside range (normalized by range width)
        CASE
            WHEN p.energy_proxy > b.energy_proxy_p95
            THEN (p.energy_proxy - b.energy_proxy_p95) / NULLIF(b.energy_proxy_p95 - b.energy_proxy_p05, 0)
            WHEN p.energy_proxy < b.energy_proxy_p05
            THEN (b.energy_proxy_p05 - p.energy_proxy) / NULLIF(b.energy_proxy_p95 - b.energy_proxy_p05, 0)
            ELSE 0
        END AS energy_proxy_exceedance,

        CASE
            WHEN p.coherence > b.coherence_p95
            THEN (p.coherence - b.coherence_p95) / NULLIF(b.coherence_p95 - b.coherence_p05, 0)
            WHEN p.coherence < b.coherence_p05
            THEN (b.coherence_p05 - p.coherence) / NULLIF(b.coherence_p95 - b.coherence_p05, 0)
            ELSE 0
        END AS coherence_exceedance,

        CASE
            WHEN p.state_distance > b.state_distance_p95
            THEN (p.state_distance - b.state_distance_p95) / NULLIF(b.state_distance_p95 - b.state_distance_p05, 0)
            WHEN p.state_distance < b.state_distance_p05
            THEN (b.state_distance_p05 - p.state_distance) / NULLIF(b.state_distance_p95 - b.state_distance_p05, 0)
            ELSE 0
        END AS state_distance_exceedance,

        CASE
            WHEN p.effective_dim > b.effective_dim_p95
            THEN (p.effective_dim - b.effective_dim_p95) / NULLIF(b.effective_dim_p95 - b.effective_dim_p05, 0)
            WHEN p.effective_dim < b.effective_dim_p05
            THEN (b.effective_dim_p05 - p.effective_dim) / NULLIF(b.effective_dim_p95 - b.effective_dim_p05, 0)
            ELSE 0
        END AS effective_dim_exceedance,

        -- Individual out-of-range flags (baseline percentile bounds)
        (p.energy_proxy < b.energy_proxy_p05 OR p.energy_proxy > b.energy_proxy_p95) AS flag_energy_proxy,
        (p.energy_velocity < b.energy_velocity_p05 OR p.energy_velocity > b.energy_velocity_p95) AS flag_energy_velocity,
        (p.dissipation_rate < b.dissipation_rate_p05 OR p.dissipation_rate > b.dissipation_rate_p95) AS flag_dissipation_rate,
        (p.coherence < b.coherence_p05 OR p.coherence > b.coherence_p95) AS flag_coherence,
        (p.coherence_velocity < b.coherence_velocity_p05 OR p.coherence_velocity > b.coherence_velocity_p95) AS flag_coherence_velocity,
        (p.effective_dim < b.effective_dim_p05 OR p.effective_dim > b.effective_dim_p95) AS flag_effective_dim,
        (p.eigenvalue_entropy < b.eigenvalue_entropy_p05 OR p.eigenvalue_entropy > b.eigenvalue_entropy_p95) AS flag_eigenvalue_entropy,
        (p.state_distance < b.state_distance_p05 OR p.state_distance > b.state_distance_p95) AS flag_state_distance,
        (p.state_velocity < b.state_velocity_p05 OR p.state_velocity > b.state_velocity_p95) AS flag_state_velocity,
        (p.state_acceleration < b.state_acceleration_p05 OR p.state_acceleration > b.state_acceleration_p95) AS flag_state_acceleration,

        -- Raw values for context
        p.energy_proxy,
        p.coherence,
        p.state_distance,

        -- Is this point in baseline period?
        p.I <= b.baseline_end AS in_baseline

    FROM physics p
    JOIN baselines b ON p.cohort = b.cohort
)
SELECT
    *,

    -- Composite deviation score: max exceedance across key metrics
    GREATEST(
        COALESCE(energy_proxy_exceedance, 0),
        COALESCE(coherence_exceedance, 0),
        COALESCE(state_distance_exceedance, 0),
        COALESCE(effective_dim_exceedance, 0)
    ) AS deviation_score,

    -- Fleet percentile rank of composite exceedance
    PERCENT_RANK() OVER (ORDER BY GREATEST(
        COALESCE(energy_proxy_exceedance, 0),
        COALESCE(coherence_exceedance, 0),
        COALESCE(state_distance_exceedance, 0),
        COALESCE(effective_dim_exceedance, 0)
    )) AS deviation_pctile

FROM raw_deviations;


-- ============================================================================
-- PHASE 3: FLAG DEVIATIONS
-- ============================================================================
-- Any metric outside baseline range? Flag it.

CREATE OR REPLACE VIEW v_deviation_flags AS
SELECT
    d.cohort,
    d.I,
    d.in_baseline,

    -- Individual flags (already computed from baseline range)
    d.flag_energy_proxy,
    d.flag_energy_velocity,
    d.flag_dissipation_rate,
    d.flag_coherence,
    d.flag_coherence_velocity,
    d.flag_effective_dim,
    d.flag_eigenvalue_entropy,
    d.flag_state_distance,
    d.flag_state_velocity,
    d.flag_state_acceleration,

    -- Count of deviating metrics
    (CASE WHEN d.flag_energy_proxy THEN 1 ELSE 0 END) +
    (CASE WHEN d.flag_energy_velocity THEN 1 ELSE 0 END) +
    (CASE WHEN d.flag_dissipation_rate THEN 1 ELSE 0 END) +
    (CASE WHEN d.flag_coherence THEN 1 ELSE 0 END) +
    (CASE WHEN d.flag_coherence_velocity THEN 1 ELSE 0 END) +
    (CASE WHEN d.flag_effective_dim THEN 1 ELSE 0 END) +
    (CASE WHEN d.flag_eigenvalue_entropy THEN 1 ELSE 0 END) +
    (CASE WHEN d.flag_state_distance THEN 1 ELSE 0 END) +
    (CASE WHEN d.flag_state_velocity THEN 1 ELSE 0 END) +
    (CASE WHEN d.flag_state_acceleration THEN 1 ELSE 0 END)
    AS n_deviating_metrics,

    -- Composite deviation score and fleet percentile
    d.deviation_score,
    d.deviation_pctile,

    -- Which metric has max exceedance?
    CASE GREATEST(
        COALESCE(d.energy_proxy_exceedance, 0),
        COALESCE(d.coherence_exceedance, 0),
        COALESCE(d.state_distance_exceedance, 0),
        COALESCE(d.effective_dim_exceedance, 0)
    )
        WHEN COALESCE(d.energy_proxy_exceedance, 0) THEN 'energy_proxy'
        WHEN COALESCE(d.coherence_exceedance, 0) THEN 'coherence'
        WHEN COALESCE(d.state_distance_exceedance, 0) THEN 'state_distance'
        WHEN COALESCE(d.effective_dim_exceedance, 0) THEN 'effective_dim'
        ELSE 'unknown'
    END AS max_deviation_metric,

    -- Overall severity (fleet-relative + count-based)
    CASE
        WHEN d.deviation_pctile > 0.99
          OR (CASE WHEN d.flag_energy_proxy THEN 1 ELSE 0 END) +
             (CASE WHEN d.flag_coherence THEN 1 ELSE 0 END) +
             (CASE WHEN d.flag_state_distance THEN 1 ELSE 0 END) >= 3
        THEN 'critical'
        WHEN d.deviation_pctile > 0.95
          OR (CASE WHEN d.flag_energy_proxy THEN 1 ELSE 0 END) +
             (CASE WHEN d.flag_coherence THEN 1 ELSE 0 END) +
             (CASE WHEN d.flag_state_distance THEN 1 ELSE 0 END) >= 1
        THEN 'warning'
        ELSE 'normal'
    END AS severity,

    -- Signed deviations for reference
    d.energy_proxy_ratio,
    d.coherence_shift,
    d.state_distance_ratio,
    d.effective_dim_shift

FROM v_deviation_scores d;


-- ============================================================================
-- DEVIATION EVENTS
-- ============================================================================
-- When does deviation first appear? When does it resolve?

CREATE OR REPLACE VIEW v_deviation_events AS
SELECT
    cohort,
    I AS event_time,
    severity,
    n_deviating_metrics,
    max_deviation_metric,
    deviation_score,
    LAG(severity) OVER w AS prev_severity,

    -- Event type
    CASE
        WHEN LAG(severity) OVER w = 'normal' AND severity != 'normal'
        THEN 'deviation_onset'
        WHEN LAG(severity) OVER w != 'normal' AND severity = 'normal'
        THEN 'return_to_normal'
        WHEN LAG(severity) OVER w = 'warning' AND severity = 'critical'
        THEN 'escalation'
        WHEN LAG(severity) OVER w = 'critical' AND severity = 'warning'
        THEN 'de_escalation'
        ELSE NULL
    END AS event_type

FROM v_deviation_flags
WHERE in_baseline = FALSE  -- Only monitor post-baseline
WINDOW w AS (PARTITION BY cohort ORDER BY I)
HAVING event_type IS NOT NULL;


-- ============================================================================
-- ENTITY DEVIATION SUMMARY
-- ============================================================================

CREATE OR REPLACE VIEW v_deviation_entity_summary AS
SELECT
    cohort,

    -- Baseline info
    MAX(n_baseline_points) AS n_baseline_points,

    -- Post-baseline counts
    SUM(CASE WHEN NOT in_baseline THEN 1 ELSE 0 END) AS n_monitored_points,

    -- Deviation counts
    SUM(CASE WHEN NOT in_baseline AND severity = 'warning' THEN 1 ELSE 0 END) AS n_warning,
    SUM(CASE WHEN NOT in_baseline AND severity = 'critical' THEN 1 ELSE 0 END) AS n_critical,
    SUM(CASE WHEN NOT in_baseline AND severity = 'normal' THEN 1 ELSE 0 END) AS n_normal,

    -- Percentages
    100.0 * SUM(CASE WHEN NOT in_baseline AND severity != 'normal' THEN 1 ELSE 0 END) /
        NULLIF(SUM(CASE WHEN NOT in_baseline THEN 1 ELSE 0 END), 0) AS pct_abnormal,

    -- Current status
    MAX(CASE WHEN I = (SELECT MAX(I) FROM physics p2 WHERE p2.cohort = v.cohort) THEN severity END)
        AS current_severity,

    -- Max deviations seen
    MAX(n_deviating_metrics) AS max_simultaneous_deviations,
    MAX(deviation_score) AS max_deviation_score,

    -- Most common deviation source
    MODE() WITHIN GROUP (ORDER BY max_deviation_metric) FILTER (WHERE severity != 'normal')
        AS most_common_deviation_source,

    -- Health assessment
    CASE
        WHEN SUM(CASE WHEN NOT in_baseline AND severity = 'critical' THEN 1 ELSE 0 END) >
             SUM(CASE WHEN NOT in_baseline THEN 1 ELSE 0 END) * 0.1
        THEN 'degraded'
        WHEN SUM(CASE WHEN NOT in_baseline AND severity != 'normal' THEN 1 ELSE 0 END) >
             SUM(CASE WHEN NOT in_baseline THEN 1 ELSE 0 END) * 0.2
        THEN 'unstable'
        WHEN SUM(CASE WHEN NOT in_baseline AND severity != 'normal' THEN 1 ELSE 0 END) >
             SUM(CASE WHEN NOT in_baseline THEN 1 ELSE 0 END) * 0.05
        THEN 'noisy'
        ELSE 'healthy'
    END AS health_assessment

FROM v_deviation_flags v
JOIN baselines b ON v.cohort = b.cohort
GROUP BY v.cohort;


-- ============================================================================
-- FLEET DEVIATION SUMMARY
-- ============================================================================

CREATE OR REPLACE VIEW v_deviation_fleet_summary AS
SELECT
    COUNT(DISTINCT cohort) AS n_entities,

    -- Health distribution
    SUM(CASE WHEN health_assessment = 'healthy' THEN 1 ELSE 0 END) AS n_healthy,
    SUM(CASE WHEN health_assessment = 'noisy' THEN 1 ELSE 0 END) AS n_noisy,
    SUM(CASE WHEN health_assessment = 'unstable' THEN 1 ELSE 0 END) AS n_unstable,
    SUM(CASE WHEN health_assessment = 'degraded' THEN 1 ELSE 0 END) AS n_degraded,

    -- Current status distribution
    SUM(CASE WHEN current_severity = 'normal' THEN 1 ELSE 0 END) AS n_currently_normal,
    SUM(CASE WHEN current_severity = 'warning' THEN 1 ELSE 0 END) AS n_currently_warning,
    SUM(CASE WHEN current_severity = 'critical' THEN 1 ELSE 0 END) AS n_currently_critical,

    -- Fleet health percentage
    100.0 * SUM(CASE WHEN health_assessment = 'healthy' THEN 1 ELSE 0 END) / COUNT(*) AS pct_healthy,

    -- Average abnormal rate
    AVG(pct_abnormal) AS avg_pct_abnormal,

    -- Most common deviation source across fleet
    MODE() WITHIN GROUP (ORDER BY most_common_deviation_source)
        AS fleet_common_deviation_source

FROM v_deviation_entity_summary;


-- ============================================================================
-- SENSITIVITY ANALYSIS (percentile-based)
-- ============================================================================
-- How does detection change across different percentile thresholds?

CREATE OR REPLACE VIEW v_sensitivity_analysis AS
SELECT
    cohort,

    -- At fleet P90
    100.0 * SUM(CASE WHEN deviation_pctile > 0.90 THEN 1 ELSE 0 END) / COUNT(*) AS pct_flagged_p90,

    -- At fleet P95 (default warning)
    100.0 * SUM(CASE WHEN deviation_pctile > 0.95 THEN 1 ELSE 0 END) / COUNT(*) AS pct_flagged_p95,

    -- At fleet P99 (default critical)
    100.0 * SUM(CASE WHEN deviation_pctile > 0.99 THEN 1 ELSE 0 END) / COUNT(*) AS pct_flagged_p99,

    -- By out-of-range count
    100.0 * SUM(CASE WHEN n_deviating_metrics >= 1 THEN 1 ELSE 0 END) / COUNT(*) AS pct_any_metric_oor,
    100.0 * SUM(CASE WHEN n_deviating_metrics >= 3 THEN 1 ELSE 0 END) / COUNT(*) AS pct_3plus_metrics_oor,

    -- Recommendation
    CASE
        WHEN 100.0 * SUM(CASE WHEN n_deviating_metrics >= 1 THEN 1 ELSE 0 END) / COUNT(*) > 50
        THEN 'High OOR rate — tighten baseline or increase min_deviations'
        WHEN 100.0 * SUM(CASE WHEN n_deviating_metrics >= 1 THEN 1 ELSE 0 END) / COUNT(*) < 1
        THEN 'Very low OOR rate — may be missing deviations'
        ELSE 'Detection rate within expected range'
    END AS recommendation

FROM v_deviation_flags
WHERE NOT in_baseline
GROUP BY cohort;


-- ============================================================================
-- EXPORT: baseline_deviation table (for downstream parquet)
-- ============================================================================
-- This table is consumed by 60_ground_truth.sql and 63_threshold_optimization.sql

CREATE OR REPLACE TABLE baseline_deviation AS
SELECT
    cohort,
    I,
    deviation_score,
    deviation_pctile,
    n_deviating_metrics,
    max_deviation_metric,
    severity,
    energy_proxy_ratio,
    coherence_shift,
    state_distance_ratio,
    effective_dim_shift,
    in_baseline
FROM v_deviation_flags
WHERE NOT in_baseline;


-- ============================================================================
-- VERIFY
-- ============================================================================

.print ''
.print '=== BASELINE & DEVIATION ANALYSIS ==='
.print ''

.print 'Baselines established:'
SELECT
    cohort,
    n_baseline_points || ' pts' AS baseline_size,
    ROUND(energy_proxy_mean, 2) AS energy_mean,
    ROUND(coherence_mean, 2) AS coherence_mean,
    ROUND(state_distance_mean, 2) AS state_mean
FROM baselines
LIMIT 10;

.print ''
.print 'Entity Health Summary:'
SELECT
    cohort,
    health_assessment,
    current_severity,
    ROUND(pct_abnormal, 1) || '%' AS pct_abnormal,
    most_common_deviation_source AS deviation_source
FROM v_deviation_entity_summary
ORDER BY pct_abnormal DESC
LIMIT 10;

.print ''
.print 'Fleet Summary:'
SELECT * FROM v_deviation_fleet_summary;

.print ''
.print 'Sensitivity Analysis:'
SELECT
    cohort,
    ROUND(pct_flagged_p90, 1) || '%' AS '>p90',
    ROUND(pct_flagged_p95, 1) || '%' AS '>p95',
    ROUND(pct_flagged_p99, 1) || '%' AS '>p99',
    recommendation
FROM v_sensitivity_analysis
LIMIT 10;

.print ''
.print 'Recent Deviation Events:'
SELECT
    cohort,
    event_time,
    event_type,
    prev_severity || ' → ' || severity AS transition,
    max_deviation_metric,
    ROUND(deviation_score, 4) AS deviation_score
FROM v_deviation_events
ORDER BY event_time DESC
LIMIT 10;
