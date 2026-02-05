-- ============================================================================
-- ORTHON SQL: 23_baseline_deviation.sql
-- ============================================================================
-- BASELINE & DEVIATION: Self-Referential Anomaly Detection
--
-- The Question: "Is this still normal?"
--
-- Phase 1 — Baseline:
--   Each signal. Each metric. Establish normal.
--   One row per signal per metric. The fingerprint of normal.
--
-- Phase 2 — Monitor:
--   Every new window. Compare to baseline. Compute z-scores.
--
-- Phase 3 — Flag:
--   Don't require all metrics to move. One is enough to flag.
--   Something changed. Something worth looking at.
--
-- No external model. No assumptions about failure modes.
-- The system defines its own normal. Deviation is self-referential.
-- ============================================================================

-- ============================================================================
-- CONFIGURATION
-- ============================================================================

CREATE OR REPLACE TABLE config_baseline AS
SELECT
    -- Baseline training window (first N% of data or first N points)
    0.10  AS baseline_pct,           -- Use first 10% of data for baseline
    100   AS baseline_min_points,    -- Minimum points for baseline

    -- Deviation thresholds
    2.0   AS z_threshold_warning,    -- |z| > 2 = warning
    3.0   AS z_threshold_critical,   -- |z| > 3 = critical

    -- Percentile bounds
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
    entity_id,
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
GROUP BY entity_id;


-- Baseline statistics per entity per metric
CREATE OR REPLACE TABLE baselines AS
WITH baseline_data AS (
    SELECT
        p.entity_id,
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
    JOIN v_baseline_windows b ON p.entity_id = b.entity_id
    WHERE p.I <= b.baseline_end_I
)
SELECT
    entity_id,

    -- Energy baseline
    AVG(energy_proxy) AS energy_proxy_mean,
    STDDEV(energy_proxy) AS energy_proxy_std,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY energy_proxy) AS energy_proxy_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY energy_proxy) AS energy_proxy_p95,

    AVG(energy_velocity) AS energy_velocity_mean,
    STDDEV(energy_velocity) AS energy_velocity_std,

    AVG(dissipation_rate) AS dissipation_rate_mean,
    STDDEV(dissipation_rate) AS dissipation_rate_std,

    -- Coherence baseline
    AVG(coherence) AS coherence_mean,
    STDDEV(coherence) AS coherence_std,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY coherence) AS coherence_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY coherence) AS coherence_p95,

    AVG(coherence_velocity) AS coherence_velocity_mean,
    STDDEV(coherence_velocity) AS coherence_velocity_std,

    AVG(effective_dim) AS effective_dim_mean,
    STDDEV(effective_dim) AS effective_dim_std,

    AVG(eigenvalue_entropy) AS eigenvalue_entropy_mean,
    STDDEV(eigenvalue_entropy) AS eigenvalue_entropy_std,

    -- State baseline
    AVG(state_distance) AS state_distance_mean,
    STDDEV(state_distance) AS state_distance_std,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY state_distance) AS state_distance_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY state_distance) AS state_distance_p95,

    AVG(state_velocity) AS state_velocity_mean,
    STDDEV(state_velocity) AS state_velocity_std,

    AVG(state_acceleration) AS state_acceleration_mean,
    STDDEV(state_acceleration) AS state_acceleration_std,

    -- Baseline metadata
    COUNT(*) AS n_baseline_points,
    MIN(I) AS baseline_start,
    MAX(I) AS baseline_end

FROM baseline_data
GROUP BY entity_id;


-- ============================================================================
-- PHASE 2: COMPUTE Z-SCORES
-- ============================================================================
-- Every point. Compare to baseline. How many standard deviations?

CREATE OR REPLACE VIEW v_deviation_scores AS
SELECT
    p.entity_id,
    p.I,

    -- Energy z-scores
    (p.energy_proxy - b.energy_proxy_mean) / NULLIF(b.energy_proxy_std, 0) AS z_energy_proxy,
    (p.energy_velocity - b.energy_velocity_mean) / NULLIF(b.energy_velocity_std, 0) AS z_energy_velocity,
    (p.dissipation_rate - b.dissipation_rate_mean) / NULLIF(b.dissipation_rate_std, 0) AS z_dissipation_rate,

    -- Coherence z-scores
    (p.coherence - b.coherence_mean) / NULLIF(b.coherence_std, 0) AS z_coherence,
    (p.coherence_velocity - b.coherence_velocity_mean) / NULLIF(b.coherence_velocity_std, 0) AS z_coherence_velocity,
    (p.effective_dim - b.effective_dim_mean) / NULLIF(b.effective_dim_std, 0) AS z_effective_dim,
    (p.eigenvalue_entropy - b.eigenvalue_entropy_mean) / NULLIF(b.eigenvalue_entropy_std, 0) AS z_eigenvalue_entropy,

    -- State z-scores
    (p.state_distance - b.state_distance_mean) / NULLIF(b.state_distance_std, 0) AS z_state_distance,
    (p.state_velocity - b.state_velocity_mean) / NULLIF(b.state_velocity_std, 0) AS z_state_velocity,
    (p.state_acceleration - b.state_acceleration_mean) / NULLIF(b.state_acceleration_std, 0) AS z_state_acceleration,

    -- Raw values for context
    p.energy_proxy,
    p.coherence,
    p.state_distance,

    -- Is this point in baseline period?
    p.I <= b.baseline_end AS in_baseline

FROM physics p
JOIN baselines b ON p.entity_id = b.entity_id;


-- ============================================================================
-- PHASE 3: FLAG DEVIATIONS
-- ============================================================================
-- Any metric exceeds threshold? Flag it.

CREATE OR REPLACE VIEW v_deviation_flags AS
SELECT
    d.entity_id,
    d.I,
    d.in_baseline,

    -- Individual flags (warning level: |z| > 2)
    ABS(d.z_energy_proxy) > c.z_threshold_warning AS flag_energy_proxy,
    ABS(d.z_energy_velocity) > c.z_threshold_warning AS flag_energy_velocity,
    ABS(d.z_dissipation_rate) > c.z_threshold_warning AS flag_dissipation_rate,
    ABS(d.z_coherence) > c.z_threshold_warning AS flag_coherence,
    ABS(d.z_coherence_velocity) > c.z_threshold_warning AS flag_coherence_velocity,
    ABS(d.z_effective_dim) > c.z_threshold_warning AS flag_effective_dim,
    ABS(d.z_eigenvalue_entropy) > c.z_threshold_warning AS flag_eigenvalue_entropy,
    ABS(d.z_state_distance) > c.z_threshold_warning AS flag_state_distance,
    ABS(d.z_state_velocity) > c.z_threshold_warning AS flag_state_velocity,
    ABS(d.z_state_acceleration) > c.z_threshold_warning AS flag_state_acceleration,

    -- Count of deviating metrics
    (CASE WHEN ABS(d.z_energy_proxy) > c.z_threshold_warning THEN 1 ELSE 0 END) +
    (CASE WHEN ABS(d.z_energy_velocity) > c.z_threshold_warning THEN 1 ELSE 0 END) +
    (CASE WHEN ABS(d.z_dissipation_rate) > c.z_threshold_warning THEN 1 ELSE 0 END) +
    (CASE WHEN ABS(d.z_coherence) > c.z_threshold_warning THEN 1 ELSE 0 END) +
    (CASE WHEN ABS(d.z_coherence_velocity) > c.z_threshold_warning THEN 1 ELSE 0 END) +
    (CASE WHEN ABS(d.z_effective_dim) > c.z_threshold_warning THEN 1 ELSE 0 END) +
    (CASE WHEN ABS(d.z_eigenvalue_entropy) > c.z_threshold_warning THEN 1 ELSE 0 END) +
    (CASE WHEN ABS(d.z_state_distance) > c.z_threshold_warning THEN 1 ELSE 0 END) +
    (CASE WHEN ABS(d.z_state_velocity) > c.z_threshold_warning THEN 1 ELSE 0 END) +
    (CASE WHEN ABS(d.z_state_acceleration) > c.z_threshold_warning THEN 1 ELSE 0 END)
    AS n_deviating_metrics,

    -- Critical count (|z| > 3)
    (CASE WHEN ABS(d.z_energy_proxy) > c.z_threshold_critical THEN 1 ELSE 0 END) +
    (CASE WHEN ABS(d.z_coherence) > c.z_threshold_critical THEN 1 ELSE 0 END) +
    (CASE WHEN ABS(d.z_state_distance) > c.z_threshold_critical THEN 1 ELSE 0 END)
    AS n_critical_metrics,

    -- Maximum absolute z-score
    GREATEST(
        ABS(COALESCE(d.z_energy_proxy, 0)),
        ABS(COALESCE(d.z_coherence, 0)),
        ABS(COALESCE(d.z_state_distance, 0)),
        ABS(COALESCE(d.z_effective_dim, 0))
    ) AS max_z_score,

    -- Which metric has max deviation?
    CASE GREATEST(
        ABS(COALESCE(d.z_energy_proxy, 0)),
        ABS(COALESCE(d.z_coherence, 0)),
        ABS(COALESCE(d.z_state_distance, 0)),
        ABS(COALESCE(d.z_effective_dim, 0))
    )
        WHEN ABS(COALESCE(d.z_energy_proxy, 0)) THEN 'energy_proxy'
        WHEN ABS(COALESCE(d.z_coherence, 0)) THEN 'coherence'
        WHEN ABS(COALESCE(d.z_state_distance, 0)) THEN 'state_distance'
        WHEN ABS(COALESCE(d.z_effective_dim, 0)) THEN 'effective_dim'
        ELSE 'unknown'
    END AS max_deviation_metric,

    -- Overall severity
    CASE
        WHEN (CASE WHEN ABS(d.z_energy_proxy) > c.z_threshold_critical THEN 1 ELSE 0 END) +
             (CASE WHEN ABS(d.z_coherence) > c.z_threshold_critical THEN 1 ELSE 0 END) +
             (CASE WHEN ABS(d.z_state_distance) > c.z_threshold_critical THEN 1 ELSE 0 END) > 0
        THEN 'critical'
        WHEN (CASE WHEN ABS(d.z_energy_proxy) > c.z_threshold_warning THEN 1 ELSE 0 END) +
             (CASE WHEN ABS(d.z_coherence) > c.z_threshold_warning THEN 1 ELSE 0 END) +
             (CASE WHEN ABS(d.z_state_distance) > c.z_threshold_warning THEN 1 ELSE 0 END) > 0
        THEN 'warning'
        ELSE 'normal'
    END AS severity,

    -- Z-scores for reference
    d.z_energy_proxy,
    d.z_coherence,
    d.z_state_distance,
    d.z_effective_dim

FROM v_deviation_scores d
CROSS JOIN config_baseline c;


-- ============================================================================
-- DEVIATION EVENTS
-- ============================================================================
-- When does deviation first appear? When does it resolve?

CREATE OR REPLACE VIEW v_deviation_events AS
SELECT
    entity_id,
    I AS event_time,
    severity,
    n_deviating_metrics,
    max_deviation_metric,
    max_z_score,
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
WINDOW w AS (PARTITION BY entity_id ORDER BY I)
HAVING event_type IS NOT NULL;


-- ============================================================================
-- ENTITY DEVIATION SUMMARY
-- ============================================================================

CREATE OR REPLACE VIEW v_deviation_entity_summary AS
SELECT
    entity_id,

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
    MAX(CASE WHEN I = (SELECT MAX(I) FROM physics p2 WHERE p2.entity_id = v.entity_id) THEN severity END)
        AS current_severity,

    -- Max deviations seen
    MAX(n_deviating_metrics) AS max_simultaneous_deviations,
    MAX(max_z_score) AS max_z_score_seen,

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
JOIN baselines b ON v.entity_id = b.entity_id
GROUP BY v.entity_id;


-- ============================================================================
-- FLEET DEVIATION SUMMARY
-- ============================================================================

CREATE OR REPLACE VIEW v_deviation_fleet_summary AS
SELECT
    COUNT(DISTINCT entity_id) AS n_entities,

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
-- SENSITIVITY ADJUSTMENT
-- ============================================================================
-- Views to help tune the sensitivity thresholds.

CREATE OR REPLACE VIEW v_sensitivity_analysis AS
SELECT
    entity_id,

    -- At z > 2 (current default)
    100.0 * SUM(CASE WHEN max_z_score > 2.0 THEN 1 ELSE 0 END) / COUNT(*) AS pct_flagged_z2,

    -- At z > 2.5
    100.0 * SUM(CASE WHEN max_z_score > 2.5 THEN 1 ELSE 0 END) / COUNT(*) AS pct_flagged_z25,

    -- At z > 3 (more conservative)
    100.0 * SUM(CASE WHEN max_z_score > 3.0 THEN 1 ELSE 0 END) / COUNT(*) AS pct_flagged_z3,

    -- At z > 4 (very conservative)
    100.0 * SUM(CASE WHEN max_z_score > 4.0 THEN 1 ELSE 0 END) / COUNT(*) AS pct_flagged_z4,

    -- Recommendation
    CASE
        WHEN 100.0 * SUM(CASE WHEN max_z_score > 2.0 THEN 1 ELSE 0 END) / COUNT(*) > 30
        THEN 'Increase threshold (too many flags)'
        WHEN 100.0 * SUM(CASE WHEN max_z_score > 3.0 THEN 1 ELSE 0 END) / COUNT(*) < 1
        THEN 'Decrease threshold (missing deviations)'
        ELSE 'Current threshold appropriate'
    END AS recommendation

FROM v_deviation_flags
WHERE NOT in_baseline
GROUP BY entity_id;


-- ============================================================================
-- VERIFY
-- ============================================================================

.print ''
.print '=== BASELINE & DEVIATION ANALYSIS ==='
.print ''

.print 'Baselines established:'
SELECT
    entity_id,
    n_baseline_points || ' pts' AS baseline_size,
    ROUND(energy_proxy_mean, 2) AS energy_mean,
    ROUND(coherence_mean, 2) AS coherence_mean,
    ROUND(state_distance_mean, 2) AS state_mean
FROM baselines
LIMIT 10;

.print ''
.print 'Entity Health Summary:'
SELECT
    entity_id,
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
    entity_id,
    ROUND(pct_flagged_z2, 1) || '%' AS 'z>2',
    ROUND(pct_flagged_z25, 1) || '%' AS 'z>2.5',
    ROUND(pct_flagged_z3, 1) || '%' AS 'z>3',
    recommendation
FROM v_sensitivity_analysis
LIMIT 10;

.print ''
.print 'Recent Deviation Events:'
SELECT
    entity_id,
    event_time,
    event_type,
    prev_severity || ' → ' || severity AS transition,
    max_deviation_metric,
    ROUND(max_z_score, 2) AS z_score
FROM v_deviation_events
ORDER BY event_time DESC
LIMIT 10;
