-- ============================================================================
-- ORTHON SQL: 24_incident_summary.sql
-- ============================================================================
-- THE INCIDENT SUMMARY: What ORTHON Returns
--
-- Ties together all analyses into a comprehensive incident report:
--   1. Baseline established — "Here's normal for each signal, each metric."
--   2. Deviation detected — "Window X. Signal Y deviated first. Z-score W."
--   3. Propagation mapped — Origin → downstream path with timing
--   4. Force attribution — Exogenous (translation) or Endogenous (deformation)
--   5. Energy accounting — Injected, absorbed, lost
--   6. The gap identified — What's unmeasured?
--
-- Output: Human-readable incident summary ready for operator review.
-- ============================================================================

-- ============================================================================
-- PREREQUISITE VIEWS (from other scripts)
-- ============================================================================
-- Requires:
--   - baselines, v_deviation_flags, v_deviation_events (23_baseline_deviation.sql)
--   - v_force_attribution, v_attribution_entity_summary (21_geometric_attribution.sql)
--   - v_orthon_signal, v_orthon_entity_summary (16_orthon_signal.sql)
--   - physics table (12_load_physics.sql)
-- ============================================================================


-- ============================================================================
-- FIRST DEVIATION DETECTION
-- ============================================================================
-- Which signal deviated first? When?

CREATE OR REPLACE VIEW v_first_deviation AS
SELECT DISTINCT ON (entity_id)
    entity_id,
    I AS first_deviation_time,
    max_deviation_metric AS first_deviation_metric,
    max_z_score AS first_z_score,
    severity AS first_severity
FROM v_deviation_flags
WHERE severity != 'normal'
  AND NOT in_baseline
ORDER BY entity_id, I;


-- ============================================================================
-- PROPAGATION PATH
-- ============================================================================
-- How did the deviation spread through the system?
-- Track which metrics deviated in sequence.

CREATE OR REPLACE VIEW v_propagation_sequence AS
WITH metric_deviations AS (
    SELECT
        entity_id,
        I,
        'energy_proxy' AS metric,
        z_energy_proxy AS z_score,
        ABS(z_energy_proxy) > 2 AS deviated
    FROM v_deviation_scores WHERE NOT in_baseline
    UNION ALL
    SELECT entity_id, I, 'coherence', z_coherence, ABS(z_coherence) > 2
    FROM v_deviation_scores WHERE NOT in_baseline
    UNION ALL
    SELECT entity_id, I, 'state_distance', z_state_distance, ABS(z_state_distance) > 2
    FROM v_deviation_scores WHERE NOT in_baseline
    UNION ALL
    SELECT entity_id, I, 'effective_dim', z_effective_dim, ABS(z_effective_dim) > 2
    FROM v_deviation_scores WHERE NOT in_baseline
),
first_deviation_per_metric AS (
    SELECT DISTINCT ON (entity_id, metric)
        entity_id,
        metric,
        I AS first_deviation_I,
        z_score
    FROM metric_deviations
    WHERE deviated
    ORDER BY entity_id, metric, I
)
SELECT
    entity_id,
    metric,
    first_deviation_I,
    z_score,
    first_deviation_I - MIN(first_deviation_I) OVER (PARTITION BY entity_id) AS lag_from_origin,
    ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY first_deviation_I) AS propagation_order
FROM first_deviation_per_metric;


-- ============================================================================
-- PROPAGATION PATH STRING
-- ============================================================================

CREATE OR REPLACE VIEW v_propagation_path AS
SELECT
    entity_id,
    STRING_AGG(
        metric || ' (I=' || first_deviation_I || ', z=' || ROUND(z_score::DECIMAL, 1) || ')',
        ' → '
        ORDER BY propagation_order
    ) AS propagation_chain,
    MAX(lag_from_origin) AS total_propagation_time,
    COUNT(*) AS n_metrics_affected
FROM v_propagation_sequence
GROUP BY entity_id;


-- ============================================================================
-- ENERGY BALANCE
-- ============================================================================
-- Account for energy: injected, absorbed, lost.

CREATE OR REPLACE VIEW v_energy_balance AS
WITH energy_changes AS (
    SELECT
        entity_id,
        I,
        energy_proxy,
        energy_velocity,
        LAG(energy_proxy) OVER w AS prev_energy,
        energy_proxy - LAG(energy_proxy) OVER w AS energy_delta
    FROM physics
    WINDOW w AS (PARTITION BY entity_id ORDER BY I)
),
period_sums AS (
    SELECT
        entity_id,
        SUM(CASE WHEN energy_delta > 0 THEN energy_delta ELSE 0 END) AS energy_injected,
        SUM(CASE WHEN energy_delta < 0 THEN ABS(energy_delta) ELSE 0 END) AS energy_dissipated,
        SUM(energy_delta) AS net_energy_change,
        MIN(energy_proxy) AS min_energy,
        MAX(energy_proxy) AS max_energy,
        AVG(energy_proxy) AS avg_energy
    FROM energy_changes
    WHERE energy_delta IS NOT NULL
    GROUP BY entity_id
)
SELECT
    entity_id,
    ROUND(energy_injected, 4) AS energy_injected,
    ROUND(energy_dissipated, 4) AS energy_dissipated,
    ROUND(net_energy_change, 4) AS net_energy_change,
    ROUND(energy_dissipated - energy_injected, 4) AS energy_gap,

    -- Gap interpretation
    CASE
        WHEN ABS(energy_dissipated - energy_injected) < 0.01 THEN 'balanced'
        WHEN energy_dissipated > energy_injected THEN 'deficit_unmeasured_sink'
        ELSE 'surplus_unmeasured_source'
    END AS energy_balance_status,

    -- Gap magnitude
    CASE
        WHEN ABS(energy_dissipated - energy_injected) / NULLIF(energy_injected, 0) > 0.2
        THEN 'significant_gap'
        WHEN ABS(energy_dissipated - energy_injected) / NULLIF(energy_injected, 0) > 0.1
        THEN 'moderate_gap'
        ELSE 'minor_gap'
    END AS gap_severity

FROM period_sums;


-- ============================================================================
-- INCIDENT STATE TIMELINE
-- ============================================================================
-- Track system state through the incident.

CREATE OR REPLACE VIEW v_incident_timeline AS
SELECT
    entity_id,
    I,
    severity,

    -- State progression
    CASE
        WHEN severity = 'normal' AND LAG(severity) OVER w != 'normal' THEN 'recovered'
        WHEN severity = 'warning' AND LAG(severity) OVER w = 'normal' THEN 'degraded'
        WHEN severity = 'critical' AND LAG(severity) OVER w = 'warning' THEN 'critical'
        WHEN severity = 'critical' AND LAG(severity) OVER w = 'normal' THEN 'rapid_failure'
        ELSE NULL
    END AS state_transition,

    -- Time in current state
    I - MAX(CASE WHEN severity != LAG(severity) OVER w THEN I END) OVER (
        PARTITION BY entity_id ORDER BY I ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS time_in_state

FROM v_deviation_flags
WHERE NOT in_baseline
WINDOW w AS (PARTITION BY entity_id ORDER BY I);


-- ============================================================================
-- INCIDENT PEAK
-- ============================================================================
-- When was the incident at its worst?

CREATE OR REPLACE VIEW v_incident_peak AS
SELECT DISTINCT ON (entity_id)
    entity_id,
    I AS peak_time,
    max_z_score AS peak_z_score,
    n_deviating_metrics AS peak_n_deviations,
    severity AS peak_severity
FROM v_deviation_flags
WHERE NOT in_baseline
ORDER BY entity_id, max_z_score DESC;


-- ============================================================================
-- THE INCIDENT SUMMARY
-- ============================================================================
-- Comprehensive summary for each entity.

CREATE OR REPLACE VIEW v_incident_summary AS
SELECT
    fd.entity_id,

    -- === DETECTION ===
    fd.first_deviation_time,
    fd.first_deviation_metric,
    fd.first_z_score,

    -- === PROPAGATION ===
    pp.propagation_chain,
    pp.total_propagation_time,
    pp.n_metrics_affected,

    -- === PEAK ===
    ip.peak_time,
    ip.peak_z_score,
    ip.peak_severity,

    -- === CURRENT STATE ===
    des.current_severity,
    des.health_assessment,
    des.pct_abnormal,

    -- === FORCE ATTRIBUTION ===
    aes.dominant_force,
    aes.attribution_ratio,

    -- === ENERGY BALANCE ===
    eb.energy_injected,
    eb.energy_dissipated,
    eb.net_energy_change,
    eb.energy_gap,
    eb.energy_balance_status,

    -- === ORTHON SIGNAL ===
    oes.current_orthon_signal,
    oes.status_message AS orthon_status

FROM v_first_deviation fd
LEFT JOIN v_propagation_path pp ON fd.entity_id = pp.entity_id
LEFT JOIN v_incident_peak ip ON fd.entity_id = ip.entity_id
LEFT JOIN v_deviation_entity_summary des ON fd.entity_id = des.entity_id
LEFT JOIN v_attribution_entity_summary aes ON fd.entity_id = aes.entity_id
LEFT JOIN v_energy_balance eb ON fd.entity_id = eb.entity_id
LEFT JOIN v_orthon_entity_summary oes ON fd.entity_id = oes.entity_id;


-- ============================================================================
-- HUMAN-READABLE INCIDENT REPORT
-- ============================================================================

CREATE OR REPLACE VIEW v_incident_report AS
SELECT
    entity_id,

    '================================================================================
INCIDENT SUMMARY: ' || entity_id || '
================================================================================

FIRST DETECTION
---------------
  Window:     ' || first_deviation_time || '
  Metric:     ' || first_deviation_metric || '
  Z-score:    ' || ROUND(first_z_score::DECIMAL, 2) || '

PROPAGATION PATH
----------------
  ' || COALESCE(propagation_chain, '(single metric affected)') || '
  Total span: ' || COALESCE(total_propagation_time::TEXT, '0') || ' windows
  Affected:   ' || n_metrics_affected || ' metrics

PEAK SEVERITY
-------------
  Window:     ' || peak_time || '
  Z-score:    ' || ROUND(peak_z_score::DECIMAL, 2) || '
  Severity:   ' || UPPER(peak_severity) || '

FORCE ATTRIBUTION
-----------------
  Type:       ' || UPPER(COALESCE(dominant_force, 'unknown')) || '
  Ratio:      ' || COALESCE(ROUND(attribution_ratio::DECIMAL, 2)::TEXT, 'N/A') ||
    CASE
        WHEN attribution_ratio > 2 THEN ' (externally driven)'
        WHEN attribution_ratio < 0.5 THEN ' (internally driven)'
        ELSE ' (mixed forces)'
    END || '

ENERGY BALANCE
--------------
  Injected:   ' || COALESCE(energy_injected::TEXT, 'N/A') || '
  Dissipated: ' || COALESCE(energy_dissipated::TEXT, 'N/A') || '
  Net change: ' || COALESCE(net_energy_change::TEXT, 'N/A') || '
  Gap:        ' || COALESCE(energy_gap::TEXT, 'N/A') ||
    CASE energy_balance_status
        WHEN 'deficit_unmeasured_sink' THEN ' ← UNMEASURED SINK DETECTED'
        WHEN 'surplus_unmeasured_source' THEN ' ← UNMEASURED SOURCE DETECTED'
        ELSE ''
    END || '

CURRENT STATE
-------------
  Severity:   ' || UPPER(COALESCE(current_severity, 'unknown')) || '
  Health:     ' || UPPER(COALESCE(health_assessment, 'unknown')) || '
  Abnormal:   ' || COALESCE(ROUND(pct_abnormal::DECIMAL, 1)::TEXT, '0') || '%

ORTHON SIGNAL
-------------
  Active:     ' || CASE WHEN current_orthon_signal THEN 'YES ⚠️' ELSE 'No' END || '
  Status:     ' || COALESCE(orthon_status, 'Normal operation') || '

================================================================================
' AS report

FROM v_incident_summary;


-- ============================================================================
-- FLEET INCIDENT OVERVIEW
-- ============================================================================

CREATE OR REPLACE VIEW v_fleet_incident_overview AS
SELECT
    COUNT(*) AS n_entities_with_incidents,

    -- Severity distribution
    SUM(CASE WHEN current_severity = 'critical' THEN 1 ELSE 0 END) AS n_critical,
    SUM(CASE WHEN current_severity = 'warning' THEN 1 ELSE 0 END) AS n_warning,
    SUM(CASE WHEN current_severity = 'normal' THEN 1 ELSE 0 END) AS n_normal,

    -- Orthon signals
    SUM(CASE WHEN current_orthon_signal THEN 1 ELSE 0 END) AS n_orthon_signals,

    -- Force attribution
    SUM(CASE WHEN dominant_force = 'exogenous_dominated' THEN 1 ELSE 0 END) AS n_external,
    SUM(CASE WHEN dominant_force = 'endogenous_dominated' THEN 1 ELSE 0 END) AS n_internal,

    -- Energy gaps
    SUM(CASE WHEN energy_balance_status = 'deficit_unmeasured_sink' THEN 1 ELSE 0 END) AS n_unmeasured_sinks,
    SUM(CASE WHEN energy_balance_status = 'surplus_unmeasured_source' THEN 1 ELSE 0 END) AS n_unmeasured_sources,

    -- Common first deviation metric
    MODE() WITHIN GROUP (ORDER BY first_deviation_metric) AS most_common_trigger

FROM v_incident_summary;


-- ============================================================================
-- VERIFY / OUTPUT
-- ============================================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║                        ORTHON INCIDENT SUMMARY                               ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'
.print ''

.print '=== FLEET OVERVIEW ==='
SELECT * FROM v_fleet_incident_overview;

.print ''
.print '=== ENTITIES WITH INCIDENTS ==='
SELECT
    entity_id,
    UPPER(current_severity) AS severity,
    first_deviation_metric AS trigger,
    ROUND(peak_z_score::DECIMAL, 1) AS peak_z,
    UPPER(COALESCE(dominant_force, 'unknown')) AS force_type,
    CASE WHEN current_orthon_signal THEN '⚠️ ACTIVE' ELSE '—' END AS orthon
FROM v_incident_summary
ORDER BY
    CASE current_severity
        WHEN 'critical' THEN 1
        WHEN 'warning' THEN 2
        ELSE 3
    END,
    peak_z_score DESC
LIMIT 20;

.print ''
.print '=== DETAILED REPORT (First Critical Entity) ==='
SELECT report
FROM v_incident_report
WHERE current_severity = 'critical'
ORDER BY peak_z_score DESC
LIMIT 1;

.print ''
.print '=== ENERGY GAPS DETECTED ==='
SELECT
    entity_id,
    energy_balance_status,
    energy_gap,
    CASE
        WHEN energy_gap > 0 THEN 'Energy leaving system unmeasured'
        ELSE 'Energy entering system unmeasured'
    END AS interpretation
FROM v_energy_balance
WHERE energy_balance_status != 'balanced'
ORDER BY ABS(energy_gap) DESC
LIMIT 10;
