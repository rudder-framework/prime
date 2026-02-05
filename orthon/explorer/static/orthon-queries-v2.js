// ORTHON SQL Query Library v2
// Organized by Four-Pillar structure: Vectors, Geometry, Physics, Dynamics, Advanced, Summary
// Maps to Y10-Y13 SQL views created by ORTHON analysis scripts
// These queries run client-side via DuckDB-WASM against user-loaded parquet files

const ORTHON_QUERIES_V2 = {

  // ============================================================
  // VECTORS - observations.parquet and vector.parquet
  // Base data inspection (pre-analysis)
  // ============================================================
  vectors: {
    obs_summary: {
      name: "Observation Summary",
      description: "Overview of loaded observations",
      sql: `
SELECT
    COUNT(*) AS total_observations,
    COUNT(DISTINCT entity_id) AS n_entities,
    COUNT(DISTINCT signal_id) AS n_signals,
    MIN(I) AS I_min,
    MAX(I) AS I_max,
    ROUND(AVG(y), 4) AS y_mean,
    ROUND(STDDEV(y), 4) AS y_std
FROM observations
      `
    },
    obs_by_entity: {
      name: "Group by Entity",
      description: "Observation counts per entity",
      sql: `
SELECT
    entity_id,
    COUNT(*) AS n_observations,
    COUNT(DISTINCT signal_id) AS n_signals,
    MIN(I) AS I_start,
    MAX(I) AS I_end,
    MAX(I) - MIN(I) AS I_span
FROM observations
GROUP BY entity_id
ORDER BY entity_id
      `
    },
    obs_by_signal: {
      name: "Group by Sensor",
      description: "Observation counts per sensor",
      sql: `
SELECT
    signal_id,
    unit,
    COUNT(*) AS n_observations,
    COUNT(DISTINCT entity_id) AS n_entities,
    ROUND(AVG(y), 4) AS y_mean,
    ROUND(MIN(y), 4) AS y_min,
    ROUND(MAX(y), 4) AS y_max
FROM observations
GROUP BY signal_id, unit
ORDER BY signal_id
      `
    },
    obs_units: {
      name: "Unit Distribution",
      description: "Distribution of units across sensors",
      sql: `
SELECT
    unit,
    COUNT(DISTINCT signal_id) AS n_signals,
    COUNT(*) AS n_observations,
    STRING_AGG(DISTINCT signal_id, ', ') AS signals
FROM observations
GROUP BY unit
ORDER BY n_signals DESC
      `
    },
    vector_summary: {
      name: "Vector Summary",
      description: "Summary of vector.parquet metrics",
      sql: `
SELECT
    entity_id,
    signal_id,
    COUNT(*) AS n_windows,
    ROUND(AVG(hurst_dfa), 3) AS avg_hurst,
    ROUND(AVG(sample_entropy), 3) AS avg_entropy,
    ROUND(AVG(trend_slope), 6) AS avg_trend
FROM vector
GROUP BY entity_id, signal_id
ORDER BY entity_id, signal_id
      `
    }
  },

  // ============================================================
  // GEOMETRY - Y10 Structure Engine (10-19)
  // Eigenvalue coherence, effective dimension, signal coupling
  // ============================================================
  geometry: {
    geo_coherence: {
      name: "Coherence Analysis",
      description: "Eigenvalue-based coherence from geometry.parquet",
      sql: `
SELECT
    entity_id,
    I,
    ROUND(coherence, 4) AS coherence,
    ROUND(eff_dim, 2) AS effective_dim,
    ROUND(hd_slope, 6) AS hd_slope,
    ROUND(eigenvalue_entropy, 4) AS eigen_entropy
FROM geometry
ORDER BY entity_id, I
LIMIT 100
      `
    },
    geo_coupling_state: {
      name: "Coupling State",
      description: "L2 coherence interpretation (from 14_l2_coherence.sql)",
      sql: `
SELECT
    entity_id,
    I,
    coupling_state,
    structure_state,
    interpretation
FROM v_l2_interpretation
ORDER BY entity_id, I
LIMIT 100
      `
    },
    geo_story_beginning: {
      name: "System Beginning State",
      description: "Initial system state (from 18_system_story.sql)",
      sql: `
SELECT
    entity_id,
    ROUND(initial_energy, 3) AS initial_energy,
    ROUND(initial_coherence, 3) AS initial_coherence,
    initial_coupling_state,
    initial_position,
    n_signals,
    opening_sentence
FROM v_story_beginning
ORDER BY entity_id
      `
    },
    geo_story_ending: {
      name: "System Ending State",
      description: "Final system state (from 18_system_story.sql)",
      sql: `
SELECT
    entity_id,
    ROUND(final_energy, 3) AS final_energy,
    ROUND(final_coherence, 3) AS final_coherence,
    final_coupling_state,
    final_motion,
    ROUND(energy_change_pct * 100, 1) AS energy_pct_change,
    closing_sentence
FROM v_story_ending
ORDER BY ABS(energy_change_pct) DESC
LIMIT 30
      `
    },
    geo_story_timeline: {
      name: "Event Timeline",
      description: "Significant events during analysis (from 18_system_story.sql)",
      sql: `
SELECT
    entity_id,
    event_time,
    event_category,
    event_type,
    ROUND(value_at_event, 3) AS value,
    description
FROM v_story_timeline
ORDER BY entity_id, event_time
LIMIT 100
      `
    },
    geo_correlation: {
      name: "Sensor Correlation",
      description: "Pairwise correlations from behavioral_geometry",
      sql: `
SELECT
    signal_a,
    signal_b,
    ROUND(correlation, 4) AS correlation,
    ROUND(coherence, 4) AS coherence
FROM behavioral_geometry
WHERE signal_a < signal_b
ORDER BY ABS(correlation) DESC
LIMIT 50
      `
    }
  },

  // ============================================================
  // PHYSICS - Y11 Physics Engine (12-26)
  // Thermodynamics, Orthon Signal, Force Attribution
  // ============================================================
  physics: {
    phys_orthon_signal: {
      name: "Orthon Signal Detection",
      description: "THE degradation signal (from 16_orthon_signal.sql)",
      sql: `
SELECT
    entity_id,
    I,
    orthon_signal,
    severity,
    severity_score,
    energy_trend,
    coupling_state,
    state_trend
FROM v_orthon_signal
WHERE orthon_signal = TRUE
ORDER BY severity_score DESC
LIMIT 50
      `
    },
    phys_entity_summary: {
      name: "Entity Health Summary",
      description: "Current status per entity (from 16_orthon_signal.sql)",
      sql: `
SELECT
    entity_id,
    current_orthon_signal AS orthon_active,
    current_severity AS severity,
    current_severity_score AS score,
    energy_trend,
    coupling_state,
    state_trend,
    ROUND(pct_in_signal, 1) AS pct_in_signal,
    status_message
FROM v_orthon_entity_summary
ORDER BY current_severity_score DESC
      `
    },
    phys_fleet_summary: {
      name: "Fleet Health Overview",
      description: "Fleet-wide summary (from 16_orthon_signal.sql)",
      sql: `
SELECT
    n_entities,
    n_with_orthon_signal,
    ROUND(pct_with_signal, 1) AS pct_signaling,
    n_critical,
    n_warning,
    n_watch,
    n_normal,
    ROUND(pct_healthy, 1) AS pct_healthy,
    n_strongly_coupled,
    n_decoupled
FROM v_orthon_fleet_summary
      `
    },
    phys_alerts: {
      name: "Active Alerts",
      description: "Current alerts (from 16_orthon_signal.sql)",
      sql: `
SELECT
    entity_id,
    I,
    alert_level,
    alert_message,
    severity_score,
    energy_trend,
    coupling_state
FROM v_orthon_alerts
ORDER BY
    CASE alert_level WHEN 'CRITICAL' THEN 1 WHEN 'WARNING' THEN 2 ELSE 3 END,
    severity_score DESC
LIMIT 50
      `
    },
    phys_attribution: {
      name: "Force Attribution",
      description: "Endogenous vs Exogenous (from 21_geometric_attribution.sql)",
      sql: `
SELECT
    entity_id,
    dominant_force,
    ROUND(endogenous_fraction * 100, 1) AS endogenous_pct,
    ROUND(exogenous_fraction * 100, 1) AS exogenous_pct,
    ROUND(attribution_ratio, 2) AS ratio,
    interpretation
FROM v_attribution_entity_summary
ORDER BY ABS(attribution_ratio - 1) DESC
      `
    },
    phys_energy_balance: {
      name: "Energy Balance",
      description: "Energy accounting (from 24_incident_summary.sql)",
      sql: `
SELECT
    entity_id,
    energy_injected,
    energy_dissipated,
    net_energy_change,
    energy_gap,
    energy_balance_status,
    gap_severity
FROM v_energy_balance
ORDER BY ABS(energy_gap) DESC
      `
    },
    phys_incident_summary: {
      name: "Incident Summary",
      description: "Comprehensive incident report (from 24_incident_summary.sql)",
      sql: `
SELECT
    entity_id,
    first_deviation_time,
    first_deviation_metric AS trigger,
    ROUND(first_z_score, 1) AS trigger_z,
    propagation_chain,
    peak_severity,
    ROUND(peak_z_score, 1) AS peak_z,
    current_severity,
    UPPER(dominant_force) AS force_type,
    CASE WHEN current_orthon_signal THEN 'ACTIVE' ELSE '-' END AS orthon
FROM v_incident_summary
ORDER BY
    CASE current_severity WHEN 'critical' THEN 1 WHEN 'warning' THEN 2 ELSE 3 END,
    peak_z_score DESC
LIMIT 30
      `
    },
    phys_deviation: {
      name: "Deviation Analysis",
      description: "Baseline deviation flags (from 23_baseline_deviation.sql)",
      sql: `
SELECT
    entity_id,
    I,
    severity,
    n_deviating_metrics,
    max_deviation_metric,
    ROUND(max_z_score, 2) AS max_z,
    ROUND(z_energy_proxy, 2) AS z_energy,
    ROUND(z_coherence, 2) AS z_coherence
FROM v_deviation_flags
WHERE severity != 'normal'
ORDER BY max_z_score DESC
LIMIT 50
      `
    }
  },

  // ============================================================
  // DYNAMICS - Y12 Dynamics Engine (30-33)
  // Lyapunov, RQA, Basin Stability, Birth Certificate
  // ============================================================
  dynamics: {
    dyn_stability: {
      name: "Stability Classification",
      description: "Entity stability from Lyapunov+RQA (from 30_dynamics_stability.sql)",
      sql: `
SELECT
    entity_id,
    entity_stability,
    dynamics_type,
    ROUND(stability_score, 3) AS stability_score,
    ROUND(mean_lyapunov, 4) AS lyapunov,
    ROUND(mean_determinism, 3) AS det,
    ROUND(mean_laminarity, 3) AS lam,
    n_windows
FROM v_dynamics_summary
ORDER BY stability_score ASC
      `
    },
    dyn_alerts: {
      name: "Dynamics Alerts",
      description: "Unstable entities (from 30_dynamics_stability.sql)",
      sql: `
SELECT
    entity_id,
    alert_level,
    alert_message,
    ROUND(severity_score, 3) AS severity
FROM v_dynamics_alerts
ORDER BY
    CASE alert_level WHEN 'CRITICAL' THEN 1 WHEN 'WARNING' THEN 2 ELSE 3 END,
    severity_score DESC
      `
    },
    dyn_temporal: {
      name: "Lyapunov Evolution",
      description: "Window-level dynamics (from 30_dynamics_stability.sql)",
      sql: `
SELECT
    entity_id,
    I,
    ROUND(lyapunov_max, 4) AS lyapunov,
    ROUND(determinism, 3) AS det,
    ROUND(laminarity, 3) AS lam,
    lyapunov_class,
    rqa_class,
    ROUND(lyap_delta, 4) AS lyap_delta
FROM v_dynamics_temporal
ORDER BY entity_id, I
LIMIT 100
      `
    },
    dyn_regime_transitions: {
      name: "Regime Transitions",
      description: "Stability changes over time (from 31_regime_transitions.sql)",
      sql: `
SELECT
    entity_id,
    transition_time,
    from_regime,
    to_regime,
    transition_type,
    ROUND(lyapunov_before, 4) AS lyap_before,
    ROUND(lyapunov_after, 4) AS lyap_after,
    interpretation
FROM v_regime_transitions
ORDER BY transition_time
LIMIT 50
      `
    },
    dyn_basin_stability: {
      name: "Basin Stability",
      description: "Composite stability score (from 32_basin_stability.sql)",
      sql: `
SELECT
    entity_id,
    basin_class,
    ROUND(basin_stability_score, 3) AS basin_score,
    ROUND(return_probability, 3) AS return_prob,
    ROUND(mean_return_time, 1) AS return_time,
    n_perturbations,
    assessment
FROM v_basin_stability
ORDER BY basin_stability_score ASC
      `
    },
    dyn_birth_certificate: {
      name: "Birth Certificate",
      description: "Early-life prognosis (from 33_birth_certificate.sql)",
      sql: `
SELECT
    entity_id,
    birth_grade,
    ROUND(birth_stability, 3) AS birth_stability,
    ROUND(birth_coherence, 3) AS birth_coherence,
    ROUND(expected_lifespan, 0) AS expected_life,
    ROUND(actual_lifespan, 0) AS actual_life,
    prognosis
FROM v_birth_prognosis
ORDER BY
    CASE birth_grade WHEN 'A' THEN 1 WHEN 'B' THEN 2 WHEN 'C' THEN 3 WHEN 'D' THEN 4 ELSE 5 END
      `
    },
    dyn_bifurcation: {
      name: "Bifurcation Detection",
      description: "Windows where stability changes sign",
      sql: `
WITH stability_changes AS (
    SELECT
        entity_id, I, lyapunov_max,
        SIGN(lyapunov_max) AS sign_now,
        LAG(SIGN(lyapunov_max)) OVER (PARTITION BY entity_id ORDER BY I) AS sign_prev
    FROM dynamics
)
SELECT
    entity_id, I,
    ROUND(lyapunov_max, 4) AS lyapunov,
    CASE
        WHEN sign_prev <= 0 AND sign_now > 0 THEN 'ONSET_OF_CHAOS'
        WHEN sign_prev > 0 AND sign_now <= 0 THEN 'RETURN_TO_STABILITY'
    END AS bifurcation_type
FROM stability_changes
WHERE sign_now != sign_prev AND sign_prev IS NOT NULL
ORDER BY I
      `
    },
    dyn_attractor: {
      name: "Attractor Properties",
      description: "Phase space attractor characteristics",
      sql: `
SELECT
    entity_id,
    I,
    ROUND(correlation_dim, 2) AS fractal_dim,
    CASE
        WHEN correlation_dim < 1.5 THEN 'POINT'
        WHEN correlation_dim < 2.5 THEN 'LIMIT_CYCLE'
        WHEN correlation_dim < 3.5 THEN 'TORUS'
        ELSE 'STRANGE'
    END AS attractor_type,
    embedding_dim,
    ROUND(lyapunov_max, 4) AS lyapunov
FROM dynamics
ORDER BY entity_id, I
LIMIT 100
      `
    }
  },

  // ============================================================
  // ADVANCED - Y13 Advanced Analysis (PR #13 Final)
  // Causality, Topology, Emergence, Integration
  // ============================================================
  advanced: {
    // ----- CAUSALITY ENGINE (30_causality_reports.sql) -----
    causal_significant_edges: {
      name: "Significant Causal Edges",
      description: "Strong causal links (from 30_causality_reports.sql)",
      sql: `
SELECT entity_id, window_id, source, target,
    ROUND(granger_f, 2) AS granger_f,
    ROUND(granger_p, 4) AS granger_p,
    ROUND(transfer_entropy, 4) AS te
FROM causality_edges
WHERE is_significant = TRUE
ORDER BY transfer_entropy DESC
LIMIT 50
      `
    },
    causal_network_summary: {
      name: "Causal Network Summary",
      description: "Network structure overview (from 30_causality_reports.sql)",
      sql: `
SELECT entity_id, window_id, timestamp_start,
    ROUND(density, 3) AS density,
    ROUND(hierarchy, 3) AS hierarchy,
    n_feedback_loops,
    top_driver, ROUND(top_driver_flow, 3) AS driver_flow,
    top_sink, bottleneck,
    CASE
        WHEN n_feedback_loops > 5 THEN 'COMPLEX'
        WHEN hierarchy < 0.3 THEN 'CIRCULAR'
        WHEN hierarchy > 0.7 THEN 'HIERARCHICAL'
        ELSE 'MODERATE'
    END AS network_type
FROM causality_network
ORDER BY entity_id, window_id
LIMIT 100
      `
    },
    causal_feedback_alerts: {
      name: "Feedback Loop Alerts",
      description: "Feedback loop warnings (from 30_causality_reports.sql)",
      sql: `
SELECT entity_id, window_id, timestamp_start,
    n_feedback_loops, ROUND(hierarchy, 3) AS hierarchy,
    CASE
        WHEN n_feedback_loops > 5 THEN 'WARNING'
        WHEN n_feedback_loops > 3 THEN 'ELEVATED'
        ELSE 'NORMAL'
    END AS loop_status
FROM causality_network
WHERE n_feedback_loops > 2
ORDER BY n_feedback_loops DESC
      `
    },
    causal_driver_changes: {
      name: "Driver Changes",
      description: "When dominant drivers shift",
      sql: `
WITH lagged AS (
    SELECT entity_id, window_id, top_driver,
        LAG(top_driver) OVER (PARTITION BY entity_id ORDER BY window_id) AS prev_driver
    FROM causality_network
)
SELECT entity_id, window_id, top_driver, prev_driver,
    'DRIVER_CHANGE' AS event_type
FROM lagged
WHERE prev_driver IS NOT NULL AND top_driver != prev_driver
ORDER BY window_id
      `
    },

    // ----- TOPOLOGY ENGINE (31_topology_reports.sql) -----
    topo_summary: {
      name: "Topology Summary",
      description: "Topological state (from 31_topology_reports.sql)",
      sql: `
SELECT entity_id, window_id, timestamp_start,
    betti_0, betti_1, betti_2,
    ROUND(topological_complexity, 2) AS complexity,
    fragmentation,
    ROUND(topology_change, 3) AS topo_change,
    CASE
        WHEN fragmentation THEN 'FRAGMENTED'
        WHEN betti_1 > 2 THEN 'COMPLEX_LOOPS'
        WHEN topology_change > 0.5 THEN 'CHANGING'
        ELSE 'STABLE'
    END AS topology_status
FROM topology
ORDER BY entity_id, window_id
LIMIT 100
      `
    },
    topo_fragmentation_alerts: {
      name: "Fragmentation Alerts",
      description: "Attractor fragmentation (from 31_topology_reports.sql)",
      sql: `
SELECT entity_id, window_id, timestamp_start,
    betti_0, ROUND(topological_complexity, 2) AS complexity,
    'FRAGMENTATION' AS alert_type
FROM topology
WHERE fragmentation = TRUE
ORDER BY topological_complexity DESC
      `
    },
    topo_evolution: {
      name: "Topology Evolution",
      description: "Betti number changes over time",
      sql: `
WITH lagged AS (
    SELECT entity_id, window_id, betti_1,
        LAG(betti_1) OVER (PARTITION BY entity_id ORDER BY window_id) AS prev_betti_1
    FROM topology
)
SELECT entity_id, window_id, betti_1, prev_betti_1,
    betti_1 - prev_betti_1 AS betti_1_change,
    CASE
        WHEN betti_1 > prev_betti_1 THEN 'LOOPS_APPEARING'
        WHEN betti_1 < prev_betti_1 THEN 'LOOPS_DISAPPEARING'
        ELSE 'STABLE'
    END AS evolution
FROM lagged
WHERE prev_betti_1 IS NOT NULL AND betti_1 != prev_betti_1
ORDER BY window_id
      `
    },

    // ----- EMERGENCE ENGINE (32_emergence_reports.sql) -----
    emergence_high_synergy: {
      name: "High Synergy Triplets",
      description: "Multi-sensor emergence (from 32_emergence_reports.sql)",
      sql: `
SELECT entity_id, window_id, timestamp_start,
    source_1, source_2, target,
    ROUND(synergy, 4) AS synergy,
    ROUND(synergy_ratio, 3) AS synergy_ratio,
    ROUND(total_info, 4) AS total_info,
    CASE
        WHEN synergy_ratio > 0.5 THEN 'HIGH_EMERGENCE'
        WHEN synergy_ratio > 0.3 THEN 'MODERATE_EMERGENCE'
        ELSE 'LOW_EMERGENCE'
    END AS emergence_level
FROM emergence_triplets
WHERE synergy_ratio > 0.2
ORDER BY synergy_ratio DESC
LIMIT 50
      `
    },
    emergence_redundant: {
      name: "Redundant Sensors",
      description: "Redundant sensor pairs (from 32_emergence_reports.sql)",
      sql: `
SELECT entity_id, window_id,
    source_1, source_2, target,
    ROUND(redundancy, 4) AS redundancy,
    ROUND(total_info, 4) AS total_info,
    ROUND(redundancy / NULLIF(total_info, 0), 3) AS redundancy_ratio
FROM emergence_triplets
WHERE redundancy / NULLIF(total_info, 0) > 0.7
ORDER BY redundancy DESC
LIMIT 50
      `
    },
    emergence_unique_sources: {
      name: "Unique Information Sources",
      description: "Sensors with unique information",
      sql: `
SELECT entity_id, window_id, source_1, source_2, target,
    ROUND(unique_1, 4) AS unique_1,
    ROUND(unique_2, 4) AS unique_2,
    CASE
        WHEN unique_1 > unique_2 * 2 THEN source_1
        WHEN unique_2 > unique_1 * 2 THEN source_2
        ELSE 'BALANCED'
    END AS dominant_source
FROM emergence_triplets
WHERE unique_1 > 0.1 OR unique_2 > 0.1
ORDER BY GREATEST(unique_1, unique_2) DESC
LIMIT 50
      `
    },

    // ----- INTEGRATION ENGINE (33_integration_reports.sql) -----
    health_dashboard: {
      name: "Health Dashboard",
      description: "Unified health view (from 33_integration_reports.sql)",
      sql: `
SELECT entity_id, window_id, timestamp_start,
    health_score, risk_level,
    ROUND(stability_score, 2) AS stability,
    ROUND(predictability_score, 2) AS predictability,
    ROUND(physics_score, 2) AS physics,
    ROUND(topology_score, 2) AS topology,
    ROUND(causality_score, 2) AS causality,
    primary_concern, secondary_concern, recommendation
FROM health
ORDER BY health_score ASC
LIMIT 50
      `
    },
    health_critical_alerts: {
      name: "Critical Health Alerts",
      description: "High/critical risk entities (from 33_integration_reports.sql)",
      sql: `
SELECT entity_id, window_id, timestamp_start,
    health_score, risk_level,
    primary_concern, recommendation
FROM health
WHERE risk_level IN ('CRITICAL', 'HIGH')
ORDER BY health_score ASC
      `
    },
    health_trends: {
      name: "Health Trends",
      description: "Health score changes over time",
      sql: `
WITH lagged AS (
    SELECT entity_id, window_id, health_score,
        LAG(health_score) OVER (PARTITION BY entity_id ORDER BY window_id) AS prev_health
    FROM health
)
SELECT entity_id, window_id,
    ROUND(health_score, 1) AS health,
    ROUND(prev_health, 1) AS prev_health,
    ROUND(health_score - prev_health, 1) AS health_change,
    CASE
        WHEN health_score < prev_health - 10 THEN 'DEGRADING'
        WHEN health_score > prev_health + 10 THEN 'IMPROVING'
        ELSE 'STABLE'
    END AS trend
FROM lagged
WHERE prev_health IS NOT NULL
ORDER BY ABS(health_score - prev_health) DESC
LIMIT 50
      `
    },
    health_entity_ranking: {
      name: "Entity Health Ranking",
      description: "Entities ranked by health",
      sql: `
SELECT entity_id,
    ROUND(AVG(health_score), 1) AS avg_health,
    ROUND(MIN(health_score), 1) AS min_health,
    ROUND(MAX(health_score), 1) AS max_health,
    COUNT(CASE WHEN risk_level = 'CRITICAL' THEN 1 END) AS critical_events
FROM health
GROUP BY entity_id
ORDER BY avg_health ASC
      `
    }
  },

  // ============================================================
  // SUMMARY - Fleet-level views and cross-pillar analysis
  // ============================================================
  summary: {
    sum_fleet: {
      name: "Fleet Health Overview",
      description: "Overall fleet status across all pillars",
      sql: `
SELECT
    n_entities,
    n_with_orthon_signal,
    ROUND(pct_with_signal, 1) AS pct_signaling,
    n_critical,
    n_warning,
    n_watch,
    n_normal,
    ROUND(pct_healthy, 1) AS pct_healthy
FROM v_orthon_fleet_summary
      `
    },
    sum_orthon: {
      name: "All Orthon Signals",
      description: "All entities with active degradation signal",
      sql: `
SELECT
    entity_id,
    current_severity AS severity,
    current_severity_score AS score,
    energy_trend,
    coupling_state,
    state_trend,
    status_message
FROM v_orthon_entity_summary
WHERE current_orthon_signal = TRUE
ORDER BY current_severity_score DESC
      `
    },
    sum_narratives: {
      name: "Entity Narratives",
      description: "Complete story for each entity (from 18_system_story.sql)",
      sql: `
SELECT
    entity_id,
    overall_trajectory,
    likely_causation,
    ROUND(initial_energy, 3) AS start_energy,
    ROUND(final_energy, 3) AS end_energy,
    ROUND(initial_coherence, 3) AS start_coherence,
    ROUND(final_coherence, 3) AS end_coherence,
    full_narrative
FROM v_story_complete
ORDER BY
    CASE overall_trajectory
        WHEN 'degradation' THEN 1
        WHEN 'ongoing_issue' THEN 2
        WHEN 'overload' THEN 3
        WHEN 'recovery' THEN 4
        ELSE 5
    END
      `
    },
    sum_story_fleet: {
      name: "Fleet Story Summary",
      description: "Fleet-level trajectory summary (from 18_system_story.sql)",
      sql: `
SELECT
    n_entities,
    n_degrading,
    n_overloaded,
    n_recovering,
    n_ongoing_issues,
    n_stable,
    n_exogenous,
    n_endogenous,
    ROUND(avg_energy_change, 4) AS avg_energy_delta,
    ROUND(avg_coherence_change, 3) AS avg_coherence_delta
FROM v_story_fleet_summary
      `
    },
    sum_cross_layer: {
      name: "Cross-Pillar Agreement",
      description: "Correlation between pillar health scores",
      sql: `
SELECT
    'CROSS-LAYER AGREEMENT' AS metric,
    ROUND(CORR(d.stability_score, t.topology_health_score), 2) AS dynamics_topology_r,
    ROUND(CORR(d.stability_score, i.information_health_score), 2) AS dynamics_info_r,
    ROUND(CORR(t.topology_health_score, i.information_health_score), 2) AS topology_info_r
FROM dynamics_entities d
JOIN topology_entities t ON d.entity_id = t.entity_id
JOIN information_entities i ON d.entity_id = i.entity_id
      `
    },
    sum_all_alerts: {
      name: "All Active Alerts",
      description: "Combined alerts from all four pillars",
      sql: `
SELECT 'PHYSICS' AS pillar, entity_id, alert_level, alert_message, severity_score
FROM v_orthon_alerts WHERE alert_level IN ('CRITICAL', 'WARNING')
UNION ALL
SELECT 'DYNAMICS' AS pillar, entity_id, alert_level, alert_message, severity_score
FROM v_dynamics_alerts WHERE alert_level IN ('CRITICAL', 'WARNING')
UNION ALL
SELECT 'TOPOLOGY' AS pillar, entity_id, alert_level, alert_message, severity_score
FROM v_topology_alerts WHERE alert_level IN ('CRITICAL', 'WARNING')
UNION ALL
SELECT 'INFORMATION' AS pillar, entity_id, alert_level, alert_message, severity_score
FROM v_information_alerts WHERE alert_level IN ('CRITICAL', 'WARNING')
ORDER BY
    CASE pillar WHEN 'PHYSICS' THEN 1 WHEN 'DYNAMICS' THEN 2 WHEN 'TOPOLOGY' THEN 3 ELSE 4 END,
    severity_score DESC
LIMIT 50
      `
    },
    sum_incident_overview: {
      name: "Incident Fleet Overview",
      description: "Fleet incident summary (from 24_incident_summary.sql)",
      sql: `
SELECT
    n_entities_with_incidents,
    n_critical,
    n_warning,
    n_normal,
    n_orthon_signals,
    n_external,
    n_internal,
    n_unmeasured_sinks,
    n_unmeasured_sources,
    most_common_trigger
FROM v_fleet_incident_overview
      `
    },
    sum_severity_ranking: {
      name: "Severity Ranking",
      description: "All entities ranked by combined severity",
      sql: `
WITH combined AS (
    SELECT
        p.entity_id,
        p.current_severity_score AS physics_score,
        COALESCE(d.stability_score, 0.5) AS dynamics_score,
        COALESCE(t.topology_health_score, 0.5) AS topology_score,
        COALESCE(i.information_health_score, 0.5) AS info_score
    FROM v_orthon_entity_summary p
    LEFT JOIN dynamics_entities d ON p.entity_id = d.entity_id
    LEFT JOIN topology_entities t ON p.entity_id = t.entity_id
    LEFT JOIN information_entities i ON p.entity_id = i.entity_id
)
SELECT
    entity_id,
    ROUND(physics_score, 2) AS physics,
    ROUND(1 - dynamics_score, 2) AS dynamics_risk,
    ROUND(1 - topology_score, 2) AS topology_risk,
    ROUND(1 - info_score, 2) AS info_risk,
    ROUND(physics_score + (1 - dynamics_score) + (1 - topology_score) + (1 - info_score), 2) AS combined_risk
FROM combined
ORDER BY combined_risk DESC
LIMIT 30
      `
    },
    // ----- INTEGRATION ENGINE FLEET SUMMARY (PR #13) -----
    sum_health_fleet: {
      name: "Health Fleet Summary",
      description: "Fleet-wide health statistics (from 33_integration_reports.sql)",
      sql: `
SELECT
    COUNT(*) AS total_entities,
    COUNT(CASE WHEN risk_level = 'LOW' THEN 1 END) AS healthy,
    COUNT(CASE WHEN risk_level = 'MODERATE' THEN 1 END) AS moderate,
    COUNT(CASE WHEN risk_level = 'HIGH' THEN 1 END) AS high_risk,
    COUNT(CASE WHEN risk_level = 'CRITICAL' THEN 1 END) AS critical,
    ROUND(AVG(health_score), 1) AS avg_health,
    ROUND(MIN(health_score), 1) AS min_health
FROM health
WHERE window_id = (SELECT MAX(window_id) FROM health)
      `
    },
    sum_complete_picture: {
      name: "Complete Health Picture",
      description: "Final unified health view (PR #13 Integration Engine)",
      sql: `
SELECT
    entity_id,
    health_score,
    risk_level,
    primary_concern,
    secondary_concern,
    recommendation,
    ROUND(stability_score * 100, 0) AS stability_pct,
    ROUND(predictability_score * 100, 0) AS predict_pct,
    ROUND(physics_score * 100, 0) AS physics_pct,
    ROUND(topology_score * 100, 0) AS topo_pct,
    ROUND(causality_score * 100, 0) AS causal_pct
FROM health
WHERE window_id = (SELECT MAX(window_id) FROM health)
ORDER BY health_score ASC
LIMIT 30
      `
    }
  },

  // ============================================================
  // STATISTICS - Y14 Statistics Engines (PR #14 Final)
  // Baselines, Anomalies, Fleet Analytics, Executive Reports
  // ============================================================
  statistics: {
    // ----- BASELINE ENGINE (40_baseline_reports.sql) -----
    baseline_overview: {
      name: "Baseline Overview",
      description: "All baseline statistics (from 40_baseline_reports.sql)",
      sql: `
SELECT
    metric_source,
    metric_name,
    entity_id,
    ROUND(mean, 4) AS mean,
    ROUND(std, 4) AS std,
    ROUND(median, 4) AS median,
    ROUND(p5, 4) AS p5,
    ROUND(p95, 4) AS p95,
    n_samples,
    ROUND(p95 - p5, 4) AS iqr_90
FROM baseline
ORDER BY metric_source, metric_name
LIMIT 100
      `
    },
    baseline_high_variance: {
      name: "High Variance Metrics",
      description: "Metrics with high coefficient of variation",
      sql: `
SELECT
    metric_source,
    metric_name,
    ROUND(mean, 4) AS mean,
    ROUND(std, 4) AS std,
    ROUND(std / NULLIF(ABS(mean), 0.001), 3) AS cv,
    CASE
        WHEN std / NULLIF(ABS(mean), 0.001) > 0.5 THEN 'HIGH_VARIANCE'
        WHEN std / NULLIF(ABS(mean), 0.001) > 0.25 THEN 'MODERATE_VARIANCE'
        ELSE 'LOW_VARIANCE'
    END AS variance_level
FROM baseline
WHERE entity_id = 'FLEET'
ORDER BY cv DESC
LIMIT 50
      `
    },
    baseline_entity_vs_fleet: {
      name: "Entity vs Fleet Baseline",
      description: "Entity deviations from fleet baseline",
      sql: `
SELECT
    e.metric_source,
    e.metric_name,
    e.entity_id,
    ROUND(e.mean, 4) AS entity_mean,
    ROUND(f.mean, 4) AS fleet_mean,
    ROUND(e.mean - f.mean, 4) AS deviation,
    ROUND((e.mean - f.mean) / NULLIF(f.std, 0.001), 2) AS z_from_fleet
FROM baseline e
JOIN baseline f ON e.metric_source = f.metric_source
    AND e.metric_name = f.metric_name
    AND f.entity_id = 'FLEET'
WHERE e.entity_id != 'FLEET'
ORDER BY ABS(z_from_fleet) DESC
LIMIT 50
      `
    },

    // ----- ANOMALY ENGINE (41_anomaly_reports.sql) -----
    anomaly_current: {
      name: "Current Anomalies",
      description: "All current anomalies (from 41_anomaly_reports.sql)",
      sql: `
SELECT
    entity_id,
    metric_source,
    metric_name,
    ROUND(value, 4) AS value,
    ROUND(baseline_mean, 4) AS baseline,
    ROUND(z_score, 2) AS z_score,
    anomaly_severity
FROM anomaly
WHERE is_anomaly = TRUE
ORDER BY ABS(z_score) DESC
LIMIT 50
      `
    },
    anomaly_by_entity: {
      name: "Anomalies by Entity",
      description: "Anomaly counts per entity",
      sql: `
SELECT
    entity_id,
    COUNT(*) AS total_anomalies,
    COUNT(CASE WHEN anomaly_severity = 'CRITICAL' THEN 1 END) AS critical,
    COUNT(CASE WHEN anomaly_severity = 'WARNING' THEN 1 END) AS warning,
    ROUND(AVG(ABS(z_score)), 2) AS avg_abs_z,
    ROUND(MAX(ABS(z_score)), 2) AS max_abs_z
FROM anomaly
WHERE is_anomaly = TRUE
GROUP BY entity_id
ORDER BY total_anomalies DESC
      `
    },
    anomaly_by_metric: {
      name: "Anomalies by Metric",
      description: "Anomaly counts per metric",
      sql: `
SELECT
    metric_source,
    metric_name,
    COUNT(*) AS total_anomalies,
    COUNT(DISTINCT entity_id) AS affected_entities,
    ROUND(AVG(z_score), 2) AS avg_z,
    ROUND(MAX(ABS(z_score)), 2) AS max_abs_z
FROM anomaly
WHERE is_anomaly = TRUE
GROUP BY metric_source, metric_name
ORDER BY total_anomalies DESC
LIMIT 30
      `
    },
    anomaly_timeline: {
      name: "Anomaly Timeline",
      description: "Anomalies over time",
      sql: `
SELECT
    window_id,
    COUNT(*) AS n_anomalies,
    COUNT(CASE WHEN anomaly_severity = 'CRITICAL' THEN 1 END) AS critical,
    COUNT(DISTINCT entity_id) AS affected_entities
FROM anomaly
WHERE is_anomaly = TRUE
GROUP BY window_id
ORDER BY window_id
      `
    },

    // ----- FLEET ENGINE (42_fleet_reports.sql) -----
    fleet_rankings: {
      name: "Entity Rankings",
      description: "All entities ranked by health (from 42_fleet_reports.sql)",
      sql: `
SELECT
    entity_id,
    health_rank,
    ROUND(avg_health, 1) AS avg_health,
    ROUND(min_health, 1) AS min_health,
    ROUND(latest_health, 1) AS latest_health,
    ROUND(health_volatility, 2) AS volatility,
    critical_events,
    high_events,
    cluster,
    health_tier,
    total_anomalies,
    critical_anomalies
FROM fleet_rankings
ORDER BY health_rank
      `
    },
    fleet_clusters: {
      name: "Cluster Summary",
      description: "Entity clusters (from 42_fleet_reports.sql)",
      sql: `
SELECT
    cluster,
    COUNT(*) AS n_entities,
    ROUND(AVG(avg_health), 1) AS cluster_avg_health,
    SUM(critical_events) AS total_critical,
    ROUND(AVG(health_volatility), 2) AS avg_volatility
FROM fleet_rankings
GROUP BY cluster
ORDER BY cluster_avg_health
      `
    },
    fleet_tier_distribution: {
      name: "Health Tier Distribution",
      description: "Entities by health tier",
      sql: `
SELECT
    health_tier,
    COUNT(*) AS n_entities,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_fleet,
    ROUND(AVG(avg_health), 1) AS tier_avg_health
FROM fleet_rankings
GROUP BY health_tier
ORDER BY tier_avg_health DESC
      `
    },
    fleet_top_bottom: {
      name: "Top/Bottom Performers",
      description: "Best and worst performing entities",
      sql: `
SELECT 'TOP' AS category, entity_id, ROUND(avg_health, 1) AS avg_health, health_tier
FROM fleet_rankings
ORDER BY avg_health DESC
LIMIT 10

UNION ALL

SELECT 'BOTTOM' AS category, entity_id, ROUND(avg_health, 1) AS avg_health, health_tier
FROM fleet_rankings
ORDER BY avg_health ASC
LIMIT 10
      `
    },

    // ----- SUMMARY ENGINE (43_summary_reports.sql) -----
    summary_kpis: {
      name: "Executive KPIs",
      description: "Key performance indicators (from 43_summary_reports.sql)",
      sql: `
SELECT
    (SELECT COUNT(DISTINCT entity_id) FROM health) AS total_entities,
    (SELECT ROUND(AVG(health_score), 1) FROM health WHERE window_id = (SELECT MAX(window_id) FROM health)) AS current_avg_health,
    (SELECT COUNT(*) FROM health WHERE window_id = (SELECT MAX(window_id) FROM health) AND risk_level = 'CRITICAL') AS critical_count,
    (SELECT COUNT(*) FROM health WHERE window_id = (SELECT MAX(window_id) FROM health) AND risk_level = 'HIGH') AS high_risk_count,
    (SELECT COUNT(*) FROM anomaly WHERE is_anomaly = TRUE AND window_id = (SELECT MAX(window_id) FROM anomaly)) AS current_anomalies
      `
    },
    summary_health_distribution: {
      name: "Health Distribution",
      description: "Health score histogram",
      sql: `
SELECT
    CASE
        WHEN health_score >= 90 THEN '90-100 (Excellent)'
        WHEN health_score >= 80 THEN '80-89 (Good)'
        WHEN health_score >= 70 THEN '70-79 (Fair)'
        WHEN health_score >= 60 THEN '60-69 (Poor)'
        WHEN health_score >= 50 THEN '50-59 (At Risk)'
        ELSE '< 50 (Critical)'
    END AS health_bucket,
    COUNT(*) AS n_entities
FROM health
WHERE window_id = (SELECT MAX(window_id) FROM health)
GROUP BY health_bucket
ORDER BY health_bucket DESC
      `
    },
    summary_risk_trend: {
      name: "Risk Level Trend",
      description: "Risk levels over time",
      sql: `
SELECT
    window_id,
    COUNT(CASE WHEN risk_level = 'LOW' THEN 1 END) AS low,
    COUNT(CASE WHEN risk_level = 'MODERATE' THEN 1 END) AS moderate,
    COUNT(CASE WHEN risk_level = 'HIGH' THEN 1 END) AS high,
    COUNT(CASE WHEN risk_level = 'CRITICAL' THEN 1 END) AS critical
FROM health
GROUP BY window_id
ORDER BY window_id
      `
    },
    summary_top_concerns: {
      name: "Top Concerns Summary",
      description: "Most common concerns across fleet",
      sql: `
SELECT
    primary_concern,
    COUNT(*) AS n_entities,
    ROUND(AVG(health_score), 1) AS avg_health_of_affected
FROM health
WHERE window_id = (SELECT MAX(window_id) FROM health)
GROUP BY primary_concern
ORDER BY n_entities DESC
LIMIT 10
      `
    }
  },

  // ============================================================
  // VALIDATION - Data Sufficiency & Finding Significance
  // Based on INTERPRETATION_THRESHOLDS.md v2.0
  // "A finding is not significant just because the math runs."
  // Updated: Revised Lyapunov (3k hard), DIV metric, coherence velocity, 4-tier fleet
  // ============================================================
  validation: {
    // ----- DATA SUFFICIENCY (Updated thresholds) -----
    data_sufficiency: {
      name: "Data Sufficiency Check",
      description: "Validates if entities have enough data for reliable engine results (updated thresholds)",
      sql: `
WITH entity_stats AS (
    SELECT
        entity_id,
        COUNT(*) AS total_obs,
        COUNT(DISTINCT signal_id) AS n_signals
    FROM observations
    GROUP BY entity_id
)
SELECT
    entity_id,
    total_obs,
    n_signals,

    -- Engine readiness (updated: Lyapunov 3k, corr_dim depends on embedding)
    CASE WHEN total_obs >= 3000 THEN 'OK' ELSE 'INSUFFICIENT' END AS lyapunov,
    CASE WHEN total_obs >= 1000 THEN 'OK' ELSE 'INSUFFICIENT' END AS corr_dim_low,
    CASE WHEN total_obs >= 5000 THEN 'OK' ELSE 'INSUFFICIENT' END AS corr_dim_high,
    CASE WHEN total_obs >= 1000 AND n_signals >= 3 THEN 'OK' ELSE 'INSUFFICIENT' END AS transfer_entropy,
    CASE WHEN total_obs >= 500 AND n_signals >= 2 THEN 'OK' ELSE 'INSUFFICIENT' END AS granger,
    CASE WHEN total_obs >= 1000 THEN 'OK' ELSE 'INSUFFICIENT' END AS rqa,
    CASE WHEN total_obs >= 500 THEN 'OK' ELSE 'INSUFFICIENT' END AS topology,
    CASE WHEN n_signals >= 3 THEN 'OK' ELSE 'INSUFFICIENT' END AS coherence,

    -- Lyapunov confidence (3k marginal, 10k reliable)
    CASE
        WHEN total_obs >= 10000 THEN 'RELIABLE'
        WHEN total_obs >= 3000 THEN 'MARGINAL'
        ELSE 'UNRELIABLE'
    END AS lyapunov_confidence,

    -- Overall capability
    CASE
        WHEN total_obs >= 5000 AND n_signals >= 3 THEN 'FULL_ANALYSIS'
        WHEN total_obs >= 1000 AND n_signals >= 2 THEN 'PARTIAL_ANALYSIS'
        ELSE 'LIMITED_ANALYSIS'
    END AS analysis_capability
FROM entity_stats
ORDER BY total_obs DESC
      `
    },
    engine_readiness_summary: {
      name: "Engine Readiness Summary",
      description: "Count of entities ready for each engine type (updated thresholds)",
      sql: `
WITH entity_stats AS (
    SELECT
        entity_id,
        COUNT(*) AS total_obs,
        COUNT(DISTINCT signal_id) AS n_signals
    FROM observations
    GROUP BY entity_id
)
SELECT
    'Lyapunov ready (3k+ obs)' AS engine,
    SUM(CASE WHEN total_obs >= 3000 THEN 1 ELSE 0 END) AS ready,
    COUNT(*) AS total,
    ROUND(100.0 * SUM(CASE WHEN total_obs >= 3000 THEN 1 ELSE 0 END) / COUNT(*), 0) AS pct_ready
FROM entity_stats
UNION ALL
SELECT 'Lyapunov reliable (10k+ obs)',
    SUM(CASE WHEN total_obs >= 10000 THEN 1 ELSE 0 END),
    COUNT(*),
    ROUND(100.0 * SUM(CASE WHEN total_obs >= 10000 THEN 1 ELSE 0 END) / COUNT(*), 0)
FROM entity_stats
UNION ALL
SELECT 'Corr Dim low-d (1k+ obs)',
    SUM(CASE WHEN total_obs >= 1000 THEN 1 ELSE 0 END),
    COUNT(*),
    ROUND(100.0 * SUM(CASE WHEN total_obs >= 1000 THEN 1 ELSE 0 END) / COUNT(*), 0)
FROM entity_stats
UNION ALL
SELECT 'Corr Dim high-d (5k+ obs)',
    SUM(CASE WHEN total_obs >= 5000 THEN 1 ELSE 0 END),
    COUNT(*),
    ROUND(100.0 * SUM(CASE WHEN total_obs >= 5000 THEN 1 ELSE 0 END) / COUNT(*), 0)
FROM entity_stats
UNION ALL
SELECT 'Transfer Entropy (1k obs, 3 sig)',
    SUM(CASE WHEN total_obs >= 1000 AND n_signals >= 3 THEN 1 ELSE 0 END),
    COUNT(*),
    ROUND(100.0 * SUM(CASE WHEN total_obs >= 1000 AND n_signals >= 3 THEN 1 ELSE 0 END) / COUNT(*), 0)
FROM entity_stats
UNION ALL
SELECT 'RQA (1k+ obs)',
    SUM(CASE WHEN total_obs >= 1000 THEN 1 ELSE 0 END),
    COUNT(*),
    ROUND(100.0 * SUM(CASE WHEN total_obs >= 1000 THEN 1 ELSE 0 END) / COUNT(*), 0)
FROM entity_stats
UNION ALL
SELECT 'Topology (500+ obs)',
    SUM(CASE WHEN total_obs >= 500 THEN 1 ELSE 0 END),
    COUNT(*),
    ROUND(100.0 * SUM(CASE WHEN total_obs >= 500 THEN 1 ELSE 0 END) / COUNT(*), 0)
FROM entity_stats
      `
    },

    // ----- FLEET VALIDITY (Updated 4-tier guidance) -----
    fleet_validity: {
      name: "Fleet Size Validity",
      description: "Validates if fleet is large enough for statistical claims (4-tier guidance)",
      sql: `
SELECT
    COUNT(DISTINCT entity_id) AS fleet_size,
    CASE
        WHEN COUNT(DISTINCT entity_id) >= 30 THEN 'RELIABLE - Full parametric stats'
        WHEN COUNT(DISTINCT entity_id) >= 10 THEN 'MARGINAL - Use percentiles, z-scores with caution'
        WHEN COUNT(DISTINCT entity_id) >= 5 THEN 'LIMITED - Robust stats only (median, IQR), NO z-scores'
        ELSE 'MINIMAL - Individual baselines only'
    END AS z_score_validity,
    CASE
        WHEN COUNT(DISTINCT entity_id) >= 50 THEN 'Full clustering'
        WHEN COUNT(DISTINCT entity_id) >= 20 THEN 'Basic clustering'
        WHEN COUNT(DISTINCT entity_id) >= 10 THEN 'Limited clustering'
        ELSE 'Clustering not recommended'
    END AS clustering_validity,
    CASE
        WHEN COUNT(DISTINCT entity_id) >= 30 THEN 'Z-scores, percentiles, clustering, anomaly detection'
        WHEN COUNT(DISTINCT entity_id) >= 10 THEN 'Percentile ranks, z-scores WITH CAUTION'
        WHEN COUNT(DISTINCT entity_id) >= 5 THEN 'Median, IQR, percentile ranks - NO z-scores'
        ELSE 'Individual entity analysis only'
    END AS valid_analyses
FROM observations
      `
    },

    // ----- BASELINE VALIDITY -----
    baseline_validity: {
      name: "Baseline Validity Check",
      description: "Validates first 20% of data as baseline (per entity/sensor)",
      sql: `
WITH time_bounds AS (
    SELECT entity_id, MIN(I) + 0.20 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations GROUP BY entity_id
),
baseline_stats AS (
    SELECT
        o.entity_id,
        o.signal_id,
        COUNT(*) AS baseline_obs,
        AVG(o.y) AS mu,
        STDDEV(o.y) AS sigma
    FROM observations o
    JOIN time_bounds t ON o.entity_id = t.entity_id
    WHERE o.I <= t.baseline_end
    GROUP BY o.entity_id, o.signal_id
)
SELECT
    entity_id,
    signal_id,
    baseline_obs,
    ROUND(mu, 4) AS baseline_mean,
    ROUND(sigma, 4) AS baseline_std,
    CASE WHEN baseline_obs >= 50 THEN 'SUFFICIENT' ELSE 'TOO_FEW' END AS sample_check,
    CASE
        WHEN sigma = 0 THEN 'CONSTANT'
        WHEN sigma / NULLIF(ABS(mu), 0.0001) < 0.5 THEN 'STABLE'
        ELSE 'VOLATILE'
    END AS stability,
    CASE
        WHEN baseline_obs >= 50 AND (sigma = 0 OR sigma / NULLIF(ABS(mu), 0.0001) < 0.5)
        THEN 'VALID' ELSE 'QUESTIONABLE'
    END AS validity
FROM baseline_stats
ORDER BY entity_id, signal_id
      `
    },
    baseline_summary: {
      name: "Baseline Quality Summary",
      description: "Per-entity baseline quality assessment",
      sql: `
WITH time_bounds AS (
    SELECT entity_id, MIN(I) + 0.20 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations GROUP BY entity_id
),
baseline_stats AS (
    SELECT
        o.entity_id,
        o.signal_id,
        COUNT(*) AS baseline_obs,
        AVG(o.y) AS mu,
        STDDEV(o.y) AS sigma,
        CASE
            WHEN COUNT(*) >= 50 AND (STDDEV(o.y) = 0 OR STDDEV(o.y) / NULLIF(ABS(AVG(o.y)), 0.0001) < 0.5)
            THEN 1 ELSE 0
        END AS is_valid
    FROM observations o
    JOIN time_bounds t ON o.entity_id = t.entity_id
    WHERE o.I <= t.baseline_end
    GROUP BY o.entity_id, o.signal_id
)
SELECT
    entity_id,
    COUNT(*) AS total_signals,
    SUM(is_valid) AS valid_baselines,
    ROUND(100.0 * SUM(is_valid) / COUNT(*), 0) AS pct_valid,
    CASE
        WHEN 100.0 * SUM(is_valid) / COUNT(*) >= 90 THEN 'GOOD'
        WHEN 100.0 * SUM(is_valid) / COUNT(*) >= 70 THEN 'ACCEPTABLE'
        ELSE 'POOR'
    END AS baseline_quality
FROM baseline_stats
GROUP BY entity_id
ORDER BY pct_valid DESC
      `
    },

    // ----- EFFECT SIZE VALIDATION -----
    effect_size_check: {
      name: "Effect Size Validation",
      description: "Checks if health changes are meaningful (not just statistically significant)",
      sql: `
WITH baseline_health AS (
    SELECT
        entity_id,
        AVG(health_score) AS baseline_health
    FROM health
    WHERE window_id <= (SELECT 0.20 * MAX(window_id) FROM health)
    GROUP BY entity_id
),
current_health AS (
    SELECT entity_id, health_score
    FROM health
    WHERE window_id = (SELECT MAX(window_id) FROM health)
)
SELECT
    c.entity_id,
    ROUND(b.baseline_health, 1) AS baseline,
    ROUND(c.health_score, 1) AS current,
    ROUND(b.baseline_health - c.health_score, 1) AS drop,
    CASE
        WHEN ABS(b.baseline_health - c.health_score) > 15 THEN 'LARGE - Actionable'
        WHEN ABS(b.baseline_health - c.health_score) > 10 THEN 'MEDIUM - Investigate'
        WHEN ABS(b.baseline_health - c.health_score) > 5 THEN 'SMALL - Watch'
        ELSE 'NEGLIGIBLE - Normal variation'
    END AS effect_size,
    CASE
        WHEN ABS(b.baseline_health - c.health_score) > 15 THEN 'Yes'
        ELSE 'No'
    END AS actionable
FROM current_health c
JOIN baseline_health b ON c.entity_id = b.entity_id
ORDER BY drop DESC
      `
    },

    // ----- MULTI-PILLAR CONFIRMATION -----
    pillar_agreement: {
      name: "Multi-Pillar Confirmation",
      description: "Checks if findings are confirmed across multiple pillars",
      sql: `
SELECT
    entity_id,
    ROUND(stability_score * 100, 0) AS stability_pct,
    ROUND(predictability_score * 100, 0) AS predict_pct,
    ROUND(physics_score * 100, 0) AS physics_pct,
    ROUND(topology_score * 100, 0) AS topo_pct,
    (CASE WHEN stability_score < 0.5 THEN 1 ELSE 0 END +
     CASE WHEN predictability_score < 0.5 THEN 1 ELSE 0 END +
     CASE WHEN physics_score < 0.5 THEN 1 ELSE 0 END +
     CASE WHEN topology_score < 0.5 THEN 1 ELSE 0 END) AS pillars_showing_issues,
    CASE
        WHEN (CASE WHEN stability_score < 0.5 THEN 1 ELSE 0 END +
              CASE WHEN predictability_score < 0.5 THEN 1 ELSE 0 END +
              CASE WHEN physics_score < 0.5 THEN 1 ELSE 0 END +
              CASE WHEN topology_score < 0.5 THEN 1 ELSE 0 END) >= 3 THEN 'HIGH (3-4 pillars)'
        WHEN (CASE WHEN stability_score < 0.5 THEN 1 ELSE 0 END +
              CASE WHEN predictability_score < 0.5 THEN 1 ELSE 0 END +
              CASE WHEN physics_score < 0.5 THEN 1 ELSE 0 END +
              CASE WHEN topology_score < 0.5 THEN 1 ELSE 0 END) >= 2 THEN 'MODERATE (2 pillars)'
        ELSE 'LOW (0-1 pillar)'
    END AS confidence
FROM health
WHERE window_id = (SELECT MAX(window_id) FROM health)
ORDER BY pillars_showing_issues DESC, health_score ASC
      `
    },

    // ----- FINDING SIGNIFICANCE -----
    finding_significance: {
      name: "Finding Significance Assessment",
      description: "Final verdict on whether findings are truly actionable",
      sql: `
WITH entity_obs AS (
    SELECT entity_id, COUNT(*) AS total_obs
    FROM observations GROUP BY entity_id
),
baseline_health AS (
    SELECT entity_id, AVG(health_score) AS baseline_health
    FROM health
    WHERE window_id <= (SELECT 0.20 * MAX(window_id) FROM health)
    GROUP BY entity_id
),
current AS (
    SELECT h.entity_id, h.health_score, h.risk_level,
           h.stability_score, h.predictability_score, h.physics_score, h.topology_score
    FROM health h
    WHERE h.window_id = (SELECT MAX(window_id) FROM health)
)
SELECT
    c.entity_id,
    ROUND(c.health_score, 1) AS health,
    c.risk_level,
    CASE WHEN e.total_obs >= 1000 THEN 'OK' ELSE 'INSUFFICIENT' END AS data_check,
    CASE WHEN ABS(b.baseline_health - c.health_score) > 10 THEN 'MEANINGFUL' ELSE 'MINOR' END AS effect_check,
    (CASE WHEN c.stability_score < 0.5 THEN 1 ELSE 0 END +
     CASE WHEN c.predictability_score < 0.5 THEN 1 ELSE 0 END +
     CASE WHEN c.physics_score < 0.5 THEN 1 ELSE 0 END +
     CASE WHEN c.topology_score < 0.5 THEN 1 ELSE 0 END) AS pillars_agree,
    CASE
        WHEN e.total_obs < 1000 THEN 'INSUFFICIENT_DATA'
        WHEN ABS(b.baseline_health - c.health_score) < 5 THEN 'NOT_SIGNIFICANT'
        WHEN (CASE WHEN c.stability_score < 0.5 THEN 1 ELSE 0 END +
              CASE WHEN c.predictability_score < 0.5 THEN 1 ELSE 0 END +
              CASE WHEN c.physics_score < 0.5 THEN 1 ELSE 0 END +
              CASE WHEN c.topology_score < 0.5 THEN 1 ELSE 0 END) < 2 THEN 'NOT_CONFIRMED'
        WHEN ABS(b.baseline_health - c.health_score) > 15 THEN 'ACTIONABLE'
        ELSE 'WATCH'
    END AS verdict
FROM current c
JOIN entity_obs e ON c.entity_id = e.entity_id
JOIN baseline_health b ON c.entity_id = b.entity_id
ORDER BY c.health_score ASC
      `
    },

    // ----- COHERENCE VELOCITY (NEW) -----
    coherence_velocity: {
      name: "Coherence Rate of Change",
      description: "Monitor coherence velocity - rapid changes are often more predictive than absolute values",
      sql: `
WITH coherence_with_lag AS (
    SELECT
        entity_id,
        window_id,
        coherence_ratio,
        LAG(coherence_ratio) OVER (PARTITION BY entity_id ORDER BY window_id) AS prev_coherence
    FROM geometry
)
SELECT
    entity_id,
    window_id,
    ROUND(coherence_ratio, 4) AS coherence,
    ROUND(coherence_ratio - prev_coherence, 4) AS delta,
    CASE
        WHEN ABS(coherence_ratio - prev_coherence) >= 0.10 THEN 'ALARM'
        WHEN ABS(coherence_ratio - prev_coherence) >= 0.05 THEN 'WARNING'
        WHEN ABS(coherence_ratio - prev_coherence) >= 0.02 THEN 'WATCH'
        ELSE 'STABLE'
    END AS status,
    CASE
        WHEN (coherence_ratio - prev_coherence) > 0.05 THEN 'Rapid COUPLING'
        WHEN (coherence_ratio - prev_coherence) < -0.05 THEN 'Rapid DECOUPLING'
        ELSE 'Stable'
    END AS direction
FROM coherence_with_lag
WHERE prev_coherence IS NOT NULL
ORDER BY ABS(coherence_ratio - prev_coherence) DESC
LIMIT 50
      `
    },

    // ----- VALIDATION SUMMARY (Updated) -----
    validation_summary: {
      name: "Dataset Validation Summary",
      description: "Overall data quality and reliability assessment (updated thresholds)",
      sql: `
SELECT
    (SELECT COUNT(DISTINCT entity_id) FROM observations) AS n_entities,
    (SELECT COUNT(DISTINCT signal_id) FROM observations) AS n_signals,
    (SELECT COUNT(*) FROM observations) AS total_obs,
    (SELECT ROUND(AVG(cnt)) FROM (SELECT entity_id, COUNT(*) AS cnt FROM observations GROUP BY entity_id)) AS avg_obs_per_entity,

    -- Fleet validity (4-tier)
    CASE
        WHEN (SELECT COUNT(DISTINCT entity_id) FROM observations) >= 30 THEN 'RELIABLE - Full stats'
        WHEN (SELECT COUNT(DISTINCT entity_id) FROM observations) >= 10 THEN 'MARGINAL - Z-scores with caution'
        WHEN (SELECT COUNT(DISTINCT entity_id) FROM observations) >= 5 THEN 'LIMITED - NO z-scores'
        ELSE 'MINIMAL - Individual only'
    END AS fleet_status,

    -- Lyapunov readiness (updated: 3k ready, 10k reliable)
    (SELECT SUM(CASE WHEN cnt >= 3000 THEN 1 ELSE 0 END) || '/' || COUNT(*) || ' ready'
     FROM (SELECT entity_id, COUNT(*) AS cnt FROM observations GROUP BY entity_id)) AS lyapunov_ready,

    (SELECT SUM(CASE WHEN cnt >= 10000 THEN 1 ELSE 0 END) || '/' || COUNT(*) || ' reliable'
     FROM (SELECT entity_id, COUNT(*) AS cnt FROM observations GROUP BY entity_id)) AS lyapunov_reliable,

    -- RQA readiness
    (SELECT SUM(CASE WHEN cnt >= 1000 THEN 1 ELSE 0 END) || '/' || COUNT(*)
     FROM (SELECT entity_id, COUNT(*) AS cnt FROM observations GROUP BY entity_id)) AS rqa_ready
      `
    }
  }
};

// Export for use in HTML
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ORTHON_QUERIES_V2;
}
