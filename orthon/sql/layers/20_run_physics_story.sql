-- ============================================================================
-- ORTHON SQL: 20_run_physics_story.sql
-- ============================================================================
-- MASTER SCRIPT: Run all four-pillar analysis and generate unified health report
--
-- THE FOUR PILLARS:
--   1. Geometry   (10-19): Coherence, effective dimension, state velocity
--   2. Dynamics   (30-33): Lyapunov, RQA, basin stability, birth certificate
--   3. Topology   (40-41): Betti numbers, persistence, attractor shape
--   4. Information (50-51): Transfer entropy, causal hierarchy, feedback loops
--
-- Usage:
--   duckdb < 20_run_physics_story.sql
--
-- Or with path substitution:
--   sed 's|{prism_output}|/path/to/output|g' 20_run_physics_story.sql | duckdb
--
-- Output: Complete four-pillar health assessment for each entity
--
-- Configuration:
--   Edit 00_config.sql to adjust thresholds
-- ============================================================================

-- Set prism output path (override with sed or environment)
-- SET prism_output = '/Users/jasonrudder/prism/data/output/JOB_ID';

.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║           ORTHON FOUR-PILLAR STRUCTURAL HEALTH ANALYZER                    ║'
.print '╠══════════════════════════════════════════════════════════════════════════════╣'
.print '║  Geometry   │ Dynamics   │ Topology   │ Information                         ║'
.print '║  Coherence  │ Lyapunov   │ Betti nums │ Transfer entropy                    ║'
.print '║  Eff. dim   │ RQA        │ Persistence│ Causal hierarchy                    ║'
.print '║  Velocity   │ Basin      │ Shape      │ Feedback loops                      ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'
.print ''

-- ============================================================================
-- STEP 0: CONFIGURATION AUDIT (MUST RUN FIRST)
-- ============================================================================
-- This detects heterogeneous signal counts that confound fingerprint metrics.
-- If detected, subsequent analyses will use normalized metrics.

.print 'Running configuration audit...'
.read 00_configuration_audit.sql

-- Load configuration (thresholds, data requirements)
.print 'Loading configuration...'
.read 00_config.sql

-- Load base data
.print 'Loading physics.parquet...'
.read 12_load_physics.sql

-- L4: Thermodynamics
.print 'Analyzing L4: Thermodynamics...'
.read 13_l4_thermodynamics.sql

-- L2: Coherence (eigenvalue-based)
.print 'Analyzing L2: Coherence (eigenvalue-based)...'
.read 14_l2_coherence.sql

-- L1: State
.print 'Analyzing L1: State...'
.read 15_l1_state.sql

-- Ørthon Signal
.print 'Detecting Ørthon Signal...'
.read 16_orthon_signal.sql

-- The Story
.print 'Generating the story...'
.read 18_system_story.sql

-- Geometric Attribution: Endogenous vs Exogenous
.print 'Computing geometric attribution...'
.read 21_geometric_attribution.sql

-- Vector Energy: Per-signal energy contribution
.print 'Computing vector energy...'
-- .read 22_vector_energy.sql  -- Uncomment if observations_enriched exists

-- Baseline & Deviation: Self-referential anomaly detection
.print 'Establishing baselines and detecting deviations...'
.read 23_baseline_deviation.sql

-- Incident Summary: Comprehensive report tying everything together
.print 'Generating incident summary...'
.read 24_incident_summary.sql

-- Sensitivity Analysis: Robustness checks
.print 'Running sensitivity analysis...'
.read 25_sensitivity_analysis.sql

-- ML Feature Export: Early warning features for predictive models
.print 'Exporting ML features...'
.read 26_ml_feature_export.sql

-- ============================================================================
-- DYNAMICAL SYSTEMS ANALYSIS (Lyapunov, basin stability, regime transitions)
-- ============================================================================
-- These require primitives.parquet with lyapunov_exponent computed

-- Dynamics Stability: Lyapunov-based classification
.print 'Analyzing dynamical stability (Lyapunov)...'
.read 30_dynamics_stability.sql

-- Regime Transitions: Detect stability changes over time
.print 'Detecting regime transitions...'
.read 31_regime_transitions.sql

-- Basin Stability: Composite stability score
.print 'Computing basin stability...'
.read 32_basin_stability.sql

-- Birth Certificate: Early-life prognosis
.print 'Generating birth certificates...'
.read 33_birth_certificate.sql

-- ============================================================================
-- TOPOLOGICAL DATA ANALYSIS (Betti numbers, persistence, attractor shape)
-- ============================================================================
-- These require topology.parquet from PRISM topology engine

-- Topology Health: Betti number classification
.print 'Analyzing topological health...'
.read 40_topology_health.sql

-- ============================================================================
-- INFORMATION FLOW ANALYSIS (Transfer entropy, causal networks)
-- ============================================================================
-- These require information_flow.parquet from PRISM information engine

-- Information Flow Health: Causal network classification
.print 'Analyzing information flow health...'
.read 50_information_health.sql

-- Try to load observations_enriched for principal actors
.print 'Identifying principal actors (if observations_enriched available)...'
-- .read 19_principal_actors.sql  -- Uncomment if observations_enriched exists

.print ''
.print '=============================================='
.print 'ANALYSIS COMPLETE'
.print '=============================================='
.print ''

-- ============================================================================
-- EXECUTIVE SUMMARY
-- ============================================================================

.print '=== FLEET SUMMARY ==='
SELECT * FROM v_orthon_fleet_summary;

.print ''
.print '=== ØRTHON SIGNALS ==='
SELECT
    entity_id,
    current_severity,
    current_severity_score,
    status_message
FROM v_orthon_entity_summary
WHERE current_orthon_signal = TRUE
ORDER BY current_severity_score DESC;

.print ''
.print '=== ENTITIES BY SEVERITY ==='
SELECT
    current_severity,
    COUNT(*) AS n_entities,
    STRING_AGG(entity_id, ', ') AS entities
FROM v_orthon_entity_summary
GROUP BY current_severity
ORDER BY
    CASE current_severity
        WHEN 'critical' THEN 1
        WHEN 'warning' THEN 2
        WHEN 'watch' THEN 3
        ELSE 4
    END;

.print ''
.print '=== STORY OVERVIEW ==='
SELECT
    n_entities,
    n_degrading,
    n_recovering,
    n_stable,
    n_exogenous || ' exogenous, ' || n_endogenous || ' endogenous' AS causation_split
FROM v_story_fleet_summary;

-- ============================================================================
-- DETAILED NARRATIVES
-- ============================================================================

.print ''
.print '=== ENTITY NARRATIVES ==='

SELECT
    entity_id,
    overall_trajectory,
    likely_causation,
    full_narrative
FROM v_story_complete
ORDER BY
    CASE overall_trajectory
        WHEN 'degradation' THEN 1
        WHEN 'ongoing_issue' THEN 2
        WHEN 'overload' THEN 3
        WHEN 'recovery' THEN 4
        ELSE 5
    END;

-- ============================================================================
-- ALERTS
-- ============================================================================

.print ''
.print '=== ACTIVE ALERTS ==='
SELECT
    alert_level,
    entity_id,
    alert_message,
    severity_score
FROM v_orthon_alerts
ORDER BY
    CASE alert_level
        WHEN 'CRITICAL' THEN 1
        WHEN 'WARNING' THEN 2
        ELSE 3
    END,
    severity_score DESC
LIMIT 20;

-- ============================================================================
-- COHERENCE INTERPRETATION
-- ============================================================================

.print ''
.print '=== COHERENCE STATE ==='
SELECT
    entity_id,
    coupling_state,
    structure_state,
    interpretation
FROM v_l2_interpretation
WHERE I = (SELECT MAX(I) FROM physics WHERE physics.entity_id = v_l2_interpretation.entity_id)
ORDER BY coupling_state;

-- ============================================================================
-- DYNAMICAL SYSTEMS SUMMARY
-- ============================================================================

.print ''
.print '=== DYNAMICS STABILITY ==='
SELECT
    entity_stability,
    COUNT(*) as n_entities,
    ROUND(AVG(stability_score), 3) as avg_stability_score
FROM v_dynamics_summary
GROUP BY entity_stability
ORDER BY avg_stability_score ASC;

.print ''
.print '=== BASIN STABILITY ==='
SELECT
    basin_class,
    COUNT(*) as n_entities,
    ROUND(AVG(basin_stability_score), 3) as avg_score
FROM v_basin_stability
GROUP BY basin_class
ORDER BY avg_score DESC;

.print ''
.print '=== BIRTH CERTIFICATE PROGNOSIS ==='
SELECT
    birth_grade,
    COUNT(*) as n_entities,
    ROUND(AVG(actual_lifespan), 0) as avg_lifespan,
    ROUND(AVG(expected_lifespan), 0) as expected_lifespan
FROM v_birth_prognosis
GROUP BY birth_grade
ORDER BY avg_lifespan DESC;

-- ============================================================================
-- TOPOLOGICAL SUMMARY
-- ============================================================================

.print ''
.print '=== TOPOLOGY HEALTH ==='
SELECT
    entity_topology,
    COUNT(*) as n_entities,
    ROUND(AVG(topology_health_score), 2) as avg_health,
    ROUND(AVG(mean_betti_0), 1) as avg_b0,
    ROUND(AVG(mean_betti_1), 1) as avg_b1
FROM v_topology_summary
GROUP BY entity_topology
ORDER BY avg_health ASC;

.print ''
.print '=== TOPOLOGY ALERTS ==='
SELECT
    alert_level,
    entity_id,
    alert_message,
    ROUND(severity_score, 2) as severity
FROM v_topology_alerts
ORDER BY severity_score DESC
LIMIT 10;

-- ============================================================================
-- INFORMATION FLOW SUMMARY
-- ============================================================================

.print ''
.print '=== INFORMATION FLOW HEALTH ==='
SELECT
    entity_network_type,
    entity_feedback_risk,
    COUNT(*) as n_entities,
    ROUND(AVG(information_health_score), 2) as avg_health,
    ROUND(AVG(mean_hierarchy), 3) as avg_hierarchy,
    ROUND(AVG(mean_feedback_loops), 0) as avg_loops
FROM v_information_summary
GROUP BY entity_network_type, entity_feedback_risk
ORDER BY avg_health ASC;

.print ''
.print '=== CAUSAL NETWORK ALERTS ==='
SELECT
    alert_level,
    entity_id,
    alert_message,
    ROUND(severity_score, 2) as severity
FROM v_information_alerts
ORDER BY severity_score DESC
LIMIT 10;

-- ============================================================================
-- UNIFIED FOUR-PILLAR SUMMARY
-- ============================================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║                    UNIFIED FOUR-PILLAR HEALTH SUMMARY                       ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'
.print ''
.print 'PILLAR STATUS:'
.print '  Geometry:    Coherence, effective dimension, state velocity'
.print '  Dynamics:    Lyapunov exponents, RQA, basin stability'
.print '  Topology:    Betti numbers, persistence, attractor shape'
.print '  Information: Transfer entropy, causal hierarchy, feedback loops'
.print ''

-- Cross-pillar correlation check
SELECT
    'CROSS-LAYER AGREEMENT' as metric,
    ROUND(CORR(d.stability_score, t.topology_health_score), 2) as dynamics_topology_r,
    ROUND(CORR(d.stability_score, i.information_health_score), 2) as dynamics_info_r,
    ROUND(CORR(t.topology_health_score, i.information_health_score), 2) as topology_info_r
FROM dynamics_entities d
JOIN topology_entities t ON d.entity_id = t.entity_id
JOIN information_entities i ON d.entity_id = i.entity_id;

.print ''
.print '=============================================='
.print 'END OF REPORT'
.print '=============================================='
