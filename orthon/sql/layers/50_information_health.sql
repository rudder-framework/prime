-- ============================================================================
-- ORTHON SQL: 50_information_health.sql
-- ============================================================================
-- INFORMATION FLOW HEALTH: Causal network interpretation
--
-- Interprets information flow metrics from PRISM information_flow.parquet:
--   - Hierarchy score: how directional is causation
--   - Feedback loops: bidirectional causal relationships
--   - Network density: how coupled is everything
--   - Transfer entropy: strength of causal influence
--
-- Classification:
--   - HIERARCHICAL: Clear causal direction (healthy)
--   - MIXED: Partial hierarchy with some coupling
--   - COUPLED: Significant bidirectional influences
--   - CIRCULAR: No clear hierarchy (critical)
--
-- Usage:
--   .read 50_information_health.sql
-- ============================================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║                    INFORMATION FLOW HEALTH ANALYSIS                         ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'

-- ============================================================================
-- SECTION 1: LOAD INFORMATION FLOW DATA
-- ============================================================================

.print ''
.print '=== SECTION 1: Causal Network Overview ==='

CREATE OR REPLACE TABLE information_windows AS
SELECT
    entity_id,
    I,
    n_causal_edges,
    network_density,
    network_reciprocity,
    hierarchy_score,
    n_feedback_loops,
    max_transfer_entropy,
    mean_transfer_entropy,
    top_driver,
    top_sink,
    network_changed,

    -- Network classification
    CASE
        WHEN hierarchy_score > 0.8 THEN 'HIERARCHICAL'
        WHEN hierarchy_score > 0.5 THEN 'MIXED'
        WHEN hierarchy_score > 0.2 THEN 'COUPLED'
        ELSE 'CIRCULAR'
    END as network_class,

    -- Feedback risk level
    CASE
        WHEN n_feedback_loops = 0 THEN 'LOW'
        WHEN n_feedback_loops <= 5 THEN 'MODERATE'
        WHEN n_feedback_loops <= 20 THEN 'HIGH'
        ELSE 'CRITICAL'
    END as feedback_risk,

    -- Information health score
    CASE
        WHEN hierarchy_score > 0.8 AND n_feedback_loops <= 2 THEN 1.0
        WHEN hierarchy_score > 0.5 AND n_feedback_loops <= 5 THEN 0.7
        WHEN hierarchy_score > 0.2 AND n_feedback_loops <= 20 THEN 0.4
        ELSE 0.2
    END as information_health

FROM read_parquet('{prism_output}/information_flow.parquet');

SELECT
    network_class,
    feedback_risk,
    COUNT(*) as n_windows,
    COUNT(DISTINCT entity_id) as n_entities,
    ROUND(AVG(hierarchy_score), 3) as avg_hierarchy,
    ROUND(AVG(n_feedback_loops), 0) as avg_loops,
    ROUND(AVG(information_health), 2) as avg_health
FROM information_windows
GROUP BY network_class, feedback_risk
ORDER BY avg_health ASC;


-- ============================================================================
-- SECTION 2: ENTITY-LEVEL INFORMATION FLOW
-- ============================================================================

.print ''
.print '=== SECTION 2: Entity-Level Causal Health ==='

CREATE OR REPLACE TABLE information_entities AS
SELECT
    entity_id,
    COUNT(*) as n_windows,

    -- Network statistics
    ROUND(AVG(hierarchy_score), 3) as mean_hierarchy,
    ROUND(AVG(network_density), 3) as mean_density,
    ROUND(AVG(network_reciprocity), 3) as mean_reciprocity,
    ROUND(AVG(n_feedback_loops), 0) as mean_feedback_loops,
    ROUND(MAX(n_feedback_loops), 0) as max_feedback_loops,

    -- Transfer entropy
    ROUND(AVG(max_transfer_entropy), 3) as mean_max_te,
    ROUND(AVG(mean_transfer_entropy), 3) as mean_te,

    -- Most common driver/sink
    MODE(top_driver) as dominant_driver,
    MODE(top_sink) as dominant_sink,

    -- Network changes
    SUM(CASE WHEN network_changed THEN 1 ELSE 0 END) as n_network_changes,

    -- Entity classification
    CASE
        WHEN AVG(hierarchy_score) > 0.8 THEN 'HIERARCHICAL'
        WHEN AVG(hierarchy_score) > 0.5 THEN 'MIXED'
        WHEN AVG(hierarchy_score) > 0.2 THEN 'COUPLED'
        ELSE 'CIRCULAR'
    END as entity_network_type,

    -- Feedback risk
    CASE
        WHEN AVG(n_feedback_loops) <= 2 THEN 'LOW'
        WHEN AVG(n_feedback_loops) <= 10 THEN 'MODERATE'
        WHEN AVG(n_feedback_loops) <= 50 THEN 'HIGH'
        ELSE 'CRITICAL'
    END as entity_feedback_risk,

    -- Information health score
    ROUND(AVG(information_health), 2) as information_health_score

FROM information_windows
GROUP BY entity_id;

SELECT
    entity_network_type,
    entity_feedback_risk,
    COUNT(*) as n_entities,
    ROUND(AVG(mean_hierarchy), 3) as avg_hierarchy,
    ROUND(AVG(mean_feedback_loops), 0) as avg_loops,
    ROUND(AVG(information_health_score), 2) as avg_health
FROM information_entities
GROUP BY entity_network_type, entity_feedback_risk
ORDER BY avg_health ASC;


-- ============================================================================
-- SECTION 3: CAUSAL DRIVERS
-- ============================================================================

.print ''
.print '=== SECTION 3: Dominant Causal Drivers ==='

SELECT
    dominant_driver,
    COUNT(*) as n_entities,
    ROUND(AVG(mean_max_te), 3) as avg_te,
    ROUND(AVG(mean_hierarchy), 3) as avg_hierarchy
FROM information_entities
GROUP BY dominant_driver
ORDER BY n_entities DESC
LIMIT 10;


-- ============================================================================
-- SECTION 4: HIERARCHY EVOLUTION
-- ============================================================================

.print ''
.print '=== SECTION 4: Hierarchy Evolution ==='

CREATE OR REPLACE VIEW v_information_evolution AS
WITH lagged AS (
    SELECT
        entity_id,
        I,
        hierarchy_score,
        n_feedback_loops,
        network_class,
        LAG(hierarchy_score, 1) OVER (PARTITION BY entity_id ORDER BY I) as prev_hierarchy,
        LAG(n_feedback_loops, 1) OVER (PARTITION BY entity_id ORDER BY I) as prev_loops,
        LAG(network_class, 1) OVER (PARTITION BY entity_id ORDER BY I) as prev_class
    FROM information_windows
)
SELECT
    entity_id,
    I,
    hierarchy_score,
    n_feedback_loops,
    network_class,
    prev_class,
    CASE
        WHEN hierarchy_score < prev_hierarchy - 0.1 THEN 'HIERARCHY_BREAKING'
        WHEN hierarchy_score > prev_hierarchy + 0.1 THEN 'HIERARCHY_STRENGTHENING'
        WHEN n_feedback_loops > prev_loops + 5 THEN 'FEEDBACK_FORMING'
        WHEN n_feedback_loops < prev_loops - 5 THEN 'FEEDBACK_RESOLVING'
        ELSE 'STABLE'
    END as causal_trend,
    CASE
        WHEN network_class != prev_class AND prev_class IS NOT NULL THEN TRUE
        ELSE FALSE
    END as class_changed
FROM lagged;

SELECT
    causal_trend,
    COUNT(*) as n_transitions,
    COUNT(DISTINCT entity_id) as n_entities
FROM v_information_evolution
WHERE causal_trend != 'STABLE'
GROUP BY causal_trend
ORDER BY n_transitions DESC;


-- ============================================================================
-- SECTION 5: INFORMATION RANKING
-- ============================================================================

.print ''
.print '=== SECTION 5: Information Health Ranking (worst first) ==='

SELECT
    entity_id,
    entity_network_type,
    entity_feedback_risk,
    information_health_score,
    mean_hierarchy,
    mean_feedback_loops as loops,
    mean_density as density,
    dominant_driver,
    n_windows
FROM information_entities
ORDER BY information_health_score ASC
LIMIT 15;


-- ============================================================================
-- CREATE VIEWS
-- ============================================================================

CREATE OR REPLACE VIEW v_information_summary AS
SELECT
    i.entity_id,
    i.entity_network_type,
    i.entity_feedback_risk,
    i.information_health_score,
    i.mean_hierarchy,
    i.mean_density,
    i.mean_reciprocity,
    i.mean_feedback_loops,
    i.max_feedback_loops,
    i.mean_max_te,
    i.dominant_driver,
    i.dominant_sink,
    i.n_network_changes,
    i.n_windows
FROM information_entities i;

CREATE OR REPLACE VIEW v_information_alerts AS
SELECT
    entity_id,
    CASE
        WHEN entity_network_type = 'CIRCULAR' AND entity_feedback_risk = 'CRITICAL' THEN 'CRITICAL'
        WHEN entity_network_type = 'CIRCULAR' THEN 'CRITICAL'
        WHEN entity_feedback_risk = 'CRITICAL' THEN 'WARNING'
        WHEN entity_network_type = 'COUPLED' THEN 'WARNING'
        WHEN entity_feedback_risk = 'HIGH' THEN 'WATCH'
        ELSE 'NORMAL'
    END as alert_level,
    entity_network_type || ': hierarchy=' || ROUND(mean_hierarchy, 2) ||
    ', loops=' || ROUND(mean_feedback_loops, 0) ||
    ', driver=' || dominant_driver as alert_message,
    1.0 - information_health_score as severity_score
FROM information_entities
WHERE entity_network_type IN ('CIRCULAR', 'COUPLED') OR entity_feedback_risk IN ('CRITICAL', 'HIGH');


.print ''
.print '=== INFORMATION FLOW ANALYSIS COMPLETE ==='
.print ''
.print 'Views created:'
.print '  v_information_summary    - Entity causal network health'
.print '  v_information_alerts     - Entities with degraded causal structure'
.print '  v_information_evolution  - Causal changes over time'
.print ''
.print 'INTERPRETATION:'
.print '  HIERARCHICAL: Clear cause-effect chain (HEALTHY)'
.print '  MIXED:        Some bidirectional coupling (NORMAL)'
.print '  COUPLED:      Significant mutual influence (WARNING)'
.print '  CIRCULAR:     No clear hierarchy - runaway risk (CRITICAL)'
.print ''
.print '  Feedback loops > 50: System approaching cascade failure'
.print '  Network density > 70%: Everything driving everything'
.print ''
