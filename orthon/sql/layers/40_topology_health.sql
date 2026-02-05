-- ============================================================================
-- ORTHON SQL: 40_topology_health.sql
-- ============================================================================
-- TOPOLOGY HEALTH: Persistent homology interpretation
--
-- Interprets topological metrics from PRISM topology.parquet:
--   - Betti numbers: β₀ (components), β₁ (loops), β₂ (voids)
--   - Persistence: how "real" the features are
--   - Complexity: total topological richness
--
-- Classification:
--   - HEALTHY_CYCLE: β₀=1, β₁=1 (clean limit cycle)
--   - QUASI_PERIODIC: β₀=1, β₁=2 (torus-like)
--   - COMPLEX: β₀=1, β₁>2 (multiple loops)
--   - COLLAPSED: β₀=1, β₁=0 (no periodic structure)
--   - FRAGMENTED: β₀>1 (disconnected attractor)
--
-- Usage:
--   .read 40_topology_health.sql
-- ============================================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║                    TOPOLOGICAL HEALTH ANALYSIS                              ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'

-- ============================================================================
-- SECTION 1: LOAD TOPOLOGY DATA
-- ============================================================================

.print ''
.print '=== SECTION 1: Topology Overview ==='

CREATE OR REPLACE TABLE topology_windows AS
SELECT
    entity_id,
    observation_idx,
    betti_0,
    betti_1,
    betti_2,
    h1_max_persistence,
    h1_persistence_entropy,
    topological_complexity,
    landscape_h1_integral,

    -- Topology classification
    CASE
        WHEN betti_0 > 1 THEN 'FRAGMENTED'
        WHEN betti_1 = 0 THEN 'COLLAPSED'
        WHEN betti_1 = 1 THEN 'HEALTHY_CYCLE'
        WHEN betti_1 = 2 THEN 'QUASI_PERIODIC'
        WHEN betti_1 > 2 THEN 'COMPLEX'
        ELSE 'UNKNOWN'
    END as topology_class,

    -- Health score based on topology
    CASE
        WHEN betti_0 > 1 THEN 0.1  -- Fragmented is worst
        WHEN betti_1 = 0 THEN 0.3  -- Collapsed is bad
        WHEN betti_1 = 1 THEN 1.0  -- Healthy cycle
        WHEN betti_1 = 2 THEN 0.8  -- Quasi-periodic is fine
        WHEN betti_1 > 2 THEN 0.5  -- Complex may be early degradation
        ELSE 0.5
    END as topology_health

FROM read_parquet('{prism_output}/topology.parquet');

SELECT
    topology_class,
    COUNT(*) as n_windows,
    COUNT(DISTINCT entity_id) as n_entities,
    ROUND(AVG(betti_0), 1) as avg_b0,
    ROUND(AVG(betti_1), 1) as avg_b1,
    ROUND(AVG(topology_health), 2) as avg_health
FROM topology_windows
GROUP BY topology_class
ORDER BY avg_health ASC;


-- ============================================================================
-- SECTION 2: ENTITY-LEVEL TOPOLOGY
-- ============================================================================

.print ''
.print '=== SECTION 2: Entity-Level Topology ==='

CREATE OR REPLACE TABLE topology_entities AS
SELECT
    entity_id,
    COUNT(*) as n_windows,

    -- Betti number statistics
    ROUND(AVG(betti_0), 1) as mean_betti_0,
    ROUND(MAX(betti_0), 0) as max_betti_0,
    ROUND(AVG(betti_1), 1) as mean_betti_1,
    ROUND(MAX(betti_1), 0) as max_betti_1,

    -- Persistence statistics
    ROUND(AVG(h1_max_persistence), 3) as mean_h1_persistence,
    ROUND(AVG(topological_complexity), 1) as mean_complexity,

    -- Entity topology classification (based on dominant state)
    CASE
        WHEN AVG(betti_0) > 2 THEN 'FRAGMENTED'
        WHEN AVG(betti_1) < 0.5 THEN 'COLLAPSED'
        WHEN AVG(betti_1) BETWEEN 0.5 AND 1.5 THEN 'HEALTHY_CYCLE'
        WHEN AVG(betti_1) BETWEEN 1.5 AND 2.5 THEN 'QUASI_PERIODIC'
        ELSE 'COMPLEX'
    END as entity_topology,

    -- Topology health score
    ROUND(AVG(topology_health), 2) as topology_health_score,

    -- Fragmentation score (higher = more disconnected)
    ROUND((AVG(betti_0) - 1) / GREATEST(AVG(betti_0), 1), 2) as fragmentation_score

FROM topology_windows
GROUP BY entity_id;

SELECT
    entity_topology,
    COUNT(*) as n_entities,
    ROUND(AVG(mean_betti_0), 1) as avg_b0,
    ROUND(AVG(mean_betti_1), 1) as avg_b1,
    ROUND(AVG(topology_health_score), 2) as avg_health,
    ROUND(AVG(fragmentation_score), 2) as avg_frag
FROM topology_entities
GROUP BY entity_topology
ORDER BY avg_health ASC;


-- ============================================================================
-- SECTION 3: TOPOLOGY EVOLUTION
-- ============================================================================

.print ''
.print '=== SECTION 3: Topology Evolution ==='

CREATE OR REPLACE VIEW v_topology_evolution AS
WITH lagged AS (
    SELECT
        entity_id,
        observation_idx,
        betti_0,
        betti_1,
        topology_class,
        topology_health,
        LAG(betti_0, 1) OVER (PARTITION BY entity_id ORDER BY observation_idx) as prev_b0,
        LAG(betti_1, 1) OVER (PARTITION BY entity_id ORDER BY observation_idx) as prev_b1,
        LAG(topology_class, 1) OVER (PARTITION BY entity_id ORDER BY observation_idx) as prev_class
    FROM topology_windows
)
SELECT
    entity_id,
    observation_idx,
    betti_0,
    betti_1,
    topology_class,
    prev_class,
    CASE
        WHEN betti_0 > prev_b0 THEN 'FRAGMENTING'
        WHEN betti_0 < prev_b0 THEN 'RECONNECTING'
        WHEN betti_1 > prev_b1 THEN 'LOOPS_FORMING'
        WHEN betti_1 < prev_b1 THEN 'LOOPS_COLLAPSING'
        ELSE 'STABLE'
    END as topology_trend,
    CASE
        WHEN topology_class != prev_class AND prev_class IS NOT NULL THEN TRUE
        ELSE FALSE
    END as class_changed
FROM lagged;

SELECT
    topology_trend,
    COUNT(*) as n_transitions,
    COUNT(DISTINCT entity_id) as n_entities
FROM v_topology_evolution
WHERE topology_trend != 'STABLE'
GROUP BY topology_trend
ORDER BY n_transitions DESC;


-- ============================================================================
-- SECTION 4: TOPOLOGY RANKING
-- ============================================================================

.print ''
.print '=== SECTION 4: Topology Ranking (worst first) ==='

SELECT
    entity_id,
    entity_topology,
    topology_health_score,
    mean_betti_0 as b0,
    mean_betti_1 as b1,
    fragmentation_score as frag,
    mean_complexity as complexity,
    n_windows
FROM topology_entities
ORDER BY topology_health_score ASC
LIMIT 15;


-- ============================================================================
-- CREATE VIEWS
-- ============================================================================

CREATE OR REPLACE VIEW v_topology_summary AS
SELECT
    t.entity_id,
    t.entity_topology,
    t.topology_health_score,
    t.mean_betti_0,
    t.mean_betti_1,
    t.mean_h1_persistence,
    t.mean_complexity,
    t.fragmentation_score,
    t.n_windows
FROM topology_entities t;

CREATE OR REPLACE VIEW v_topology_alerts AS
SELECT
    entity_id,
    CASE entity_topology
        WHEN 'FRAGMENTED' THEN 'CRITICAL'
        WHEN 'COLLAPSED' THEN 'WARNING'
        WHEN 'COMPLEX' THEN 'WATCH'
        ELSE 'NORMAL'
    END as alert_level,
    entity_topology || ': β₀=' || ROUND(mean_betti_0, 0) ||
    ', β₁=' || ROUND(mean_betti_1, 0) ||
    ', frag=' || ROUND(fragmentation_score, 2) as alert_message,
    1.0 - topology_health_score as severity_score
FROM topology_entities
WHERE entity_topology IN ('FRAGMENTED', 'COLLAPSED', 'COMPLEX');


.print ''
.print '=== TOPOLOGY ANALYSIS COMPLETE ==='
.print ''
.print 'Views created:'
.print '  v_topology_summary    - Entity topology health'
.print '  v_topology_alerts     - Entities with degraded topology'
.print '  v_topology_evolution  - Topology changes over time'
.print ''
.print 'INTERPRETATION:'
.print '  β₀ > 1:  FRAGMENTED - attractor broken into pieces (CRITICAL)'
.print '  β₁ = 0:  COLLAPSED - lost periodic structure (WARNING)'
.print '  β₁ = 1:  HEALTHY_CYCLE - clean limit cycle (GOOD)'
.print '  β₁ = 2:  QUASI_PERIODIC - torus attractor (NORMAL)'
.print '  β₁ > 2:  COMPLEX - multiple loops (WATCH)'
.print ''
