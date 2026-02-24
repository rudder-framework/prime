-- ============================================================================
-- REPORT 20: TOPOLOGY SUMMARY (Persistent Homology)
-- ============================================================================
-- Betti numbers describe the "shape" of the data in feature space:
--   betti_0 — connected components (clusters)
--   betti_1 — loops (cycles in the dynamics)
--
-- Persistence measures how robust these topological features are:
--   High persistence = real structure
--   Low persistence = noise artifact
--
-- Source: persistent_homology
-- ============================================================================


-- ============================================================================
-- SECTION 1: TOPOLOGICAL FEATURE SUMMARY
-- Average topological complexity per cohort
-- ============================================================================

SELECT
    cohort,
    ROUND(AVG(betti_0), 2) AS avg_betti_0,
    ROUND(AVG(betti_1), 2) AS avg_betti_1,
    MAX(betti_0) AS max_betti_0,
    MAX(betti_1) AS max_betti_1,
    ROUND(AVG(total_persistence_0), 4) AS avg_persistence_0,
    ROUND(AVG(total_persistence_1), 4) AS avg_persistence_1,
    AVG(n_points) AS avg_n_points,
    COUNT(*) AS n_windows,
    CASE
        WHEN AVG(betti_1) > 1.0 THEN 'MULTI_LOOP'
        WHEN AVG(betti_1) > 0.5 THEN 'LOOP_PRESENT'
        WHEN AVG(betti_1) > 0.1 THEN 'OCCASIONAL_LOOP'
        ELSE 'NO_LOOPS'
    END AS topology_class
FROM persistent_homology
GROUP BY cohort
ORDER BY AVG(betti_1) DESC;


-- ============================================================================
-- SECTION 2: TOPOLOGICAL EVOLUTION OVER TIME
-- How does the topology change across windows?
-- ============================================================================

WITH topo_trajectory AS (
    SELECT
        cohort,
        signal_0_end,
        betti_0,
        betti_1,
        total_persistence_0,
        total_persistence_1,
        betti_1 - LAG(betti_1) OVER (PARTITION BY cohort ORDER BY signal_0_end) AS betti_1_delta,
        total_persistence_1 - LAG(total_persistence_1) OVER (
            PARTITION BY cohort ORDER BY signal_0_end) AS persistence_1_delta
    FROM persistent_homology
)
SELECT
    cohort,
    ROUND(REGR_SLOPE(betti_0, signal_0_end), 8) AS betti_0_trend,
    ROUND(REGR_SLOPE(betti_1, signal_0_end), 8) AS betti_1_trend,
    ROUND(REGR_SLOPE(total_persistence_1, signal_0_end), 8) AS persistence_1_trend,
    SUM(CASE WHEN betti_1_delta > 0 THEN 1 ELSE 0 END) AS windows_gaining_loops,
    SUM(CASE WHEN betti_1_delta < 0 THEN 1 ELSE 0 END) AS windows_losing_loops,
    COUNT(*) AS n_windows,
    CASE
        WHEN REGR_SLOPE(betti_1, signal_0_end) > 0.0001 THEN 'COMPLEXIFYING'
        WHEN REGR_SLOPE(betti_1, signal_0_end) < -0.0001 THEN 'SIMPLIFYING'
        ELSE 'TOPOLOGICALLY_STABLE'
    END AS topology_trend
FROM topo_trajectory
GROUP BY cohort
ORDER BY ABS(REGR_SLOPE(betti_1, signal_0_end)) DESC;


-- ============================================================================
-- SECTION 3: PERSISTENCE DIAGRAM SUMMARY
-- Windows with the most robust topological features
-- ============================================================================

SELECT
    cohort,
    signal_0_end,
    betti_0,
    betti_1,
    ROUND(total_persistence_0, 4) AS persistence_0,
    ROUND(max_persistence_0, 4) AS max_pers_0,
    ROUND(total_persistence_1, 4) AS persistence_1,
    ROUND(max_persistence_1, 4) AS max_pers_1,
    CASE
        WHEN max_persistence_1 > 0 AND betti_1 > 0 THEN 'ROBUST_LOOP'
        WHEN betti_1 > 0 THEN 'WEAK_LOOP'
        ELSE 'NO_LOOP'
    END AS loop_quality
FROM persistent_homology
WHERE betti_1 > 0
ORDER BY max_persistence_1 DESC
LIMIT 30;


-- ============================================================================
-- SECTION 4: TOPOLOGICAL EARLY VS LATE COMPARISON
-- Does the topology change between early and late life?
-- ============================================================================

WITH lifecycle AS (
    SELECT
        cohort,
        MIN(signal_0_end) AS min_I,
        MAX(signal_0_end) AS max_I
    FROM persistent_homology
    GROUP BY cohort
),
early_late AS (
    SELECT
        ph.cohort,
        AVG(CASE WHEN ph.signal_0_end <= lc.min_I + (lc.max_I - lc.min_I) * 0.3
            THEN ph.betti_0 END) AS early_betti_0,
        AVG(CASE WHEN ph.signal_0_end >= lc.max_I - (lc.max_I - lc.min_I) * 0.3
            THEN ph.betti_0 END) AS late_betti_0,
        AVG(CASE WHEN ph.signal_0_end <= lc.min_I + (lc.max_I - lc.min_I) * 0.3
            THEN ph.betti_1 END) AS early_betti_1,
        AVG(CASE WHEN ph.signal_0_end >= lc.max_I - (lc.max_I - lc.min_I) * 0.3
            THEN ph.betti_1 END) AS late_betti_1,
        AVG(CASE WHEN ph.signal_0_end <= lc.min_I + (lc.max_I - lc.min_I) * 0.3
            THEN ph.total_persistence_1 END) AS early_persistence_1,
        AVG(CASE WHEN ph.signal_0_end >= lc.max_I - (lc.max_I - lc.min_I) * 0.3
            THEN ph.total_persistence_1 END) AS late_persistence_1
    FROM persistent_homology ph
    JOIN lifecycle lc ON ph.cohort = lc.cohort
    GROUP BY ph.cohort
)
SELECT
    cohort,
    ROUND(early_betti_0, 2) AS early_b0,
    ROUND(late_betti_0, 2) AS late_b0,
    ROUND(early_betti_1, 2) AS early_b1,
    ROUND(late_betti_1, 2) AS late_b1,
    ROUND(late_betti_1 - early_betti_1, 2) AS betti_1_change,
    CASE
        WHEN late_betti_1 - early_betti_1 > 0.5 THEN 'LOOPS_EMERGED'
        WHEN late_betti_1 - early_betti_1 < -0.5 THEN 'LOOPS_COLLAPSED'
        ELSE 'TOPOLOGY_PRESERVED'
    END AS topology_evolution
FROM early_late
WHERE early_betti_0 IS NOT NULL AND late_betti_0 IS NOT NULL
ORDER BY ABS(late_betti_1 - early_betti_1) DESC;
