-- ============================================================================
-- ENGINES: CAUSAL MECHANICS
-- ============================================================================
-- Determine what drives what. SOURCE, SINK, CONDUIT roles.
-- Granger-like causality, transfer entropy proxies, causal chains.
-- ============================================================================

-- ============================================================================
-- 001: GRANGER CAUSALITY PROXY
-- ============================================================================
-- Does lagged X improve prediction of Y beyond Y's own history?
-- Simplified: compare corr(X_lag, Y) vs corr(Y_lag, Y)

CREATE OR REPLACE VIEW v_granger_proxy AS
WITH lagged AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        value,
        LAG(value, 10) OVER (PARTITION BY cohort, signal_id ORDER BY signal_0) AS value_lag10
    FROM v_base
),
own_prediction AS (
    SELECT
        cohort,
        signal_id,
        CORR(value, value_lag10) AS self_prediction
    FROM lagged
    WHERE value_lag10 IS NOT NULL
    GROUP BY cohort, signal_id
),
cross_prediction AS (
    SELECT
        a.cohort,
        a.signal_id AS target,
        b.signal_id AS source,
        CORR(a.value, b.value_lag) AS cross_prediction
    FROM v_base a
    JOIN (
        SELECT cohort, signal_id, signal_0, LAG(value, 10) OVER (PARTITION BY cohort, signal_id ORDER BY signal_0) AS value_lag
        FROM v_base
    ) b ON a.signal_0 = b.signal_0 AND a.signal_id != b.signal_id AND a.cohort = b.cohort
    WHERE b.value_lag IS NOT NULL
    GROUP BY a.cohort, a.signal_id, b.signal_id
)
SELECT
    cp.cohort,
    cp.source,
    cp.target,
    cp.cross_prediction,
    op.self_prediction AS target_self_prediction,
    ABS(cp.cross_prediction) - ABS(op.self_prediction) AS granger_score,
    CASE
        WHEN ABS(cp.cross_prediction) > ABS(op.self_prediction) + 0.1 THEN TRUE
        ELSE FALSE
    END AS granger_causes
FROM cross_prediction cp
JOIN own_prediction op ON cp.target = op.signal_id AND cp.cohort = op.cohort;


-- ============================================================================
-- 002: BIDIRECTIONAL CAUSALITY CHECK
-- ============================================================================
-- Does A cause B, B cause A, both, or neither?

CREATE OR REPLACE VIEW v_bidirectional_causality AS
WITH a_causes_b AS (
    SELECT cohort, source, target, granger_score AS score_a_to_b, granger_causes AS a_causes_b
    FROM v_granger_proxy
),
b_causes_a AS (
    SELECT cohort, target AS source, source AS target, granger_score AS score_b_to_a, granger_causes AS b_causes_a
    FROM v_granger_proxy
)
SELECT
    ab.cohort,
    ab.source AS signal_a,
    ab.target AS signal_b,
    ab.score_a_to_b,
    ba.score_b_to_a,
    ab.a_causes_b,
    ba.b_causes_a,
    CASE
        WHEN ab.a_causes_b AND NOT ba.b_causes_a THEN 'A_causes_B'
        WHEN ba.b_causes_a AND NOT ab.a_causes_b THEN 'B_causes_A'
        WHEN ab.a_causes_b AND ba.b_causes_a THEN 'bidirectional'
        ELSE 'no_causation'
    END AS causal_direction,
    ab.score_a_to_b - COALESCE(ba.score_b_to_a, 0) AS net_causal_flow
FROM a_causes_b ab
LEFT JOIN b_causes_a ba ON ab.source = ba.source AND ab.target = ba.target AND ab.cohort = ba.cohort;


-- ============================================================================
-- 003: TRANSFER ENTROPY PROXY
-- ============================================================================
-- Information flow approximation using conditional dependencies

CREATE OR REPLACE VIEW v_transfer_entropy_proxy AS
WITH lagged_data AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        value,
        LAG(value, 1) OVER w AS value_lag1,
        LAG(value, 5) OVER w AS value_lag5,
        LAG(value, 10) OVER w AS value_lag10
    FROM v_base
    WINDOW w AS (PARTITION BY cohort, signal_id ORDER BY signal_0)
),
binned AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        NTILE(5) OVER (PARTITION BY cohort, signal_id ORDER BY value) AS value_bin,
        NTILE(5) OVER (PARTITION BY cohort, signal_id ORDER BY value_lag1) AS value_lag_bin
    FROM lagged_data
    WHERE value_lag1 IS NOT NULL
),
binned_with_changes AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        value_bin,
        value_bin - COALESCE(LAG(value_bin) OVER (PARTITION BY cohort, signal_id ORDER BY signal_0), value_bin) AS value_bin_change
    FROM binned
)
-- Cross-signal conditional entropy approximation
SELECT
    a.cohort,
    a.signal_id AS source,
    b.signal_id AS target,
    -- Simplified: correlation of changes as proxy for information transfer
    CORR(a.value_bin_change, b.value_bin_change) AS change_correlation,
    ABS(CORR(a.value_bin_change, b.value_bin_change)) AS transfer_entropy_proxy
FROM binned_with_changes a
JOIN binned_with_changes b ON a.signal_0 = b.signal_0 + 5 AND a.signal_id != b.signal_id AND a.cohort = b.cohort
GROUP BY a.cohort, a.signal_id, b.signal_id;


-- ============================================================================
-- 004: CAUSAL ROLE ASSIGNMENT
-- ============================================================================
-- SOURCE: drives others, not driven
-- SINK: driven by others, doesn't drive
-- CONDUIT: both drives and is driven

CREATE OR REPLACE VIEW v_causal_roles AS
WITH out_degree AS (
    SELECT cohort, signal_a AS signal_id, COUNT(*) AS n_causes
    FROM v_bidirectional_causality
    WHERE causal_direction IN ('A_causes_B', 'bidirectional')
    GROUP BY cohort, signal_a
),
in_degree AS (
    SELECT cohort, signal_b AS signal_id, COUNT(*) AS n_caused_by
    FROM v_bidirectional_causality
    WHERE causal_direction IN ('A_causes_B', 'bidirectional')
    GROUP BY cohort, signal_b
),
total_influence AS (
    SELECT cohort, signal_a AS signal_id, SUM(net_causal_flow) AS total_out_flow
    FROM v_bidirectional_causality
    GROUP BY cohort, signal_a
)
SELECT
    COALESCE(o.cohort, i.cohort) AS cohort,
    COALESCE(o.signal_id, i.signal_id) AS signal_id,
    COALESCE(o.n_causes, 0) AS out_degree,
    COALESCE(i.n_caused_by, 0) AS in_degree,
    COALESCE(o.n_causes, 0) - COALESCE(i.n_caused_by, 0) AS net_influence,
    COALESCE(tf.total_out_flow, 0) AS total_causal_flow,
    CASE
        WHEN COALESCE(o.n_causes, 0) > COALESCE(i.n_caused_by, 0) * 2 AND COALESCE(i.n_caused_by, 0) = 0 THEN 'SOURCE'
        WHEN COALESCE(i.n_caused_by, 0) > COALESCE(o.n_causes, 0) * 2 AND COALESCE(o.n_causes, 0) = 0 THEN 'SINK'
        WHEN COALESCE(o.n_causes, 0) > 0 AND COALESCE(i.n_caused_by, 0) > 0 THEN 'CONDUIT'
        WHEN COALESCE(o.n_causes, 0) > COALESCE(i.n_caused_by, 0) THEN 'DRIVER'
        WHEN COALESCE(i.n_caused_by, 0) > COALESCE(o.n_causes, 0) THEN 'FOLLOWER'
        ELSE 'ISOLATED'
    END AS causal_role
FROM out_degree o
FULL OUTER JOIN in_degree i USING (cohort, signal_id)
LEFT JOIN total_influence tf USING (cohort, signal_id);


-- ============================================================================
-- 005: CAUSAL CHAIN DETECTION
-- ============================================================================
-- Find chains: A -> B -> C

CREATE OR REPLACE VIEW v_causal_chains AS
WITH direct_causes AS (
    SELECT cohort, signal_a AS source, signal_b AS target
    FROM v_bidirectional_causality
    WHERE causal_direction IN ('A_causes_B')
)
SELECT
    a.cohort,
    a.source AS root,
    a.target AS middle,
    b.target AS leaf,
    a.source || ' -> ' || a.target || ' -> ' || b.target AS chain_path
FROM direct_causes a
JOIN direct_causes b ON a.target = b.source AND a.cohort = b.cohort
WHERE a.source != b.target;  -- No cycles


-- ============================================================================
-- 006: CAUSAL IMPACT TIMING
-- ============================================================================
-- How quickly does a change in source affect target?

CREATE OR REPLACE VIEW v_causal_timing AS
WITH change_events AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        ABS(dvalue) AS abs_dvalue,
        PERCENT_RANK() OVER (
            PARTITION BY cohort, signal_id
            ORDER BY ABS(dvalue)
        ) AS change_pctile
    FROM v_dvalue
    WHERE dvalue IS NOT NULL
),
significant_changes AS (
    SELECT cohort, signal_id, signal_0
    FROM change_events
    WHERE change_pctile > 0.95
)
SELECT
    s.cohort,
    s.signal_id AS source,
    t.signal_id AS target,
    AVG(t.signal_0 - s.signal_0) AS avg_response_lag,
    MIN(t.signal_0 - s.signal_0) AS min_response_lag,
    COUNT(*) AS n_linked_events
FROM significant_changes s
JOIN significant_changes t ON t.signal_id != s.signal_id
    AND t.signal_0 > s.signal_0
    AND t.signal_0 < s.signal_0 + 50
    AND t.cohort = s.cohort
GROUP BY s.cohort, s.signal_id, t.signal_id
HAVING COUNT(*) > 5;


-- ============================================================================
-- 007: INTERVENTION EFFECT ESTIMATION
-- ============================================================================
-- What happens to the system when a signal changes significantly?

CREATE OR REPLACE VIEW v_intervention_effects AS
WITH source_shocks AS (
    SELECT
        cohort,
        r.signal_id AS source,
        r.signal_0 AS shock_I,
        r.change_score AS shock_magnitude  -- Use change_score instead of mean_change
    FROM v_regime_changes r
    WHERE r.is_regime_change
),
target_responses AS (
    SELECT
        b.cohort,
        b.signal_id AS target,
        b.signal_0 AS response_I,
        s.source,
        s.shock_I,
        s.shock_magnitude,
        b.signal_0 - s.shock_I AS response_lag,
        b.value - LAG(b.value, 10) OVER (PARTITION BY b.cohort, b.signal_id ORDER BY b.signal_0) AS target_change
    FROM v_base b
    JOIN source_shocks s ON b.signal_id != s.source
      AND b.signal_0 > s.shock_I
      AND b.signal_0 < s.shock_I + 30
      AND b.cohort = s.cohort
)
SELECT
    cohort,
    source,
    target,
    AVG(response_lag) AS avg_response_lag,
    CORR(shock_magnitude, target_change) AS intervention_correlation,
    COUNT(*) AS n_interventions
FROM target_responses
WHERE target_change IS NOT NULL
GROUP BY cohort, source, target
HAVING COUNT(*) > 3;


-- ============================================================================
-- 008: ROOT CAUSE CANDIDATES
-- ============================================================================
-- For a given target signal's regime change, what likely caused it?

CREATE OR REPLACE VIEW v_root_cause_candidates AS
WITH target_changes AS (
    SELECT cohort, signal_id AS target, signal_0 AS change_I
    FROM v_regime_changes
    WHERE is_regime_change
),
candidate_sources AS (
    SELECT
        tc.cohort,
        tc.target,
        tc.change_I,
        rc.signal_id AS candidate_source,
        rc.signal_0 AS source_change_I,
        tc.change_I - rc.signal_0 AS lead_time
    FROM target_changes tc
    JOIN v_regime_changes rc ON rc.signal_id != tc.target
        AND rc.is_regime_change
        AND rc.signal_0 < tc.change_I
        AND rc.signal_0 > tc.change_I - 50
        AND rc.cohort = tc.cohort
)
SELECT
    cohort,
    target,
    candidate_source,
    change_I AS target_change_at,
    source_change_I AS source_changed_at,
    lead_time,
    -- Check if this pair has known causal relationship
    EXISTS (
        SELECT 1 FROM v_granger_proxy g
        WHERE g.source = candidate_source AND g.target = cs.target AND g.granger_causes AND g.cohort = cs.cohort
    ) AS has_granger_support
FROM candidate_sources cs;


-- ============================================================================
-- 009: CAUSAL STRENGTH RANKING
-- ============================================================================
-- Rank all causal relationships by strength

CREATE OR REPLACE VIEW v_causal_strength AS
SELECT
    bc.cohort,
    bc.signal_a AS source,
    bc.signal_b AS target,
    bc.causal_direction,
    bc.score_a_to_b AS granger_score,
    ol.optimal_correlation AS coupling_strength,
    ol.optimal_lag AS coupling_lag,
    te.transfer_entropy_proxy,
    -- Combined causal strength score
    (COALESCE(ABS(bc.score_a_to_b), 0) +
     COALESCE(ABS(ol.optimal_correlation), 0) +
     COALESCE(te.transfer_entropy_proxy, 0)) / 3 AS combined_causal_strength
FROM v_bidirectional_causality bc
LEFT JOIN v_optimal_lag ol ON bc.signal_a = ol.signal_a AND bc.signal_b = ol.signal_b AND bc.cohort = ol.cohort
LEFT JOIN v_transfer_entropy_proxy te ON bc.signal_a = te.source AND bc.signal_b = te.target AND bc.cohort = te.cohort
WHERE bc.causal_direction IN ('A_causes_B', 'bidirectional')
ORDER BY combined_causal_strength DESC;


-- ============================================================================
-- 010: CAUSAL GRAPH EDGES (for visualization)
-- ============================================================================

CREATE OR REPLACE VIEW v_causal_graph AS
SELECT
    cohort,
    source,
    target,
    combined_causal_strength AS weight,
    granger_score,
    coupling_strength,
    coupling_lag AS lag,
    'causal' AS edge_type
FROM v_causal_strength
WHERE combined_causal_strength > 0.2

UNION ALL

-- Add coupling edges that don't have strong causality
SELECT
    cohort,
    signal_a AS source,
    signal_b AS target,
    optimal_abs_correlation AS weight,
    NULL AS granger_score,
    optimal_correlation AS coupling_strength,
    optimal_lag AS lag,
    'coupling' AS edge_type
FROM v_optimal_lag
WHERE optimal_abs_correlation > 0.5
  AND NOT EXISTS (
      SELECT 1 FROM v_causal_strength cs
      WHERE (cs.source = signal_a AND cs.target = signal_b AND cs.cohort = v_optimal_lag.cohort)
         OR (cs.source = signal_b AND cs.target = signal_a AND cs.cohort = v_optimal_lag.cohort)
  );


-- ============================================================================
-- CAUSAL MECHANICS SUMMARY
-- ============================================================================

CREATE OR REPLACE VIEW v_causality_complete AS
SELECT
    cr.cohort,
    cr.signal_id,
    cr.causal_role,
    cr.out_degree,
    cr.in_degree,
    cr.net_influence,
    cr.total_causal_flow,

    -- What this signal drives
    (SELECT STRING_AGG(target, ', ')
     FROM v_causal_strength cs
     WHERE cs.source = cr.signal_id AND cs.cohort = cr.cohort) AS drives,

    -- What drives this signal
    (SELECT STRING_AGG(source, ', ')
     FROM v_causal_strength cs
     WHERE cs.target = cr.signal_id AND cs.cohort = cr.cohort) AS driven_by

FROM v_causal_roles cr;


-- ============================================================================
-- SYSTEM CAUSAL STRUCTURE SUMMARY
-- ============================================================================

CREATE OR REPLACE VIEW v_system_causal_structure AS
SELECT
    -- Count by role
    (SELECT COUNT(*) FROM v_causal_roles WHERE causal_role = 'SOURCE') AS n_sources,
    (SELECT COUNT(*) FROM v_causal_roles WHERE causal_role = 'SINK') AS n_sinks,
    (SELECT COUNT(*) FROM v_causal_roles WHERE causal_role = 'CONDUIT') AS n_conduits,
    (SELECT COUNT(*) FROM v_causal_roles WHERE causal_role = 'ISOLATED') AS n_isolated,

    -- Network density
    (SELECT COUNT(*) FROM v_causal_graph) AS n_edges,
    (SELECT COUNT(DISTINCT signal_id) FROM v_base) AS n_nodes,
    (SELECT COUNT(*)::FLOAT / NULLIF(COUNT(DISTINCT signal_id) * (COUNT(DISTINCT signal_id) - 1), 0)
     FROM v_causal_graph, v_base) AS causal_density,

    -- Chain depth
    (SELECT MAX(LENGTH(chain_path) - LENGTH(REPLACE(chain_path, '->', '')) + 1)
     FROM v_causal_chains) AS max_chain_depth,

    -- Top sources
    (SELECT STRING_AGG(signal_id, ', ' ORDER BY total_causal_flow DESC)
     FROM (SELECT signal_id, total_causal_flow FROM v_causal_roles WHERE causal_role IN ('SOURCE', 'DRIVER') LIMIT 3) sub
    ) AS top_sources;
