-- ============================================================================
-- Engines CAUSALITY AND INFLUENCE REPORTS
-- ============================================================================
--
-- Analyzes lead-lag relationships and directional influence between signals.
-- Helps understand cause-effect chains in the process.
--
-- Usage: Run against observations table
-- ============================================================================


-- ============================================================================
-- REPORT 1: LEAD-LAG RELATIONSHIPS
-- Identifies which signals lead/lag others
-- ============================================================================

WITH
lagged AS (
    SELECT
        a.cohort,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        a.I,
        a.value AS y_a,
        b.value AS y_b,
        LAG(a.value, 1) OVER wa AS y_a_lag1,
        LAG(a.value, 5) OVER wa AS y_a_lag5,
        LEAD(a.value, 1) OVER wa AS y_a_lead1,
        LEAD(a.value, 5) OVER wa AS y_a_lead5
    FROM observations a
    JOIN observations b
        ON a.cohort = b.cohort
        AND a.I = b.I
        AND a.signal_id < b.signal_id
    WINDOW wa AS (PARTITION BY a.cohort, a.signal_id, b.signal_id ORDER BY a.I)
),

cross_correlations AS (
    SELECT
        cohort,
        signal_a,
        signal_b,
        CORR(y_a, y_b) AS corr_0,
        CORR(y_a_lag1, y_b) AS corr_a_leads_1,
        CORR(y_a_lag5, y_b) AS corr_a_leads_5,
        CORR(y_a_lead1, y_b) AS corr_a_lags_1,
        CORR(y_a_lead5, y_b) AS corr_a_lags_5
    FROM lagged
    WHERE y_a_lag5 IS NOT NULL AND y_a_lead5 IS NOT NULL
    GROUP BY cohort, signal_a, signal_b
)

SELECT
    cohort,
    signal_a,
    signal_b,
    ROUND(corr_0, 3) AS sync_corr,
    ROUND(corr_a_leads_1, 3) AS a_leads_1,
    ROUND(corr_a_leads_5, 3) AS a_leads_5,
    ROUND(corr_a_lags_1, 3) AS a_lags_1,
    ROUND(corr_a_lags_5, 3) AS a_lags_5,
    -- Determine lead-lag relationship
    CASE
        WHEN corr_a_leads_5 > corr_0 + 0.1 AND corr_a_leads_5 > corr_a_lags_5 + 0.1 THEN signal_a || ' LEADS'
        WHEN corr_a_lags_5 > corr_0 + 0.1 AND corr_a_lags_5 > corr_a_leads_5 + 0.1 THEN signal_b || ' LEADS'
        WHEN ABS(corr_0) > 0.7 THEN 'SYNCHRONOUS'
        ELSE 'NO_CLEAR_RELATIONSHIP'
    END AS lead_lag,
    -- Best lag
    CASE
        WHEN GREATEST(ABS(corr_a_leads_5), ABS(corr_a_leads_1), ABS(corr_0), ABS(corr_a_lags_1), ABS(corr_a_lags_5)) = ABS(corr_a_leads_5) THEN '-5'
        WHEN GREATEST(ABS(corr_a_leads_5), ABS(corr_a_leads_1), ABS(corr_0), ABS(corr_a_lags_1), ABS(corr_a_lags_5)) = ABS(corr_a_leads_1) THEN '-1'
        WHEN GREATEST(ABS(corr_a_leads_5), ABS(corr_a_leads_1), ABS(corr_0), ABS(corr_a_lags_1), ABS(corr_a_lags_5)) = ABS(corr_0) THEN '0'
        WHEN GREATEST(ABS(corr_a_leads_5), ABS(corr_a_leads_1), ABS(corr_0), ABS(corr_a_lags_1), ABS(corr_a_lags_5)) = ABS(corr_a_lags_1) THEN '+1'
        ELSE '+5'
    END AS best_lag
FROM cross_correlations
WHERE ABS(corr_0) > 0.3 OR ABS(corr_a_leads_5) > 0.3 OR ABS(corr_a_lags_5) > 0.3
ORDER BY cohort, GREATEST(ABS(corr_a_leads_5), ABS(corr_a_leads_1), ABS(corr_0)) DESC;


-- ============================================================================
-- REPORT 2: INFLUENCE PROPAGATION
-- Tracks how changes propagate through signal network
-- ============================================================================

WITH
derivatives AS (
    SELECT
        cohort,
        signal_id,
        I,
        value - LAG(value) OVER w AS dy
    FROM observations
    WINDOW w AS (PARTITION BY cohort, signal_id ORDER BY I)
),

change_events AS (
    SELECT
        cohort,
        signal_id,
        I,
        dy,
        -- Mark significant changes
        CASE
            WHEN ABS(dy) > 2 * AVG(ABS(dy)) OVER (PARTITION BY cohort, signal_id) THEN 1
            ELSE 0
        END AS is_significant_change
    FROM derivatives
    WHERE dy IS NOT NULL
),

propagation AS (
    SELECT
        a.cohort,
        a.signal_id AS source_signal,
        b.signal_id AS target_signal,
        a.I AS source_time,
        MIN(b.I) AS target_time
    FROM change_events a
    JOIN change_events b
        ON a.cohort = b.cohort
        AND a.signal_id < b.signal_id
        AND b.I > a.I
        AND b.I <= a.I + 10  -- Look for response within 10 time units
        AND a.is_significant_change = 1
        AND b.is_significant_change = 1
    GROUP BY a.cohort, a.signal_id, b.signal_id, a.I
)

SELECT
    cohort,
    source_signal,
    target_signal,
    COUNT(*) AS n_propagations,
    ROUND(AVG(target_time - source_time), 2) AS avg_delay,
    ROUND(MIN(target_time - source_time), 2) AS min_delay,
    ROUND(MAX(target_time - source_time), 2) AS max_delay,
    CASE
        WHEN COUNT(*) > 10 AND AVG(target_time - source_time) < 3 THEN 'STRONG_DIRECT'
        WHEN COUNT(*) > 10 THEN 'STRONG_DELAYED'
        WHEN COUNT(*) > 5 THEN 'MODERATE'
        ELSE 'WEAK'
    END AS influence_strength
FROM propagation
GROUP BY cohort, source_signal, target_signal
HAVING COUNT(*) >= 3
ORDER BY cohort, n_propagations DESC;


-- ============================================================================
-- REPORT 3: SIGNAL INFLUENCE RANKING
-- Ranks signals by how much they influence others
-- ============================================================================

WITH
lagged AS (
    SELECT
        a.cohort,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        a.I,
        LAG(a.value, 3) OVER wa AS y_a_lag3,
        b.value AS y_b
    FROM observations a
    JOIN observations b
        ON a.cohort = b.cohort
        AND a.I = b.I
        AND a.signal_id != b.signal_id
    WINDOW wa AS (PARTITION BY a.cohort, a.signal_id, b.signal_id ORDER BY a.I)
),

predictive_corr AS (
    SELECT
        cohort,
        signal_a,
        signal_b,
        CORR(y_a_lag3, y_b) AS predictive_correlation
    FROM lagged
    WHERE y_a_lag3 IS NOT NULL
    GROUP BY cohort, signal_a, signal_b
),

influence_scores AS (
    SELECT
        cohort,
        signal_a AS signal_id,
        -- How well does this signal predict others?
        AVG(ABS(predictive_correlation)) AS avg_influence,
        MAX(ABS(predictive_correlation)) AS max_influence,
        SUM(CASE WHEN ABS(predictive_correlation) > 0.5 THEN 1 ELSE 0 END) AS n_strong_influences
    FROM predictive_corr
    GROUP BY cohort, signal_a
)

SELECT
    cohort,
    signal_id,
    ROUND(avg_influence, 3) AS avg_predictive_power,
    ROUND(max_influence, 3) AS max_predictive_power,
    n_strong_influences,
    RANK() OVER (PARTITION BY cohort ORDER BY avg_influence DESC) AS influence_rank,
    CASE
        WHEN avg_influence > 0.4 THEN 'HIGH_INFLUENCE'
        WHEN avg_influence > 0.2 THEN 'MODERATE_INFLUENCE'
        ELSE 'LOW_INFLUENCE'
    END AS influence_class,
    CASE
        WHEN avg_influence > 0.4 THEN 'Key driver - monitor closely'
        WHEN avg_influence > 0.2 THEN 'Secondary driver'
        ELSE 'Follower signal'
    END AS interpretation
FROM influence_scores
ORDER BY cohort, avg_influence DESC;


-- ============================================================================
-- REPORT 4: RESPONSIVE SIGNALS
-- Identifies signals that respond to changes in others
-- ============================================================================

WITH
lagged AS (
    SELECT
        a.cohort,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        a.I,
        a.value AS y_a,
        LEAD(b.value, 3) OVER wb AS y_b_future
    FROM observations a
    JOIN observations b
        ON a.cohort = b.cohort
        AND a.I = b.I
        AND a.signal_id != b.signal_id
    WINDOW wb AS (PARTITION BY a.cohort, a.signal_id, b.signal_id ORDER BY b.I)
),

response_corr AS (
    SELECT
        cohort,
        signal_b AS signal_id,
        AVG(ABS(CORR(y_a, y_b_future))) AS avg_responsiveness,
        MAX(ABS(CORR(y_a, y_b_future))) AS max_responsiveness
    FROM lagged
    WHERE y_b_future IS NOT NULL
    GROUP BY cohort, signal_b
)

SELECT
    cohort,
    signal_id,
    ROUND(avg_responsiveness, 3) AS avg_response,
    ROUND(max_responsiveness, 3) AS max_response,
    RANK() OVER (PARTITION BY cohort ORDER BY avg_responsiveness DESC) AS response_rank,
    CASE
        WHEN avg_responsiveness > 0.4 THEN 'HIGHLY_RESPONSIVE'
        WHEN avg_responsiveness > 0.2 THEN 'MODERATELY_RESPONSIVE'
        ELSE 'INDEPENDENT'
    END AS response_class
FROM response_corr
ORDER BY cohort, avg_responsiveness DESC;


-- ============================================================================
-- REPORT 5: FEEDBACK LOOP DETECTION
-- Identifies potential feedback relationships (A affects B, B affects A)
-- ============================================================================

WITH
lagged AS (
    SELECT
        a.cohort,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        a.I,
        LAG(a.value, 3) OVER wa AS y_a_lag,
        b.value AS y_b,
        LAG(b.value, 3) OVER wb AS y_b_lag,
        a.value AS y_a
    FROM observations a
    JOIN observations b
        ON a.cohort = b.cohort
        AND a.I = b.I
        AND a.signal_id < b.signal_id
    WINDOW
        wa AS (PARTITION BY a.cohort, a.signal_id, b.signal_id ORDER BY a.I),
        wb AS (PARTITION BY a.cohort, a.signal_id, b.signal_id ORDER BY b.I)
),

bidirectional AS (
    SELECT
        cohort,
        signal_a,
        signal_b,
        CORR(y_a_lag, y_b) AS a_predicts_b,
        CORR(y_b_lag, y_a) AS b_predicts_a
    FROM lagged
    WHERE y_a_lag IS NOT NULL AND y_b_lag IS NOT NULL
    GROUP BY cohort, signal_a, signal_b
)

SELECT
    cohort,
    signal_a,
    signal_b,
    ROUND(a_predicts_b, 3) AS a_to_b,
    ROUND(b_predicts_a, 3) AS b_to_a,
    ROUND(ABS(a_predicts_b) + ABS(b_predicts_a), 3) AS total_coupling,
    CASE
        WHEN ABS(a_predicts_b) > 0.3 AND ABS(b_predicts_a) > 0.3 THEN 'FEEDBACK_LOOP'
        WHEN ABS(a_predicts_b) > 0.3 THEN signal_a || ' -> ' || signal_b
        WHEN ABS(b_predicts_a) > 0.3 THEN signal_b || ' -> ' || signal_a
        ELSE 'NO_CAUSAL_LINK'
    END AS relationship_type,
    CASE
        WHEN ABS(a_predicts_b) > 0.3 AND ABS(b_predicts_a) > 0.3
         AND SIGN(a_predicts_b) = SIGN(b_predicts_a) THEN 'POSITIVE_FEEDBACK'
        WHEN ABS(a_predicts_b) > 0.3 AND ABS(b_predicts_a) > 0.3
         AND SIGN(a_predicts_b) != SIGN(b_predicts_a) THEN 'NEGATIVE_FEEDBACK'
        ELSE 'N/A'
    END AS feedback_type
FROM bidirectional
WHERE ABS(a_predicts_b) > 0.2 OR ABS(b_predicts_a) > 0.2
ORDER BY cohort, ABS(a_predicts_b) + ABS(b_predicts_a) DESC;


-- ============================================================================
-- REPORT 6: CAUSAL CHAIN CANDIDATES
-- Identifies potential A -> B -> C causal chains
-- ============================================================================

WITH
lagged AS (
    SELECT
        a.cohort,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        a.I,
        LAG(a.value, 3) OVER wa AS y_a_lag,
        b.value AS y_b
    FROM observations a
    JOIN observations b
        ON a.cohort = b.cohort
        AND a.I = b.I
        AND a.signal_id != b.signal_id
    WINDOW wa AS (PARTITION BY a.cohort, a.signal_id, b.signal_id ORDER BY a.I)
),

pairwise AS (
    SELECT
        cohort,
        signal_a,
        signal_b,
        CORR(y_a_lag, y_b) AS predictive_corr
    FROM lagged
    WHERE y_a_lag IS NOT NULL
    GROUP BY cohort, signal_a, signal_b
    HAVING ABS(CORR(y_a_lag, y_b)) > 0.3
)

-- Find chains: A -> B and B -> C
SELECT
    ab.cohort,
    ab.signal_a || ' -> ' || ab.signal_b || ' -> ' || bc.signal_b AS causal_chain,
    ROUND(ab.predictive_corr, 3) AS link_1_strength,
    ROUND(bc.predictive_corr, 3) AS link_2_strength,
    ROUND((ABS(ab.predictive_corr) + ABS(bc.predictive_corr)) / 2, 3) AS chain_strength,
    CASE
        WHEN SIGN(ab.predictive_corr) = SIGN(bc.predictive_corr) THEN 'AMPLIFYING'
        ELSE 'INVERTING'
    END AS chain_type
FROM pairwise ab
JOIN pairwise bc
    ON ab.cohort = bc.cohort
    AND ab.signal_b = bc.signal_a
    AND ab.signal_a != bc.signal_b
ORDER BY ab.cohort, (ABS(ab.predictive_corr) + ABS(bc.predictive_corr)) DESC
LIMIT 50;
