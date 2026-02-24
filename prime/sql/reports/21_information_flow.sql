-- ============================================================================
-- REPORT 21: INFORMATION FLOW & TRANSFER ENTROPY
-- ============================================================================
-- Directional information transfer between signals:
--   Granger causality — does past of A predict future of B beyond B's own past?
--   Transfer entropy — non-parametric version (bits transferred)
--   KL divergence — distributional distance between signals
--   JS divergence — symmetric divergence measure
--
-- Source: cohort_information_flow (alias: information_flow)
-- ============================================================================


-- ============================================================================
-- SECTION 1: PAIRWISE INFORMATION FLOW
-- Full matrix of directional information transfer
-- ============================================================================

SELECT
    signal_a,
    signal_b,
    cohort,
    n_samples,
    ROUND(granger_f_a_to_b, 2) AS granger_a_to_b,
    ROUND(granger_f_b_to_a, 2) AS granger_b_to_a,
    ROUND(granger_p_a_to_b, 4) AS p_a_to_b,
    ROUND(granger_p_b_to_a, 4) AS p_b_to_a,
    ROUND(transfer_entropy_a_to_b, 4) AS te_a_to_b,
    ROUND(transfer_entropy_b_to_a, 4) AS te_b_to_a,
    -- Net transfer entropy: positive = A drives B, negative = B drives A
    ROUND(transfer_entropy_a_to_b - transfer_entropy_b_to_a, 4) AS net_te,
    CASE
        WHEN granger_p_a_to_b < 0.05 AND granger_p_b_to_a < 0.05 THEN 'BIDIRECTIONAL'
        WHEN granger_p_a_to_b < 0.05 THEN 'A_DRIVES_B'
        WHEN granger_p_b_to_a < 0.05 THEN 'B_DRIVES_A'
        ELSE 'INDEPENDENT'
    END AS causal_direction,
    CASE
        WHEN granger_p_a_to_b < 0.05 AND granger_p_b_to_a < 0.05
            AND ABS(transfer_entropy_a_to_b - transfer_entropy_b_to_a) < 0.005
        THEN 'SYMMETRIC_FEEDBACK'
        WHEN granger_p_a_to_b < 0.05 AND granger_p_b_to_a < 0.05
        THEN 'ASYMMETRIC_FEEDBACK'
        ELSE NULL
    END AS feedback_type
FROM information_flow
ORDER BY GREATEST(granger_f_a_to_b, granger_f_b_to_a) DESC;


-- ============================================================================
-- SECTION 2: STRONGEST CAUSAL DRIVERS
-- Which signals drive the most other signals?
-- ============================================================================

WITH drivers AS (
    SELECT signal_a AS signal_id, signal_b AS target, cohort,
        granger_f_a_to_b AS granger_f, granger_p_a_to_b AS p_val,
        transfer_entropy_a_to_b AS te
    FROM information_flow
    WHERE granger_p_a_to_b < 0.05
    UNION ALL
    SELECT signal_b AS signal_id, signal_a AS target, cohort,
        granger_f_b_to_a AS granger_f, granger_p_b_to_a AS p_val,
        transfer_entropy_b_to_a AS te
    FROM information_flow
    WHERE granger_p_b_to_a < 0.05
)
SELECT
    signal_id,
    cohort,
    COUNT(*) AS n_targets_driven,
    ROUND(AVG(granger_f), 2) AS avg_granger_f,
    ROUND(AVG(te), 4) AS avg_transfer_entropy,
    STRING_AGG(target, ', ' ORDER BY granger_f DESC) AS driven_signals
FROM drivers
GROUP BY signal_id, cohort
ORDER BY n_targets_driven DESC, avg_granger_f DESC;


-- ============================================================================
-- SECTION 3: CAUSAL CHAIN DETECTION
-- A → B → C chains (transitive causality)
-- ============================================================================

WITH directed_edges AS (
    SELECT signal_a AS src, signal_b AS dst, cohort,
        granger_f_a_to_b AS strength, transfer_entropy_a_to_b AS te
    FROM information_flow WHERE granger_p_a_to_b < 0.05
    UNION ALL
    SELECT signal_b AS src, signal_a AS dst, cohort,
        granger_f_b_to_a AS strength, transfer_entropy_b_to_a AS te
    FROM information_flow WHERE granger_p_b_to_a < 0.05
)
SELECT
    e1.src AS origin,
    e1.dst AS intermediate,
    e2.dst AS endpoint,
    e1.cohort,
    ROUND(e1.strength, 2) AS f_step1,
    ROUND(e2.strength, 2) AS f_step2,
    ROUND(LEAST(e1.te, e2.te), 4) AS bottleneck_te,
    e1.src || ' → ' || e1.dst || ' → ' || e2.dst AS chain
FROM directed_edges e1
JOIN directed_edges e2
    ON e1.dst = e2.src AND e1.cohort = e2.cohort AND e1.src != e2.dst
ORDER BY LEAST(e1.strength, e2.strength) DESC
LIMIT 20;


-- ============================================================================
-- SECTION 4: DISTRIBUTIONAL DISTANCES
-- KL and JS divergence between signal pairs
-- ============================================================================

SELECT
    signal_a,
    signal_b,
    cohort,
    ROUND(kl_divergence_a_to_b, 4) AS kl_a_to_b,
    ROUND(kl_divergence_b_to_a, 4) AS kl_b_to_a,
    ROUND(js_divergence, 4) AS js_divergence,
    CASE
        WHEN js_divergence < 0.01 THEN 'NEAR_IDENTICAL'
        WHEN js_divergence < 0.1 THEN 'SIMILAR'
        WHEN js_divergence < 0.5 THEN 'DIFFERENT'
        ELSE 'HIGHLY_DIVERGENT'
    END AS distributional_similarity,
    -- Asymmetry in KL divergence indicates which signal has heavier tails
    ROUND(ABS(kl_divergence_a_to_b - kl_divergence_b_to_a), 4) AS kl_asymmetry
FROM information_flow
ORDER BY js_divergence DESC;
