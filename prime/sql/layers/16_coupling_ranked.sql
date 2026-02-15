-- ============================================================================
-- COUPLING RANKED (replaces gated coupling_strength)
-- Filters sign-flip artifacts from eigenvector ordering ambiguity
-- Excludes highly redundant signal pairs (|corr| > 0.99 across fleet)
-- ============================================================================

-- Identify redundant pairs: signals with near-perfect correlation across the fleet
CREATE OR REPLACE VIEW v_redundant_pairs AS
WITH pair_stats AS (
    SELECT
        LEAST(signal_a, signal_b) AS sig_lo,
        GREATEST(signal_a, signal_b) AS sig_hi,
        AVG(ABS(correlation)) AS mean_abs_corr,
        COUNT(*) AS n_observations
    FROM signal_pairwise
    WHERE correlation IS NOT NULL
    GROUP BY LEAST(signal_a, signal_b), GREATEST(signal_a, signal_b)
)
SELECT DISTINCT sig_lo, sig_hi
FROM pair_stats
WHERE mean_abs_corr > 0.99;

CREATE OR REPLACE VIEW v_coupling_ranked AS
WITH coupling_with_delta AS (
    SELECT
        signal_a,
        signal_b,
        cohort,
        I,
        correlation,
        ABS(correlation) AS coupling_magnitude,

        RANK() OVER (
            PARTITION BY cohort, I
            ORDER BY ABS(correlation) DESC
        ) AS coupling_rank,

        -- How much coupling changed from previous window
        correlation - LAG(correlation) OVER (
            PARTITION BY cohort, signal_a, signal_b ORDER BY I
        ) AS coupling_delta

    FROM signal_pairwise
    WHERE correlation IS NOT NULL
),
-- Exclude redundant pairs
coupling_filtered AS (
    SELECT c.*
    FROM coupling_with_delta c
    LEFT JOIN v_redundant_pairs rp
        ON LEAST(c.signal_a, c.signal_b) = rp.sig_lo
        AND GREATEST(c.signal_a, c.signal_b) = rp.sig_hi
    WHERE rp.sig_lo IS NULL  -- exclude redundant pairs
)
SELECT
    *,
    -- Flag sign flips (delta magnitude > 1.0 is eigenvector ambiguity, not physics)
    CASE
        WHEN ABS(coupling_delta) > 1.0 THEN TRUE
        ELSE FALSE
    END AS is_sign_flip
FROM coupling_filtered;

-- Most decoupled pairs (biggest delta, excluding sign flips and redundant pairs)
SELECT
    cohort,
    signal_a,
    signal_b,
    I,
    ROUND(correlation, 4) AS correlation,
    ROUND(coupling_delta, 4) AS delta,
    coupling_rank
FROM v_coupling_ranked
WHERE coupling_delta IS NOT NULL
  AND is_sign_flip = FALSE
ORDER BY ABS(coupling_delta) DESC
LIMIT 30;

-- Sign flips summary (for reference -- these are computational, not physical)
SELECT
    signal_a,
    signal_b,
    COUNT(*) AS n_sign_flips,
    COUNT(DISTINCT cohort) AS n_cohorts_affected
FROM v_coupling_ranked
WHERE is_sign_flip = TRUE
GROUP BY signal_a, signal_b
ORDER BY n_sign_flips DESC
LIMIT 20;
