-- ============================================================================
-- COUPLING RANKED (replaces gated coupling_strength)
-- ============================================================================

CREATE OR REPLACE VIEW v_coupling_ranked AS
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
WHERE correlation IS NOT NULL;

-- Most decoupled pairs (biggest delta)
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
ORDER BY ABS(coupling_delta) DESC
LIMIT 30;
