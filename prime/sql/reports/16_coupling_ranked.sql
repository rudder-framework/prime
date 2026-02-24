-- ============================================================================
-- 16_coupling_ranked.sql
-- ============================================================================
-- Coupling analysis report
-- Depends on: v_redundant_pairs, v_coupling_ranked (layer 16)
-- ============================================================================

.print ''
.print '============================================================================'
.print '                     COUPLING RANKED ANALYSIS                              '
.print '============================================================================'

-- Most decoupled pairs (biggest delta, excluding sign flips and redundant pairs)
SELECT
    cohort,
    signal_a,
    signal_b,
    signal_0_end,
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
