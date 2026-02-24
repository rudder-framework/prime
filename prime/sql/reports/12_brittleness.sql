-- ============================================================================
-- 12_brittleness.sql
-- ============================================================================
-- Brittleness analysis report
-- Depends on: v_brittleness (layer 12), signal_pairwise
-- ============================================================================

.print ''
.print '============================================================================'
.print '                     BRITTLENESS ANALYSIS                                  '
.print '============================================================================'

-- Summary: most brittle cohorts (with coupling context)
WITH brittleness_agg AS (
    SELECT
        cohort,
        ROUND(AVG(brittleness_score), 4) AS avg_brittleness,
        ROUND(MAX(brittleness_score), 4) AS max_brittleness,
        ROUND(AVG(condition_number), 2) AS avg_condition_number,
        ROUND(AVG(eigenvalue_gap), 6) AS avg_eigenvalue_gap,
        ROUND(AVG(temperature), 4) AS avg_temperature,
        ROUND(AVG(effective_dim), 2) AS avg_effective_dim
    FROM v_brittleness
    GROUP BY cohort
),
coupling_agg AS (
    SELECT
        cohort,
        ROUND(AVG(ABS(correlation)), 3) AS avg_abs_corr,
        CASE
            WHEN AVG(ABS(correlation)) > 0.4 THEN 'HIGHLY_COUPLED'
            WHEN AVG(ABS(correlation)) > 0.25 THEN 'MODERATELY_COUPLED'
            ELSE 'LOOSELY_COUPLED'
        END AS coupling_level
    FROM signal_pairwise
    WHERE correlation IS NOT NULL AND NOT isnan(correlation)
    GROUP BY cohort
)
SELECT
    b.*,
    c.avg_abs_corr,
    c.coupling_level,
    CASE
        WHEN b.avg_brittleness IS NULL THEN 'NO_DATA'
        WHEN c.avg_abs_corr IS NOT NULL AND c.avg_abs_corr < 0.05 AND b.avg_brittleness > 50
            THEN 'DECOUPLED_SYSTEM'
        WHEN b.avg_brittleness > 100 THEN 'CRITICALLY_BRITTLE'
        WHEN b.avg_brittleness > 30  THEN 'ELEVATED'
        ELSE 'HEALTHY'
    END AS brittleness_interpretation
FROM brittleness_agg b
LEFT JOIN coupling_agg c ON b.cohort = c.cohort
ORDER BY b.avg_brittleness DESC NULLS LAST
LIMIT 30;

-- Brittleness over time (fleet average per window)
SELECT
    signal_0_center,
    ROUND(AVG(brittleness_score), 4) AS fleet_avg_brittleness,
    ROUND(MAX(brittleness_score), 4) AS fleet_max_brittleness,
    COUNT(*) AS n_cohorts
FROM v_brittleness
GROUP BY signal_0_center
ORDER BY signal_0_center;
