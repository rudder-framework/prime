-- ============================================================================
-- BRITTLENESS SCORE
-- High condition number + tight eigenvalue gap + low temperature = brittle
-- ============================================================================

CREATE OR REPLACE VIEW v_brittleness AS
WITH geometry_metrics AS (
    SELECT
        cohort,
        I,
        condition_number,
        eigenvalue_1,
        eigenvalue_2,
        eigenvalue_1 - eigenvalue_2 AS eigenvalue_gap,
        effective_dim
    FROM state_geometry
    WHERE condition_number IS NOT NULL
      AND NOT isnan(condition_number)
),
thermo_metrics AS (
    SELECT
        cohort,
        temperature
    FROM cohort_thermodynamics
    WHERE temperature IS NOT NULL
      AND temperature > 0
)
SELECT
    g.cohort,
    g.I,
    g.condition_number,
    g.eigenvalue_gap,
    g.effective_dim,
    t.temperature,

    -- Brittleness score
    g.condition_number
        * (1.0 / NULLIF(ABS(g.eigenvalue_gap), 0))
        * (1.0 / NULLIF(t.temperature, 0))
    AS brittleness_score,

    -- Rank by brittleness across fleet
    RANK() OVER (
        ORDER BY g.condition_number
            * (1.0 / NULLIF(ABS(g.eigenvalue_gap), 0))
            * (1.0 / NULLIF(t.temperature, 0))
        DESC NULLS LAST
    ) AS brittleness_rank,

    -- Percentile within cohort history
    PERCENT_RANK() OVER (
        PARTITION BY g.cohort
        ORDER BY g.condition_number
            * (1.0 / NULLIF(ABS(g.eigenvalue_gap), 0))
            * (1.0 / NULLIF(t.temperature, 0))
    ) AS brittleness_percentile

FROM geometry_metrics g
LEFT JOIN thermo_metrics t ON g.cohort = t.cohort;

-- Summary: most brittle cohorts
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
ORDER BY avg_brittleness DESC NULLS LAST
LIMIT 30;

-- Brittleness over time (fleet average per window)
SELECT
    I,
    ROUND(AVG(brittleness_score), 4) AS fleet_avg_brittleness,
    ROUND(MAX(brittleness_score), 4) AS fleet_max_brittleness,
    COUNT(*) AS n_cohorts
FROM v_brittleness
GROUP BY I
ORDER BY I;
