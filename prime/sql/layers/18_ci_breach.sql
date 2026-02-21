-- ============================================================================
-- LAYER 18: CONFIDENCE INTERVAL BREACH DETECTION
-- When effective_dim moves outside its jackknife CI, geometry is changing
-- faster than measurement uncertainty can explain. That's a canary.
--
-- One view. Three columns on top of state_geometry.
-- Source: state_geometry (eff_dim_ci_low, eff_dim_ci_high from jackknife)
-- ============================================================================

CREATE OR REPLACE VIEW v_ci_breach AS
SELECT
    *,
    -- Breach: effective_dim outside [ci_low, ci_high]
    (effective_dim < eff_dim_ci_low OR effective_dim > eff_dim_ci_high) AS ci_breach,
    -- Direction: which way did it escape?
    CASE
        WHEN effective_dim < eff_dim_ci_low THEN 'below'
        WHEN effective_dim > eff_dim_ci_high THEN 'above'
        ELSE NULL
    END AS ci_breach_direction,
    -- Magnitude normalized by CI width.
    -- 5.0 means geometry moved 5× the width of the CI. That's not noise.
    CASE
        WHEN (eff_dim_ci_high - eff_dim_ci_low) > 0 THEN
            CASE
                WHEN effective_dim < eff_dim_ci_low
                    THEN (eff_dim_ci_low - effective_dim) / (eff_dim_ci_high - eff_dim_ci_low)
                WHEN effective_dim > eff_dim_ci_high
                    THEN (effective_dim - eff_dim_ci_high) / (eff_dim_ci_high - eff_dim_ci_low)
                ELSE 0.0
            END
        ELSE NULL
    END AS ci_breach_magnitude,
FROM state_geometry
WHERE eff_dim_ci_low IS NOT NULL
  AND eff_dim_ci_high IS NOT NULL;


-- ============================================================================
-- ANALYSIS QUERIES (not views — run ad hoc)
-- ============================================================================

-- Breach rate per cohort, ranked. Cross-reference against lifecycle length.
WITH breach_rates AS (
    SELECT
        cohort,
        engine,
        COUNT(*) AS n_windows,
        SUM(CASE WHEN ci_breach THEN 1 ELSE 0 END) AS n_breaches,
        ROUND(SUM(CASE WHEN ci_breach THEN 1 ELSE 0 END)::DOUBLE / COUNT(*), 4) AS breach_rate,
        ROUND(AVG(CASE WHEN ci_breach THEN ci_breach_magnitude END), 3) AS mean_magnitude,
    FROM v_ci_breach
    GROUP BY cohort, engine
),
lifecycle AS (
    SELECT cohort, MAX(signal_0_center) + 1 AS lifecycle_length
    FROM v_ci_breach
    WHERE engine = (SELECT MIN(engine) FROM v_ci_breach)
    GROUP BY cohort
)
SELECT
    br.cohort,
    br.engine,
    br.breach_rate,
    br.n_breaches,
    br.mean_magnitude,
    lc.lifecycle_length,
    RANK() OVER (PARTITION BY br.engine ORDER BY br.breach_rate DESC) AS breach_rank,
    RANK() OVER (PARTITION BY br.engine ORDER BY lc.lifecycle_length ASC) AS rul_rank,
FROM breach_rates br
JOIN lifecycle lc USING (cohort)
ORDER BY br.engine, br.breach_rate DESC;

-- The money question: do high-breach engines die sooner?
WITH per_cohort AS (
    SELECT
        cohort,
        AVG(CASE WHEN ci_breach THEN 1.0 ELSE 0.0 END) AS breach_rate,
    FROM v_ci_breach
    GROUP BY cohort
),
lifecycle AS (
    SELECT cohort, MAX(signal_0_center) + 1 AS lifecycle_length
    FROM v_ci_breach
    WHERE engine = (SELECT MIN(engine) FROM v_ci_breach)
    GROUP BY cohort
)
SELECT
    ROUND(CORR(pc.breach_rate, lc.lifecycle_length), 4) AS breach_vs_life_corr,
    COUNT(*) AS n_cohorts,
FROM per_cohort pc
JOIN lifecycle lc USING (cohort);

-- Do breaches cluster in late life?
WITH lifecycle AS (
    SELECT cohort, MAX(signal_0_center) AS max_I
    FROM v_ci_breach
    WHERE engine = (SELECT MIN(engine) FROM v_ci_breach)
    GROUP BY cohort
)
SELECT
    CASE
        WHEN b.signal_0_center::DOUBLE / NULLIF(lc.max_I, 0) < 0.5 THEN 'early'
        ELSE 'late'
    END AS half,
    b.engine,
    ROUND(AVG(CASE WHEN b.ci_breach THEN 1.0 ELSE 0.0 END), 4) AS breach_rate,
    ROUND(AVG(CASE WHEN b.ci_breach THEN b.ci_breach_magnitude END), 3) AS mean_magnitude,
FROM v_ci_breach b
JOIN lifecycle lc USING (cohort)
GROUP BY half, b.engine
ORDER BY half, b.engine;
