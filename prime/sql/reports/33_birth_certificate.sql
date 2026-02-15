-- ============================================================================
-- Rudder SQL: 33_birth_certificate.sql
-- ============================================================================
-- BIRTH CERTIFICATE: Early-life stability predicts lifespan
--
-- Key finding from turbofan analysis:
--   Early-life metrics (first 20% of lifecycle) correlate with total lifespan.
--   Systems with "good" birth certificates live longer.
--
-- This report computes the "birth certificate" for each entity and
-- (for run-to-failure data) correlates it with actual lifespan.
-- ============================================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║                    BIRTH CERTIFICATE ANALYSIS                               ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'

-- ============================================================================
-- SECTION 1: EXTRACT EARLY-LIFE METRICS
-- ============================================================================

.print ''
.print '=== SECTION 1: Early-Life Metrics (First 20% of observations) ==='

CREATE OR REPLACE TABLE early_life AS
WITH lifecycle_info AS (
    SELECT
        cohort,
        MIN(I) as first_I,
        MAX(I) as last_I,
        COUNT(*) as total_obs
    FROM read_parquet('{prism_output}/physics.parquet')
    GROUP BY cohort
),
early_obs AS (
    SELECT
        p.cohort,
        p.I,
        p.coherence,
        p.state_velocity,
        p.dissipation_rate,
        p.effective_dim,
        p.eigenvalue_entropy,
        l.total_obs,
        ROW_NUMBER() OVER (PARTITION BY p.cohort ORDER BY p.I) as obs_num
    FROM read_parquet('{prism_output}/physics.parquet') p
    JOIN lifecycle_info l ON p.cohort = l.cohort
)
SELECT
    cohort,
    total_obs as lifespan,

    -- Early-life coherence
    AVG(coherence) as early_coherence,
    STDDEV(coherence) as early_coherence_std,

    -- Early-life velocity
    AVG(state_velocity) as early_velocity,
    STDDEV(state_velocity) as early_velocity_std,

    -- Early-life dissipation
    AVG(dissipation_rate) as early_dissipation,

    -- Early-life dimension
    AVG(effective_dim) as early_effective_dim,

    -- Early-life entropy
    AVG(eigenvalue_entropy) as early_entropy

FROM early_obs
WHERE obs_num <= total_obs * 0.2  -- First 20%
GROUP BY cohort, total_obs;

SELECT
    COUNT(*) as n_entities,
    ROUND(AVG(lifespan), 1) as avg_lifespan,
    ROUND(AVG(early_coherence), 3) as avg_early_coherence,
    ROUND(AVG(early_velocity), 4) as avg_early_velocity,
    ROUND(AVG(early_dissipation), 2) as avg_early_dissipation
FROM early_life;


-- ============================================================================
-- SECTION 2: BIRTH CERTIFICATE SCORE
-- ============================================================================

.print ''
.print '=== SECTION 2: Birth Certificate Score ==='

CREATE OR REPLACE TABLE birth_certificate AS
WITH ranked AS (
    SELECT
        e.cohort,
        e.lifespan,
        e.early_coherence,
        e.early_velocity,
        e.early_dissipation,
        e.early_coherence_std,

        -- Fleet percentile ranks (no z-scores, no Gaussian assumption)
        -- High early coherence = good
        ROUND(PERCENT_RANK() OVER (ORDER BY e.early_coherence), 3)
            as early_coupling_score,

        -- Low early velocity = good (invert)
        ROUND(PERCENT_RANK() OVER (ORDER BY e.early_velocity DESC), 3)
            as early_stability_score,

        -- Low early volatility = good (invert)
        ROUND(PERCENT_RANK() OVER (ORDER BY e.early_coherence_std DESC), 3)
            as early_consistency_score

    FROM early_life e
)
SELECT
    cohort,
    lifespan,
    early_coherence,
    early_velocity,
    early_dissipation,
    early_coherence_std,

    early_coupling_score,
    early_stability_score,
    early_consistency_score,

    -- BIRTH CERTIFICATE SCORE: weighted combination
    ROUND(
        early_coupling_score * 0.4 +
        early_stability_score * 0.4 +
        early_consistency_score * 0.2
    , 3) as birth_certificate_score,

    -- Classification
    CASE
        WHEN early_coupling_score * 0.4 + early_stability_score * 0.4 + early_consistency_score * 0.2
             > 0.65 THEN 'EXCELLENT'
        WHEN early_coupling_score * 0.4 + early_stability_score * 0.4 + early_consistency_score * 0.2
             > 0.5 THEN 'GOOD'
        WHEN early_coupling_score * 0.4 + early_stability_score * 0.4 + early_consistency_score * 0.2
             > 0.35 THEN 'FAIR'
        ELSE 'POOR'
    END as birth_grade

FROM ranked;

SELECT
    birth_grade,
    COUNT(*) as n_entities,
    ROUND(AVG(birth_certificate_score), 3) as avg_score,
    ROUND(AVG(lifespan), 1) as avg_lifespan,
    ROUND(MIN(lifespan), 0) as min_lifespan,
    ROUND(MAX(lifespan), 0) as max_lifespan
FROM birth_certificate
GROUP BY birth_grade
ORDER BY avg_score DESC;


-- ============================================================================
-- SECTION 3: BIRTH CERTIFICATE vs LIFESPAN CORRELATION
-- ============================================================================

.print ''
.print '=== SECTION 3: Birth Certificate vs Lifespan ==='

-- Compute correlation
SELECT
    ROUND(CORR(birth_certificate_score, lifespan), 3) as score_lifespan_corr,
    ROUND(CORR(early_coherence, lifespan), 3) as coherence_lifespan_corr,
    ROUND(CORR(early_velocity, lifespan), 3) as velocity_lifespan_corr,
    ROUND(CORR(early_coherence_std, lifespan), 3) as volatility_lifespan_corr
FROM birth_certificate;


-- ============================================================================
-- SECTION 4: PROGNOSIS BY BIRTH GRADE
-- ============================================================================

.print ''
.print '=== SECTION 4: Prognosis by Birth Grade ==='

SELECT
    birth_grade,
    COUNT(*) as n_entities,
    ROUND(AVG(lifespan), 0) as expected_lifespan,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY lifespan), 0) as p25_lifespan,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY lifespan), 0) as p75_lifespan,

    -- Survival rate at various points
    ROUND(SUM(CASE WHEN lifespan >= 150 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 0) as pct_survive_150,
    ROUND(SUM(CASE WHEN lifespan >= 200 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 0) as pct_survive_200,
    ROUND(SUM(CASE WHEN lifespan >= 250 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 0) as pct_survive_250

FROM birth_certificate
GROUP BY birth_grade
ORDER BY expected_lifespan DESC;


-- ============================================================================
-- SECTION 5: INDIVIDUAL BIRTH CERTIFICATES
-- ============================================================================

.print ''
.print '=== SECTION 5: Individual Birth Certificates ==='

SELECT
    cohort,
    birth_grade,
    birth_certificate_score as score,
    lifespan,
    ROUND(early_coherence, 3) as early_coh,
    ROUND(early_velocity, 4) as early_vel,
    ROUND(early_coherence_std, 4) as early_vol
FROM birth_certificate
ORDER BY birth_certificate_score DESC
LIMIT 15;


-- ============================================================================
-- CREATE VIEWS
-- ============================================================================

CREATE OR REPLACE VIEW v_birth_certificate AS
SELECT * FROM birth_certificate;

CREATE OR REPLACE VIEW v_birth_prognosis AS
SELECT
    b.cohort,
    b.birth_grade,
    b.birth_certificate_score,
    b.lifespan as actual_lifespan,

    -- Expected lifespan based on birth grade peers
    (SELECT ROUND(AVG(lifespan), 0) FROM birth_certificate
     WHERE birth_grade = b.birth_grade) as expected_lifespan,

    -- Prognosis message
    CASE b.birth_grade
        WHEN 'EXCELLENT' THEN 'Strong early-life metrics. Expected long operational life.'
        WHEN 'GOOD' THEN 'Healthy start. Above-average lifespan expected.'
        WHEN 'FAIR' THEN 'Average early-life metrics. Monitor for early degradation.'
        WHEN 'POOR' THEN 'Weak early-life metrics. High risk of early failure. Prioritize monitoring.'
    END as prognosis

FROM birth_certificate b;


.print ''
.print '=== BIRTH CERTIFICATE ANALYSIS COMPLETE ==='
.print ''
.print 'Views created:'
.print '  v_birth_certificate  - Early-life scores for each entity'
.print '  v_birth_prognosis    - Lifespan prognosis based on birth grade'
.print ''
.print 'KEY INSIGHT:'
.print '  Early-life stability metrics predict total lifespan.'
.print '  "POOR" birth grades indicate systems likely to fail early.'
.print '  Use for proactive maintenance prioritization.'
.print ''
