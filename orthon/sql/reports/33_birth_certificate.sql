-- ============================================================================
-- ORTHON SQL: 33_birth_certificate.sql
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
        entity_id,
        MIN(I) as first_I,
        MAX(I) as last_I,
        COUNT(*) as total_obs
    FROM read_parquet('{prism_output}/physics.parquet')
    GROUP BY entity_id
),
early_obs AS (
    SELECT
        p.entity_id,
        p.I,
        p.coherence,
        p.state_velocity,
        p.dissipation_rate,
        p.effective_dim,
        p.eigenvalue_entropy,
        l.total_obs,
        ROW_NUMBER() OVER (PARTITION BY p.entity_id ORDER BY p.I) as obs_num
    FROM read_parquet('{prism_output}/physics.parquet') p
    JOIN lifecycle_info l ON p.entity_id = l.entity_id
)
SELECT
    entity_id,
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
GROUP BY entity_id, total_obs;

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
WITH fleet_early AS (
    SELECT
        AVG(early_coherence) as fleet_coh,
        STDDEV(early_coherence) as fleet_coh_std,
        AVG(early_velocity) as fleet_vel,
        STDDEV(early_velocity) as fleet_vel_std,
        AVG(early_coherence_std) as fleet_coh_vol,
        STDDEV(early_coherence_std) as fleet_coh_vol_std
    FROM early_life
)
SELECT
    e.entity_id,
    e.lifespan,
    e.early_coherence,
    e.early_velocity,
    e.early_dissipation,
    e.early_coherence_std,

    -- Birth certificate components (higher = healthier start)
    -- High early coherence = good
    ROUND(1.0 / (1.0 + EXP(-(e.early_coherence - f.fleet_coh) / NULLIF(f.fleet_coh_std, 0))), 3)
        as early_coupling_score,

    -- Low early velocity = good (invert)
    ROUND(1.0 / (1.0 + EXP((e.early_velocity - f.fleet_vel) / NULLIF(f.fleet_vel_std, 0))), 3)
        as early_stability_score,

    -- Low early volatility = good (invert)
    ROUND(1.0 / (1.0 + EXP((e.early_coherence_std - f.fleet_coh_vol) / NULLIF(f.fleet_coh_vol_std, 0))), 3)
        as early_consistency_score,

    -- BIRTH CERTIFICATE SCORE: weighted combination
    ROUND(
        (1.0 / (1.0 + EXP(-(e.early_coherence - f.fleet_coh) / NULLIF(f.fleet_coh_std, 0)))) * 0.4 +
        (1.0 / (1.0 + EXP((e.early_velocity - f.fleet_vel) / NULLIF(f.fleet_vel_std, 0)))) * 0.4 +
        (1.0 / (1.0 + EXP((e.early_coherence_std - f.fleet_coh_vol) / NULLIF(f.fleet_coh_vol_std, 0)))) * 0.2
    , 3) as birth_certificate_score,

    -- Classification
    CASE
        WHEN (1.0 / (1.0 + EXP(-(e.early_coherence - f.fleet_coh) / NULLIF(f.fleet_coh_std, 0)))) * 0.4 +
             (1.0 / (1.0 + EXP((e.early_velocity - f.fleet_vel) / NULLIF(f.fleet_vel_std, 0)))) * 0.4 +
             (1.0 / (1.0 + EXP((e.early_coherence_std - f.fleet_coh_vol) / NULLIF(f.fleet_coh_vol_std, 0)))) * 0.2
             > 0.65 THEN 'EXCELLENT'
        WHEN (1.0 / (1.0 + EXP(-(e.early_coherence - f.fleet_coh) / NULLIF(f.fleet_coh_std, 0)))) * 0.4 +
             (1.0 / (1.0 + EXP((e.early_velocity - f.fleet_vel) / NULLIF(f.fleet_vel_std, 0)))) * 0.4 +
             (1.0 / (1.0 + EXP((e.early_coherence_std - f.fleet_coh_vol) / NULLIF(f.fleet_coh_vol_std, 0)))) * 0.2
             > 0.5 THEN 'GOOD'
        WHEN (1.0 / (1.0 + EXP(-(e.early_coherence - f.fleet_coh) / NULLIF(f.fleet_coh_std, 0)))) * 0.4 +
             (1.0 / (1.0 + EXP((e.early_velocity - f.fleet_vel) / NULLIF(f.fleet_vel_std, 0)))) * 0.4 +
             (1.0 / (1.0 + EXP((e.early_coherence_std - f.fleet_coh_vol) / NULLIF(f.fleet_coh_vol_std, 0)))) * 0.2
             > 0.35 THEN 'FAIR'
        ELSE 'POOR'
    END as birth_grade

FROM early_life e
CROSS JOIN fleet_early f;

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
    entity_id,
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
    b.entity_id,
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
