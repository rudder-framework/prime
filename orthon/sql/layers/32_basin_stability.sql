-- ============================================================================
-- ORTHON SQL: 32_basin_stability.sql
-- ============================================================================
-- BASIN STABILITY: Composite stability score from dynamics
--
-- Basin stability measures how "deep" the system's attractor basin is:
--   - Deeper basin = more stable = tolerates larger perturbations
--   - Shallow basin = less stable = small perturbations cause transitions
--
-- Combines:
--   - Lyapunov (trajectory divergence)
--   - Coherence stability (coupling consistency)
--   - State velocity (movement rate)
--   - Volatility (variability)
-- ============================================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║                    BASIN STABILITY ANALYSIS                                 ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'

-- ============================================================================
-- SECTION 1: COMPUTE BASIN STABILITY COMPONENTS
-- ============================================================================

.print ''
.print '=== SECTION 1: Stability Components ==='

CREATE OR REPLACE TABLE basin_components AS
SELECT
    p.entity_id,

    -- Coherence stability (higher = more stable)
    AVG(p.coherence) as mean_coherence,
    STDDEV(p.coherence) as coherence_volatility,

    -- State stability (lower velocity = more stable)
    AVG(p.state_velocity) as mean_velocity,
    STDDEV(p.state_velocity) as velocity_volatility,

    -- Energy stability (lower dissipation = more stable)
    AVG(p.dissipation_rate) as mean_dissipation,

    -- Dimensional stability (consistent dim = more stable)
    AVG(p.effective_dim) as mean_effective_dim,
    STDDEV(p.effective_dim) as dim_volatility,

    -- Count
    COUNT(*) as n_observations

FROM read_parquet('{prism_output}/physics.parquet') p
GROUP BY p.entity_id;

SELECT
    ROUND(AVG(mean_coherence), 3) as fleet_mean_coherence,
    ROUND(AVG(coherence_volatility), 4) as fleet_coherence_vol,
    ROUND(AVG(mean_velocity), 3) as fleet_mean_velocity,
    ROUND(AVG(velocity_volatility), 4) as fleet_velocity_vol
FROM basin_components;


-- ============================================================================
-- SECTION 2: COMPUTE BASIN STABILITY SCORE
-- ============================================================================

.print ''
.print '=== SECTION 2: Basin Stability Score ==='

-- Normalize each component to 0-1 scale relative to fleet
CREATE OR REPLACE TABLE basin_stability AS
WITH fleet_stats AS (
    SELECT
        AVG(mean_coherence) as fleet_coh,
        STDDEV(mean_coherence) as fleet_coh_std,
        AVG(mean_velocity) as fleet_vel,
        STDDEV(mean_velocity) as fleet_vel_std,
        AVG(coherence_volatility) as fleet_coh_vol,
        STDDEV(coherence_volatility) as fleet_coh_vol_std,
        AVG(velocity_volatility) as fleet_vel_vol,
        STDDEV(velocity_volatility) as fleet_vel_vol_std
    FROM basin_components
),
normalized AS (
    SELECT
        b.entity_id,
        b.mean_coherence,
        b.mean_velocity,
        b.coherence_volatility,
        b.velocity_volatility,
        b.n_observations,

        -- Normalized scores (z-score, then sigmoid to 0-1)
        -- Higher coherence = more stable
        1.0 / (1.0 + EXP(-(b.mean_coherence - f.fleet_coh) / NULLIF(f.fleet_coh_std, 0))) as coherence_score,

        -- Lower velocity = more stable (invert)
        1.0 / (1.0 + EXP((b.mean_velocity - f.fleet_vel) / NULLIF(f.fleet_vel_std, 0))) as velocity_score,

        -- Lower coherence volatility = more stable (invert)
        1.0 / (1.0 + EXP((b.coherence_volatility - f.fleet_coh_vol) / NULLIF(f.fleet_coh_vol_std, 0))) as coherence_stability_score,

        -- Lower velocity volatility = more stable (invert)
        1.0 / (1.0 + EXP((b.velocity_volatility - f.fleet_vel_vol) / NULLIF(f.fleet_vel_vol_std, 0))) as velocity_stability_score

    FROM basin_components b
    CROSS JOIN fleet_stats f
)
SELECT
    entity_id,
    n_observations,
    mean_coherence,
    mean_velocity,
    coherence_volatility,
    velocity_volatility,

    -- Component scores
    ROUND(coherence_score, 3) as coherence_score,
    ROUND(velocity_score, 3) as velocity_score,
    ROUND(coherence_stability_score, 3) as coherence_stability_score,
    ROUND(velocity_stability_score, 3) as velocity_stability_score,

    -- BASIN STABILITY SCORE: weighted combination
    -- Weights: coherence (30%), velocity (30%), coherence_stability (20%), velocity_stability (20%)
    ROUND(
        coherence_score * 0.30 +
        velocity_score * 0.30 +
        coherence_stability_score * 0.20 +
        velocity_stability_score * 0.20
    , 3) as basin_stability_score,

    -- Classification
    CASE
        WHEN coherence_score * 0.30 + velocity_score * 0.30 +
             coherence_stability_score * 0.20 + velocity_stability_score * 0.20 > 0.7
        THEN 'DEEP_BASIN'
        WHEN coherence_score * 0.30 + velocity_score * 0.30 +
             coherence_stability_score * 0.20 + velocity_stability_score * 0.20 > 0.5
        THEN 'MODERATE_BASIN'
        WHEN coherence_score * 0.30 + velocity_score * 0.30 +
             coherence_stability_score * 0.20 + velocity_stability_score * 0.20 > 0.3
        THEN 'SHALLOW_BASIN'
        ELSE 'UNSTABLE'
    END as basin_class

FROM normalized;

SELECT
    basin_class,
    COUNT(*) as n_entities,
    ROUND(AVG(basin_stability_score), 3) as avg_score,
    ROUND(AVG(mean_coherence), 3) as avg_coherence,
    ROUND(AVG(mean_velocity), 3) as avg_velocity
FROM basin_stability
GROUP BY basin_class
ORDER BY avg_score DESC;


-- ============================================================================
-- SECTION 3: BASIN STABILITY RANKING
-- ============================================================================

.print ''
.print '=== SECTION 3: Stability Ranking ==='

SELECT
    entity_id,
    basin_class,
    basin_stability_score,
    coherence_score,
    velocity_score,
    ROUND(mean_coherence, 3) as coherence,
    ROUND(mean_velocity, 3) as velocity
FROM basin_stability
ORDER BY basin_stability_score ASC
LIMIT 15;


-- ============================================================================
-- SECTION 4: CREATE VIEWS
-- ============================================================================

CREATE OR REPLACE VIEW v_basin_stability AS
SELECT * FROM basin_stability;

CREATE OR REPLACE VIEW v_basin_alerts AS
SELECT
    entity_id,
    CASE basin_class
        WHEN 'UNSTABLE' THEN 'CRITICAL'
        WHEN 'SHALLOW_BASIN' THEN 'WARNING'
        WHEN 'MODERATE_BASIN' THEN 'WATCH'
        ELSE 'NORMAL'
    END as alert_level,
    basin_class || ': score=' || basin_stability_score as alert_message,
    1.0 - basin_stability_score as severity_score
FROM basin_stability
WHERE basin_class IN ('UNSTABLE', 'SHALLOW_BASIN');


.print ''
.print '=== BASIN STABILITY ANALYSIS COMPLETE ==='
.print ''
.print 'Views created:'
.print '  v_basin_stability  - Full basin stability metrics'
.print '  v_basin_alerts     - Entities with shallow/unstable basins'
.print ''
.print 'INTERPRETATION:'
.print '  DEEP_BASIN:     System strongly attracted to operating point'
.print '  MODERATE_BASIN: Normal stability, some perturbation tolerance'
.print '  SHALLOW_BASIN:  Weak attraction, vulnerable to transitions'
.print '  UNSTABLE:       Near or past stability boundary'
.print ''
