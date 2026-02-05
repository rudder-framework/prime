-- ============================================================================
-- ORTHON SQL: 31_regime_transitions.sql
-- ============================================================================
-- REGIME TRANSITIONS: Detect stability changes over time
--
-- Identifies when systems transition between dynamical regimes:
--   - DESTABILIZING: Lyapunov trending positive
--   - STABILIZING: Lyapunov trending negative
--   - REGIME_SHIFT: Sudden change in stability
--
-- Requires: physics.parquet with temporal data
-- ============================================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║                    REGIME TRANSITION DETECTION                              ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'

-- ============================================================================
-- SECTION 1: TEMPORAL STABILITY METRICS
-- ============================================================================

.print ''
.print '=== SECTION 1: Temporal Stability Analysis ==='

-- Use coherence velocity and state acceleration as stability proxies
CREATE OR REPLACE TABLE regime_dynamics AS
WITH windowed AS (
    SELECT
        entity_id,
        I,
        coherence,
        coherence_velocity,
        state_velocity,
        state_acceleration,
        effective_dim,
        dissipation_rate,

        -- Rolling averages for trend detection
        AVG(coherence_velocity) OVER w20 as coherence_vel_ma,
        AVG(state_acceleration) OVER w20 as state_accel_ma,

        -- Rolling volatility
        STDDEV(coherence) OVER w20 as coherence_volatility,
        STDDEV(state_velocity) OVER w20 as velocity_volatility,

        -- Lifecycle position
        ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY I) as obs_num,
        COUNT(*) OVER (PARTITION BY entity_id) as total_obs

    FROM read_parquet('{prism_output}/physics.parquet')
    WINDOW w20 AS (PARTITION BY entity_id ORDER BY I ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
)
SELECT
    entity_id,
    I,
    obs_num,
    total_obs,
    obs_num * 100.0 / total_obs as pct_life,
    coherence,
    coherence_velocity,
    state_velocity,
    state_acceleration,
    coherence_vel_ma,
    state_accel_ma,
    coherence_volatility,
    velocity_volatility,

    -- Stability indicators
    CASE
        WHEN coherence_vel_ma < -0.01 AND state_accel_ma > 0.01 THEN 'DESTABILIZING'
        WHEN coherence_vel_ma > 0.01 AND state_accel_ma < -0.01 THEN 'STABILIZING'
        WHEN ABS(coherence_vel_ma) < 0.005 AND ABS(state_accel_ma) < 0.005 THEN 'STABLE'
        ELSE 'TRANSIENT'
    END as regime_state,

    -- Volatility spike detection
    CASE
        WHEN coherence_volatility > 0.1 OR velocity_volatility > 0.5 THEN TRUE
        ELSE FALSE
    END as high_volatility

FROM windowed
WHERE obs_num >= 20;  -- Need window to compute

SELECT
    entity_id,
    regime_state,
    COUNT(*) as n_observations,
    ROUND(MIN(pct_life), 1) as start_pct,
    ROUND(MAX(pct_life), 1) as end_pct
FROM regime_dynamics
GROUP BY entity_id, regime_state
ORDER BY entity_id, start_pct;


-- ============================================================================
-- SECTION 2: REGIME TRANSITION DETECTION
-- ============================================================================

.print ''
.print '=== SECTION 2: Regime Transitions ==='

CREATE OR REPLACE TABLE regime_transitions AS
WITH regime_changes AS (
    SELECT
        entity_id,
        I,
        obs_num,
        pct_life,
        regime_state,
        LAG(regime_state) OVER (PARTITION BY entity_id ORDER BY I) as prev_regime,
        coherence,
        state_velocity
    FROM regime_dynamics
)
SELECT
    entity_id,
    I as transition_I,
    obs_num as transition_obs,
    pct_life as transition_pct_life,
    prev_regime as from_regime,
    regime_state as to_regime,
    coherence as coherence_at_transition,
    state_velocity as velocity_at_transition,

    -- Transition type
    CASE
        WHEN prev_regime = 'STABLE' AND regime_state = 'DESTABILIZING' THEN 'ONSET_DEGRADATION'
        WHEN prev_regime = 'STABLE' AND regime_state = 'TRANSIENT' THEN 'PERTURBATION'
        WHEN prev_regime = 'DESTABILIZING' AND regime_state = 'STABLE' THEN 'RECOVERY'
        WHEN prev_regime = 'TRANSIENT' AND regime_state = 'DESTABILIZING' THEN 'CONFIRMED_DEGRADATION'
        ELSE 'OTHER'
    END as transition_type

FROM regime_changes
WHERE regime_state != prev_regime
  AND prev_regime IS NOT NULL;

SELECT
    transition_type,
    COUNT(*) as n_transitions,
    COUNT(DISTINCT entity_id) as n_entities,
    ROUND(AVG(transition_pct_life), 1) as avg_pct_life,
    ROUND(AVG(coherence_at_transition), 3) as avg_coherence,
    ROUND(AVG(velocity_at_transition), 3) as avg_velocity
FROM regime_transitions
GROUP BY transition_type
ORDER BY n_transitions DESC;


-- ============================================================================
-- SECTION 3: FIRST DEGRADATION ONSET
-- ============================================================================

.print ''
.print '=== SECTION 3: First Degradation Onset ==='

CREATE OR REPLACE TABLE first_degradation AS
SELECT
    entity_id,
    MIN(transition_I) as first_degrad_I,
    MIN(transition_obs) as first_degrad_obs,
    MIN(transition_pct_life) as first_degrad_pct_life
FROM regime_transitions
WHERE transition_type IN ('ONSET_DEGRADATION', 'CONFIRMED_DEGRADATION')
GROUP BY entity_id;

SELECT
    CASE
        WHEN first_degrad_pct_life < 25 THEN 'EARLY (< 25%)'
        WHEN first_degrad_pct_life < 50 THEN 'MID-EARLY (25-50%)'
        WHEN first_degrad_pct_life < 75 THEN 'MID-LATE (50-75%)'
        ELSE 'LATE (> 75%)'
    END as degradation_onset_group,
    COUNT(*) as n_entities,
    ROUND(AVG(first_degrad_pct_life), 1) as avg_onset_pct
FROM first_degradation
GROUP BY 1
ORDER BY avg_onset_pct;


-- ============================================================================
-- SECTION 4: ENTITY REGIME SUMMARY
-- ============================================================================

.print ''
.print '=== SECTION 4: Entity Regime Summary ==='

CREATE OR REPLACE VIEW v_regime_summary AS
SELECT
    r.entity_id,

    -- Time in each regime
    SUM(CASE WHEN regime_state = 'STABLE' THEN 1 ELSE 0 END) * 100.0 /
        COUNT(*) as pct_stable,
    SUM(CASE WHEN regime_state = 'DESTABILIZING' THEN 1 ELSE 0 END) * 100.0 /
        COUNT(*) as pct_destabilizing,
    SUM(CASE WHEN regime_state = 'TRANSIENT' THEN 1 ELSE 0 END) * 100.0 /
        COUNT(*) as pct_transient,

    -- Number of transitions
    (SELECT COUNT(*) FROM regime_transitions t WHERE t.entity_id = r.entity_id) as n_transitions,

    -- First degradation
    f.first_degrad_pct_life,

    -- Current regime (latest observation)
    LAST(regime_state ORDER BY I) as current_regime,

    -- Overall trajectory
    CASE
        WHEN SUM(CASE WHEN regime_state = 'DESTABILIZING' THEN 1 ELSE 0 END) * 100.0 /
             COUNT(*) > 30 THEN 'DEGRADING'
        WHEN SUM(CASE WHEN regime_state = 'STABLE' THEN 1 ELSE 0 END) * 100.0 /
             COUNT(*) > 70 THEN 'HEALTHY'
        ELSE 'MIXED'
    END as overall_trajectory

FROM regime_dynamics r
LEFT JOIN first_degradation f ON r.entity_id = f.entity_id
GROUP BY r.entity_id, f.first_degrad_pct_life;

SELECT
    overall_trajectory,
    COUNT(*) as n_entities,
    ROUND(AVG(pct_stable), 1) as avg_pct_stable,
    ROUND(AVG(pct_destabilizing), 1) as avg_pct_destabilizing,
    ROUND(AVG(first_degrad_pct_life), 1) as avg_first_degrad_pct
FROM v_regime_summary
GROUP BY overall_trajectory;


.print ''
.print '=== REGIME ANALYSIS COMPLETE ==='
.print ''
.print 'Views created:'
.print '  v_regime_summary  - Entity regime breakdown'
.print ''
