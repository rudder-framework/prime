-- ============================================================================
-- 31_regime_transitions.sql — Ranked Views
-- ============================================================================
-- REGIME TRANSITIONS: Rank stability changes over time
--
-- No categorical gates. Rank by magnitude of change.
-- The analyst queries WHERE transition_violence_rank = 1.
--
-- Requires: physics.parquet with temporal data
-- ============================================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║                    REGIME TRANSITION DETECTION                              ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'

-- ============================================================================
-- SECTION 1: TEMPORAL STABILITY METRICS (ranked)
-- ============================================================================

.print ''
.print '=== SECTION 1: Temporal Stability Analysis ==='

CREATE OR REPLACE TABLE regime_dynamics AS
WITH windowed AS (
    SELECT
        cohort,
        signal_0_center,
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
        ROW_NUMBER() OVER (PARTITION BY cohort ORDER BY signal_0_center) as obs_num,
        COUNT(*) OVER (PARTITION BY cohort) as total_obs

    FROM read_parquet('{manifold_output}/physics.parquet')
    WINDOW w20 AS (PARTITION BY cohort ORDER BY signal_0_center ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
)
SELECT
    cohort,
    signal_0_center,
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

    -- Destabilization magnitude (positive = destabilizing)
    (-coherence_vel_ma + state_accel_ma) AS destabilization_signal,
    ABS(coherence_vel_ma) + ABS(state_accel_ma) AS regime_activity,

    -- Rank by destabilization signal within each cohort
    RANK() OVER (
        PARTITION BY cohort
        ORDER BY (-coherence_vel_ma + state_accel_ma) DESC NULLS LAST
    ) AS destabilization_rank,

    -- Rank by regime activity (most dynamic moments)
    RANK() OVER (
        PARTITION BY cohort
        ORDER BY ABS(coherence_vel_ma) + ABS(state_accel_ma) DESC NULLS LAST
    ) AS activity_rank,

    -- Percentile of destabilization within cohort
    PERCENT_RANK() OVER (
        PARTITION BY cohort
        ORDER BY (-coherence_vel_ma + state_accel_ma) NULLS FIRST
    ) AS destabilization_percentile,

    -- Volatility magnitude
    COALESCE(coherence_volatility, 0) + COALESCE(velocity_volatility, 0) AS total_volatility,

    -- Rank by volatility within cohort
    RANK() OVER (
        PARTITION BY cohort
        ORDER BY COALESCE(coherence_volatility, 0) + COALESCE(velocity_volatility, 0) DESC NULLS LAST
    ) AS volatility_rank

FROM windowed
WHERE obs_num >= 20;  -- Need window to compute

SELECT
    cohort,
    COUNT(*) as n_observations,
    ROUND(AVG(destabilization_signal), 4) as avg_destabilization,
    ROUND(MAX(destabilization_signal), 4) as max_destabilization,
    MIN(destabilization_rank) as best_destab_rank
FROM regime_dynamics
GROUP BY cohort
ORDER BY avg_destabilization DESC;


-- ============================================================================
-- SECTION 2: REGIME TRANSITIONS (ranked by magnitude)
-- ============================================================================

.print ''
.print '=== SECTION 2: Regime Transitions ==='

CREATE OR REPLACE TABLE regime_transitions AS
WITH regime_changes AS (
    SELECT
        cohort,
        signal_0_center,
        obs_num,
        pct_life,
        destabilization_signal,
        regime_activity,
        LAG(destabilization_signal) OVER (PARTITION BY cohort ORDER BY signal_0_center) as prev_destab,
        coherence,
        state_velocity
    FROM regime_dynamics
)
SELECT
    cohort,
    signal_0_center as transition_I,
    obs_num as transition_obs,
    pct_life as transition_pct_life,
    coherence as coherence_at_transition,
    state_velocity as velocity_at_transition,
    ABS(state_velocity) AS transition_magnitude,

    -- Change in destabilization signal (how much did the regime shift)
    ABS(destabilization_signal - prev_destab) AS regime_shift_magnitude,

    -- Rank transitions by violence (magnitude of velocity at transition)
    RANK() OVER (
        PARTITION BY cohort
        ORDER BY ABS(state_velocity) DESC
    ) AS transition_violence_rank,

    -- Rank by regime shift magnitude
    RANK() OVER (
        PARTITION BY cohort
        ORDER BY ABS(destabilization_signal - prev_destab) DESC NULLS LAST
    ) AS regime_shift_rank,

    -- Rank by how early in life (earlier = more interesting)
    RANK() OVER (
        ORDER BY pct_life ASC
    ) AS earliest_transition_rank,

    -- Order within each cohort (first, second, third transition)
    ROW_NUMBER() OVER (
        PARTITION BY cohort ORDER BY signal_0_center ASC
    ) AS transition_sequence,

    -- Percentile of transition violence within fleet
    PERCENT_RANK() OVER (
        ORDER BY ABS(state_velocity)
    ) AS violence_percentile

FROM regime_changes
WHERE prev_destab IS NOT NULL
  AND ABS(destabilization_signal - prev_destab) > 0;

SELECT
    COUNT(*) as n_transitions,
    COUNT(DISTINCT cohort) as n_entities,
    ROUND(AVG(transition_pct_life), 1) as avg_pct_life,
    ROUND(AVG(coherence_at_transition), 3) as avg_coherence,
    ROUND(AVG(velocity_at_transition), 3) as avg_velocity
FROM regime_transitions;


-- ============================================================================
-- SECTION 3: FIRST DEGRADATION ONSET (ranked)
-- ============================================================================

.print ''
.print '=== SECTION 3: First Degradation Onset ==='

CREATE OR REPLACE TABLE first_degradation AS
SELECT
    cohort,
    MIN(transition_I) as first_degrad_I,
    MIN(transition_obs) as first_degrad_obs,
    MIN(transition_pct_life) as first_degrad_pct_life,

    -- Rank cohorts by earliest degradation onset
    RANK() OVER (
        ORDER BY MIN(transition_pct_life) ASC
    ) AS earliest_onset_rank

FROM regime_transitions
WHERE transition_violence_rank <= 3  -- Top 3 most violent transitions per cohort
GROUP BY cohort;

SELECT
    cohort,
    ROUND(first_degrad_pct_life, 1) as onset_pct,
    earliest_onset_rank
FROM first_degradation
ORDER BY earliest_onset_rank
LIMIT 20;


-- ============================================================================
-- SECTION 4: ENTITY REGIME SUMMARY (ranked)
-- ============================================================================

.print ''
.print '=== SECTION 4: Entity Regime Summary ==='

CREATE OR REPLACE VIEW v_regime_summary AS
SELECT
    r.cohort,

    -- Activity distribution
    AVG(r.destabilization_signal) AS mean_destabilization,
    MAX(r.destabilization_signal) AS max_destabilization,
    AVG(r.regime_activity) AS mean_activity,

    -- Volatility distribution
    AVG(r.total_volatility) AS mean_volatility,
    MAX(r.total_volatility) AS max_volatility,

    -- Number of significant transitions (top quartile by violence)
    (SELECT COUNT(*) FROM regime_transitions t
     WHERE t.cohort = r.cohort AND t.violence_percentile > 0.75) AS n_significant_transitions,

    -- First degradation
    f.first_degrad_pct_life,
    f.earliest_onset_rank,

    -- Rank cohorts by overall instability
    RANK() OVER (
        ORDER BY AVG(r.destabilization_signal) DESC
    ) AS fleet_instability_rank,

    -- Rank cohorts by volatility
    RANK() OVER (
        ORDER BY AVG(r.total_volatility) DESC
    ) AS fleet_volatility_rank

FROM regime_dynamics r
LEFT JOIN first_degradation f ON r.cohort = f.cohort
GROUP BY r.cohort, f.first_degrad_pct_life, f.earliest_onset_rank;

SELECT
    cohort,
    ROUND(mean_destabilization, 4) as mean_destab,
    ROUND(mean_volatility, 4) as mean_vol,
    fleet_instability_rank,
    fleet_volatility_rank,
    ROUND(first_degrad_pct_life, 1) as onset_pct
FROM v_regime_summary
ORDER BY fleet_instability_rank
LIMIT 20;


.print ''
.print '=== REGIME ANALYSIS COMPLETE ==='
.print ''
.print 'Views created:'
.print '  v_regime_summary  - Entity regime breakdown (ranked)'
.print ''
