-- ============================================================================
-- ORTHON SQL: 17_dynamical_systems_prism.sql
-- ============================================================================
-- Load and interpret PRISM dynamical_systems.parquet output
--
-- This loads PRISM-computed dynamics (not ORTHON-computed from calculus).
-- PRISM computes: regime detection, Lyapunov exponents, attractor dimensions.
-- ORTHON interprets: stability classes, regime transitions, phase portraits.
--
-- IMPORTANT: Lyapunov exponents require ~10,000 observations for reliability.
-- See 00_config.sql for data sufficiency thresholds.
-- ============================================================================

-- Load dynamical systems data
CREATE OR REPLACE TABLE dynamical_systems AS
SELECT * FROM read_parquet('{prism_output}/dynamical_systems.parquet');

-- Count observations per entity for data sufficiency checks
CREATE OR REPLACE VIEW v_dynamics_entity_obs AS
SELECT
    entity_id,
    COUNT(*) AS n_observations,
    MAX(regime_end_idx) AS max_idx
FROM dynamical_systems
GROUP BY entity_id;

-- Basic info
CREATE OR REPLACE VIEW v_dynamics_info AS
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT entity_id) AS n_entities,
    COUNT(DISTINCT signal_id) AS n_signals,
    COUNT(DISTINCT regime_id) AS n_regimes
FROM dynamical_systems;


-- ============================================================================
-- REGIME ANALYSIS
-- ============================================================================
-- NOTE: Lyapunov interpretation requires sufficient data (default: 10,000 obs).
-- Regimes from entities with insufficient data get lyapunov_reliable = FALSE.
-- ============================================================================

-- Configurable threshold (adjust as needed)
-- To change: UPDATE config_lyapunov SET min_observations = 5000;
CREATE OR REPLACE TABLE config_lyapunov AS
SELECT 10000 AS min_observations;  -- Wolf et al. (1985) recommendation

CREATE OR REPLACE VIEW v_dynamics_regimes AS
SELECT
    d.entity_id,
    d.signal_id,
    d.regime_id,
    d.regime_start_idx,
    d.regime_end_idx,
    d.regime_end_idx - d.regime_start_idx AS regime_duration,

    -- Regime statistics (if available)
    d.regime_mean,
    d.regime_std,
    d.regime_min,
    d.regime_max,

    -- Data sufficiency
    e.n_observations,
    e.n_observations >= c.min_observations AS lyapunov_reliable,

    -- Stability metrics (raw value always available)
    d.lyapunov,

    -- Stability class (only meaningful when lyapunov_reliable = TRUE)
    CASE
        WHEN e.n_observations < c.min_observations THEN 'insufficient_data'
        WHEN d.lyapunov > 0.1 THEN 'chaotic'
        WHEN d.lyapunov > 0 THEN 'weakly_chaotic'
        WHEN d.lyapunov < -0.1 THEN 'strongly_stable'
        WHEN d.lyapunov < 0 THEN 'stable'
        ELSE 'edge_of_chaos'
    END AS stability_class,

    -- Attractor info
    d.attractor_dimension

FROM dynamical_systems d
LEFT JOIN v_dynamics_entity_obs e ON d.entity_id = e.entity_id
CROSS JOIN config_lyapunov c;


-- Regime transitions
CREATE OR REPLACE VIEW v_dynamics_transitions AS
SELECT
    entity_id,
    signal_id,
    regime_id AS from_regime,
    LEAD(regime_id) OVER w AS to_regime,
    regime_end_idx AS transition_point,
    regime_mean AS from_mean,
    LEAD(regime_mean) OVER w AS to_mean,
    LEAD(regime_mean) OVER w - regime_mean AS mean_jump,
    lyapunov AS from_lyapunov,
    LEAD(lyapunov) OVER w AS to_lyapunov,

    -- Transition type
    CASE
        WHEN LEAD(lyapunov) OVER w > lyapunov + 0.1 THEN 'destabilizing'
        WHEN LEAD(lyapunov) OVER w < lyapunov - 0.1 THEN 'stabilizing'
        WHEN ABS(LEAD(regime_mean) OVER w - regime_mean) > regime_std THEN 'level_shift'
        ELSE 'minor_change'
    END AS transition_type

FROM dynamical_systems
WINDOW w AS (PARTITION BY entity_id, signal_id ORDER BY regime_start_idx)
HAVING to_regime IS NOT NULL;


-- Entity-level dynamics summary
-- NOTE: Lyapunov metrics only included when data is sufficient
CREATE OR REPLACE VIEW v_dynamics_entity_summary AS
SELECT
    entity_id,

    -- Regime counts
    COUNT(DISTINCT signal_id) AS n_signals,
    COUNT(DISTINCT regime_id) AS n_distinct_regimes,
    AVG(regime_duration) AS avg_regime_duration,

    -- Data sufficiency
    MAX(n_observations) AS n_observations,
    MAX(lyapunov_reliable::INT)::BOOL AS lyapunov_reliable,

    -- Stability summary (NULL if insufficient data)
    CASE WHEN MAX(lyapunov_reliable::INT) = 1 THEN AVG(lyapunov) ELSE NULL END AS mean_lyapunov,
    CASE WHEN MAX(lyapunov_reliable::INT) = 1 THEN MAX(lyapunov) ELSE NULL END AS max_lyapunov,
    CASE WHEN MAX(lyapunov_reliable::INT) = 1 THEN MIN(lyapunov) ELSE NULL END AS min_lyapunov,

    -- Stability distribution (only count if reliable)
    SUM(CASE WHEN lyapunov_reliable AND lyapunov > 0.1 THEN 1 ELSE 0 END) AS n_chaotic_regimes,
    SUM(CASE WHEN lyapunov_reliable AND lyapunov < -0.1 THEN 1 ELSE 0 END) AS n_stable_regimes,

    -- Overall stability
    CASE
        WHEN MAX(lyapunov_reliable::INT) = 0 THEN 'insufficient_data'
        WHEN MAX(lyapunov) > 0.1 THEN 'has_chaos'
        WHEN AVG(lyapunov) > 0 THEN 'weakly_unstable'
        WHEN AVG(lyapunov) < -0.1 THEN 'strongly_stable'
        ELSE 'marginally_stable'
    END AS overall_stability,

    -- Attractor complexity
    AVG(attractor_dimension) AS mean_attractor_dim,
    MAX(attractor_dimension) AS max_attractor_dim

FROM v_dynamics_regimes
GROUP BY entity_id;


-- Signal-level dynamics summary
CREATE OR REPLACE VIEW v_dynamics_signal_summary AS
SELECT
    entity_id,
    signal_id,

    -- Regime summary
    COUNT(DISTINCT regime_id) AS n_regimes,
    SUM(regime_duration) AS total_duration,
    AVG(regime_duration) AS avg_regime_duration,

    -- Data sufficiency
    MAX(n_observations) AS n_observations,
    MAX(lyapunov_reliable::INT)::BOOL AS lyapunov_reliable,

    -- Stability (NULL if insufficient data)
    CASE WHEN MAX(lyapunov_reliable::INT) = 1 THEN AVG(lyapunov) ELSE NULL END AS mean_lyapunov,
    CASE
        WHEN MAX(lyapunov_reliable::INT) = 0 THEN 'insufficient_data'
        WHEN AVG(lyapunov) > 0.1 THEN 'chaotic'
        WHEN AVG(lyapunov) > 0 THEN 'weakly_unstable'
        WHEN AVG(lyapunov) < -0.1 THEN 'strongly_stable'
        ELSE 'marginally_stable'
    END AS stability_class,

    -- Attractor
    AVG(attractor_dimension) AS mean_attractor_dim,

    -- Regime variability
    STDDEV(regime_mean) AS regime_mean_variability

FROM v_dynamics_regimes
GROUP BY entity_id, signal_id;


-- ============================================================================
-- PHASE SPACE ANALYSIS
-- ============================================================================

-- Join with physics for phase space context
CREATE OR REPLACE VIEW v_dynamics_phase_space AS
SELECT
    d.entity_id,
    d.signal_id,
    d.regime_id,
    d.lyapunov,
    d.stability_class,
    d.attractor_dimension,

    -- From physics layer
    p.state_distance,
    p.state_velocity,
    p.coherence,
    p.effective_dim,

    -- Phase space state
    CASE
        WHEN d.lyapunov > 0 AND p.state_velocity > 0.01 THEN 'chaotic_diverging'
        WHEN d.lyapunov > 0 AND p.state_velocity < -0.01 THEN 'chaotic_converging'
        WHEN d.lyapunov < 0 AND p.state_velocity > 0.01 THEN 'stable_but_drifting'
        WHEN d.lyapunov < 0 AND p.state_velocity < -0.01 THEN 'stable_returning'
        ELSE 'equilibrium'
    END AS phase_space_state

FROM v_dynamics_regimes d
LEFT JOIN physics p ON d.entity_id = p.entity_id
WHERE p.I BETWEEN d.regime_start_idx AND d.regime_end_idx;


-- ============================================================================
-- SYSTEM-LEVEL REGIME DETECTION
-- ============================================================================
-- When multiple signals enter chaotic regimes together
-- NOTE: Only includes entities with sufficient data for reliable Lyapunov

CREATE OR REPLACE VIEW v_dynamics_system_chaos AS
SELECT
    entity_id,
    regime_start_idx AS chaos_start,
    COUNT(DISTINCT signal_id) AS n_signals_chaotic,
    AVG(lyapunov) AS mean_lyapunov,
    MAX(lyapunov) AS max_lyapunov

FROM v_dynamics_regimes
WHERE lyapunov > 0.1
  AND lyapunov_reliable = TRUE  -- Only include reliable Lyapunov estimates
GROUP BY entity_id, regime_start_idx
HAVING COUNT(DISTINCT signal_id) > 1
ORDER BY entity_id, regime_start_idx;


-- Fleet dynamics summary
CREATE OR REPLACE VIEW v_dynamics_fleet_summary AS
SELECT
    COUNT(DISTINCT entity_id) AS n_entities,

    -- Data sufficiency
    SUM(CASE WHEN lyapunov_reliable THEN 1 ELSE 0 END) AS n_lyapunov_reliable,
    SUM(CASE WHEN NOT lyapunov_reliable THEN 1 ELSE 0 END) AS n_lyapunov_insufficient,

    -- Stability distribution (only reliable entities)
    SUM(CASE WHEN overall_stability = 'has_chaos' THEN 1 ELSE 0 END) AS n_with_chaos,
    SUM(CASE WHEN overall_stability = 'weakly_unstable' THEN 1 ELSE 0 END) AS n_weakly_unstable,
    SUM(CASE WHEN overall_stability = 'marginally_stable' THEN 1 ELSE 0 END) AS n_marginally_stable,
    SUM(CASE WHEN overall_stability = 'strongly_stable' THEN 1 ELSE 0 END) AS n_strongly_stable,
    SUM(CASE WHEN overall_stability = 'insufficient_data' THEN 1 ELSE 0 END) AS n_insufficient_data,

    -- Lyapunov stats (only from reliable entities)
    AVG(mean_lyapunov) AS fleet_mean_lyapunov,  -- NULLs excluded automatically
    MAX(max_lyapunov) AS fleet_max_lyapunov,

    -- Complexity
    AVG(mean_attractor_dim) AS fleet_mean_attractor_dim

FROM v_dynamics_entity_summary;


-- Verify
SELECT * FROM v_dynamics_info;
