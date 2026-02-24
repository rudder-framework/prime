-- ============================================================================
-- REPORT 19: THERMODYNAMICS SUMMARY
-- ============================================================================
-- Statistical mechanics analogy applied to the feature space:
--   Entropy     — disorder in the eigenvalue distribution
--   Energy      — total variance (sum of eigenvalues)
--   Temperature — energy per effective degree of freedom
--   Free energy — energy - T*S (available for organized motion)
--
-- Sources: thermodynamics, geometry_dynamics, state_geometry
-- ============================================================================


-- ============================================================================
-- SECTION 1: COHORT THERMODYNAMIC STATE
-- Global thermodynamic summary per cohort
-- ============================================================================

SELECT
    cohort,
    ROUND(entropy, 4) AS entropy,
    ROUND(energy, 4) AS energy,
    ROUND(temperature, 4) AS temperature,
    ROUND(free_energy, 4) AS free_energy,
    ROUND(energy_std, 4) AS energy_std,
    n_samples,
    CASE
        WHEN temperature < 0.01 THEN 'FROZEN'
        WHEN temperature < 0.1 THEN 'COLD'
        WHEN temperature < 1.0 THEN 'WARM'
        ELSE 'HOT'
    END AS thermal_state,
    CASE
        WHEN free_energy / NULLIF(energy, 0) > 0.9 THEN 'HIGH_ORDER'
        WHEN free_energy / NULLIF(energy, 0) > 0.5 THEN 'MODERATE_ORDER'
        ELSE 'DISORDERED'
    END AS order_level
FROM thermodynamics
ORDER BY temperature DESC;


-- ============================================================================
-- SECTION 2: ENERGY TRAJECTORY
-- How total variance (energy) evolves over time
-- ============================================================================

WITH energy_trajectory AS (
    SELECT
        cohort,
        signal_0_center,
        total_variance AS energy,
        variance_velocity AS energy_velocity,
        effective_dim,
        effective_dim_velocity,
        total_variance / NULLIF(effective_dim, 0) AS temperature_instantaneous
    FROM geometry_dynamics
    WHERE total_variance IS NOT NULL
)
SELECT
    cohort,
    ROUND(AVG(energy), 4) AS avg_energy,
    ROUND(STDDEV_POP(energy), 4) AS energy_std,
    ROUND(AVG(energy_velocity), 6) AS avg_energy_velocity,
    ROUND(AVG(temperature_instantaneous), 4) AS avg_temperature,
    ROUND(REGR_SLOPE(energy, signal_0_center), 8) AS energy_trend,
    COUNT(*) AS n_windows,
    CASE
        WHEN REGR_SLOPE(energy, signal_0_center) > 0.001 THEN 'HEATING'
        WHEN REGR_SLOPE(energy, signal_0_center) < -0.001 THEN 'COOLING'
        ELSE 'THERMAL_EQUILIBRIUM'
    END AS thermal_trend
FROM energy_trajectory
GROUP BY cohort
ORDER BY ABS(REGR_SLOPE(energy, signal_0_center)) DESC;


-- ============================================================================
-- SECTION 3: ENTROPY EVOLUTION
-- Entropy from eigenvalue distribution across windows
-- ============================================================================

SELECT
    cohort,
    ROUND(AVG(eigenvalue_entropy_normalized), 4) AS avg_entropy_norm,
    ROUND(MIN(eigenvalue_entropy_normalized), 4) AS min_entropy,
    ROUND(MAX(eigenvalue_entropy_normalized), 4) AS max_entropy,
    ROUND(STDDEV_POP(eigenvalue_entropy_normalized), 4) AS entropy_std,
    ROUND(REGR_SLOPE(eigenvalue_entropy_normalized, signal_0_center), 8) AS entropy_trend,
    COUNT(*) AS n_windows,
    CASE
        WHEN REGR_SLOPE(eigenvalue_entropy_normalized, signal_0_center) > 0.0001
            THEN 'INCREASING_DISORDER'
        WHEN REGR_SLOPE(eigenvalue_entropy_normalized, signal_0_center) < -0.0001
            THEN 'INCREASING_ORDER'
        ELSE 'ENTROPY_STABLE'
    END AS entropy_trend_label
FROM state_geometry
WHERE eigenvalue_entropy_normalized IS NOT NULL
  AND NOT isnan(eigenvalue_entropy_normalized)
GROUP BY cohort
ORDER BY entropy_trend DESC;


-- ============================================================================
-- SECTION 4: FREE ENERGY AND PHASE TRANSITIONS
-- Detect phase transitions via discontinuities in effective_dim trajectory
-- ============================================================================

WITH dim_changes AS (
    SELECT
        cohort,
        signal_0_center,
        effective_dim,
        effective_dim_velocity,
        effective_dim_acceleration,
        effective_dim_jerk,
        effective_dim_curvature
    FROM geometry_dynamics
    WHERE effective_dim_velocity IS NOT NULL
),
phase_events AS (
    SELECT
        cohort,
        signal_0_center,
        effective_dim,
        effective_dim_velocity,
        effective_dim_acceleration,
        CASE
            WHEN ABS(effective_dim_acceleration) > 3 * STDDEV_POP(effective_dim_acceleration) OVER (PARTITION BY cohort)
            THEN 'PHASE_TRANSITION_CANDIDATE'
            ELSE NULL
        END AS phase_event
    FROM dim_changes
)
SELECT
    cohort,
    signal_0_center,
    ROUND(effective_dim, 3) AS effective_dim,
    ROUND(effective_dim_velocity, 4) AS dim_velocity,
    ROUND(effective_dim_acceleration, 4) AS dim_acceleration
FROM phase_events
WHERE phase_event IS NOT NULL
ORDER BY cohort, signal_0_center
LIMIT 30;
