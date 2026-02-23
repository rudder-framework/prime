-- ============================================================================
-- PHYSICS COMPATIBILITY VIEW
-- ============================================================================
-- Constructs a `physics` view from cohort_geometry + cohort_vector to serve
-- SQL files that originally depended on the retired physics.parquet.
--
-- Pattern follows reports/23_baseline_deviation.sql lines 30-84.
--
-- Columns provided:
--   cohort, signal_0_center, engine, n_signals,
--   coherence, effective_dim, eigenvalue_entropy, condition_number,
--   energy_proxy, energy_velocity, dissipation_rate,
--   coherence_velocity,
--   state_distance, state_velocity, state_acceleration
-- ============================================================================

CREATE OR REPLACE VIEW physics AS
WITH geo_raw AS (
    SELECT
        cohort,
        signal_0_center,
        engine,
        n_signals,
        CASE WHEN isnan(effective_dim) THEN NULL ELSE effective_dim END AS effective_dim,
        CASE WHEN isnan(eigenvalue_entropy_norm) THEN NULL ELSE eigenvalue_entropy_norm END AS eigenvalue_entropy,
        CASE WHEN isnan(total_variance) THEN NULL ELSE total_variance END AS energy_proxy,
        CASE WHEN isnan(explained_1) THEN NULL ELSE explained_1 END AS coherence,
        CASE WHEN isnan(condition_number) THEN NULL ELSE condition_number END AS condition_number
    FROM cohort_geometry
    WHERE engine = (SELECT MIN(engine) FROM cohort_geometry)
),
geo_with_velocity AS (
    SELECT
        *,
        energy_proxy - LAG(energy_proxy) OVER w AS energy_velocity,
        -(energy_proxy - LAG(energy_proxy) OVER w) AS dissipation_rate,
        coherence - LAG(coherence) OVER w AS coherence_velocity
    FROM geo_raw
    WINDOW w AS (PARTITION BY cohort ORDER BY signal_0_center)
),
sv_raw AS (
    SELECT
        cohort,
        signal_0_center,
        CASE WHEN isnan(mean_distance) THEN NULL ELSE mean_distance END AS state_distance
    FROM cohort_vector
),
sv_with_velocity AS (
    SELECT
        *,
        state_distance - LAG(state_distance) OVER w AS state_velocity
    FROM sv_raw
    WINDOW w AS (PARTITION BY cohort ORDER BY signal_0_center)
),
sv_with_acceleration AS (
    SELECT
        *,
        state_velocity - LAG(state_velocity) OVER w AS state_acceleration
    FROM sv_with_velocity
    WINDOW w AS (PARTITION BY cohort ORDER BY signal_0_center)
)
SELECT
    g.cohort,
    g.signal_0_center,
    g.engine,
    g.n_signals,
    g.coherence,
    g.effective_dim,
    g.eigenvalue_entropy,
    g.condition_number,
    g.energy_proxy,
    g.energy_velocity,
    g.dissipation_rate,
    g.coherence_velocity,
    s.state_distance,
    s.state_velocity,
    s.state_acceleration
FROM geo_with_velocity g
JOIN sv_with_acceleration s
  ON g.cohort = s.cohort
 AND g.signal_0_center = s.signal_0_center;
