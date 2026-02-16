-- ============================================================================
-- 12_ranked_derived.sql — New Ranked Derived Views
-- ============================================================================
-- Canary Sequence: first-mover signals per cohort
-- Curvature Ranked: second-derivative early warning
-- Brittleness Score: geometry x thermodynamics fragility metric
--
-- Source tables:
--   signal_vector   — per-signal statistical features (for percentile canary)
--   ftle_rolling    — per-signal FTLE with acceleration (for curvature)
--   state_geometry  — eigendecomposition per window
--   v_cohort_thermodynamics — from 05_manifold_derived.sql
-- ============================================================================


-- ============================================================================
-- CANARY SEQUENCE
-- ============================================================================
-- Identifies the first-mover signals per cohort — the canaries.
-- Which signal deviated first in each engine? Which signal is most
-- often the canary across the fleet?
-- Uses PERCENT_RANK of spectral_entropy (no z-scores).

CREATE OR REPLACE VIEW v_canary_sequence AS
WITH signal_extremes AS (
    SELECT
        cohort,
        signal_id,
        signal_0_center,
        spectral_entropy AS value,
        PERCENT_RANK() OVER (
            PARTITION BY cohort, signal_id
            ORDER BY spectral_entropy
        ) AS signal_percentile
    FROM signal_vector
    WHERE spectral_entropy IS NOT NULL
      AND NOT isnan(spectral_entropy)
),
first_deviation AS (
    SELECT
        cohort,
        signal_id,
        MIN(signal_0_center) AS first_extreme_I
    FROM signal_extremes
    WHERE signal_percentile > 0.95
    GROUP BY cohort, signal_id
),
with_counts AS (
    SELECT
        cohort,
        signal_id,
        first_extreme_I,
        RANK() OVER (
            PARTITION BY cohort
            ORDER BY first_extreme_I ASC
        ) AS canary_rank,
        COUNT(*) OVER (
            PARTITION BY signal_id
        ) AS times_first_mover
    FROM first_deviation
    WHERE first_extreme_I IS NOT NULL
)
SELECT
    cohort,
    signal_id,
    first_extreme_I,
    canary_rank,
    times_first_mover,
    RANK() OVER (
        ORDER BY times_first_mover DESC
    ) AS fleet_canary_rank
FROM with_counts
ORDER BY cohort, canary_rank;


-- ============================================================================
-- CURVATURE RANKED (second derivative early warning)
-- ============================================================================
-- Ranks by FTLE acceleration magnitude (second derivative of Lyapunov).
-- High curvature = dynamics are bending — potential inflection point.
-- Source: ftle_rolling.parquet

CREATE OR REPLACE VIEW v_curvature_ranked AS
SELECT
    signal_0_center,
    signal_id,
    cohort,
    ftle_acceleration AS d2y,
    ABS(ftle_acceleration) AS curvature_magnitude,

    -- Rank by curvature within each timestep per cohort
    RANK() OVER (
        PARTITION BY cohort, signal_0_center
        ORDER BY ABS(ftle_acceleration) DESC NULLS LAST
    ) AS curvature_rank,

    -- Rank within signal history (how unusual is this curvature)
    PERCENT_RANK() OVER (
        PARTITION BY cohort, signal_id
        ORDER BY ABS(ftle_acceleration)
    ) AS curvature_percentile,

    -- Fleet-wide rank at this timestep
    RANK() OVER (
        PARTITION BY signal_0_center
        ORDER BY ABS(ftle_acceleration) DESC NULLS LAST
    ) AS fleet_curvature_rank,

    -- Sign of curvature (bending toward or away from failure)
    SIGN(ftle_acceleration) AS curvature_direction

FROM ftle_rolling
WHERE ftle_acceleration IS NOT NULL
  AND direction = 'forward';


-- ============================================================================
-- BRITTLENESS SCORE (Layer 12)
-- ============================================================================
-- Joins geometry and thermodynamics.
-- Brittleness = high condition number x tight eigenvalue gap x low temperature.
-- A brittle system is rigid, concentrated, and cold — it will shatter, not bend.

CREATE OR REPLACE VIEW v_brittleness AS
WITH geometry_metrics AS (
    SELECT
        cohort,
        signal_0_center,
        engine,
        condition_number,
        eigenvalue_1,
        -- Eigenvalue gap: ratio of first eigenvalue to total variance
        -- (proxy for gap when eigenvalue_2 not always available)
        CASE WHEN total_variance > 0
            THEN eigenvalue_1 / total_variance
            ELSE NULL
        END AS energy_concentration
    FROM state_geometry
    WHERE condition_number IS NOT NULL
),
thermo_metrics AS (
    SELECT
        cohort,
        signal_0_center,
        engine,
        effective_temperature
    FROM v_cohort_thermodynamics
    WHERE effective_temperature IS NOT NULL
      AND effective_temperature > 0
)
SELECT
    g.cohort,
    g.signal_0_center,
    g.engine,
    g.condition_number,
    g.energy_concentration,
    t.effective_temperature,

    -- Brittleness: high condition number x high energy concentration x low temperature
    g.condition_number
        * COALESCE(g.energy_concentration, 1.0)
        * (1.0 / NULLIF(t.effective_temperature, 0))
    AS brittleness_score,

    -- Rank by brittleness within cohort
    RANK() OVER (
        PARTITION BY g.cohort
        ORDER BY g.condition_number
            * COALESCE(g.energy_concentration, 1.0)
            * (1.0 / NULLIF(t.effective_temperature, 0))
            DESC NULLS LAST
    ) AS brittleness_rank,

    -- Fleet-wide brittleness rank
    RANK() OVER (
        ORDER BY g.condition_number
            * COALESCE(g.energy_concentration, 1.0)
            * (1.0 / NULLIF(t.effective_temperature, 0))
            DESC NULLS LAST
    ) AS fleet_brittleness_rank,

    -- Percentile within cohort history
    PERCENT_RANK() OVER (
        PARTITION BY g.cohort
        ORDER BY g.condition_number
            * COALESCE(g.energy_concentration, 1.0)
            * (1.0 / NULLIF(t.effective_temperature, 0))
            NULLS FIRST
    ) AS brittleness_percentile

FROM geometry_metrics g
LEFT JOIN thermo_metrics t
    ON g.cohort = t.cohort
    AND g.signal_0_center = t.signal_0_center
    AND g.engine = t.engine;
