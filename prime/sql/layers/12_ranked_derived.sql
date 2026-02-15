-- ============================================================================
-- Rudder SQL: 12_ranked_derived.sql — New Ranked Derived Views
-- ============================================================================
-- Canary Sequence: first-mover signals per cohort
-- Curvature Ranked: second-derivative early warning
-- Brittleness Score: geometry x thermodynamics fragility metric
-- ============================================================================


-- ============================================================================
-- CANARY SEQUENCE
-- ============================================================================
-- Identifies the first-mover signals per cohort — the canaries.
-- Which signal deviated first in each engine? Which signal is most
-- often the canary across the fleet?

CREATE OR REPLACE VIEW v_canary_sequence AS
WITH signal_extremes AS (
    SELECT
        cohort,
        signal_id,
        I,
        ABS(z_score) AS z_magnitude,
        PERCENT_RANK() OVER (
            PARTITION BY cohort, signal_id
            ORDER BY ABS(z_score)
        ) AS signal_percentile
    FROM zscore
    WHERE z_score IS NOT NULL
),
first_deviation AS (
    SELECT
        cohort,
        signal_id,
        MIN(I) AS first_extreme_I
    FROM signal_extremes
    WHERE signal_percentile > 0.95
    GROUP BY cohort, signal_id
)
SELECT
    cohort,
    signal_id,
    first_extreme_I,

    -- Canary rank: which signal deviated first in this engine
    RANK() OVER (
        PARTITION BY cohort
        ORDER BY first_extreme_I ASC
    ) AS canary_rank,

    -- Fleet-wide: how many cohorts did this signal canary for
    COUNT(*) OVER (
        PARTITION BY signal_id
    ) AS times_first_mover,

    -- Fleet-wide: rank signals by how often they're the canary
    RANK() OVER (
        ORDER BY COUNT(*) OVER (PARTITION BY signal_id) DESC
    ) AS fleet_canary_rank

FROM first_deviation
WHERE first_extreme_I IS NOT NULL
ORDER BY cohort, canary_rank;


-- ============================================================================
-- CURVATURE RANKED (second derivative early warning)
-- ============================================================================
-- Ranks by curvature magnitude (d2y = second derivative).
-- High curvature = the signal is bending — potential inflection point.

CREATE OR REPLACE VIEW v_curvature_ranked AS
SELECT
    c.I,
    c.signal_id,
    c.cohort,
    c.d2y,
    ABS(c.d2y) AS curvature_magnitude,

    -- Rank by curvature within each timestep per cohort
    RANK() OVER (
        PARTITION BY c.cohort, c.I
        ORDER BY ABS(c.d2y) DESC
    ) AS curvature_rank,

    -- Rank within signal history (how unusual is this curvature)
    PERCENT_RANK() OVER (
        PARTITION BY c.cohort, c.signal_id
        ORDER BY ABS(c.d2y)
    ) AS curvature_percentile,

    -- Fleet-wide rank at this timestep
    RANK() OVER (
        PARTITION BY c.I
        ORDER BY ABS(c.d2y) DESC
    ) AS fleet_curvature_rank,

    -- Sign of curvature (bending toward or away from failure)
    SIGN(c.d2y) AS curvature_direction

FROM v_d2y c
WHERE c.d2y IS NOT NULL;


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
        I,
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
        I,
        engine,
        effective_temperature
    FROM v_cohort_thermodynamics
    WHERE effective_temperature IS NOT NULL
      AND effective_temperature > 0
)
SELECT
    g.cohort,
    g.I,
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
    AND g.I = t.I
    AND g.engine = t.engine;
