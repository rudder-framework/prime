-- ============================================================================
-- GEOMETRY RANKED (replaces gated geometry_status)
-- Deduplicated: one row per cohort per window
-- ============================================================================

CREATE OR REPLACE VIEW v_geometry_ranked AS
WITH deduped AS (
    SELECT
        g.cohort,
        g.signal_0_center,
        g.effective_dim,
        g.effective_dim_velocity,
        g.effective_dim_acceleration,
        ABS(g.effective_dim_velocity) AS velocity_magnitude,
        g.eigenvalue_1,
        g.eigenvalue_1_velocity,
        g.eigenvalue_1_acceleration,
        g.condition_number,
        g.condition_number_velocity,
        g.condition_number_acceleration,
        ROW_NUMBER() OVER (PARTITION BY g.cohort, g.signal_0_center ORDER BY g.condition_number DESC NULLS LAST) AS rn
    FROM geometry_dynamics g
    WHERE g.effective_dim IS NOT NULL
      AND NOT isnan(g.effective_dim)
)
SELECT
    cohort,
    signal_0_center,
    effective_dim,
    effective_dim_velocity,
    effective_dim_acceleration,
    velocity_magnitude,
    eigenvalue_1,
    eigenvalue_1_velocity,
    eigenvalue_1_acceleration,
    condition_number,
    condition_number_velocity,
    condition_number_acceleration,

    RANK() OVER (
        PARTITION BY cohort
        ORDER BY velocity_magnitude DESC NULLS LAST
    ) AS velocity_rank,

    PERCENT_RANK() OVER (
        PARTITION BY cohort
        ORDER BY velocity_magnitude
    ) AS velocity_percentile,

    RANK() OVER (
        PARTITION BY signal_0_center
        ORDER BY velocity_magnitude DESC NULLS LAST
    ) AS fleet_velocity_rank,

    -- Eigenvalue_1 velocity ranking
    RANK() OVER (
        PARTITION BY cohort
        ORDER BY ABS(eigenvalue_1_velocity) DESC NULLS LAST
    ) AS eig1_velocity_rank,

    RANK() OVER (
        ORDER BY ABS(eigenvalue_1_velocity) DESC NULLS LAST
    ) AS eig1_fleet_velocity_rank

FROM deduped
WHERE rn = 1;
