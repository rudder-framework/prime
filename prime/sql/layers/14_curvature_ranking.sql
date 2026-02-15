-- ============================================================================
-- CURVATURE RANKING (Second Derivative)
-- Bending of trajectory — early warning before velocity changes
-- ============================================================================

CREATE OR REPLACE VIEW v_geometry_curvature AS
SELECT
    cohort,
    I,
    effective_dim,
    effective_dim_velocity,
    effective_dim_acceleration,
    ABS(effective_dim_acceleration) AS curvature_magnitude,

    -- Rank by curvature within each cohort at each window
    RANK() OVER (
        PARTITION BY cohort
        ORDER BY ABS(effective_dim_acceleration) DESC NULLS LAST
    ) AS curvature_rank,

    -- Percentile within cohort history
    PERCENT_RANK() OVER (
        PARTITION BY cohort
        ORDER BY ABS(effective_dim_acceleration)
    ) AS curvature_percentile,

    -- Fleet-wide rank at each window
    RANK() OVER (
        PARTITION BY I
        ORDER BY ABS(effective_dim_acceleration) DESC NULLS LAST
    ) AS fleet_curvature_rank,

    -- Direction: bending toward collapse or away
    SIGN(effective_dim_acceleration) AS curvature_direction

FROM geometry_dynamics
WHERE effective_dim_acceleration IS NOT NULL
  AND NOT isnan(effective_dim_acceleration);

-- Biggest curvature events across fleet
SELECT
    cohort,
    I,
    ROUND(effective_dim, 3) AS eff_dim,
    ROUND(effective_dim_velocity, 4) AS velocity,
    ROUND(effective_dim_acceleration, 4) AS acceleration,
    ROUND(curvature_magnitude, 4) AS magnitude,
    curvature_rank,
    curvature_direction
FROM v_geometry_curvature
WHERE curvature_rank <= 3
ORDER BY curvature_magnitude DESC
LIMIT 30;
