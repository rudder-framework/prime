-- ============================================================================
-- GEOMETRY RANKED (replaces gated geometry_status)
-- ============================================================================

CREATE OR REPLACE VIEW v_geometry_ranked AS
SELECT
    g.cohort,
    g.I,
    g.effective_dim,
    g.effective_dim_velocity,
    ABS(g.effective_dim_velocity) AS velocity_magnitude,
    s.condition_number,

    RANK() OVER (
        PARTITION BY g.cohort
        ORDER BY ABS(g.effective_dim_velocity) DESC NULLS LAST
    ) AS velocity_rank,

    PERCENT_RANK() OVER (
        PARTITION BY g.cohort
        ORDER BY ABS(g.effective_dim_velocity)
    ) AS velocity_percentile,

    RANK() OVER (
        PARTITION BY g.I
        ORDER BY ABS(g.effective_dim_velocity) DESC NULLS LAST
    ) AS fleet_velocity_rank

FROM geometry_dynamics g
LEFT JOIN state_geometry s ON g.cohort = s.cohort AND g.I = s.I
WHERE g.effective_dim IS NOT NULL
  AND NOT isnan(g.effective_dim);

-- Fastest changing geometry across fleet
SELECT
    cohort,
    I,
    ROUND(effective_dim, 3) AS eff_dim,
    ROUND(effective_dim_velocity, 4) AS velocity,
    ROUND(condition_number, 2) AS cond_num,
    velocity_rank,
    fleet_velocity_rank
FROM v_geometry_ranked
WHERE fleet_velocity_rank <= 5
ORDER BY velocity_magnitude DESC
LIMIT 30;
