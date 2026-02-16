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
        ABS(g.effective_dim_velocity) AS velocity_magnitude,
        s.condition_number,
        ROW_NUMBER() OVER (PARTITION BY g.cohort, g.signal_0_center ORDER BY s.condition_number DESC) AS rn
    FROM geometry_dynamics g
    LEFT JOIN state_geometry s ON g.cohort = s.cohort AND g.signal_0_center = s.signal_0_center
    WHERE g.effective_dim IS NOT NULL
      AND NOT isnan(g.effective_dim)
)
SELECT
    cohort,
    signal_0_center,
    effective_dim,
    effective_dim_velocity,
    velocity_magnitude,
    condition_number,

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
    ) AS fleet_velocity_rank

FROM deduped
WHERE rn = 1;

-- Fastest changing geometry across fleet
SELECT
    cohort,
    signal_0_center,
    ROUND(effective_dim, 3) AS eff_dim,
    ROUND(effective_dim_velocity, 4) AS velocity,
    ROUND(condition_number, 2) AS cond_num,
    velocity_rank,
    fleet_velocity_rank
FROM v_geometry_ranked
WHERE fleet_velocity_rank <= 5
ORDER BY velocity_magnitude DESC
LIMIT 30;
