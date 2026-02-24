-- ============================================================================
-- 15_geometry_ranked.sql
-- ============================================================================
-- Geometry ranking report
-- Depends on: v_geometry_ranked (layer 15)
-- ============================================================================

.print ''
.print '============================================================================'
.print '                     GEOMETRY RANKED ANALYSIS                              '
.print '============================================================================'

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
