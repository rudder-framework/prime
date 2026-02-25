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

-- Fastest changing effective_dim across fleet
SELECT
    cohort,
    signal_0_center,
    ROUND(effective_dim, 3) AS eff_dim,
    ROUND(effective_dim_velocity, 4) AS velocity,
    ROUND(effective_dim_acceleration, 4) AS acceleration,
    ROUND(condition_number, 2) AS cond_num,
    ROUND(condition_number_velocity, 4) AS cond_velocity,
    ROUND(condition_number_acceleration, 4) AS cond_accel,
    velocity_rank,
    fleet_velocity_rank
FROM v_geometry_ranked
WHERE fleet_velocity_rank <= 5
ORDER BY velocity_magnitude DESC
LIMIT 30;

-- Eigenvalue_1 velocity + acceleration ranking
SELECT
    cohort,
    signal_0_center,
    ROUND(eigenvalue_1, 4) AS eigenvalue_1,
    ROUND(eigenvalue_1_velocity, 4) AS eig1_velocity,
    ROUND(eigenvalue_1_acceleration, 4) AS eig1_accel,
    eig1_velocity_rank,
    eig1_fleet_velocity_rank
FROM v_geometry_ranked
WHERE eigenvalue_1_velocity IS NOT NULL
  AND (eig1_velocity_rank <= 2 OR eig1_fleet_velocity_rank <= 5)
ORDER BY eig1_fleet_velocity_rank, eig1_velocity_rank
LIMIT 30;
