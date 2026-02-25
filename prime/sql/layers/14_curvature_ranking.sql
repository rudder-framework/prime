-- ============================================================================
-- CURVATURE RANKING (Second Derivative)
-- Bending of trajectory — early warning before velocity changes
-- ============================================================================

CREATE OR REPLACE VIEW v_geometry_curvature AS
SELECT
    cohort,
    signal_0_center,
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
        PARTITION BY signal_0_center
        ORDER BY ABS(effective_dim_acceleration) DESC NULLS LAST
    ) AS fleet_curvature_rank,

    -- Direction: bending toward collapse or away
    SIGN(effective_dim_acceleration) AS curvature_direction

FROM geometry_dynamics
WHERE effective_dim_acceleration IS NOT NULL
  AND NOT isnan(effective_dim_acceleration);

-- Biggest effective_dim curvature events across fleet
SELECT
    cohort,
    signal_0_center,
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


-- ============================================================================
-- EIGENVALUE_1 CURVATURE RANKING
-- Dominant eigenvalue acceleration — precedes RUL cliff
-- ============================================================================

CREATE OR REPLACE VIEW v_eigenvalue_1_curvature AS
SELECT
    cohort,
    signal_0_center,
    eigenvalue_1,
    eigenvalue_1_velocity,
    eigenvalue_1_acceleration,
    ABS(eigenvalue_1_acceleration) AS eig1_curvature_magnitude,

    RANK() OVER (
        PARTITION BY cohort
        ORDER BY ABS(eigenvalue_1_acceleration) DESC NULLS LAST
    ) AS eig1_curvature_rank,

    RANK() OVER (
        PARTITION BY signal_0_center
        ORDER BY ABS(eigenvalue_1_acceleration) DESC NULLS LAST
    ) AS eig1_fleet_curvature_rank,

    CASE WHEN eigenvalue_1_acceleration < 0 THEN -1 ELSE 1 END AS eig1_curvature_direction

FROM geometry_dynamics
WHERE eigenvalue_1_acceleration IS NOT NULL
  AND NOT isnan(eigenvalue_1_acceleration);

-- Biggest eigenvalue_1 curvature events across fleet
SELECT
    cohort,
    signal_0_center,
    ROUND(eigenvalue_1, 4) AS eigenvalue_1,
    ROUND(eigenvalue_1_velocity, 4) AS eig1_velocity,
    ROUND(eigenvalue_1_acceleration, 4) AS eig1_acceleration,
    ROUND(eig1_curvature_magnitude, 4) AS magnitude,
    eig1_curvature_rank,
    eig1_curvature_direction
FROM v_eigenvalue_1_curvature
WHERE eig1_curvature_rank <= 3
ORDER BY eig1_curvature_magnitude DESC
LIMIT 30;


-- ============================================================================
-- CONDITION NUMBER CURVATURE RANKING
-- Numerical stability acceleration — ill-conditioning onset
-- ============================================================================

CREATE OR REPLACE VIEW v_condition_number_curvature AS
SELECT
    cohort,
    signal_0_center,
    condition_number,
    condition_number_velocity,
    condition_number_acceleration,
    ABS(condition_number_acceleration) AS cond_curvature_magnitude,

    RANK() OVER (
        PARTITION BY cohort
        ORDER BY ABS(condition_number_acceleration) DESC NULLS LAST
    ) AS cond_curvature_rank,

    RANK() OVER (
        PARTITION BY signal_0_center
        ORDER BY ABS(condition_number_acceleration) DESC NULLS LAST
    ) AS cond_fleet_curvature_rank

FROM geometry_dynamics
WHERE condition_number_acceleration IS NOT NULL
  AND NOT isnan(condition_number_acceleration);

-- Biggest condition_number curvature events across fleet
SELECT
    cohort,
    signal_0_center,
    ROUND(condition_number, 2) AS cond_num,
    ROUND(condition_number_velocity, 4) AS cond_velocity,
    ROUND(condition_number_acceleration, 4) AS cond_acceleration,
    ROUND(cond_curvature_magnitude, 4) AS magnitude,
    cond_curvature_rank
FROM v_condition_number_curvature
WHERE cond_curvature_rank <= 3
ORDER BY cond_curvature_magnitude DESC
LIMIT 30;
