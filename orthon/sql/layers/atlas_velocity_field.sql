-- ============================================================
-- Atlas Velocity Field Classification
-- Classifies motion from velocity_field.parquet
--
-- Engines computes velocity vectors. ORTHON classifies motion.
-- ============================================================

-- ------------------------------------------------------------
-- MOTION CLASSIFICATION
-- Classifies trajectory motion type from speed, acceleration,
-- and curvature of the state-space trajectory.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_motion_class AS
SELECT
    I,
    cohort,
    speed,
    acceleration_magnitude,
    acceleration_parallel,
    acceleration_perpendicular,
    curvature,
    motion_dimensionality,
    dominant_motion_signal,
    dominant_motion_fraction,
    dominant_accel_signal,

    -- Motion type from speed + acceleration
    CASE
        WHEN speed < 0.001 THEN 'stationary'
        WHEN ABS(acceleration_parallel) < 0.01 * speed THEN 'constant'
        WHEN acceleration_parallel > 0 THEN 'accelerating'
        ELSE 'decelerating'
    END AS motion_type,

    -- Trajectory shape from curvature
    CASE
        WHEN curvature IS NULL THEN 'unknown'
        WHEN curvature < 0.01 THEN 'straight'
        WHEN curvature < 0.5 THEN 'curved'
        ELSE 'spiral'
    END AS trajectory_shape,

    -- Motion complexity from dimensionality
    CASE
        WHEN motion_dimensionality IS NULL THEN 'unknown'
        WHEN motion_dimensionality < 1.5 THEN '1D'
        WHEN motion_dimensionality < 2.5 THEN '2D'
        ELSE '3D'
    END AS motion_complexity,

    -- Dominant signal concentration
    CASE
        WHEN dominant_motion_fraction IS NULL THEN 'unknown'
        WHEN dominant_motion_fraction > 0.8 THEN 'single_signal'
        WHEN dominant_motion_fraction > 0.5 THEN 'dominant_signal'
        ELSE 'distributed'
    END AS motion_concentration

FROM velocity_field;
