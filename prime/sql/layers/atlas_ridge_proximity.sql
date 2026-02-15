-- ============================================================
-- Atlas Ridge Proximity Classification
-- Classifies urgency from ridge_proximity.parquet
--
-- Engines computes ridge metrics. Prime classifies urgency.
-- ============================================================

-- ------------------------------------------------------------
-- URGENCY CLASSIFICATION
-- Classifies how urgently a signal is approaching a regime
-- boundary (FTLE ridge) based on proximity and approach rate.
--
-- The engine already provides urgency_class, but Prime adds
-- warning horizon and trend classifications.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_urgency_class AS
SELECT
    I,
    cohort,
    signal_id,
    ftle_current,
    ftle_gradient,
    ftle_acceleration,
    speed,
    urgency,
    time_to_ridge,
    urgency_class,

    -- Warning horizon from time_to_ridge
    CASE
        WHEN time_to_ridge IS NULL THEN 'none'
        WHEN time_to_ridge < 10 THEN 'imminent'
        WHEN time_to_ridge < 50 THEN 'near'
        WHEN time_to_ridge < 200 THEN 'distant'
        ELSE 'none'
    END AS warning_horizon,

    -- Trend from FTLE gradient
    CASE
        WHEN ftle_gradient IS NULL THEN 'unknown'
        WHEN ftle_gradient > 0.01 THEN 'approaching_ridge'
        WHEN ftle_gradient < -0.01 THEN 'leaving_ridge'
        ELSE 'stable'
    END AS ridge_trend,

    -- Acceleration: is approach speeding up?
    CASE
        WHEN ftle_acceleration IS NULL THEN 'unknown'
        WHEN ftle_acceleration > 0.001 THEN 'accelerating_toward'
        WHEN ftle_acceleration < -0.001 THEN 'decelerating'
        ELSE 'constant_rate'
    END AS approach_dynamics

FROM ridge_proximity;
