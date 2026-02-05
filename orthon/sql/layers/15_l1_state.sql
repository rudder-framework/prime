-- ============================================================================
-- ORTHON SQL: 15_l1_state.sql
-- ============================================================================
-- L1: STATE - Phase Space Position
--
-- Fourth question in physics stack: Where is the system in phase space?
--
-- State is the CONSEQUENCE of energy dynamics.
--   state_distance = Mahalanobis distance from baseline (using ALL metrics)
--   state_velocity = generalized hd_slope (rate of change through phase space)
--   state_acceleration = is degradation speeding up?
-- ============================================================================

-- Basic state metrics
CREATE OR REPLACE VIEW v_l1_state AS
SELECT
    entity_id,
    I,
    -- Raw metrics
    state_distance,
    state_velocity,
    state_acceleration,
    n_metrics_used,

    -- Stability check
    CASE
        WHEN ABS(state_velocity) < 0.01 AND state_distance < 2.0 THEN TRUE
        ELSE FALSE
    END AS is_stable,

    -- Trend classification
    CASE
        WHEN state_velocity > 0.01 THEN 'diverging'
        WHEN state_velocity < -0.01 THEN 'converging'
        ELSE 'stable'
    END AS state_trend,

    -- Distance severity (σ from baseline)
    CASE
        WHEN state_distance > 3.0 THEN 'critical'
        WHEN state_distance > 2.0 THEN 'warning'
        WHEN state_distance > 1.0 THEN 'watch'
        ELSE 'normal'
    END AS distance_severity,

    -- Motion state (acceleration analysis)
    CASE
        WHEN state_velocity > 0 AND state_acceleration > 0 THEN 'accelerating_away'
        WHEN state_velocity > 0 AND state_acceleration < 0 THEN 'decelerating_away'
        WHEN state_velocity < 0 AND state_acceleration < 0 THEN 'accelerating_return'
        WHEN state_velocity < 0 AND state_acceleration > 0 THEN 'decelerating_return'
        ELSE 'stable'
    END AS motion_state

FROM physics
WHERE state_distance IS NOT NULL;


-- State trajectory summary per entity
CREATE OR REPLACE VIEW v_l1_trajectory AS
SELECT
    entity_id,

    -- Distance statistics
    AVG(state_distance) AS mean_distance,
    MAX(state_distance) AS max_distance,
    MIN(state_distance) AS min_distance,

    -- Velocity statistics
    AVG(state_velocity) AS mean_velocity,
    STDDEV(state_velocity) AS velocity_std,

    -- Time above thresholds
    100.0 * SUM(CASE WHEN state_distance > 2 THEN 1 ELSE 0 END) / COUNT(*) AS pct_above_2sigma,
    100.0 * SUM(CASE WHEN state_distance > 3 THEN 1 ELSE 0 END) / COUNT(*) AS pct_above_3sigma,

    -- Trend
    REGR_SLOPE(state_distance, I) AS distance_trend,
    CASE
        WHEN REGR_SLOPE(state_distance, I) > 0.01 THEN 'degrading'
        WHEN REGR_SLOPE(state_distance, I) < -0.01 THEN 'recovering'
        ELSE 'stable'
    END AS overall_trend

FROM physics
WHERE state_distance IS NOT NULL
GROUP BY entity_id;


-- Current state per entity
CREATE OR REPLACE VIEW v_l1_current AS
WITH latest AS (
    SELECT entity_id, MAX(I) AS max_I
    FROM physics WHERE state_distance IS NOT NULL
    GROUP BY entity_id
)
SELECT
    p.entity_id,

    -- Current state
    p.state_distance AS current_distance,
    p.state_velocity AS current_velocity,
    p.state_acceleration AS current_acceleration,
    p.n_metrics_used,

    -- Classifications
    s.is_stable,
    s.state_trend,
    s.distance_severity,
    s.motion_state,

    -- Trajectory context
    t.mean_distance,
    t.max_distance,
    t.pct_above_2sigma,
    t.overall_trend

FROM physics p
JOIN latest l ON p.entity_id = l.entity_id AND p.I = l.max_I
JOIN v_l1_state s ON p.entity_id = s.entity_id AND p.I = s.I
JOIN v_l1_trajectory t USING (entity_id);


-- State transition events
CREATE OR REPLACE VIEW v_l1_transitions AS
SELECT
    entity_id,
    I,
    state_distance,
    state_velocity,
    LAG(distance_severity) OVER w AS prev_severity,
    distance_severity AS curr_severity,
    CASE
        WHEN LAG(distance_severity) OVER w = 'normal' AND distance_severity = 'watch' THEN 'entering_watch'
        WHEN LAG(distance_severity) OVER w = 'watch' AND distance_severity = 'warning' THEN 'entering_warning'
        WHEN LAG(distance_severity) OVER w = 'warning' AND distance_severity = 'critical' THEN 'entering_critical'
        WHEN LAG(distance_severity) OVER w = 'critical' AND distance_severity = 'warning' THEN 'recovering_from_critical'
        WHEN LAG(distance_severity) OVER w = 'warning' AND distance_severity = 'watch' THEN 'recovering_from_warning'
        WHEN LAG(distance_severity) OVER w = 'watch' AND distance_severity = 'normal' THEN 'returned_to_normal'
        ELSE NULL
    END AS transition_type
FROM v_l1_state
WINDOW w AS (PARTITION BY entity_id ORDER BY I)
HAVING transition_type IS NOT NULL;


-- Anomaly detection (rapid state changes)
CREATE OR REPLACE VIEW v_l1_anomalies AS
WITH stats AS (
    SELECT
        entity_id,
        AVG(state_velocity) AS mean_vel,
        STDDEV(state_velocity) AS std_vel
    FROM physics
    WHERE state_velocity IS NOT NULL
    GROUP BY entity_id
)
SELECT
    p.entity_id,
    p.I,
    p.state_distance,
    p.state_velocity,
    s.mean_vel,
    s.std_vel,
    (p.state_velocity - s.mean_vel) / NULLIF(s.std_vel, 0) AS velocity_zscore,
    CASE
        WHEN p.state_velocity > s.mean_vel + 3 * s.std_vel THEN 'rapid_divergence'
        WHEN p.state_velocity < s.mean_vel - 3 * s.std_vel THEN 'rapid_convergence'
        ELSE 'normal'
    END AS anomaly_type
FROM physics p
JOIN stats s USING (entity_id)
WHERE ABS((p.state_velocity - s.mean_vel) / NULLIF(s.std_vel, 0)) > 3;


-- Fleet state summary
CREATE OR REPLACE VIEW v_l1_fleet_summary AS
SELECT
    COUNT(DISTINCT entity_id) AS n_entities,

    -- Severity distribution
    SUM(CASE WHEN distance_severity = 'critical' THEN 1 ELSE 0 END) AS n_critical,
    SUM(CASE WHEN distance_severity = 'warning' THEN 1 ELSE 0 END) AS n_warning,
    SUM(CASE WHEN distance_severity = 'watch' THEN 1 ELSE 0 END) AS n_watch,
    SUM(CASE WHEN distance_severity = 'normal' THEN 1 ELSE 0 END) AS n_normal,

    -- Stability
    SUM(CASE WHEN is_stable THEN 1 ELSE 0 END) AS n_stable,
    100.0 * SUM(CASE WHEN is_stable THEN 1 ELSE 0 END) / COUNT(*) AS pct_stable,

    -- Trend distribution
    SUM(CASE WHEN state_trend = 'diverging' THEN 1 ELSE 0 END) AS n_diverging,
    SUM(CASE WHEN state_trend = 'converging' THEN 1 ELSE 0 END) AS n_converging,
    SUM(CASE WHEN state_trend = 'stable' THEN 1 ELSE 0 END) AS n_trend_stable,

    -- Average metrics
    AVG(current_distance) AS avg_distance,
    MAX(current_distance) AS max_distance

FROM v_l1_current;


-- Verify
SELECT COUNT(*) AS state_rows FROM v_l1_state;
