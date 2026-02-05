-- ============================================================================
-- ORTHON SQL: 21_geometric_attribution.sql
-- ============================================================================
-- GEOMETRIC ATTRIBUTION: Endogenous vs Exogenous Force Detection
--
-- The Insight:
--   Endogenous: Vectors push against each other. Shape deforms—stretches here,
--               compresses there. But the centroid stays put.
--               Internal tension, zero net translation.
--
--   Exogenous:  The whole geometry translates. Vectors move together.
--               Shape might stay intact, but it's now somewhere else.
--
-- The Measurement:
--   - Centroid stationary + shape changing = endogenous
--   - Centroid moving + shape intact = exogenous
--   - Both moving = mixed forces
--
-- No new engine. No new compute. SQL asks a new question of existing columns.
-- ============================================================================

-- ============================================================================
-- CONFIGURABLE THRESHOLDS
-- ============================================================================
-- Adjust these based on your system's noise floor and sensitivity needs.

CREATE OR REPLACE TABLE config_geometric AS
SELECT
    0.01  AS centroid_drift_threshold,    -- Movement threshold for centroid
    0.01  AS dispersion_change_threshold, -- Change threshold for shape
    0.1   AS delta_correlation_threshold, -- Correlation threshold for aligned changes
    0.7   AS coherence_exogenous_threshold, -- High coherence = moving together
    0.3   AS coherence_endogenous_threshold -- Low coherence = independent deformation
;

-- ============================================================================
-- CENTROID AND DISPERSION PER WINDOW
-- ============================================================================
-- The geometry's position (centroid) and shape (dispersion) at each timestep.
-- Uses observations_enriched if available, otherwise falls back to physics.

CREATE OR REPLACE VIEW v_geometry_centroid AS
SELECT
    entity_id,
    I,

    -- Centroid: Where is the geometry?
    -- Using energy_proxy as our "signal value" from physics
    energy_proxy AS centroid,

    -- Dispersion: What's the shape?
    -- Using effective_dim as a proxy for shape spread (higher = more spread out)
    effective_dim AS dispersion,

    -- Coherence: How aligned are the signals?
    coherence,

    -- State distance: How far from baseline?
    state_distance

FROM physics
WHERE energy_proxy IS NOT NULL;


-- ============================================================================
-- CENTROID DYNAMICS
-- ============================================================================
-- Track how the centroid moves through state space.

CREATE OR REPLACE VIEW v_centroid_dynamics AS
SELECT
    entity_id,
    I,
    centroid,
    dispersion,
    coherence,
    state_distance,

    -- Centroid drift: Did it move?
    centroid - LAG(centroid) OVER w AS centroid_drift,
    ABS(centroid - LAG(centroid) OVER w) AS abs_centroid_drift,

    -- Centroid velocity (smoothed)
    AVG(centroid - LAG(centroid) OVER w) OVER (
        PARTITION BY entity_id ORDER BY I ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) AS centroid_velocity,

    -- Dispersion change: Is shape deforming?
    dispersion - LAG(dispersion) OVER w AS dispersion_change,
    ABS(dispersion - LAG(dispersion) OVER w) AS abs_dispersion_change,

    -- Coherence change: Is alignment shifting?
    coherence - LAG(coherence) OVER w AS coherence_change

FROM v_geometry_centroid
WINDOW w AS (PARTITION BY entity_id ORDER BY I);


-- ============================================================================
-- FORCE ATTRIBUTION
-- ============================================================================
-- Classify each timestep as endogenous, exogenous, or mixed.

CREATE OR REPLACE VIEW v_force_attribution AS
SELECT
    d.entity_id,
    d.I,
    d.centroid,
    d.centroid_drift,
    d.centroid_velocity,
    d.dispersion,
    d.dispersion_change,
    d.coherence,
    d.coherence_change,
    d.state_distance,

    -- Force type based on centroid vs dispersion changes
    CASE
        -- Exogenous: Centroid moves, shape stays intact
        WHEN d.abs_centroid_drift > c.centroid_drift_threshold
         AND d.abs_dispersion_change < c.dispersion_change_threshold
        THEN 'exogenous'

        -- Endogenous: Centroid stable, shape deforms
        WHEN d.abs_centroid_drift < c.centroid_drift_threshold
         AND d.abs_dispersion_change > c.dispersion_change_threshold
        THEN 'endogenous'

        -- Both: Mixed forces
        WHEN d.abs_centroid_drift > c.centroid_drift_threshold
         AND d.abs_dispersion_change > c.dispersion_change_threshold
        THEN 'mixed'

        -- Neither: Stable
        ELSE 'stable'
    END AS force_type,

    -- Coherence-based attribution (from eigenvalue analysis)
    -- High coherence = signals moving together = exogenous
    -- Low coherence = signals moving independently = endogenous
    CASE
        WHEN d.coherence > c.coherence_exogenous_threshold THEN 'coherent_motion'
        WHEN d.coherence < c.coherence_endogenous_threshold THEN 'fragmented_motion'
        ELSE 'moderate_coupling'
    END AS coherence_attribution,

    -- Combined signal
    CASE
        WHEN d.abs_centroid_drift > c.centroid_drift_threshold
         AND d.coherence > c.coherence_exogenous_threshold
        THEN 'strong_exogenous'

        WHEN d.abs_dispersion_change > c.dispersion_change_threshold
         AND d.coherence < c.coherence_endogenous_threshold
        THEN 'strong_endogenous'

        WHEN d.abs_centroid_drift > c.centroid_drift_threshold
         OR d.abs_dispersion_change > c.dispersion_change_threshold
        THEN 'weak_mixed'

        ELSE 'stable'
    END AS combined_attribution

FROM v_centroid_dynamics d
CROSS JOIN config_geometric c
WHERE d.centroid_drift IS NOT NULL;


-- ============================================================================
-- ATTRIBUTION TIMELINE
-- ============================================================================
-- Track attribution changes over time to identify regime shifts.

CREATE OR REPLACE VIEW v_attribution_timeline AS
SELECT
    entity_id,
    I,
    force_type,
    combined_attribution,
    coherence,
    state_distance,

    -- Is this a regime change?
    CASE
        WHEN force_type != LAG(force_type) OVER w THEN TRUE
        ELSE FALSE
    END AS regime_change,

    -- Duration in current regime
    I - MAX(CASE WHEN force_type != LAG(force_type) OVER w THEN I END)
        OVER (PARTITION BY entity_id ORDER BY I ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    AS regime_duration,

    -- Consecutive same-type count
    SUM(1) OVER (
        PARTITION BY entity_id, force_type
        ORDER BY I
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS consecutive_count

FROM v_force_attribution
WINDOW w AS (PARTITION BY entity_id ORDER BY I);


-- ============================================================================
-- ATTRIBUTION EVENTS
-- ============================================================================
-- Major events: when attribution changes significantly.

CREATE OR REPLACE VIEW v_attribution_events AS
SELECT
    entity_id,
    I AS event_time,
    LAG(force_type) OVER w AS from_type,
    force_type AS to_type,
    LAG(combined_attribution) OVER w || ' → ' || combined_attribution AS transition,
    coherence,
    state_distance,

    -- Event classification
    CASE
        WHEN LAG(force_type) OVER w = 'stable' AND force_type = 'exogenous'
        THEN 'external_shock'
        WHEN LAG(force_type) OVER w = 'stable' AND force_type = 'endogenous'
        THEN 'internal_instability'
        WHEN LAG(force_type) OVER w = 'exogenous' AND force_type = 'endogenous'
        THEN 'internalization'
        WHEN LAG(force_type) OVER w = 'endogenous' AND force_type = 'exogenous'
        THEN 'externalization'
        WHEN force_type = 'mixed'
        THEN 'compound_event'
        ELSE 'regime_shift'
    END AS event_type

FROM v_force_attribution
WINDOW w AS (PARTITION BY entity_id ORDER BY I)
HAVING force_type != LAG(force_type) OVER w;


-- ============================================================================
-- ENTITY ATTRIBUTION SUMMARY
-- ============================================================================
-- Summary statistics for each entity.

CREATE OR REPLACE VIEW v_attribution_entity_summary AS
SELECT
    entity_id,

    -- Total windows analyzed
    COUNT(*) AS n_windows,

    -- Force type distribution
    SUM(CASE WHEN force_type = 'exogenous' THEN 1 ELSE 0 END) AS n_exogenous,
    SUM(CASE WHEN force_type = 'endogenous' THEN 1 ELSE 0 END) AS n_endogenous,
    SUM(CASE WHEN force_type = 'mixed' THEN 1 ELSE 0 END) AS n_mixed,
    SUM(CASE WHEN force_type = 'stable' THEN 1 ELSE 0 END) AS n_stable,

    -- Percentages
    100.0 * SUM(CASE WHEN force_type = 'exogenous' THEN 1 ELSE 0 END) / COUNT(*) AS pct_exogenous,
    100.0 * SUM(CASE WHEN force_type = 'endogenous' THEN 1 ELSE 0 END) / COUNT(*) AS pct_endogenous,
    100.0 * SUM(CASE WHEN force_type = 'mixed' THEN 1 ELSE 0 END) / COUNT(*) AS pct_mixed,
    100.0 * SUM(CASE WHEN force_type = 'stable' THEN 1 ELSE 0 END) / COUNT(*) AS pct_stable,

    -- Dominant force type
    CASE
        WHEN SUM(CASE WHEN force_type = 'exogenous' THEN 1 ELSE 0 END) >
             SUM(CASE WHEN force_type = 'endogenous' THEN 1 ELSE 0 END)
        THEN 'exogenous_dominated'
        WHEN SUM(CASE WHEN force_type = 'endogenous' THEN 1 ELSE 0 END) >
             SUM(CASE WHEN force_type = 'exogenous' THEN 1 ELSE 0 END)
        THEN 'endogenous_dominated'
        WHEN SUM(CASE WHEN force_type = 'stable' THEN 1 ELSE 0 END) > COUNT(*) * 0.5
        THEN 'stable_system'
        ELSE 'mixed_system'
    END AS dominant_force,

    -- Attribution score: ratio of translation to deformation
    -- > 1 means more exogenous, < 1 means more endogenous
    CAST(SUM(CASE WHEN force_type = 'exogenous' THEN 1 ELSE 0 END) + 0.5 AS FLOAT) /
    CAST(SUM(CASE WHEN force_type = 'endogenous' THEN 1 ELSE 0 END) + 0.5 AS FLOAT)
    AS attribution_ratio,

    -- Mean coherence
    AVG(coherence) AS mean_coherence,

    -- Number of regime changes
    SUM(CASE WHEN force_type != LAG(force_type) OVER (PARTITION BY entity_id ORDER BY I) THEN 1 ELSE 0 END)
    AS n_regime_changes

FROM v_force_attribution
GROUP BY entity_id;


-- ============================================================================
-- FLEET ATTRIBUTION SUMMARY
-- ============================================================================

CREATE OR REPLACE VIEW v_attribution_fleet_summary AS
SELECT
    COUNT(DISTINCT entity_id) AS n_entities,

    -- Dominant force distribution
    SUM(CASE WHEN dominant_force = 'exogenous_dominated' THEN 1 ELSE 0 END) AS n_exogenous_dominated,
    SUM(CASE WHEN dominant_force = 'endogenous_dominated' THEN 1 ELSE 0 END) AS n_endogenous_dominated,
    SUM(CASE WHEN dominant_force = 'mixed_system' THEN 1 ELSE 0 END) AS n_mixed_system,
    SUM(CASE WHEN dominant_force = 'stable_system' THEN 1 ELSE 0 END) AS n_stable_system,

    -- Average attribution ratio
    AVG(attribution_ratio) AS fleet_attribution_ratio,

    -- Interpretation
    CASE
        WHEN AVG(attribution_ratio) > 2.0 THEN 'Fleet primarily driven by external forces'
        WHEN AVG(attribution_ratio) < 0.5 THEN 'Fleet primarily driven by internal dynamics'
        ELSE 'Fleet shows mixed force patterns'
    END AS fleet_interpretation

FROM v_attribution_entity_summary;


-- ============================================================================
-- NARRATIVE VIEW
-- ============================================================================
-- Human-readable narrative for each entity.

CREATE OR REPLACE VIEW v_attribution_narrative AS
SELECT
    entity_id,
    dominant_force,
    attribution_ratio,
    n_exogenous,
    n_endogenous,
    n_regime_changes,

    -- The story
    CASE dominant_force
        WHEN 'exogenous_dominated' THEN
            'Entity ' || entity_id || ' is primarily driven by external forces. ' ||
            'The geometry translates through state space (' || n_exogenous || ' exogenous windows) ' ||
            'while maintaining its internal shape. Signals move together. ' ||
            CASE WHEN n_regime_changes > 5 THEN 'Multiple external shocks detected.'
                 WHEN n_regime_changes > 0 THEN 'Some regime transitions observed.'
                 ELSE 'Consistent external driving.'
            END

        WHEN 'endogenous_dominated' THEN
            'Entity ' || entity_id || ' is primarily driven by internal dynamics. ' ||
            'The geometry deforms in place (' || n_endogenous || ' endogenous windows). ' ||
            'Signals push against each other—internal tension without net translation. ' ||
            CASE WHEN attribution_ratio < 0.1 THEN 'Strong internal stress patterns.'
                 ELSE 'Moderate internal dynamics.'
            END

        WHEN 'stable_system' THEN
            'Entity ' || entity_id || ' is stable. ' ||
            'Neither significant translation nor deformation observed. ' ||
            'The geometry holds its position and shape.'

        ELSE
            'Entity ' || entity_id || ' shows mixed force patterns. ' ||
            'Both external forces and internal dynamics are active. ' ||
            'Attribution ratio: ' || ROUND(attribution_ratio, 2) || '. ' ||
            n_regime_changes || ' regime changes detected.'
    END AS narrative

FROM v_attribution_entity_summary;


-- ============================================================================
-- VERIFY
-- ============================================================================

.print ''
.print '=== GEOMETRIC ATTRIBUTION ==='
.print ''

SELECT
    entity_id,
    dominant_force,
    ROUND(attribution_ratio, 2) AS attr_ratio,
    n_exogenous || ' exo / ' || n_endogenous || ' endo / ' || n_stable || ' stable' AS distribution
FROM v_attribution_entity_summary
ORDER BY attribution_ratio DESC
LIMIT 10;

.print ''
.print '=== FLEET SUMMARY ==='

SELECT * FROM v_attribution_fleet_summary;

.print ''
.print '=== ATTRIBUTION EVENTS (Recent) ==='

SELECT
    entity_id,
    event_time,
    event_type,
    transition
FROM v_attribution_events
ORDER BY event_time DESC
LIMIT 10;
