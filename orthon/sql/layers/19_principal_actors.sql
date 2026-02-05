-- ============================================================================
-- ORTHON SQL: 19_principal_actors.sql
-- ============================================================================
-- WHO DROVE THE CHANGE?
--
-- Identifies which signals were the principal actors in system changes.
-- Requires: observations_enriched.parquet (per-signal metrics)
--
-- The questions:
--   1. Which signals' energy changed the most?
--   2. Which signals decoupled first?
--   3. Which signals drove the state divergence?
--   4. What is the causal chain?
-- ============================================================================

-- Load observations enriched (per-signal data)
CREATE OR REPLACE TABLE observations_enriched AS
SELECT * FROM read_parquet('{prism_output}/observations_enriched.parquet');

-- ============================================================================
-- SIGNAL ENERGY ANALYSIS
-- ============================================================================
-- Which signals gained or lost the most energy?

CREATE OR REPLACE VIEW v_signal_energy_change AS
WITH signal_bounds AS (
    SELECT
        entity_id,
        signal_id,
        MIN(I) AS first_I,
        MAX(I) AS last_I
    FROM observations_enriched
    GROUP BY entity_id, signal_id
),
initial_energy AS (
    SELECT
        o.entity_id,
        o.signal_id,
        AVG(o.y * o.y + COALESCE(o.dy, 0) * COALESCE(o.dy, 0)) AS initial_energy_proxy
    FROM observations_enriched o
    JOIN signal_bounds b ON o.entity_id = b.entity_id AND o.signal_id = b.signal_id
    WHERE o.I < b.first_I + 10
    GROUP BY o.entity_id, o.signal_id
),
final_energy AS (
    SELECT
        o.entity_id,
        o.signal_id,
        AVG(o.y * o.y + COALESCE(o.dy, 0) * COALESCE(o.dy, 0)) AS final_energy_proxy
    FROM observations_enriched o
    JOIN signal_bounds b ON o.entity_id = b.entity_id AND o.signal_id = b.signal_id
    WHERE o.I > b.last_I - 10
    GROUP BY o.entity_id, o.signal_id
)
SELECT
    i.entity_id,
    i.signal_id,
    i.initial_energy_proxy,
    f.final_energy_proxy,
    f.final_energy_proxy - i.initial_energy_proxy AS energy_change,
    (f.final_energy_proxy - i.initial_energy_proxy) / NULLIF(i.initial_energy_proxy, 0) AS energy_change_pct,
    CASE
        WHEN (f.final_energy_proxy - i.initial_energy_proxy) / NULLIF(i.initial_energy_proxy, 0) > 0.2 THEN 'source'
        WHEN (f.final_energy_proxy - i.initial_energy_proxy) / NULLIF(i.initial_energy_proxy, 0) < -0.2 THEN 'sink'
        ELSE 'neutral'
    END AS energy_role
FROM initial_energy i
JOIN final_energy f USING (entity_id, signal_id);


-- Rank signals by energy contribution
CREATE OR REPLACE VIEW v_signal_energy_ranking AS
SELECT
    entity_id,
    signal_id,
    energy_change,
    energy_change_pct,
    energy_role,
    RANK() OVER (PARTITION BY entity_id ORDER BY ABS(energy_change) DESC) AS energy_change_rank,
    RANK() OVER (PARTITION BY entity_id ORDER BY energy_change DESC) AS energy_source_rank,
    RANK() OVER (PARTITION BY entity_id ORDER BY energy_change ASC) AS energy_sink_rank
FROM v_signal_energy_change;


-- ============================================================================
-- SIGNAL COUPLING ANALYSIS
-- ============================================================================
-- Which signals decoupled from the group?

CREATE OR REPLACE VIEW v_signal_coupling AS
WITH signal_correlations AS (
    -- Correlation with entity mean
    SELECT
        o.entity_id,
        o.signal_id,
        o.I,
        o.y,
        AVG(o.y) OVER (PARTITION BY o.entity_id, o.I) AS entity_mean_y
    FROM observations_enriched o
),
coupling_over_time AS (
    SELECT
        entity_id,
        signal_id,
        I,
        y,
        entity_mean_y,
        -- Rolling correlation with entity mean (proxy for coupling)
        CORR(y, entity_mean_y) OVER (
            PARTITION BY entity_id, signal_id
            ORDER BY I
            ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
        ) AS rolling_coupling
    FROM signal_correlations
)
SELECT
    entity_id,
    signal_id,
    I,
    rolling_coupling,
    CASE
        WHEN rolling_coupling > 0.7 THEN 'strongly_coupled'
        WHEN rolling_coupling > 0.4 THEN 'moderately_coupled'
        WHEN rolling_coupling > 0 THEN 'weakly_coupled'
        ELSE 'anti_coupled'
    END AS coupling_state,
    LAG(rolling_coupling) OVER (PARTITION BY entity_id, signal_id ORDER BY I) AS prev_coupling,
    rolling_coupling - LAG(rolling_coupling) OVER (PARTITION BY entity_id, signal_id ORDER BY I) AS coupling_change
FROM coupling_over_time
WHERE rolling_coupling IS NOT NULL;


-- Decoupling events per signal
CREATE OR REPLACE VIEW v_signal_decoupling_events AS
SELECT
    entity_id,
    signal_id,
    I AS decoupling_time,
    rolling_coupling,
    coupling_change
FROM v_signal_coupling
WHERE coupling_change < -0.1  -- Significant drop in coupling
ORDER BY entity_id, I;


-- First decoupler
CREATE OR REPLACE VIEW v_first_decoupler AS
SELECT DISTINCT ON (entity_id)
    entity_id,
    signal_id AS first_decoupler,
    decoupling_time,
    rolling_coupling AS coupling_at_event
FROM v_signal_decoupling_events
ORDER BY entity_id, decoupling_time;


-- ============================================================================
-- SIGNAL STATE CONTRIBUTION
-- ============================================================================
-- Which signals contributed most to state divergence?

CREATE OR REPLACE VIEW v_signal_state_contribution AS
WITH signal_variance AS (
    SELECT
        entity_id,
        signal_id,
        VAR_POP(y) AS signal_variance,
        STDDEV(y) AS signal_std
    FROM observations_enriched
    GROUP BY entity_id, signal_id
),
entity_total_variance AS (
    SELECT
        entity_id,
        SUM(signal_variance) AS total_variance
    FROM signal_variance
    GROUP BY entity_id
)
SELECT
    v.entity_id,
    v.signal_id,
    v.signal_variance,
    v.signal_std,
    e.total_variance,
    v.signal_variance / NULLIF(e.total_variance, 0) AS variance_contribution,
    RANK() OVER (PARTITION BY v.entity_id ORDER BY v.signal_variance DESC) AS variance_rank
FROM signal_variance v
JOIN entity_total_variance e USING (entity_id);


-- ============================================================================
-- SIGNAL CHANGE TIMING
-- ============================================================================
-- When did each signal start changing?

CREATE OR REPLACE VIEW v_signal_change_onset AS
WITH signal_velocity AS (
    SELECT
        entity_id,
        signal_id,
        I,
        y,
        y - LAG(y) OVER (PARTITION BY entity_id, signal_id ORDER BY I) AS dy,
        ABS(y - LAG(y) OVER (PARTITION BY entity_id, signal_id ORDER BY I)) AS abs_dy
    FROM observations_enriched
),
signal_activity AS (
    SELECT
        entity_id,
        signal_id,
        I,
        abs_dy,
        AVG(abs_dy) OVER (PARTITION BY entity_id, signal_id) AS mean_activity,
        STDDEV(abs_dy) OVER (PARTITION BY entity_id, signal_id) AS std_activity
    FROM signal_velocity
    WHERE dy IS NOT NULL
)
SELECT
    entity_id,
    signal_id,
    MIN(I) AS change_onset_time
FROM signal_activity
WHERE abs_dy > mean_activity + 2 * std_activity  -- Activity spike
GROUP BY entity_id, signal_id;


-- First mover
CREATE OR REPLACE VIEW v_first_mover AS
SELECT DISTINCT ON (entity_id)
    entity_id,
    signal_id AS first_mover,
    change_onset_time
FROM v_signal_change_onset
ORDER BY entity_id, change_onset_time;


-- ============================================================================
-- THE PRINCIPAL ACTORS
-- ============================================================================
-- Combine all analyses to identify key signals

CREATE OR REPLACE VIEW v_principal_actors AS
SELECT
    e.entity_id,
    e.signal_id,

    -- Energy role
    e.energy_change,
    e.energy_change_pct,
    e.energy_role,
    e.energy_change_rank,

    -- State contribution
    s.variance_contribution,
    s.variance_rank,

    -- Decoupling
    COALESCE(d.first_decoupler = e.signal_id, FALSE) AS is_first_decoupler,

    -- First mover
    COALESCE(m.first_mover = e.signal_id, FALSE) AS is_first_mover,

    -- Principal actor score
    (10.0 / e.energy_change_rank) +                             -- High energy change
    (10.0 / s.variance_rank) +                                  -- High variance contribution
    (CASE WHEN d.first_decoupler = e.signal_id THEN 20 ELSE 0 END) +  -- First to decouple
    (CASE WHEN m.first_mover = e.signal_id THEN 15 ELSE 0 END)        -- First to change
    AS principal_actor_score,

    -- Role classification
    CASE
        WHEN d.first_decoupler = e.signal_id AND e.energy_role = 'sink'
        THEN 'degradation_driver'
        WHEN m.first_mover = e.signal_id AND e.energy_role = 'source'
        THEN 'excitation_source'
        WHEN e.energy_change_rank = 1 AND e.energy_role = 'sink'
        THEN 'primary_energy_sink'
        WHEN e.energy_change_rank = 1 AND e.energy_role = 'source'
        THEN 'primary_energy_source'
        WHEN s.variance_rank = 1
        THEN 'primary_variance_contributor'
        ELSE 'secondary_actor'
    END AS actor_classification

FROM v_signal_energy_ranking e
JOIN v_signal_state_contribution s USING (entity_id, signal_id)
LEFT JOIN v_first_decoupler d ON e.entity_id = d.entity_id
LEFT JOIN v_first_mover m ON e.entity_id = m.entity_id;


-- Top actors per entity
CREATE OR REPLACE VIEW v_top_actors AS
SELECT
    entity_id,
    signal_id,
    principal_actor_score,
    actor_classification,
    energy_role,
    is_first_decoupler,
    is_first_mover,
    RANK() OVER (PARTITION BY entity_id ORDER BY principal_actor_score DESC) AS actor_rank
FROM v_principal_actors;


-- Entity actor summary
CREATE OR REPLACE VIEW v_entity_actor_summary AS
SELECT
    entity_id,
    COUNT(*) AS n_signals,
    MAX(CASE WHEN actor_rank = 1 THEN signal_id END) AS primary_actor,
    MAX(CASE WHEN actor_rank = 1 THEN actor_classification END) AS primary_actor_role,
    MAX(CASE WHEN is_first_mover THEN signal_id END) AS first_mover,
    MAX(CASE WHEN is_first_decoupler THEN signal_id END) AS first_decoupler,
    SUM(CASE WHEN energy_role = 'source' THEN 1 ELSE 0 END) AS n_sources,
    SUM(CASE WHEN energy_role = 'sink' THEN 1 ELSE 0 END) AS n_sinks,

    -- Narrative
    'Primary actor: ' || MAX(CASE WHEN actor_rank = 1 THEN signal_id END) ||
    ' (' || MAX(CASE WHEN actor_rank = 1 THEN actor_classification END) || '). ' ||
    COALESCE('First to move: ' || MAX(CASE WHEN is_first_mover THEN signal_id END) || '. ', '') ||
    COALESCE('First to decouple: ' || MAX(CASE WHEN is_first_decoupler THEN signal_id END) || '.', '')
    AS actor_narrative

FROM v_top_actors
GROUP BY entity_id;


-- ============================================================================
-- THE CAUSAL CHAIN
-- ============================================================================
-- Reconstruct the sequence of events

CREATE OR REPLACE VIEW v_causal_chain AS
WITH events AS (
    -- First mover
    SELECT entity_id, change_onset_time AS event_time, first_mover AS signal_id, 'started_changing' AS event, 1 AS seq
    FROM v_first_mover

    UNION ALL

    -- First decoupler
    SELECT entity_id, decoupling_time AS event_time, first_decoupler AS signal_id, 'decoupled' AS event, 2 AS seq
    FROM v_first_decoupler

    UNION ALL

    -- Energy events (from main timeline if available)
    SELECT entity_id, event_time, NULL AS signal_id, event_type AS event, 3 AS seq
    FROM v_story_energy_events
)
SELECT
    entity_id,
    event_time,
    signal_id,
    event,
    ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY event_time, seq) AS chain_position
FROM events
ORDER BY entity_id, event_time;


-- Verify
SELECT entity_id, primary_actor, primary_actor_role, actor_narrative
FROM v_entity_actor_summary
LIMIT 5;
