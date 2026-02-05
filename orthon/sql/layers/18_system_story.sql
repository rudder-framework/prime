-- ============================================================================
-- ORTHON SQL: 18_system_story.sql
-- ============================================================================
-- THE SYSTEM STORY
--
-- This script tells the narrative of what happened to each entity:
--   1. Where did we start? (Initial state, energy, structure)
--   2. What changed? (Events, transitions, turning points)
--   3. Where did we end up? (Final state, trajectory)
--   4. Who drove the change? (Principal signals)
--   5. Was it internal or external? (Endogenous vs exogenous)
-- ============================================================================

-- ============================================================================
-- CHAPTER 1: THE BEGINNING
-- ============================================================================
-- What did the system look like at the start?

CREATE OR REPLACE VIEW v_story_beginning AS
WITH first_window AS (
    SELECT
        entity_id,
        MIN(I) AS start_I
    FROM physics
    GROUP BY entity_id
),
initial_state AS (
    SELECT
        p.entity_id,
        p.I,
        p.energy_proxy,
        p.coherence,
        p.effective_dim,
        p.eigenvalue_entropy,
        p.state_distance,
        p.n_signals,
        ROW_NUMBER() OVER (PARTITION BY p.entity_id ORDER BY p.I) AS rn
    FROM physics p
    JOIN first_window f ON p.entity_id = f.entity_id
    WHERE p.I < f.start_I + 10  -- First 10 time points
)
SELECT
    entity_id,

    -- Initial energy state
    AVG(energy_proxy) AS initial_energy,
    STDDEV(energy_proxy) / NULLIF(AVG(energy_proxy), 0) AS initial_energy_stability,

    -- Initial structure
    AVG(coherence) AS initial_coherence,
    AVG(effective_dim) AS initial_effective_dim,
    CASE
        WHEN AVG(coherence) > 0.7 THEN 'strongly_coupled'
        WHEN AVG(coherence) > 0.4 THEN 'moderately_coupled'
        ELSE 'loosely_coupled'
    END AS initial_coupling_state,

    -- Initial state
    AVG(state_distance) AS initial_state_distance,
    CASE
        WHEN AVG(state_distance) < 1 THEN 'at_baseline'
        WHEN AVG(state_distance) < 2 THEN 'near_baseline'
        ELSE 'away_from_baseline'
    END AS initial_position,

    -- System size
    MAX(n_signals) AS n_signals,

    -- The story begins...
    'System started with ' || ROUND(AVG(energy_proxy)::DECIMAL, 2) || ' energy, ' ||
    CASE WHEN AVG(coherence) > 0.7 THEN 'strong' WHEN AVG(coherence) > 0.4 THEN 'moderate' ELSE 'weak' END ||
    ' coupling (coherence ' || ROUND(AVG(coherence)::DECIMAL, 2) || '), and ' ||
    CASE WHEN AVG(state_distance) < 1 THEN 'was at baseline' WHEN AVG(state_distance) < 2 THEN 'was near baseline' ELSE 'was already drifting' END ||
    '.' AS opening_sentence

FROM initial_state
GROUP BY entity_id;


-- ============================================================================
-- CHAPTER 2: THE EVENTS
-- ============================================================================
-- What significant changes occurred?

-- Energy change events
CREATE OR REPLACE VIEW v_story_energy_events AS
WITH energy_changes AS (
    SELECT
        entity_id,
        I,
        energy_proxy,
        energy_velocity,
        LAG(energy_proxy, 5) OVER w AS energy_5_ago,
        (energy_proxy - LAG(energy_proxy, 5) OVER w) / NULLIF(LAG(energy_proxy, 5) OVER w, 0) AS energy_pct_change_5,
        CASE
            WHEN energy_velocity < -0.01 AND LAG(energy_velocity) OVER w >= -0.01 THEN 'dissipation_began'
            WHEN energy_velocity > 0.01 AND LAG(energy_velocity) OVER w <= 0.01 THEN 'accumulation_began'
            WHEN ABS(energy_velocity) < 0.001 AND ABS(LAG(energy_velocity) OVER w) >= 0.01 THEN 'energy_stabilized'
            ELSE NULL
        END AS energy_event
    FROM physics
    WINDOW w AS (PARTITION BY entity_id ORDER BY I)
)
SELECT
    entity_id,
    I AS event_time,
    'energy' AS event_category,
    energy_event AS event_type,
    energy_proxy AS value_at_event,
    energy_pct_change_5 AS magnitude,
    CASE energy_event
        WHEN 'dissipation_began' THEN 'Energy began dissipating'
        WHEN 'accumulation_began' THEN 'Energy began accumulating'
        WHEN 'energy_stabilized' THEN 'Energy stabilized'
    END AS description
FROM energy_changes
WHERE energy_event IS NOT NULL;


-- Coherence change events
CREATE OR REPLACE VIEW v_story_coherence_events AS
WITH coherence_changes AS (
    SELECT
        entity_id,
        I,
        coherence,
        coherence_velocity,
        effective_dim,
        LAG(coherence) OVER w AS prev_coherence,
        CASE
            WHEN coherence < 0.4 AND LAG(coherence) OVER w >= 0.4 THEN 'system_decoupled'
            WHEN coherence > 0.7 AND LAG(coherence) OVER w <= 0.7 THEN 'coupling_strengthened'
            WHEN effective_dim > LAG(effective_dim) OVER w * 1.5 THEN 'fragmentation_event'
            WHEN coherence_velocity < -0.02 THEN 'rapid_decoupling'
            ELSE NULL
        END AS coherence_event
    FROM physics
    WINDOW w AS (PARTITION BY entity_id ORDER BY I)
)
SELECT
    entity_id,
    I AS event_time,
    'coherence' AS event_category,
    coherence_event AS event_type,
    coherence AS value_at_event,
    coherence - prev_coherence AS magnitude,
    CASE coherence_event
        WHEN 'system_decoupled' THEN 'System became decoupled (coherence dropped below 0.4)'
        WHEN 'coupling_strengthened' THEN 'Coupling strengthened significantly'
        WHEN 'fragmentation_event' THEN 'System fragmented into more independent modes'
        WHEN 'rapid_decoupling' THEN 'Rapid decoupling detected'
    END AS description
FROM coherence_changes
WHERE coherence_event IS NOT NULL;


-- State change events
CREATE OR REPLACE VIEW v_story_state_events AS
WITH state_changes AS (
    SELECT
        entity_id,
        I,
        state_distance,
        state_velocity,
        state_acceleration,
        LAG(state_velocity) OVER w AS prev_velocity,
        CASE
            WHEN state_distance > 3 AND LAG(state_distance) OVER w <= 3 THEN 'entered_critical'
            WHEN state_distance > 2 AND LAG(state_distance) OVER w <= 2 THEN 'entered_warning'
            WHEN state_distance <= 2 AND LAG(state_distance) OVER w > 2 THEN 'exited_warning'
            WHEN state_velocity > 0.01 AND LAG(state_velocity) OVER w <= 0.01 THEN 'began_diverging'
            WHEN state_velocity < -0.01 AND LAG(state_velocity) OVER w >= -0.01 THEN 'began_converging'
            ELSE NULL
        END AS state_event
    FROM physics
    WINDOW w AS (PARTITION BY entity_id ORDER BY I)
)
SELECT
    entity_id,
    I AS event_time,
    'state' AS event_category,
    state_event AS event_type,
    state_distance AS value_at_event,
    state_velocity AS magnitude,
    CASE state_event
        WHEN 'entered_critical' THEN 'State entered critical zone (>3σ from baseline)'
        WHEN 'entered_warning' THEN 'State entered warning zone (>2σ from baseline)'
        WHEN 'exited_warning' THEN 'State recovered from warning zone'
        WHEN 'began_diverging' THEN 'State began diverging from baseline'
        WHEN 'began_converging' THEN 'State began returning to baseline'
    END AS description
FROM state_changes
WHERE state_event IS NOT NULL;


-- Combined event timeline
CREATE OR REPLACE VIEW v_story_timeline AS
SELECT * FROM v_story_energy_events
UNION ALL
SELECT * FROM v_story_coherence_events
UNION ALL
SELECT * FROM v_story_state_events
ORDER BY entity_id, event_time;


-- ============================================================================
-- CHAPTER 3: THE TURNING POINTS
-- ============================================================================
-- Key moments where the trajectory changed

CREATE OR REPLACE VIEW v_story_turning_points AS
WITH all_events AS (
    SELECT
        entity_id,
        event_time,
        event_category,
        event_type,
        description,
        ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY event_time) AS event_sequence
    FROM v_story_timeline
)
SELECT
    entity_id,
    event_time,
    event_sequence,
    event_category,
    event_type,
    description,

    -- What was happening simultaneously?
    (SELECT coherence FROM physics p WHERE p.entity_id = e.entity_id AND p.I = e.event_time) AS coherence_at_event,
    (SELECT state_distance FROM physics p WHERE p.entity_id = e.entity_id AND p.I = e.event_time) AS state_at_event,
    (SELECT energy_proxy FROM physics p WHERE p.entity_id = e.entity_id AND p.I = e.event_time) AS energy_at_event,

    -- Time since last event
    event_time - LAG(event_time) OVER (PARTITION BY entity_id ORDER BY event_time) AS time_since_last_event

FROM all_events e;


-- ============================================================================
-- CHAPTER 4: THE ENDING
-- ============================================================================
-- Where did we end up?

CREATE OR REPLACE VIEW v_story_ending AS
WITH last_window AS (
    SELECT
        entity_id,
        MAX(I) AS end_I
    FROM physics
    GROUP BY entity_id
),
final_state AS (
    SELECT
        p.entity_id,
        p.I,
        p.energy_proxy,
        p.coherence,
        p.effective_dim,
        p.state_distance,
        p.state_velocity,
        ROW_NUMBER() OVER (PARTITION BY p.entity_id ORDER BY p.I DESC) AS rn
    FROM physics p
    JOIN last_window l ON p.entity_id = l.entity_id
    WHERE p.I > l.end_I - 10  -- Last 10 time points
)
SELECT
    f.entity_id,

    -- Final energy
    AVG(f.energy_proxy) AS final_energy,
    b.initial_energy,
    (AVG(f.energy_proxy) - b.initial_energy) / NULLIF(b.initial_energy, 0) AS energy_change_pct,

    -- Final structure
    AVG(f.coherence) AS final_coherence,
    b.initial_coherence,
    AVG(f.coherence) - b.initial_coherence AS coherence_change,
    CASE
        WHEN AVG(f.coherence) > 0.7 THEN 'strongly_coupled'
        WHEN AVG(f.coherence) > 0.4 THEN 'moderately_coupled'
        ELSE 'decoupled'
    END AS final_coupling_state,

    -- Final state
    AVG(f.state_distance) AS final_state_distance,
    AVG(f.state_velocity) AS final_trajectory,
    CASE
        WHEN AVG(f.state_velocity) > 0.01 THEN 'still_diverging'
        WHEN AVG(f.state_velocity) < -0.01 THEN 'recovering'
        ELSE 'stabilized'
    END AS final_motion,

    -- The story ends...
    'System ended with ' || ROUND(AVG(f.energy_proxy)::DECIMAL, 2) || ' energy (' ||
    CASE
        WHEN (AVG(f.energy_proxy) - b.initial_energy) / NULLIF(b.initial_energy, 0) < -0.1 THEN 'dissipated ' || ROUND(ABS((AVG(f.energy_proxy) - b.initial_energy) / NULLIF(b.initial_energy, 0) * 100)::DECIMAL, 1) || '%'
        WHEN (AVG(f.energy_proxy) - b.initial_energy) / NULLIF(b.initial_energy, 0) > 0.1 THEN 'accumulated ' || ROUND(((AVG(f.energy_proxy) - b.initial_energy) / NULLIF(b.initial_energy, 0) * 100)::DECIMAL, 1) || '%'
        ELSE 'roughly conserved'
    END || '), ' ||
    CASE WHEN AVG(f.coherence) > b.initial_coherence + 0.1 THEN 'strengthened coupling'
         WHEN AVG(f.coherence) < b.initial_coherence - 0.1 THEN 'weakened coupling'
         ELSE 'maintained coupling'
    END || ', and ' ||
    CASE WHEN AVG(f.state_velocity) > 0.01 THEN 'is still diverging'
         WHEN AVG(f.state_velocity) < -0.01 THEN 'is recovering'
         ELSE 'has stabilized'
    END || '.' AS closing_sentence

FROM final_state f
JOIN v_story_beginning b USING (entity_id)
GROUP BY f.entity_id, b.initial_energy, b.initial_coherence;


-- ============================================================================
-- CHAPTER 5: THE PRINCIPAL ACTORS
-- ============================================================================
-- Which signals drove the changes?

-- This requires observations_enriched with per-signal data
-- For now, we identify based on physics layer metrics

CREATE OR REPLACE VIEW v_story_drivers AS
WITH signal_energy_trends AS (
    -- If we have per-signal energy data
    SELECT
        entity_id,
        'Signal analysis requires observations_enriched' AS note
    FROM physics
    LIMIT 1
)
SELECT * FROM signal_energy_trends;


-- ============================================================================
-- CHAPTER 6: ENDOGENOUS VS EXOGENOUS
-- ============================================================================
-- Was the change internal or driven by external factors?

CREATE OR REPLACE VIEW v_story_causation AS
SELECT
    entity_id,

    -- Count of each event type
    SUM(CASE WHEN event_category = 'energy' THEN 1 ELSE 0 END) AS n_energy_events,
    SUM(CASE WHEN event_category = 'coherence' THEN 1 ELSE 0 END) AS n_coherence_events,
    SUM(CASE WHEN event_category = 'state' THEN 1 ELSE 0 END) AS n_state_events,

    -- What led?
    (SELECT event_category FROM v_story_timeline t WHERE t.entity_id = e.entity_id ORDER BY event_time LIMIT 1) AS first_event_type,

    -- Interpretation
    CASE
        WHEN (SELECT event_type FROM v_story_timeline t WHERE t.entity_id = e.entity_id ORDER BY event_time LIMIT 1) LIKE '%energy%'
        THEN 'exogenous'  -- Energy changed first (external force)
        WHEN (SELECT event_type FROM v_story_timeline t WHERE t.entity_id = e.entity_id ORDER BY event_time LIMIT 1) LIKE '%coherence%'
        THEN 'endogenous'  -- Structure changed first (internal breakdown)
        ELSE 'unclear'
    END AS likely_causation,

    -- Narrative
    CASE
        WHEN (SELECT event_type FROM v_story_timeline t WHERE t.entity_id = e.entity_id ORDER BY event_time LIMIT 1) LIKE '%dissipation%'
        THEN 'Change appears to be exogenously driven - energy dissipation began first, suggesting external loading or loss.'
        WHEN (SELECT event_type FROM v_story_timeline t WHERE t.entity_id = e.entity_id ORDER BY event_time LIMIT 1) LIKE '%decoupl%'
        THEN 'Change appears to be endogenously driven - decoupling began before energy changes, suggesting internal structural breakdown.'
        WHEN (SELECT event_type FROM v_story_timeline t WHERE t.entity_id = e.entity_id ORDER BY event_time LIMIT 1) LIKE '%diverging%'
        THEN 'Change appears state-initiated - system began drifting before other changes, suggesting gradual process.'
        ELSE 'Causation pattern is unclear - multiple factors may be involved.'
    END AS causation_narrative

FROM (SELECT DISTINCT entity_id FROM v_story_timeline) e;


-- ============================================================================
-- THE COMPLETE STORY
-- ============================================================================

CREATE OR REPLACE VIEW v_story_complete AS
SELECT
    b.entity_id,

    -- Chapter 1: Beginning
    b.opening_sentence AS ch1_beginning,
    b.initial_energy,
    b.initial_coherence,
    b.initial_state_distance,

    -- Chapter 2-3: Events count
    (SELECT COUNT(*) FROM v_story_timeline t WHERE t.entity_id = b.entity_id) AS n_total_events,

    -- Chapter 4: Ending
    e.closing_sentence AS ch4_ending,
    e.final_energy,
    e.final_coherence,
    e.final_state_distance,
    e.final_motion,

    -- Chapter 6: Causation
    c.likely_causation,
    c.causation_narrative AS ch6_causation,

    -- Overall trajectory
    CASE
        WHEN e.energy_change_pct < -0.2 AND e.coherence_change < -0.2 THEN 'degradation'
        WHEN e.energy_change_pct > 0.2 AND e.coherence_change < -0.2 THEN 'overload'
        WHEN e.final_motion = 'recovering' THEN 'recovery'
        WHEN e.final_motion = 'still_diverging' THEN 'ongoing_issue'
        ELSE 'stable'
    END AS overall_trajectory,

    -- The full story
    b.opening_sentence || ' ' ||
    COALESCE(c.causation_narrative, '') || ' ' ||
    e.closing_sentence AS full_narrative

FROM v_story_beginning b
JOIN v_story_ending e USING (entity_id)
LEFT JOIN v_story_causation c USING (entity_id);


-- ============================================================================
-- STORY SUMMARY FOR FLEET
-- ============================================================================

CREATE OR REPLACE VIEW v_story_fleet_summary AS
SELECT
    COUNT(*) AS n_entities,

    -- Trajectory distribution
    SUM(CASE WHEN overall_trajectory = 'degradation' THEN 1 ELSE 0 END) AS n_degrading,
    SUM(CASE WHEN overall_trajectory = 'overload' THEN 1 ELSE 0 END) AS n_overloaded,
    SUM(CASE WHEN overall_trajectory = 'recovery' THEN 1 ELSE 0 END) AS n_recovering,
    SUM(CASE WHEN overall_trajectory = 'ongoing_issue' THEN 1 ELSE 0 END) AS n_ongoing_issues,
    SUM(CASE WHEN overall_trajectory = 'stable' THEN 1 ELSE 0 END) AS n_stable,

    -- Causation distribution
    SUM(CASE WHEN likely_causation = 'exogenous' THEN 1 ELSE 0 END) AS n_exogenous,
    SUM(CASE WHEN likely_causation = 'endogenous' THEN 1 ELSE 0 END) AS n_endogenous,

    -- Average changes
    AVG(final_energy - initial_energy) AS avg_energy_change,
    AVG(final_coherence - initial_coherence) AS avg_coherence_change

FROM v_story_complete;


-- Verify
SELECT entity_id, LEFT(full_narrative, 200) AS narrative_preview
FROM v_story_complete
LIMIT 5;
