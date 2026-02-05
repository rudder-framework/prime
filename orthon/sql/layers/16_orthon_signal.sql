-- ============================================================================
-- ORTHON SQL: 16_orthon_signal.sql
-- ============================================================================
-- THE ØRTHON SIGNAL
--
-- dissipating + decoupling + diverging = symplectic structure loss
--
-- This is THE degradation signal. When all three occur together:
--   - Energy is leaving the system (L4)
--   - Signals are decoupling (L2)
--   - State is diverging from baseline (L1)
--
-- The system's geometric structure is breaking down.
-- ============================================================================

-- Combine all physics layers for signal detection
CREATE OR REPLACE VIEW v_orthon_signal AS
SELECT
    p.entity_id,
    p.I,

    -- L4: Thermodynamics
    p.energy_proxy,
    p.energy_velocity,
    p.dissipation_rate,
    CASE
        WHEN p.energy_velocity < -0.001 THEN 'dissipating'
        WHEN p.energy_velocity > 0.001 THEN 'accumulating'
        ELSE 'stable'
    END AS energy_trend,

    -- L2: Coherence (eigenvalue-based)
    p.coherence,
    p.effective_dim,
    p.eigenvalue_entropy,
    CASE
        WHEN p.coherence > 0.7 THEN 'strongly_coupled'
        WHEN p.coherence > 0.4 THEN 'weakly_coupled'
        ELSE 'decoupled'
    END AS coupling_state,

    -- L1: State
    p.state_distance,
    p.state_velocity,
    CASE
        WHEN p.state_velocity > 0.01 THEN 'diverging'
        WHEN p.state_velocity < -0.01 THEN 'converging'
        ELSE 'stable'
    END AS state_trend,

    -- THE ØRTHON SIGNAL
    CASE
        WHEN p.energy_velocity < -0.001                              -- Energy dissipating
         AND (p.coherence < 0.4 OR p.coherence_velocity < -0.001)    -- Decoupling
         AND p.state_velocity > 0.01                                  -- State diverging
        THEN TRUE
        ELSE FALSE
    END AS orthon_signal,

    -- Severity score (0-7)
    (CASE WHEN p.energy_velocity < -0.001 THEN 1 ELSE 0 END) +
    (CASE WHEN p.coherence_velocity < -0.001 THEN 1 ELSE 0 END) +
    (CASE WHEN p.effective_dim > p.n_signals / 2.0 THEN 1 ELSE 0 END) +
    (CASE WHEN p.coherence < 0.4 THEN 2 ELSE 0 END) +
    (CASE WHEN p.state_velocity > 0.01 THEN 1 ELSE 0 END) +
    (CASE WHEN p.state_distance > 3 THEN 1 ELSE 0 END) AS severity_score,

    -- Overall severity classification
    CASE
        WHEN p.energy_velocity < -0.001
         AND (p.coherence < 0.4 OR p.coherence_velocity < -0.001)
         AND p.state_velocity > 0.01
        THEN 'critical'
        WHEN p.energy_velocity < -0.001 AND p.state_velocity > 0.01
        THEN 'warning'
        WHEN p.coherence_velocity < -0.001 OR p.state_velocity > 0.01
        THEN 'watch'
        ELSE 'normal'
    END AS severity

FROM physics p
WHERE p.coherence IS NOT NULL AND p.state_distance IS NOT NULL;


-- Ørthon signal events (when signal first appears)
CREATE OR REPLACE VIEW v_orthon_signal_events AS
SELECT
    entity_id,
    I,
    orthon_signal,
    LAG(orthon_signal) OVER w AS prev_signal,
    severity_score,
    energy_trend,
    coupling_state,
    state_trend,
    CASE
        WHEN orthon_signal AND NOT COALESCE(LAG(orthon_signal) OVER w, FALSE) THEN 'signal_onset'
        WHEN NOT orthon_signal AND COALESCE(LAG(orthon_signal) OVER w, FALSE) THEN 'signal_cleared'
        ELSE NULL
    END AS event_type
FROM v_orthon_signal
WINDOW w AS (PARTITION BY entity_id ORDER BY I)
HAVING event_type IS NOT NULL;


-- Entity summary with Ørthon signal status
CREATE OR REPLACE VIEW v_orthon_entity_summary AS
WITH latest AS (
    SELECT entity_id, MAX(I) AS max_I
    FROM v_orthon_signal
    GROUP BY entity_id
),
signal_history AS (
    SELECT
        entity_id,
        SUM(CASE WHEN orthon_signal THEN 1 ELSE 0 END) AS total_signal_points,
        COUNT(*) AS total_points,
        100.0 * SUM(CASE WHEN orthon_signal THEN 1 ELSE 0 END) / COUNT(*) AS pct_in_signal
    FROM v_orthon_signal
    GROUP BY entity_id
)
SELECT
    o.entity_id,

    -- Current status
    o.orthon_signal AS current_orthon_signal,
    o.severity AS current_severity,
    o.severity_score AS current_severity_score,

    -- Components
    o.energy_trend,
    o.coupling_state,
    o.state_trend,

    -- Current values
    o.energy_proxy,
    o.coherence,
    o.effective_dim,
    o.state_distance,
    o.state_velocity,

    -- History
    h.total_signal_points,
    h.pct_in_signal,

    -- Interpretation
    CASE
        WHEN o.orthon_signal THEN 'CRITICAL: Symplectic structure loss detected'
        WHEN o.severity = 'warning' THEN 'WARNING: Energy dissipating and state diverging'
        WHEN o.severity = 'watch' THEN 'WATCH: Early degradation indicators'
        ELSE 'NORMAL: System stable'
    END AS status_message

FROM v_orthon_signal o
JOIN latest l ON o.entity_id = l.entity_id AND o.I = l.max_I
JOIN signal_history h USING (entity_id);


-- Fleet summary
CREATE OR REPLACE VIEW v_orthon_fleet_summary AS
SELECT
    COUNT(DISTINCT entity_id) AS n_entities,

    -- Signal counts
    SUM(CASE WHEN current_orthon_signal THEN 1 ELSE 0 END) AS n_with_orthon_signal,
    100.0 * SUM(CASE WHEN current_orthon_signal THEN 1 ELSE 0 END) / COUNT(*) AS pct_with_signal,

    -- Severity distribution
    SUM(CASE WHEN current_severity = 'critical' THEN 1 ELSE 0 END) AS n_critical,
    SUM(CASE WHEN current_severity = 'warning' THEN 1 ELSE 0 END) AS n_warning,
    SUM(CASE WHEN current_severity = 'watch' THEN 1 ELSE 0 END) AS n_watch,
    SUM(CASE WHEN current_severity = 'normal' THEN 1 ELSE 0 END) AS n_normal,

    -- Percent healthy
    100.0 * SUM(CASE WHEN current_severity = 'normal' THEN 1 ELSE 0 END) / COUNT(*) AS pct_healthy,

    -- Coupling state distribution
    SUM(CASE WHEN coupling_state = 'strongly_coupled' THEN 1 ELSE 0 END) AS n_strongly_coupled,
    SUM(CASE WHEN coupling_state = 'weakly_coupled' THEN 1 ELSE 0 END) AS n_weakly_coupled,
    SUM(CASE WHEN coupling_state = 'decoupled' THEN 1 ELSE 0 END) AS n_decoupled,

    -- Averages
    AVG(coherence) AS avg_coherence,
    AVG(effective_dim) AS avg_effective_dim,
    AVG(state_distance) AS avg_state_distance

FROM v_orthon_entity_summary;


-- Alerts view
CREATE OR REPLACE VIEW v_orthon_alerts AS
SELECT
    entity_id,
    I,
    'CRITICAL' AS alert_level,
    'ØRTHON SIGNAL: Symplectic structure loss detected' AS alert_message,
    severity_score,
    energy_trend,
    coupling_state,
    state_trend
FROM v_orthon_signal
WHERE orthon_signal = TRUE

UNION ALL

SELECT
    entity_id,
    I,
    'WARNING' AS alert_level,
    'Energy dissipating while state diverging' AS alert_message,
    severity_score,
    energy_trend,
    coupling_state,
    state_trend
FROM v_orthon_signal
WHERE severity = 'warning' AND orthon_signal = FALSE

UNION ALL

SELECT
    entity_id,
    I,
    'WATCH' AS alert_level,
    CASE
        WHEN coherence < 0.4 THEN 'System decoupled (coherence ' || ROUND(coherence::DECIMAL, 2) || ')'
        WHEN coherence_velocity < -0.001 THEN 'Coherence dropping'
        WHEN state_velocity > 0.01 THEN 'State diverging from baseline'
        ELSE 'Early degradation indicator'
    END AS alert_message,
    severity_score,
    energy_trend,
    coupling_state,
    state_trend
FROM v_orthon_signal o
JOIN physics p USING (entity_id, I)
WHERE o.severity = 'watch';


-- Verify
SELECT
    COUNT(*) AS total_points,
    SUM(CASE WHEN orthon_signal THEN 1 ELSE 0 END) AS orthon_signal_points,
    SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) AS critical_points
FROM v_orthon_signal;
