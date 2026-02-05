-- ============================================================================
-- ORTHON SQL: 05_summaries.sql
-- ============================================================================
-- Aggregate summaries for export and display
-- These are the final views consumed by the UI and API
-- ============================================================================

-- =========================================================================
-- LAYER SUMMARIES (for 4-layer display)
-- =========================================================================

-- Layer 1: Signal Typology Summary
CREATE OR REPLACE VIEW v_summary_typology AS
SELECT
    'Signal Typology' AS layer_name,
    'WHAT type of signal is this?' AS layer_question,
    COUNT(*) AS n_signals,
    COUNT(*) FILTER (WHERE persistence_class = 'persistent') AS n_persistent,
    COUNT(*) FILTER (WHERE persistence_class = 'anti-persistent') AS n_anti_persistent,
    COUNT(*) FILTER (WHERE persistence_class = 'random-walk') AS n_random_walk,
    COUNT(*) FILTER (WHERE stationarity_class = 'stationary') AS n_stationary,
    ROUND(AVG(entropy), 3) AS avg_entropy,
    ROUND(AVG(hurst), 3) AS avg_hurst
FROM v_typology;

-- Layer 2: Behavioral Geometry Summary
CREATE OR REPLACE VIEW v_summary_geometry AS
SELECT
    'Behavioral Geometry' AS layer_name,
    'HOW do signals relate to each other?' AS layer_question,
    COUNT(*) AS n_pairs,
    COUNT(*) FILTER (WHERE correlation_strength = 'strong') AS n_strong_correlations,
    COUNT(*) FILTER (WHERE correlation_strength = 'moderate') AS n_moderate_correlations,
    COUNT(*) FILTER (WHERE correlation_strength = 'weak') AS n_weak_correlations,
    ROUND(AVG(ABS(instant_correlation)), 3) AS avg_abs_correlation,
    ROUND(AVG(dtw_distance), 3) AS avg_dtw_distance
FROM v_geometry;

-- Layer 3: Dynamical Systems Summary
CREATE OR REPLACE VIEW v_summary_dynamics AS
SELECT
    'Dynamical Systems' AS layer_name,
    'WHEN and HOW does behavior change?' AS layer_question,
    COUNT(DISTINCT signal_id) AS n_signals,
    COUNT(DISTINCT regime_id) AS n_unique_regimes,
    COUNT(*) FILTER (WHERE stability_class = 'stable') AS n_stable,
    COUNT(*) FILTER (WHERE stability_class = 'chaotic') AS n_chaotic,
    COUNT(*) FILTER (WHERE stability_class = 'edge-of-chaos') AS n_edge,
    ROUND(AVG(lyapunov), 4) AS avg_lyapunov
FROM v_dynamics;

-- Layer 4: Causal Mechanics Summary
CREATE OR REPLACE VIEW v_summary_causality AS
SELECT
    'Causal Mechanics' AS layer_name,
    'WHY do signals behave this way?' AS layer_question,
    COUNT(*) AS n_signals,
    COUNT(*) FILTER (WHERE causal_role = 'SOURCE') AS n_sources,
    COUNT(*) FILTER (WHERE causal_role = 'SINK') AS n_sinks,
    COUNT(*) FILTER (WHERE causal_role = 'HUB') AS n_hubs,
    COUNT(*) FILTER (WHERE causal_role = 'ISOLATE') AS n_isolates,
    SUM(n_drives) AS total_causal_links
FROM v_causality;

-- Combined layer summary
CREATE OR REPLACE VIEW v_summary_all_layers AS
SELECT 1 AS layer_order, * FROM v_summary_typology
UNION ALL
SELECT 2, * FROM v_summary_geometry
UNION ALL
SELECT 3, * FROM v_summary_dynamics
UNION ALL
SELECT 4, * FROM v_summary_causality
ORDER BY layer_order;

-- =========================================================================
-- EXPORT VIEWS (JSON format for API)
-- =========================================================================

-- Signal analysis as JSON
CREATE OR REPLACE VIEW v_export_signal_json AS
SELECT
    signal_id,
    TO_JSON({
        'signal_id': signal_id,
        'classification': {
            'class': signal_class,
            'quantity': quantity,
            'unit': unit
        },
        'typology': {
            'persistence': persistence_class,
            'entropy': entropy_class,
            'stationarity': stationarity_class
        },
        'dynamics': {
            'stability': stability_class,
            'regime_id': regime_id,
            'attractor_dimension': attractor_dimension
        },
        'causality': {
            'role': causal_role,
            'n_drives': n_drives,
            'n_driven_by': n_driven_by
        },
        'health': health_status
    }) AS signal_json
FROM v_dashboard_signal_cards;

-- System summary as JSON
CREATE OR REPLACE VIEW v_export_system_json AS
SELECT TO_JSON({
    'health': (SELECT * FROM v_dashboard_system_health),
    'alerts': (SELECT ARRAY_AGG(* ORDER BY severity DESC) FROM v_dashboard_alerts),
    'layer_summary': (SELECT ARRAY_AGG(* ORDER BY layer_order) FROM v_summary_all_layers)
}) AS system_json;

-- =========================================================================
-- INTERPRETATION HELPERS (for LLM Concierge)
-- =========================================================================

-- Signals that need attention (for LLM to explain)
CREATE OR REPLACE VIEW v_signals_needing_attention AS
SELECT
    signal_id,
    health_status,
    ARRAY_AGG(reason) AS reasons
FROM (
    SELECT signal_id, health_status, 'Chaotic dynamics detected' AS reason
    FROM v_dashboard_signal_cards WHERE stability_class = 'chaotic'
    UNION ALL
    SELECT signal_id, health_status, 'Non-stationary behavior' AS reason
    FROM v_dashboard_signal_cards WHERE stationarity_class = 'non-stationary'
    UNION ALL
    SELECT signal_id, health_status, 'Driving multiple downstream signals' AS reason
    FROM v_dashboard_signal_cards WHERE n_drives > 3
    UNION ALL
    SELECT signal_id, health_status, 'Anti-persistent (mean-reverting)' AS reason
    FROM v_dashboard_signal_cards WHERE persistence_class = 'anti-persistent'
) attention
GROUP BY signal_id, health_status
ORDER BY
    CASE health_status
        WHEN 'critical' THEN 1
        WHEN 'warning' THEN 2
        WHEN 'monitor' THEN 3
        ELSE 4
    END;

-- Key causal chains (for LLM to explain)
CREATE OR REPLACE VIEW v_key_causal_chains AS
SELECT
    source.signal_id AS source_signal,
    sink.signal_id AS sink_signal,
    source.n_drives AS source_influence,
    sink.n_driven_by AS sink_dependency
FROM v_causality source
JOIN v_causality sink ON sink.signal_id = ANY(source.drives_signals)
WHERE source.causal_role = 'SOURCE'
  AND sink.causal_role = 'SINK'
ORDER BY source.n_drives DESC, sink.n_driven_by DESC
LIMIT 10;
