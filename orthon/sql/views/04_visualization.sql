-- ============================================================================
-- ORTHON SQL: 04_visualization.sql
-- ============================================================================
-- Views for UI visualization - NO COMPUTE, just queries
--
-- These views provide data for:
--   - Signal charts
--   - Correlation heatmaps
--   - Causal graphs
--   - Regime timelines
--   - Summary dashboards
-- ============================================================================

-- =========================================================================
-- SIGNAL CHARTS
-- =========================================================================

-- Raw signal data for time series plots
CREATE OR REPLACE VIEW v_chart_signals AS
SELECT
    o.signal_id,
    o.I AS x,
    o.y,
    o.unit,
    s.signal_class
FROM observations o
LEFT JOIN v_signal_class_unit s USING (signal_id)
ORDER BY o.signal_id, o.I;

-- Signal with regime overlay (for highlighting regime changes)
CREATE OR REPLACE VIEW v_chart_signals_regime AS
SELECT
    o.signal_id,
    o.I AS x,
    o.y,
    o.unit,
    d.regime_id,
    d.regime_start_idx,
    d.regime_end_idx
FROM observations o
LEFT JOIN dynamical_systems d ON o.signal_id = d.signal_id
    AND o.row_idx BETWEEN d.regime_start_idx AND d.regime_end_idx
ORDER BY o.signal_id, o.I;

-- =========================================================================
-- CORRELATION HEATMAP
-- =========================================================================

-- Correlation matrix for heatmap visualization
CREATE OR REPLACE VIEW v_chart_correlation_matrix AS
SELECT
    signal_a,
    signal_b,
    instant_correlation AS correlation,
    correlation_strength,
    correlation_direction
FROM v_geometry
ORDER BY signal_a, signal_b;

-- Top correlations for highlighting
CREATE OR REPLACE VIEW v_chart_top_correlations AS
SELECT
    signal_a,
    signal_b,
    instant_correlation AS correlation,
    optimal_lag
FROM behavioral_geometry
WHERE ABS(instant_correlation) > 0.5
ORDER BY ABS(instant_correlation) DESC
LIMIT 20;

-- =========================================================================
-- CAUSAL GRAPH
-- =========================================================================

-- Nodes for causal graph (signals with causal role)
CREATE OR REPLACE VIEW v_graph_nodes AS
SELECT
    signal_id AS id,
    signal_id AS label,
    causal_role,
    n_drives,
    n_driven_by,
    -- Node size based on causal importance
    n_drives + n_driven_by AS causal_weight,
    -- Node color based on role
    CASE causal_role
        WHEN 'SOURCE' THEN '#ff6b6b'   -- Red: drives others
        WHEN 'SINK' THEN '#4ecdc4'     -- Teal: driven by others
        WHEN 'HUB' THEN '#ffe66d'      -- Yellow: both
        WHEN 'ISOLATE' THEN '#95a5a6'  -- Gray: neither
        ELSE '#7f8c8d'
    END AS color
FROM v_causality
WHERE causal_role IS NOT NULL;

-- Edges for causal graph (from lead-lag or granger)
CREATE OR REPLACE VIEW v_graph_edges AS
SELECT
    signal_a AS source,
    signal_b AS target,
    optimal_lag AS weight,
    instant_correlation AS strength
FROM behavioral_geometry
WHERE optimal_lag != 0  -- Only show lead-lag relationships
ORDER BY ABS(instant_correlation) DESC;

-- =========================================================================
-- REGIME TIMELINE
-- =========================================================================

-- Regime changes for timeline visualization
CREATE OR REPLACE VIEW v_chart_regime_timeline AS
SELECT
    signal_id,
    regime_id,
    regime_start_idx,
    regime_end_idx,
    regime_end_idx - regime_start_idx AS regime_duration,
    stability_class,
    lyapunov
FROM v_dynamics
ORDER BY signal_id, regime_start_idx;

-- Regime summary per signal
CREATE OR REPLACE VIEW v_chart_regime_summary AS
SELECT
    signal_id,
    COUNT(DISTINCT regime_id) AS n_regimes,
    AVG(regime_end_idx - regime_start_idx) AS avg_regime_duration,
    MAX(regime_end_idx - regime_start_idx) AS max_regime_duration,
    MIN(regime_end_idx - regime_start_idx) AS min_regime_duration
FROM dynamical_systems
GROUP BY signal_id;

-- =========================================================================
-- SIGNAL CARDS (for dashboard)
-- =========================================================================

-- Signal summary cards
CREATE OR REPLACE VIEW v_dashboard_signal_cards AS
SELECT
    a.signal_id,
    -- Classification
    c.signal_class,
    c.quantity,
    c.unit,
    -- Typology
    a.persistence_class,
    a.entropy_class,
    a.stationarity_class,
    -- Dynamics
    a.stability_class,
    a.regime_id,
    -- Causality
    a.causal_role,
    a.n_drives,
    a.n_driven_by,
    -- Health indicator
    CASE
        WHEN a.stability_class = 'chaotic' THEN 'critical'
        WHEN a.stationarity_class = 'non-stationary' THEN 'warning'
        WHEN a.causal_role = 'SINK' THEN 'monitor'
        ELSE 'normal'
    END AS health_status
FROM v_signal_analysis a
LEFT JOIN v_signal_class_unit c USING (signal_id);

-- =========================================================================
-- SYSTEM OVERVIEW (for main dashboard)
-- =========================================================================

-- Overall system health
CREATE OR REPLACE VIEW v_dashboard_system_health AS
SELECT
    COUNT(*) AS total_signals,
    COUNT(*) FILTER (WHERE stability_class = 'chaotic') AS chaotic_signals,
    COUNT(*) FILTER (WHERE stability_class = 'stable') AS stable_signals,
    COUNT(*) FILTER (WHERE stationarity_class = 'non-stationary') AS nonstationary_signals,
    COUNT(*) FILTER (WHERE causal_role = 'SOURCE') AS source_signals,
    COUNT(*) FILTER (WHERE causal_role = 'SINK') AS sink_signals,
    COUNT(*) FILTER (WHERE causal_role = 'HUB') AS hub_signals,
    -- Overall health score (0-100)
    ROUND(100.0 * (
        COUNT(*) FILTER (WHERE stability_class = 'stable')
    ) / NULLIF(COUNT(*), 0), 1) AS stability_score
FROM v_signal_analysis;

-- Alerts (signals needing attention)
CREATE OR REPLACE VIEW v_dashboard_alerts AS
SELECT
    signal_id,
    'Chaotic behavior detected' AS alert_type,
    'critical' AS severity,
    lyapunov AS metric_value
FROM v_dynamics
WHERE stability_class = 'chaotic'

UNION ALL

SELECT
    signal_id,
    'Non-stationary signal' AS alert_type,
    'warning' AS severity,
    stationarity_pvalue AS metric_value
FROM v_typology
WHERE stationarity_class = 'non-stationary'

ORDER BY severity DESC, signal_id;
