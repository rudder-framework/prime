-- ============================================================================
-- TEST: 06_general_views.sql
-- ============================================================================
-- Tests all dashboard views with pump station sample data
-- ============================================================================

-- Load test data and create observations with canonical schema
CREATE OR REPLACE TABLE observations AS
WITH raw AS (
    SELECT * FROM read_csv('/Users/jasonrudder/Domains/pump_station_24h.csv', auto_detect=true)
),
indexed AS (
    SELECT
        entity_id,
        ROW_NUMBER() OVER (ORDER BY timestamp) AS I,
        flow_gpm, pressure_psi, temp_F, motor_rpm, power_kW
    FROM raw
)
SELECT entity_id, 'flow_gpm' AS signal_id, I, flow_gpm AS y, 'gpm' AS unit FROM indexed
UNION ALL
SELECT entity_id, 'pressure_psi' AS signal_id, I, pressure_psi AS y, 'psi' AS unit FROM indexed
UNION ALL
SELECT entity_id, 'temp_F' AS signal_id, I, temp_F AS y, 'F' AS unit FROM indexed
UNION ALL
SELECT entity_id, 'motor_rpm' AS signal_id, I, motor_rpm AS y, 'rpm' AS unit FROM indexed
UNION ALL
SELECT entity_id, 'power_kW' AS signal_id, I, power_kW AS y, 'kW' AS unit FROM indexed;

SELECT 'observations' AS table_name, COUNT(*) AS n_rows FROM observations;

-- Create mock primitives table (simulating PRISM output)
-- First pass: compute dy
CREATE OR REPLACE TABLE primitives_temp AS
SELECT
    entity_id, signal_id, I, y,
    y - LAG(y) OVER (PARTITION BY signal_id, entity_id ORDER BY I) AS dy
FROM observations;

-- Second pass: compute d2y and add other fields
CREATE OR REPLACE TABLE primitives AS
SELECT
    entity_id, signal_id, I, y, dy,
    dy - LAG(dy) OVER (PARTITION BY signal_id, entity_id ORDER BY I) AS d2y,
    NTILE(3) OVER (PARTITION BY signal_id ORDER BY y) AS regime_id,
    CASE NTILE(3) OVER (PARTITION BY signal_id ORDER BY y)
        WHEN 1 THEN 'low' WHEN 2 THEN 'normal' WHEN 3 THEN 'high'
    END AS regime_label,
    0.5 + RANDOM() * 0.3 AS hurst,
    -0.05 + RANDOM() * 0.15 AS lyapunov,
    0.3 + RANDOM() * 0.5 AS entropy,
    CASE (ABS(HASH(signal_id)) % 4)
        WHEN 0 THEN 'trending' WHEN 1 THEN 'mean_reverting'
        WHEN 2 THEN 'oscillating' ELSE 'stationary'
    END AS behavioral_type,
    CASE (ABS(HASH(signal_id)) % 5)
        WHEN 0 THEN 'SOURCE' WHEN 1 THEN 'SINK' WHEN 2 THEN 'HUB'
        WHEN 3 THEN 'MEDIATOR' ELSE 'ISOLATE'
    END AS causal_role,
    (ABS(HASH(signal_id)) % 5)::INT AS causal_out_degree,
    (ABS(HASH(signal_id || '_in')) % 4)::INT AS causal_in_degree
FROM primitives_temp;

DROP TABLE primitives_temp;

SELECT 'primitives' AS table_name, COUNT(*) AS n_rows FROM primitives;

-- ============================================================================
-- TEST VIEWS
-- ============================================================================

.print
.print ===== v_dataset_overview =====
CREATE OR REPLACE VIEW v_dataset_overview AS
SELECT
    COUNT(DISTINCT entity_id) AS n_entities,
    COUNT(DISTINCT signal_id) AS n_signals,
    COUNT(*) AS n_observations,
    MIN(I) AS index_start,
    MAX(I) AS index_end,
    MAX(I) - MIN(I) AS index_span,
    COUNT(*) / NULLIF(COUNT(DISTINCT signal_id), 0) AS avg_points_per_signal,
    COUNT(DISTINCT entity_id || '|' || signal_id) AS n_unique_series
FROM observations;
SELECT * FROM v_dataset_overview;


.print
.print ===== v_signal_profile =====
CREATE OR REPLACE VIEW v_signal_profile AS
SELECT
    o.signal_id,
    COUNT(*) AS n_points,
    ROUND(AVG(o.y), 2) AS mean,
    ROUND(MEDIAN(o.y), 2) AS median,
    ROUND(STDDEV(o.y), 2) AS std,
    ROUND(MIN(o.y), 2) AS min_val,
    ROUND(MAX(o.y), 2) AS max_val,
    COUNT(DISTINCT o.y) AS n_unique_values,
    ROUND(AVG(ABS(p.dy)), 4) AS avg_velocity,
    CASE
        WHEN COUNT(DISTINCT o.y) <= 10 THEN 'likely_digital'
        WHEN COUNT(DISTINCT o.y)::FLOAT / COUNT(*) < 0.01 THEN 'likely_digital'
        ELSE 'likely_analog'
    END AS inferred_class
FROM observations o
LEFT JOIN primitives p USING (signal_id, entity_id, I)
GROUP BY o.signal_id;
SELECT * FROM v_signal_profile;


.print
.print ===== v_signal_health =====
CREATE OR REPLACE VIEW v_signal_health AS
WITH signal_stats AS (
    SELECT signal_id, COUNT(*) AS n_points,
        COUNT(*) FILTER (WHERE y IS NULL) AS n_nulls,
        AVG(y) AS mean_val, STDDEV(y) AS std_val
    FROM observations GROUP BY signal_id
),
outlier_counts AS (
    SELECT signal_id,
        COUNT(*) FILTER (WHERE ABS(y - mean_val) > 3 * std_val) AS n_outliers_3sigma
    FROM observations o JOIN signal_stats s USING (signal_id)
    GROUP BY signal_id, mean_val, std_val
)
SELECT s.signal_id, s.n_points, s.n_nulls, o.n_outliers_3sigma,
    ROUND(GREATEST(0, 100 - (s.n_nulls::FLOAT / NULLIF(s.n_points, 0) * 100)
        - (o.n_outliers_3sigma::FLOAT / NULLIF(s.n_points, 0) * 50)), 1) AS health_score
FROM signal_stats s
LEFT JOIN outlier_counts o USING (signal_id);
SELECT * FROM v_signal_health;


.print
.print ===== v_signal_correlations =====
CREATE OR REPLACE VIEW v_signal_correlations AS
WITH paired AS (
    SELECT a.signal_id AS signal_a, b.signal_id AS signal_b,
        CORR(a.y, b.y) AS correlation, COUNT(*) AS n_overlap
    FROM observations a
    JOIN observations b ON a.entity_id = b.entity_id AND a.I = b.I AND a.signal_id < b.signal_id
    GROUP BY a.signal_id, b.signal_id
    HAVING COUNT(*) > 30
)
SELECT signal_a, signal_b, ROUND(correlation, 3) AS correlation, n_overlap,
    CASE
        WHEN correlation > 0.8 THEN 'strong_positive'
        WHEN correlation > 0.5 THEN 'moderate_positive'
        WHEN correlation > 0.2 THEN 'weak_positive'
        WHEN correlation > -0.2 THEN 'uncorrelated'
        WHEN correlation > -0.5 THEN 'weak_negative'
        WHEN correlation > -0.8 THEN 'moderate_negative'
        ELSE 'strong_negative'
    END AS relationship
FROM paired WHERE ABS(correlation) > 0.3
ORDER BY ABS(correlation) DESC LIMIT 20;
SELECT * FROM v_signal_correlations;


.print
.print ===== v_regime_summary =====
CREATE OR REPLACE VIEW v_regime_summary AS
SELECT regime_id, regime_label,
    COUNT(DISTINCT signal_id) AS n_signals,
    COUNT(*) AS n_points,
    ROUND(AVG(y), 2) AS mean_value,
    ROUND(STDDEV(y), 2) AS volatility
FROM primitives WHERE regime_id IS NOT NULL
GROUP BY regime_id, regime_label ORDER BY regime_id;
SELECT * FROM v_regime_summary;


.print
.print ===== v_causal_summary =====
CREATE OR REPLACE VIEW v_causal_summary AS
SELECT signal_id, causal_role,
    MAX(causal_out_degree) AS drives_n_signals,
    MAX(causal_in_degree) AS driven_by_n_signals,
    CASE
        WHEN MAX(causal_out_degree) > 0 AND MAX(causal_in_degree) = 0 THEN 'root_cause'
        WHEN MAX(causal_out_degree) = 0 AND MAX(causal_in_degree) > 0 THEN 'effect'
        WHEN MAX(causal_out_degree) > MAX(causal_in_degree) THEN 'driver'
        WHEN MAX(causal_out_degree) < MAX(causal_in_degree) THEN 'follower'
        ELSE 'isolated'
    END AS network_role
FROM primitives
GROUP BY signal_id, causal_role
ORDER BY (MAX(causal_out_degree) + MAX(causal_in_degree)) DESC;
SELECT * FROM v_causal_summary;


.print
.print ===== v_anomalies =====
CREATE OR REPLACE VIEW v_anomalies AS
SELECT 'outlier' AS anomaly_type, o.signal_id, o.I AS index_at,
    ROUND(o.y, 2) AS value, ROUND(ABS(o.y - s.mean) / NULLIF(s.std, 0), 1) AS severity
FROM observations o
JOIN (SELECT signal_id, AVG(y) AS mean, STDDEV(y) AS std FROM observations GROUP BY signal_id) s USING (signal_id)
WHERE ABS(o.y - s.mean) > 3 * s.std
ORDER BY severity DESC LIMIT 20;
SELECT * FROM v_anomalies;


.print
.print ===== v_signal_behavior =====
CREATE OR REPLACE VIEW v_signal_behavior AS
SELECT signal_id,
    ROUND(MAX(hurst), 3) AS hurst,
    ROUND(MAX(lyapunov), 4) AS lyapunov,
    ROUND(MAX(entropy), 3) AS entropy,
    MAX(behavioral_type) AS behavioral_type,
    CASE WHEN MAX(hurst) > 0.6 THEN 'trending'
         WHEN MAX(hurst) < 0.4 THEN 'mean_reverting'
         ELSE 'random_walk' END AS persistence_class,
    CASE WHEN MAX(lyapunov) > 0.1 THEN 'chaotic'
         WHEN MAX(lyapunov) > 0 THEN 'sensitive'
         ELSE 'stable' END AS stability_class,
    COUNT(DISTINCT regime_id) AS n_regimes,
    MAX(causal_role) AS causal_role
FROM primitives GROUP BY signal_id;
SELECT * FROM v_signal_behavior;


.print
.print ===== v_system_dashboard =====
CREATE OR REPLACE VIEW v_system_dashboard AS
SELECT
    (SELECT COUNT(DISTINCT entity_id) FROM observations) AS n_entities,
    (SELECT COUNT(DISTINCT signal_id) FROM observations) AS n_signals,
    (SELECT COUNT(*) FROM observations) AS n_observations,
    (SELECT COUNT(*) FROM v_signal_profile WHERE inferred_class = 'likely_analog') AS n_analog,
    (SELECT COUNT(*) FROM v_signal_profile WHERE inferred_class = 'likely_digital') AS n_digital,
    (SELECT ROUND(AVG(health_score), 1) FROM v_signal_health) AS avg_health_score,
    (SELECT COUNT(*) FROM v_signal_health WHERE health_score < 70) AS n_unhealthy_signals,
    (SELECT COUNT(*) FROM v_anomalies) AS n_outliers,
    (SELECT COUNT(DISTINCT regime_id) FROM primitives) AS n_regimes,
    (SELECT COUNT(*) FROM v_signal_correlations WHERE relationship = 'strong_positive') AS n_strong_correlations;
SELECT * FROM v_system_dashboard;


.print
.print ===== v_insight_cards =====
CREATE OR REPLACE VIEW v_insight_cards AS
SELECT 'volatility' AS insight_type, signal_id,
    signal_id || ' has highest volatility (std=' || std || ')' AS description
FROM v_signal_profile ORDER BY std DESC LIMIT 1;
SELECT * FROM v_insight_cards;

-- Additional insight: strongest correlation
SELECT 'correlation' AS insight_type,
    signal_a || ' <-> ' || signal_b AS signals,
    'Correlation: ' || correlation AS description
FROM v_signal_correlations ORDER BY ABS(correlation) DESC LIMIT 1;


.print
.print ===== v_sparkline_data (sample) =====
CREATE OR REPLACE VIEW v_sparkline_data AS
WITH ranked AS (
    SELECT signal_id, entity_id, I, y,
        ROW_NUMBER() OVER (PARTITION BY signal_id, entity_id ORDER BY I) AS rn,
        COUNT(*) OVER (PARTITION BY signal_id, entity_id) AS total
    FROM observations
)
SELECT signal_id, entity_id, I, y
FROM ranked WHERE rn % GREATEST(1, total / 100) = 1  -- Use = 1 instead of = 0
ORDER BY signal_id, entity_id, I;
SELECT signal_id, COUNT(*) AS sampled_points FROM v_sparkline_data GROUP BY signal_id;


.print
.print ===== v_api_overview =====
CREATE OR REPLACE VIEW v_api_overview AS
SELECT json_object(
    'entities', (SELECT COUNT(DISTINCT entity_id) FROM observations),
    'signals', (SELECT COUNT(DISTINCT signal_id) FROM observations),
    'observations', (SELECT COUNT(*) FROM observations),
    'health', json_object(
        'avg_score', (SELECT ROUND(AVG(health_score), 1) FROM v_signal_health),
        'unhealthy_count', (SELECT COUNT(*) FROM v_signal_health WHERE health_score < 70)
    ),
    'regimes', (SELECT COUNT(DISTINCT regime_id) FROM primitives),
    'anomalies', (SELECT COUNT(*) FROM v_anomalies)
) AS api_json;
SELECT * FROM v_api_overview;


.print
.print ========================================
.print ALL VIEWS TESTED SUCCESSFULLY
.print ========================================
