-- ============================================================================
-- GENERAL SQL VIEWS
-- Domain-agnostic insights from any dataset
-- These power the website dashboard
-- ============================================================================

-- ============================================================================
-- 1. DATA OVERVIEW
-- What's in this dataset at a glance?
-- ============================================================================

CREATE OR REPLACE VIEW v_dataset_overview AS
SELECT
    COUNT(DISTINCT cohort) AS n_entities,
    COUNT(DISTINCT signal_id) AS n_signals,
    COUNT(*) AS n_observations,
    MIN(I) AS index_start,
    MAX(I) AS index_end,
    MAX(I) - MIN(I) AS index_span,
    COUNT(*) / NULLIF(COUNT(DISTINCT signal_id), 0) AS avg_points_per_signal,
    COUNT(DISTINCT cohort || '|' || signal_id) AS n_unique_series
FROM observations;


-- ============================================================================
-- 2. SIGNAL PROFILE
-- Statistical fingerprint of each signal
-- ============================================================================

CREATE OR REPLACE VIEW v_signal_profile AS
SELECT
    signal_id,
    COUNT(*) AS n_points,

    -- Central tendency
    AVG(y) AS mean,
    MEDIAN(y) AS median,
    MODE(y) AS mode,

    -- Spread
    STDDEV(y) AS std,
    MIN(y) AS min_val,
    MAX(y) AS max_val,
    MAX(y) - MIN(y) AS range,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY y) AS q1,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY y) AS q3,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY y) -
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY y) AS iqr,

    -- Shape
    (AVG(y) - MEDIAN(y)) / NULLIF(STDDEV(y), 0) AS skewness_proxy,
    COUNT(DISTINCT y) AS n_unique_values,
    COUNT(DISTINCT y)::FLOAT / NULLIF(COUNT(*), 0) AS unique_ratio,

    -- Dynamics (from primitives if available)
    AVG(ABS(dy)) AS avg_velocity,
    MAX(ABS(dy)) AS max_velocity,
    AVG(ABS(d2y)) AS avg_acceleration,

    -- Classification hints
    CASE
        WHEN COUNT(DISTINCT y) <= 10 THEN 'likely_digital'
        WHEN COUNT(DISTINCT y)::FLOAT / COUNT(*) < 0.01 THEN 'likely_digital'
        ELSE 'likely_analog'
    END AS inferred_class

FROM observations o
LEFT JOIN primitives p USING (signal_id, cohort, I)
GROUP BY signal_id;


-- ============================================================================
-- 3. SIGNAL DEPARTURE
-- Data quality assessment per signal
-- ============================================================================

CREATE OR REPLACE VIEW v_signal_departure AS
WITH signal_stats AS (
    SELECT
        signal_id,
        COUNT(*) AS n_points,
        COUNT(*) FILTER (WHERE y IS NULL) AS n_nulls,
        COUNT(*) FILTER (WHERE y = 0) AS n_zeros,
        AVG(y) AS mean_val,
        STDDEV(y) AS std_val,
        MIN(I) AS first_index,
        MAX(I) AS last_index
    FROM observations
    GROUP BY signal_id
),
outlier_counts AS (
    SELECT
        signal_id,
        COUNT(*) FILTER (WHERE pctile > 0.99 OR pctile < 0.01) AS n_outliers_p99,
        COUNT(*) FILTER (WHERE pctile > 0.999 OR pctile < 0.001) AS n_outliers_p999
    FROM (
        SELECT signal_id, y,
            PERCENT_RANK() OVER (PARTITION BY signal_id ORDER BY y) AS pctile
        FROM observations
    )
    GROUP BY signal_id
),
gap_analysis AS (
    SELECT
        signal_id,
        MAX(I - prev_I) AS max_gap,
        AVG(I - prev_I) AS avg_gap,
        STDDEV(I - prev_I) AS gap_std,
        COUNT(*) FILTER (WHERE (I - prev_I) > 2 * AVG(I - prev_I) OVER (PARTITION BY signal_id)) AS n_gaps
    FROM (
        SELECT signal_id, I, LAG(I) OVER (PARTITION BY signal_id ORDER BY I) AS prev_I
        FROM observations
    ) t
    WHERE prev_I IS NOT NULL
    GROUP BY signal_id
)
SELECT
    s.signal_id,
    s.n_points,

    -- Completeness
    s.n_nulls,
    s.n_nulls::FLOAT / NULLIF(s.n_points, 0) AS null_rate,

    -- Zeros (might indicate sensor issues)
    s.n_zeros,
    s.n_zeros::FLOAT / NULLIF(s.n_points, 0) AS zero_rate,

    -- Outliers (percentile-based)
    o.n_outliers_p99,
    o.n_outliers_p99::FLOAT / NULLIF(s.n_points, 0) AS outlier_rate_p99,
    o.n_outliers_p999,

    -- Gaps
    g.max_gap,
    g.avg_gap,
    g.n_gaps,

    -- Coverage
    s.first_index,
    s.last_index,
    s.last_index - s.first_index AS coverage_span,

    -- Overall departure score (0-100)
    GREATEST(0, 100
        - (s.n_nulls::FLOAT / NULLIF(s.n_points, 0) * 100)  -- Penalize nulls
        - (o.n_outliers_p99::FLOAT / NULLIF(s.n_points, 0) * 50)  -- Penalize outliers
        - (CASE WHEN g.n_gaps > 10 THEN 20 ELSE g.n_gaps * 2 END)  -- Penalize gaps
    ) AS departure_score

FROM signal_stats s
LEFT JOIN outlier_counts o USING (signal_id)
LEFT JOIN gap_analysis g USING (signal_id);


-- ============================================================================
-- 4. INDEX COVERAGE
-- Time/space coverage analysis
-- ============================================================================

CREATE OR REPLACE VIEW v_index_coverage AS
WITH bounds AS (
    SELECT
        cohort,
        signal_id,
        MIN(I) AS start_index,
        MAX(I) AS end_index,
        COUNT(*) AS n_points,
        MAX(I) - MIN(I) AS span
    FROM observations
    GROUP BY cohort, signal_id
),
global_bounds AS (
    SELECT MIN(start_index) AS global_start, MAX(end_index) AS global_end
    FROM bounds
)
SELECT
    b.cohort,
    b.signal_id,
    b.start_index,
    b.end_index,
    b.span,
    b.n_points,
    b.n_points::FLOAT / NULLIF(b.span, 0) AS density,

    -- Coverage relative to global bounds
    (b.start_index - g.global_start) AS start_offset,
    (g.global_end - b.end_index) AS end_offset,
    b.span / NULLIF(g.global_end - g.global_start, 0) AS coverage_fraction

FROM bounds b
CROSS JOIN global_bounds g;


-- ============================================================================
-- 5. SIGNAL CORRELATIONS (Top pairs)
-- Which signals move together?
-- ============================================================================

CREATE OR REPLACE VIEW v_signal_correlations AS
WITH paired AS (
    SELECT
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        CORR(a.y, b.y) AS correlation,
        COUNT(*) AS n_overlap
    FROM observations a
    JOIN observations b
        ON a.cohort = b.cohort
        AND a.I = b.I
        AND a.signal_id < b.signal_id
    GROUP BY a.signal_id, b.signal_id
    HAVING COUNT(*) > 30  -- Minimum overlap for meaningful correlation
)
SELECT
    signal_a,
    signal_b,
    correlation,
    ABS(correlation) AS abs_correlation,
    n_overlap,

    -- Classify relationship
    CASE
        WHEN correlation > 0.8 THEN 'strong_positive'
        WHEN correlation > 0.5 THEN 'moderate_positive'
        WHEN correlation > 0.2 THEN 'weak_positive'
        WHEN correlation > -0.2 THEN 'uncorrelated'
        WHEN correlation > -0.5 THEN 'weak_negative'
        WHEN correlation > -0.8 THEN 'moderate_negative'
        ELSE 'strong_negative'
    END AS relationship

FROM paired
WHERE ABS(correlation) > 0.3  -- Only show meaningful correlations
ORDER BY ABS(correlation) DESC
LIMIT 50;


-- ============================================================================
-- 6. REGIME SUMMARY
-- What operating states exist in the data?
-- ============================================================================

CREATE OR REPLACE VIEW v_regime_summary AS
SELECT
    regime_id,
    regime_label,

    -- Size
    COUNT(DISTINCT signal_id) AS n_signals_affected,
    COUNT(DISTINCT cohort) AS n_entities_affected,
    COUNT(*) AS n_points,

    -- Timing
    MIN(I) AS regime_start,
    MAX(I) AS regime_end,
    MAX(I) - MIN(I) AS duration,

    -- Characteristics
    AVG(y) AS mean_value,
    STDDEV(y) AS volatility,
    AVG(ABS(dy)) AS avg_velocity,

    -- Percentage of total data
    COUNT(*)::FLOAT / (SELECT COUNT(*) FROM primitives) AS fraction_of_data

FROM primitives
WHERE regime_id IS NOT NULL
GROUP BY regime_id, regime_label
ORDER BY MIN(I);


-- ============================================================================
-- 7. REGIME TRANSITIONS
-- When and how did the system change states?
-- ============================================================================

CREATE OR REPLACE VIEW v_regime_transitions AS
WITH transitions AS (
    SELECT
        signal_id,
        cohort,
        I AS transition_at,
        LAG(regime_id) OVER w AS from_regime,
        regime_id AS to_regime,
        LAG(regime_label) OVER w AS from_label,
        regime_label AS to_label,
        y - LAG(y) OVER w AS value_jump,
        dy - LAG(dy) OVER w AS velocity_jump
    FROM primitives
    WHERE regime_id IS NOT NULL
    WINDOW w AS (PARTITION BY signal_id, cohort ORDER BY I)
)
SELECT
    transition_at,
    from_regime,
    to_regime,
    from_label,
    to_label,
    COUNT(DISTINCT signal_id) AS n_signals_transitioning,
    AVG(ABS(value_jump)) AS avg_value_jump,
    AVG(ABS(velocity_jump)) AS avg_velocity_jump,

    -- Transition type
    CASE
        WHEN AVG(value_jump) > 0 THEN 'upward'
        WHEN AVG(value_jump) < 0 THEN 'downward'
        ELSE 'lateral'
    END AS direction

FROM transitions
WHERE from_regime IS NOT NULL
  AND from_regime != to_regime
GROUP BY transition_at, from_regime, to_regime, from_label, to_label
ORDER BY transition_at;


-- ============================================================================
-- 8. CAUSAL NETWORK SUMMARY
-- What drives what in this system?
-- ============================================================================

CREATE OR REPLACE VIEW v_causal_summary AS
SELECT
    signal_id,

    -- Causal role
    causal_role,

    -- Influence metrics
    causal_out_degree AS drives_n_signals,
    causal_in_degree AS driven_by_n_signals,
    causal_out_degree - causal_in_degree AS net_influence,

    -- Classification
    CASE
        WHEN causal_out_degree > 0 AND causal_in_degree = 0 THEN 'root_cause'
        WHEN causal_out_degree = 0 AND causal_in_degree > 0 THEN 'effect'
        WHEN causal_out_degree > causal_in_degree THEN 'driver'
        WHEN causal_out_degree < causal_in_degree THEN 'follower'
        WHEN causal_out_degree > 0 AND causal_in_degree > 0 THEN 'mediator'
        ELSE 'isolated'
    END AS network_role

FROM primitives
GROUP BY signal_id, causal_role, causal_out_degree, causal_in_degree
ORDER BY (causal_out_degree + causal_in_degree) DESC;


-- ============================================================================
-- 9. DEVIATION FEED
-- What's unusual in this data? (For alerts panel)
-- ============================================================================

CREATE OR REPLACE VIEW v_deviations AS

-- Outlier values (IQR-based, no sigma thresholds)
SELECT
    'outlier' AS deviation_type,
    signal_id,
    cohort,
    I AS index_at,
    y AS value,
    'Value ' || ROUND(y, 2) || ' exceeds IQR bounds (Q1=' || ROUND(s.q1, 2) || ', Q3=' || ROUND(s.q3, 2) || ')' AS description,
    GREATEST(
        CASE WHEN y > s.q3 THEN (y - s.q3) / NULLIF(s.iqr, 0) ELSE 0 END,
        CASE WHEN y < s.q1 THEN (s.q1 - y) / NULLIF(s.iqr, 0) ELSE 0 END
    ) AS severity
FROM observations o
JOIN (
    SELECT signal_id,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY y) AS q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY y) AS q3,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY y) -
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY y) AS iqr
    FROM observations GROUP BY signal_id
) s USING (signal_id)
WHERE y > s.q3 + 3.0 * s.iqr OR y < s.q1 - 3.0 * s.iqr

UNION ALL

-- Regime changes
SELECT
    'regime_change' AS deviation_type,
    signal_id,
    cohort,
    I AS index_at,
    y AS value,
    'Regime changed from ' || COALESCE(LAG(regime_label) OVER w, 'unknown') || ' to ' || regime_label AS description,
    1.0 AS severity
FROM primitives
WHERE regime_id != LAG(regime_id) OVER w
WINDOW w AS (PARTITION BY signal_id, cohort ORDER BY I)

UNION ALL

-- Velocity spikes
SELECT
    'velocity_spike' AS deviation_type,
    signal_id,
    cohort,
    I AS index_at,
    y AS value,
    'Sudden change: dy = ' || ROUND(dy, 3) AS description,
    ABS(dy) / NULLIF(AVG(ABS(dy)) OVER (PARTITION BY signal_id), 0) AS severity
FROM primitives
WHERE ABS(dy) > 5 * AVG(ABS(dy)) OVER (PARTITION BY signal_id)

ORDER BY severity DESC
LIMIT 100;


-- ============================================================================
-- 10. ENTITY COMPARISON
-- How do different entities (engines, patients, etc.) compare?
-- ============================================================================

CREATE OR REPLACE VIEW v_entity_comparison AS
WITH entity_stats AS (
    SELECT
        cohort,
        signal_id,
        AVG(y) AS mean_val,
        STDDEV(y) AS std_val,
        MIN(y) AS min_val,
        MAX(y) AS max_val,
        COUNT(*) AS n_points
    FROM observations
    GROUP BY cohort, signal_id
),
entity_ranked AS (
    SELECT
        cohort,
        signal_id,
        mean_val,
        std_val,
        -- Percentile rank among all entities for this signal
        PERCENT_RANK() OVER (
            PARTITION BY signal_id
            ORDER BY mean_val
        ) AS entity_pctile
    FROM entity_stats
)
SELECT
    cohort,
    signal_id,
    mean_val,
    std_val,

    -- Fleet percentile rank for this entity's mean
    entity_pctile,

    -- Is this entity unusual? (based on fleet percentile)
    CASE
        WHEN entity_pctile > 0.975 OR entity_pctile < 0.025 THEN 'unusual'
        WHEN entity_pctile > 0.85 OR entity_pctile < 0.15 THEN 'different'
        ELSE 'typical'
    END AS entity_status

FROM entity_ranked
ORDER BY ABS(entity_pctile - 0.5) DESC;


-- ============================================================================
-- 11. SIGNAL BEHAVIOR SUMMARY
-- One-row-per-signal behavioral classification
-- ============================================================================

CREATE OR REPLACE VIEW v_signal_behavior AS
SELECT
    p.signal_id,

    -- From primitives (Engines computed)
    MAX(p.hurst) AS hurst,
    MAX(p.lyapunov) AS lyapunov,
    MAX(p.entropy) AS entropy,
    MAX(p.behavioral_type) AS behavioral_type,

    -- Derived classifications
    CASE
        WHEN MAX(p.hurst) > 0.6 THEN 'trending'
        WHEN MAX(p.hurst) < 0.4 THEN 'mean_reverting'
        ELSE 'random_walk'
    END AS persistence_class,

    CASE
        WHEN MAX(p.lyapunov) > 0.1 THEN 'chaotic'
        WHEN MAX(p.lyapunov) > 0 THEN 'sensitive'
        ELSE 'stable'
    END AS stability_class,

    CASE
        WHEN MAX(p.entropy) > 0.8 THEN 'high_complexity'
        WHEN MAX(p.entropy) > 0.4 THEN 'moderate_complexity'
        ELSE 'low_complexity'
    END AS complexity_class,

    -- Regime activity
    COUNT(DISTINCT p.regime_id) AS n_regimes,

    -- Causal role
    MAX(p.causal_role) AS causal_role

FROM primitives p
GROUP BY p.signal_id;


-- ============================================================================
-- 12. SYSTEM DEPARTURE DASHBOARD
-- Overall system status at a glance
-- ============================================================================

CREATE OR REPLACE VIEW v_system_dashboard AS
SELECT
    -- Data summary
    (SELECT COUNT(DISTINCT cohort) FROM observations) AS n_entities,
    (SELECT COUNT(DISTINCT signal_id) FROM observations) AS n_signals,
    (SELECT COUNT(*) FROM observations) AS n_observations,

    -- Signal classification breakdown
    (SELECT COUNT(*) FROM v_signal_profile WHERE inferred_class = 'likely_analog') AS n_analog,
    (SELECT COUNT(*) FROM v_signal_profile WHERE inferred_class = 'likely_digital') AS n_digital,

    -- Departure summary
    (SELECT AVG(departure_score) FROM v_signal_departure) AS avg_departure_score,
    (SELECT COUNT(*) FROM v_signal_departure WHERE departure_score < 70) AS n_departed_signals,

    -- Anomaly counts
    (SELECT COUNT(*) FROM v_deviations WHERE deviation_type = 'outlier') AS n_outliers,
    (SELECT COUNT(*) FROM v_deviations WHERE deviation_type = 'regime_change') AS n_regime_changes,
    (SELECT COUNT(*) FROM v_deviations WHERE deviation_type = 'velocity_spike') AS n_velocity_spikes,

    -- Regime summary
    (SELECT COUNT(DISTINCT regime_id) FROM primitives WHERE regime_id IS NOT NULL) AS n_regimes,

    -- Correlation summary
    (SELECT COUNT(*) FROM v_signal_correlations WHERE relationship = 'strong_positive') AS n_strong_correlations,

    -- Causal structure
    (SELECT COUNT(*) FROM v_causal_summary WHERE network_role = 'root_cause') AS n_root_causes,
    (SELECT COUNT(*) FROM v_causal_summary WHERE network_role = 'effect') AS n_effects;


-- ============================================================================
-- 13. INSIGHT CARDS
-- Pre-generated insights for the dashboard (natural language ready)
-- ============================================================================

CREATE OR REPLACE VIEW v_insight_cards AS

-- Most volatile signal
SELECT
    'volatility' AS insight_type,
    signal_id,
    'Highest volatility' AS title,
    signal_id || ' has the highest volatility (std=' || ROUND(std, 2) || ')' AS description,
    std AS metric
FROM v_signal_profile
ORDER BY std DESC
LIMIT 1

UNION ALL

-- Strongest correlation
SELECT
    'correlation' AS insight_type,
    signal_a || ' ↔ ' || signal_b AS signal_id,
    'Strongest relationship' AS title,
    signal_a || ' and ' || signal_b || ' are strongly ' ||
        CASE WHEN correlation > 0 THEN 'positively' ELSE 'negatively' END ||
        ' correlated (r=' || ROUND(correlation, 2) || ')' AS description,
    ABS(correlation) AS metric
FROM v_signal_correlations
ORDER BY ABS(correlation) DESC
LIMIT 1

UNION ALL

-- Primary driver (highest causal influence)
SELECT
    'causality' AS insight_type,
    signal_id,
    'Primary driver' AS title,
    signal_id || ' drives ' || drives_n_signals || ' other signals' AS description,
    drives_n_signals AS metric
FROM v_causal_summary
WHERE drives_n_signals > 0
ORDER BY drives_n_signals DESC
LIMIT 1

UNION ALL

-- Most regime changes
SELECT
    'regime' AS insight_type,
    signal_id,
    'Most dynamic' AS title,
    signal_id || ' experienced ' || n_regimes || ' regime changes' AS description,
    n_regimes AS metric
FROM v_signal_behavior
ORDER BY n_regimes DESC
LIMIT 1

UNION ALL

-- Most departed signal
SELECT
    'departure' AS insight_type,
    signal_id,
    'Needs attention' AS title,
    signal_id || ' has data quality issues (score=' || ROUND(departure_score, 0) || ')' AS description,
    100 - departure_score AS metric
FROM v_signal_departure
ORDER BY departure_score ASC
LIMIT 1;


-- ============================================================================
-- 14. TIME SERIES SAMPLES
-- Sample data for sparklines / mini-charts
-- ============================================================================

CREATE OR REPLACE VIEW v_sparkline_data AS
WITH ranked AS (
    SELECT
        signal_id,
        cohort,
        I,
        y,
        ROW_NUMBER() OVER (PARTITION BY signal_id, cohort ORDER BY I) AS rn,
        COUNT(*) OVER (PARTITION BY signal_id, cohort) AS total
    FROM observations
)
SELECT
    signal_id,
    cohort,
    I,
    y
FROM ranked
WHERE rn % GREATEST(1, total / 100) = 0  -- Sample ~100 points per series
ORDER BY signal_id, cohort, I;


-- ============================================================================
-- 15. EXPORT: JSON-READY VIEWS
-- Pre-formatted for API responses
-- ============================================================================

CREATE OR REPLACE VIEW v_api_overview AS
SELECT json_object(
    'entities': (SELECT COUNT(DISTINCT cohort) FROM observations),
    'signals': (SELECT COUNT(DISTINCT signal_id) FROM observations),
    'observations': (SELECT COUNT(*) FROM observations),
    'index_range': json_object(
        'start': (SELECT MIN(I) FROM observations),
        'end': (SELECT MAX(I) FROM observations)
    ),
    'departure': json_object(
        'avg_score': (SELECT ROUND(AVG(departure_score), 1) FROM v_signal_departure),
        'departed_count': (SELECT COUNT(*) FROM v_signal_departure WHERE departure_score < 70)
    ),
    'regimes': (SELECT COUNT(DISTINCT regime_id) FROM primitives WHERE regime_id IS NOT NULL),
    'deviations': (SELECT COUNT(*) FROM v_deviations)
);


-- ============================================================================
-- VIEW SUMMARY
-- ============================================================================
--
-- v_dataset_overview      - What's in this dataset?
-- v_signal_profile        - Statistical fingerprint per signal
-- v_signal_departure      - Data quality assessment
-- v_index_coverage        - Time/space coverage analysis
-- v_signal_correlations   - Top correlated pairs
-- v_regime_summary        - Operating states
-- v_regime_transitions    - State changes
-- v_causal_summary        - What drives what
-- v_deviations              - Unusual events feed
-- v_entity_comparison     - Cross-entity analysis
-- v_signal_behavior       - Behavioral classification
-- v_system_dashboard      - One-row system status
-- v_insight_cards         - Pre-generated insights
-- v_sparkline_data        - Mini-chart data
-- v_api_overview          - JSON-ready summary
--
-- ============================================================================
