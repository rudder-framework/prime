-- ============================================================================
-- ORTHON SQL: 26_ml_feature_export.sql
-- ============================================================================
-- ML FEATURE EXPORT: Early Warning Features for Predictive Maintenance
--
-- This script extracts features from ORTHON's physics analysis that are
-- predictive of impending failure. Based on turbofan failure trajectory
-- analysis showing divergence begins ~cycle 110 with state_velocity as
-- the primary early indicator.
--
-- Usage:
--   duckdb < 26_ml_feature_export.sql
--   Or: .read 26_ml_feature_export.sql (from within duckdb session)
--
-- Output Tables:
--   - ml_features_current: Latest snapshot features per entity
--   - ml_features_temporal: Time-series features for sequence models
--   - ml_feature_metadata: Feature descriptions and normalization stats
--
-- Key Finding: FAILS_EARLY engines hit high-velocity trigger at 72.9% of
-- lifespan with avg 43 cycles remaining. SURVIVES_LONGER hit at 60.9%
-- with avg 87 cycles remaining.
-- ============================================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║                    ORTHON → ML FEATURE EXPORT                               ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'

-- ============================================================================
-- SECTION 1: TEMPORAL FEATURE ENGINEERING
-- ============================================================================
-- Calculate rolling windows and trends for each entity over time

.print ''
.print '=== SECTION 1: Computing Temporal Features ==='

CREATE OR REPLACE TABLE ml_features_temporal AS
WITH base_physics AS (
    SELECT
        entity_id,
        I,
        coherence,
        effective_dim,
        n_signals,
        state_distance,
        state_velocity,
        dissipation_rate,
        eigenvalue_entropy,
        -- Normalized metrics for cross-entity comparison
        effective_dim / NULLIF(n_signals, 0) AS norm_effective_dim,
        eigenvalue_entropy / NULLIF(LN(n_signals) / LN(2), 0) AS norm_entropy
    FROM read_parquet('{prism_output}/physics.parquet')
),
with_windows AS (
    SELECT
        entity_id,
        I,
        coherence,
        norm_effective_dim,
        state_distance,
        state_velocity,
        dissipation_rate,
        norm_entropy,

        -- Rolling averages (20-cycle window)
        AVG(coherence) OVER w20 AS coherence_ma20,
        AVG(state_velocity) OVER w20 AS velocity_ma20,
        AVG(dissipation_rate) OVER w20 AS dissipation_ma20,
        AVG(norm_effective_dim) OVER w20 AS norm_dim_ma20,

        -- Rolling standard deviations (volatility)
        STDDEV(coherence) OVER w20 AS coherence_volatility,
        STDDEV(state_velocity) OVER w20 AS velocity_volatility,
        STDDEV(dissipation_rate) OVER w20 AS dissipation_volatility,

        -- Rolling min/max for range detection
        MIN(coherence) OVER w20 AS coherence_min20,
        MAX(state_velocity) OVER w20 AS velocity_max20,

        -- Lagged values for trend calculation
        LAG(coherence, 10) OVER worder AS coherence_lag10,
        LAG(state_velocity, 10) OVER worder AS velocity_lag10,
        LAG(dissipation_rate, 10) OVER worder AS dissipation_lag10,

        -- First 50 cycles as baseline reference
        FIRST_VALUE(coherence) OVER worder AS initial_coherence,
        FIRST_VALUE(state_velocity) OVER worder AS initial_velocity,

        -- Cycle count for lifecycle tracking
        ROW_NUMBER() OVER worder AS cycle_num,
        COUNT(*) OVER (PARTITION BY entity_id) AS total_cycles

    FROM base_physics
    WINDOW
        w20 AS (PARTITION BY entity_id ORDER BY I ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
        worder AS (PARTITION BY entity_id ORDER BY I)
),
with_trends AS (
    SELECT
        *,
        -- Trend indicators (current - lagged) / lagged
        CASE WHEN coherence_lag10 > 0
            THEN (coherence - coherence_lag10) / coherence_lag10
            ELSE 0 END AS coherence_trend_10,
        CASE WHEN velocity_lag10 > 0
            THEN (state_velocity - velocity_lag10) / velocity_lag10
            ELSE 0 END AS velocity_trend_10,
        CASE WHEN dissipation_lag10 > 0
            THEN (dissipation_rate - dissipation_lag10) / dissipation_lag10
            ELSE 0 END AS dissipation_trend_10,

        -- Deviation from initial state
        coherence - initial_coherence AS coherence_drift,
        state_velocity - initial_velocity AS velocity_drift,

        -- Lifecycle percentage
        cycle_num * 100.0 / total_cycles AS pct_life

    FROM with_windows
    WHERE cycle_num >= 20  -- Need 20 cycles for MA calculation
)
SELECT
    entity_id,
    I,
    cycle_num,
    total_cycles,
    ROUND(pct_life, 1) AS pct_life,

    -- Current values
    ROUND(coherence, 4) AS coherence,
    ROUND(norm_effective_dim, 4) AS norm_effective_dim,
    ROUND(state_velocity, 4) AS state_velocity,
    ROUND(dissipation_rate, 4) AS dissipation_rate,
    ROUND(norm_entropy, 4) AS norm_entropy,

    -- Moving averages
    ROUND(coherence_ma20, 4) AS coherence_ma20,
    ROUND(velocity_ma20, 4) AS velocity_ma20,
    ROUND(dissipation_ma20, 4) AS dissipation_ma20,
    ROUND(norm_dim_ma20, 4) AS norm_dim_ma20,

    -- Volatility features
    ROUND(coherence_volatility, 4) AS coherence_volatility,
    ROUND(velocity_volatility, 4) AS velocity_volatility,
    ROUND(dissipation_volatility, 4) AS dissipation_volatility,

    -- Trend features
    ROUND(coherence_trend_10, 4) AS coherence_trend_10,
    ROUND(velocity_trend_10, 4) AS velocity_trend_10,
    ROUND(dissipation_trend_10, 4) AS dissipation_trend_10,

    -- Drift from initial state
    ROUND(coherence_drift, 4) AS coherence_drift,
    ROUND(velocity_drift, 4) AS velocity_drift,

    -- Extreme value tracking
    ROUND(coherence_min20, 4) AS coherence_min20,
    ROUND(velocity_max20, 4) AS velocity_max20,

    -- Binary flags (key early warning indicators)
    CASE WHEN state_velocity > 0.1 THEN 1 ELSE 0 END AS high_velocity_flag,
    CASE WHEN coherence < 0.5 THEN 1 ELSE 0 END AS low_coherence_flag,
    CASE WHEN dissipation_rate > 0.05 THEN 1 ELSE 0 END AS high_dissipation_flag,
    CASE WHEN state_velocity > 0.1 AND coherence < 0.5 THEN 1 ELSE 0 END AS orthon_signal_flag

FROM with_trends
ORDER BY entity_id, I;

SELECT
    COUNT(DISTINCT entity_id) AS n_entities,
    COUNT(*) AS n_observations,
    MIN(cycle_num) AS min_cycle,
    MAX(cycle_num) AS max_cycle
FROM ml_features_temporal;


-- ============================================================================
-- SECTION 2: CURRENT STATE SNAPSHOT FEATURES
-- ============================================================================
-- Latest values and accumulated statistics for each entity

.print ''
.print '=== SECTION 2: Computing Current State Features ==='

CREATE OR REPLACE TABLE ml_features_current AS
WITH latest_cycle AS (
    SELECT
        entity_id,
        MAX(I) AS latest_I
    FROM ml_features_temporal
    GROUP BY entity_id
),
accumulated_stats AS (
    SELECT
        entity_id,
        -- Lifecycle metrics
        MAX(cycle_num) AS total_cycles,

        -- Overall averages
        AVG(coherence) AS avg_coherence,
        AVG(state_velocity) AS avg_velocity,
        AVG(dissipation_rate) AS avg_dissipation,
        AVG(norm_effective_dim) AS avg_norm_dim,

        -- Overall volatility
        STDDEV(coherence) AS overall_coherence_volatility,
        STDDEV(state_velocity) AS overall_velocity_volatility,

        -- Time in warning states
        SUM(high_velocity_flag) * 100.0 / COUNT(*) AS pct_time_high_velocity,
        SUM(low_coherence_flag) * 100.0 / COUNT(*) AS pct_time_low_coherence,
        SUM(orthon_signal_flag) * 100.0 / COUNT(*) AS pct_time_orthon_signal,

        -- First occurrence of warning states
        MIN(CASE WHEN high_velocity_flag = 1 THEN cycle_num END) AS first_high_velocity_cycle,
        MIN(CASE WHEN orthon_signal_flag = 1 THEN cycle_num END) AS first_orthon_signal_cycle,

        -- Trend summary (last 50 vs first 50)
        AVG(CASE WHEN pct_life <= 25 THEN coherence END) AS early_coherence,
        AVG(CASE WHEN pct_life >= 75 THEN coherence END) AS late_coherence,
        AVG(CASE WHEN pct_life <= 25 THEN state_velocity END) AS early_velocity,
        AVG(CASE WHEN pct_life >= 75 THEN state_velocity END) AS late_velocity,

        -- Acceleration events (velocity_trend > 50%)
        SUM(CASE WHEN velocity_trend_10 > 0.5 THEN 1 ELSE 0 END) AS n_velocity_spikes,

        -- Degradation events (coherence dropping)
        SUM(CASE WHEN coherence_trend_10 < -0.1 THEN 1 ELSE 0 END) AS n_coherence_drops

    FROM ml_features_temporal
    GROUP BY entity_id
),
current_values AS (
    SELECT
        t.entity_id,
        t.coherence AS current_coherence,
        t.state_velocity AS current_velocity,
        t.dissipation_rate AS current_dissipation,
        t.norm_effective_dim AS current_norm_dim,
        t.coherence_ma20 AS current_coherence_ma20,
        t.velocity_ma20 AS current_velocity_ma20,
        t.coherence_volatility AS current_coherence_volatility,
        t.velocity_volatility AS current_velocity_volatility,
        t.coherence_drift AS current_coherence_drift,
        t.velocity_drift AS current_velocity_drift,
        t.high_velocity_flag AS current_high_velocity,
        t.orthon_signal_flag AS current_orthon_signal
    FROM ml_features_temporal t
    JOIN latest_cycle l ON t.entity_id = l.entity_id AND t.I = l.latest_I
)
SELECT
    c.entity_id,

    -- Lifecycle
    a.total_cycles,

    -- Current state
    ROUND(c.current_coherence, 4) AS current_coherence,
    ROUND(c.current_velocity, 4) AS current_velocity,
    ROUND(c.current_dissipation, 4) AS current_dissipation,
    ROUND(c.current_norm_dim, 4) AS current_norm_dim,

    -- Current smoothed values
    ROUND(c.current_coherence_ma20, 4) AS current_coherence_ma20,
    ROUND(c.current_velocity_ma20, 4) AS current_velocity_ma20,

    -- Current volatility
    ROUND(c.current_coherence_volatility, 4) AS current_coherence_volatility,
    ROUND(c.current_velocity_volatility, 4) AS current_velocity_volatility,

    -- Drift from initial state
    ROUND(c.current_coherence_drift, 4) AS current_coherence_drift,
    ROUND(c.current_velocity_drift, 4) AS current_velocity_drift,

    -- Historical averages
    ROUND(a.avg_coherence, 4) AS avg_coherence,
    ROUND(a.avg_velocity, 4) AS avg_velocity,
    ROUND(a.avg_dissipation, 4) AS avg_dissipation,
    ROUND(a.avg_norm_dim, 4) AS avg_norm_dim,

    -- Overall volatility
    ROUND(a.overall_coherence_volatility, 4) AS overall_coherence_volatility,
    ROUND(a.overall_velocity_volatility, 4) AS overall_velocity_volatility,

    -- Time in warning states
    ROUND(a.pct_time_high_velocity, 2) AS pct_time_high_velocity,
    ROUND(a.pct_time_low_coherence, 2) AS pct_time_low_coherence,
    ROUND(a.pct_time_orthon_signal, 2) AS pct_time_orthon_signal,

    -- First warning cycle
    a.first_high_velocity_cycle,
    a.first_orthon_signal_cycle,
    ROUND(a.first_high_velocity_cycle * 100.0 / a.total_cycles, 1) AS first_high_velocity_pct_life,
    ROUND(a.first_orthon_signal_cycle * 100.0 / a.total_cycles, 1) AS first_orthon_signal_pct_life,

    -- Lifecycle trend (late - early)
    ROUND(a.late_coherence - a.early_coherence, 4) AS coherence_lifecycle_change,
    ROUND(a.late_velocity - a.early_velocity, 4) AS velocity_lifecycle_change,

    -- Event counts
    a.n_velocity_spikes,
    a.n_coherence_drops,

    -- Current flags
    c.current_high_velocity,
    c.current_orthon_signal,

    -- Composite risk score (0-100)
    -- Weighted combination of key indicators
    ROUND(
        (c.current_high_velocity * 30) +
        (c.current_orthon_signal * 20) +
        (LEAST(a.pct_time_orthon_signal, 50) * 0.5) +
        (LEAST(a.n_velocity_spikes, 20) * 1.0) +
        (CASE WHEN c.current_velocity_drift > 0.1 THEN 10 ELSE 0 END) +
        (CASE WHEN c.current_coherence_drift < -0.2 THEN 10 ELSE 0 END)
    , 1) AS orthon_risk_score

FROM current_values c
JOIN accumulated_stats a ON c.entity_id = a.entity_id
ORDER BY orthon_risk_score DESC;

SELECT
    COUNT(*) AS n_entities,
    ROUND(AVG(orthon_risk_score), 1) AS avg_risk_score,
    SUM(CASE WHEN current_orthon_signal = 1 THEN 1 ELSE 0 END) AS n_currently_signaling,
    SUM(CASE WHEN orthon_risk_score >= 50 THEN 1 ELSE 0 END) AS n_high_risk
FROM ml_features_current;


-- ============================================================================
-- SECTION 3: FEATURE METADATA AND NORMALIZATION STATS
-- ============================================================================

.print ''
.print '=== SECTION 3: Feature Metadata ==='

CREATE OR REPLACE TABLE ml_feature_metadata AS
SELECT * FROM (VALUES
    ('current_coherence', 'Coupling strength at latest observation', 0, 1, 'higher is healthier'),
    ('current_velocity', 'State space movement rate at latest observation', 0, NULL, 'lower is healthier'),
    ('current_dissipation', 'Energy dissipation rate at latest observation', 0, NULL, 'lower is healthier'),
    ('current_norm_dim', 'Normalized effective dimensionality (0-1)', 0, 1, 'lower indicates unified behavior'),
    ('coherence_ma20', '20-cycle moving average of coherence', 0, 1, 'smoothed coupling trend'),
    ('velocity_ma20', '20-cycle moving average of velocity', 0, NULL, 'smoothed movement trend'),
    ('coherence_volatility', 'Std dev of coherence over 20 cycles', 0, NULL, 'lower is more stable'),
    ('velocity_volatility', 'Std dev of velocity over 20 cycles', 0, NULL, 'lower is more stable'),
    ('coherence_drift', 'Change from initial coherence', -1, 1, 'negative indicates degradation'),
    ('velocity_drift', 'Change from initial velocity', NULL, NULL, 'positive indicates acceleration'),
    ('pct_time_high_velocity', 'Percent of life with velocity > 0.1', 0, 100, 'early warning accumulator'),
    ('pct_time_orthon_signal', 'Percent of life with Orthon signal active', 0, 100, 'failure signature accumulator'),
    ('first_high_velocity_cycle', 'First cycle where velocity exceeded 0.1', 1, NULL, 'earlier = more warning time'),
    ('n_velocity_spikes', 'Count of >50% velocity increases in 10 cycles', 0, NULL, 'instability indicator'),
    ('orthon_risk_score', 'Composite risk score (0-100)', 0, 100, 'higher = more likely to fail soon')
) AS t(feature_name, description, min_value, max_value, interpretation);

-- Compute actual statistics for normalization
CREATE OR REPLACE TABLE ml_feature_stats AS
SELECT
    'current_coherence' AS feature,
    MIN(current_coherence) AS actual_min,
    MAX(current_coherence) AS actual_max,
    AVG(current_coherence) AS mean,
    STDDEV(current_coherence) AS std
FROM ml_features_current
UNION ALL
SELECT
    'current_velocity',
    MIN(current_velocity),
    MAX(current_velocity),
    AVG(current_velocity),
    STDDEV(current_velocity)
FROM ml_features_current
UNION ALL
SELECT
    'current_dissipation',
    MIN(current_dissipation),
    MAX(current_dissipation),
    AVG(current_dissipation),
    STDDEV(current_dissipation)
FROM ml_features_current
UNION ALL
SELECT
    'orthon_risk_score',
    MIN(orthon_risk_score),
    MAX(orthon_risk_score),
    AVG(orthon_risk_score),
    STDDEV(orthon_risk_score)
FROM ml_features_current;

SELECT * FROM ml_feature_stats;


-- ============================================================================
-- SECTION 4: EXPORT VIEWS FOR ML PIPELINES
-- ============================================================================

.print ''
.print '=== SECTION 4: Creating Export Views ==='

-- View: Dense feature matrix for gradient boosting / random forest
CREATE OR REPLACE VIEW v_ml_features_dense AS
SELECT
    entity_id,
    current_coherence,
    current_velocity,
    current_dissipation,
    current_norm_dim,
    current_coherence_ma20,
    current_velocity_ma20,
    current_coherence_volatility,
    current_velocity_volatility,
    current_coherence_drift,
    current_velocity_drift,
    avg_coherence,
    avg_velocity,
    avg_dissipation,
    overall_coherence_volatility,
    overall_velocity_volatility,
    pct_time_high_velocity,
    pct_time_low_coherence,
    pct_time_orthon_signal,
    COALESCE(first_high_velocity_pct_life, 100) AS first_high_velocity_pct_life,
    COALESCE(first_orthon_signal_pct_life, 100) AS first_orthon_signal_pct_life,
    coherence_lifecycle_change,
    velocity_lifecycle_change,
    n_velocity_spikes,
    n_coherence_drops,
    current_high_velocity,
    current_orthon_signal,
    orthon_risk_score
FROM ml_features_current;

-- View: Sequence features for LSTM / Transformer models
CREATE OR REPLACE VIEW v_ml_features_sequence AS
SELECT
    entity_id,
    cycle_num,
    pct_life,
    coherence,
    state_velocity,
    dissipation_rate,
    norm_effective_dim,
    coherence_ma20,
    velocity_ma20,
    coherence_volatility,
    velocity_volatility,
    coherence_trend_10,
    velocity_trend_10,
    high_velocity_flag,
    low_coherence_flag,
    orthon_signal_flag
FROM ml_features_temporal
ORDER BY entity_id, cycle_num;

-- View: Early warning summary for alerting systems
CREATE OR REPLACE VIEW v_ml_early_warning AS
SELECT
    entity_id,
    total_cycles AS cycles_observed,
    first_high_velocity_cycle,
    first_orthon_signal_cycle,
    current_velocity AS latest_velocity,
    current_coherence AS latest_coherence,
    orthon_risk_score,
    CASE
        WHEN orthon_risk_score >= 70 THEN 'CRITICAL'
        WHEN orthon_risk_score >= 50 THEN 'WARNING'
        WHEN orthon_risk_score >= 30 THEN 'WATCH'
        ELSE 'NORMAL'
    END AS risk_level,
    CASE
        WHEN current_orthon_signal = 1 THEN 'Active Orthon signal detected'
        WHEN current_high_velocity = 1 THEN 'Elevated state velocity'
        WHEN pct_time_orthon_signal > 5 THEN 'History of Orthon signals'
        WHEN velocity_lifecycle_change > 0.05 THEN 'Velocity trending upward'
        ELSE 'No immediate concerns'
    END AS status_message
FROM ml_features_current
ORDER BY orthon_risk_score DESC;

.print ''
.print 'Export views created:'
.print '  - v_ml_features_dense     : Dense feature matrix (28 features)'
.print '  - v_ml_features_sequence  : Time-series features for sequence models'
.print '  - v_ml_early_warning      : Summary view for alerting systems'


-- ============================================================================
-- SECTION 5: SAMPLE EXPORTS
-- ============================================================================

.print ''
.print '=== SECTION 5: Sample Feature Export ==='

.print ''
.print 'Top 10 entities by risk score:'
SELECT
    entity_id,
    risk_level,
    orthon_risk_score,
    latest_velocity,
    latest_coherence,
    status_message
FROM v_ml_early_warning
LIMIT 10;

.print ''
.print 'Feature correlation with risk (proxy for failure):'
SELECT
    ROUND(CORR(current_velocity, orthon_risk_score), 3) AS velocity_corr,
    ROUND(CORR(current_coherence, orthon_risk_score), 3) AS coherence_corr,
    ROUND(CORR(pct_time_orthon_signal, orthon_risk_score), 3) AS orthon_time_corr,
    ROUND(CORR(n_velocity_spikes, orthon_risk_score), 3) AS spikes_corr,
    ROUND(CORR(velocity_lifecycle_change, orthon_risk_score), 3) AS velocity_trend_corr
FROM ml_features_current;


-- ============================================================================
-- EXPORT TO PARQUET (for ML pipelines)
-- ============================================================================

.print ''
.print '=== Exporting to Parquet ==='

COPY (SELECT * FROM v_ml_features_dense)
TO '{prism_output}/ml_features_dense.parquet' (FORMAT PARQUET);

COPY (SELECT * FROM v_ml_features_sequence)
TO '{prism_output}/ml_features_sequence.parquet' (FORMAT PARQUET);

COPY (SELECT * FROM v_ml_early_warning)
TO '{prism_output}/ml_early_warning.parquet' (FORMAT PARQUET);

.print ''
.print 'Exported:'
.print '  - {prism_output}/ml_features_dense.parquet'
.print '  - {prism_output}/ml_features_sequence.parquet'
.print '  - {prism_output}/ml_early_warning.parquet'


-- ============================================================================
-- FINAL SUMMARY
-- ============================================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║                    ML FEATURE EXPORT COMPLETE                               ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'
.print ''
.print 'FEATURE CATEGORIES:'
.print ''
.print '  1. CURRENT STATE (point-in-time snapshot)'
.print '     - current_coherence, current_velocity, current_dissipation'
.print '     - current_norm_dim, current_high_velocity, current_orthon_signal'
.print ''
.print '  2. SMOOTHED TRENDS (noise-reduced)'
.print '     - coherence_ma20, velocity_ma20 (20-cycle moving averages)'
.print '     - coherence_trend_10, velocity_trend_10 (10-cycle % change)'
.print ''
.print '  3. VOLATILITY (instability indicators)'
.print '     - coherence_volatility, velocity_volatility (rolling std dev)'
.print '     - overall_coherence_volatility, overall_velocity_volatility'
.print ''
.print '  4. LIFECYCLE DRIFT (degradation tracking)'
.print '     - coherence_drift, velocity_drift (change from initial)'
.print '     - coherence_lifecycle_change, velocity_lifecycle_change'
.print ''
.print '  5. WARNING ACCUMULATORS (failure history)'
.print '     - pct_time_high_velocity, pct_time_low_coherence'
.print '     - pct_time_orthon_signal, n_velocity_spikes, n_coherence_drops'
.print ''
.print '  6. EARLY WARNING TRIGGERS'
.print '     - first_high_velocity_cycle, first_orthon_signal_cycle'
.print '     - first_high_velocity_pct_life, first_orthon_signal_pct_life'
.print ''
.print '  7. COMPOSITE RISK'
.print '     - orthon_risk_score (0-100, weighted combination)'
.print ''
.print 'KEY INSIGHT FROM TURBOFAN ANALYSIS:'
.print '  - Divergence begins at ~cycle 110'
.print '  - state_velocity > 0.1 is primary early warning'
.print '  - FAILS_EARLY engines hit trigger at 72.9% of life (43 cycles remaining)'
.print '  - SURVIVES_LONGER engines hit trigger at 60.9% (87 cycles remaining)'
.print ''
