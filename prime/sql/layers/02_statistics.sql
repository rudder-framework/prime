-- ============================================================================
-- ENGINES: STATISTICS (CALCULUS-BASED)
-- ============================================================================
-- Global statistics and calculus-derived metrics.
-- NO ROLLING WINDOWS - use derivatives for local behavior.
-- ============================================================================

-- ============================================================================
-- 001: GLOBAL STATISTICS (per signal) - KEEP: aggregation for summary
-- ============================================================================

CREATE OR REPLACE VIEW v_stats_global AS
SELECT
    signal_id,
    index_dimension,
    signal_class,
    COUNT(*) AS n_points,
    MIN(value) AS value_min,
    MAX(value) AS value_max,
    MAX(value) - MIN(value) AS value_range,
    AVG(value) AS value_mean,
    STDDEV(value) AS value_std,
    VARIANCE(value) AS value_var,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) AS value_q1,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY value) AS value_median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) AS value_q3,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) -
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) AS value_iqr
FROM v_base
GROUP BY signal_id, index_dimension, signal_class;


-- ============================================================================
-- 002: DERIVATIVE STATISTICS (per signal)
-- ============================================================================
-- Global stats on dvalue and d2value for threshold calibration

CREATE OR REPLACE VIEW v_derivative_stats AS
SELECT
    signal_id,
    AVG(ABS(dvalue)) AS dvalue_mean_abs,
    STDDEV(dvalue) AS dvalue_std,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ABS(dvalue)) AS dvalue_median_abs,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS(dvalue)) AS dvalue_p95_abs,
    AVG(ABS(d2value)) AS d2value_mean_abs,
    STDDEV(d2value) AS d2value_std,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ABS(d2value)) AS d2value_median_abs,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS(d2value)) AS d2value_p95_abs
FROM v_d2value
WHERE dvalue IS NOT NULL AND d2value IS NOT NULL
GROUP BY signal_id;


-- ============================================================================
-- 003: TRAJECTORY DEVIATION (percentile-based, not z-score)
-- ============================================================================
-- Ranks each value within its signal's distribution using PERCENT_RANK.
-- No Gaussian assumption — works for any distribution shape.

CREATE OR REPLACE VIEW v_trajectory_deviation AS
SELECT
    b.signal_id,
    b.signal_0,
    b.value,
    PERCENT_RANK() OVER (
        PARTITION BY b.signal_id
        ORDER BY b.value
    ) AS value_percentile,
    CASE
        WHEN PERCENT_RANK() OVER (PARTITION BY b.signal_id ORDER BY b.value) > 0.99
          OR PERCENT_RANK() OVER (PARTITION BY b.signal_id ORDER BY b.value) < 0.01
        THEN 'extreme'
        WHEN PERCENT_RANK() OVER (PARTITION BY b.signal_id ORDER BY b.value) > 0.95
          OR PERCENT_RANK() OVER (PARTITION BY b.signal_id ORDER BY b.value) < 0.05
        THEN 'outlier'
        ELSE 'normal'
    END AS deviation_category
FROM v_base b;


-- ============================================================================
-- 004: DERIVATIVE-BASED ANOMALY DETECTION
-- ============================================================================
-- Anomaly = |dvalue| or |d2value| exceeds threshold relative to signal

CREATE OR REPLACE VIEW v_derivative_anomaly AS
SELECT
    d.signal_id,
    d.signal_0,
    d.dvalue,
    d.d2value,
    ds.dvalue_median_abs,
    ds.d2value_median_abs,
    -- Anomaly if derivative exceeds 3x median
    CASE WHEN ABS(d.dvalue) > 3 * ds.dvalue_median_abs THEN TRUE ELSE FALSE END AS dvalue_anomaly,
    CASE WHEN ABS(d.d2value) > 3 * ds.d2value_median_abs THEN TRUE ELSE FALSE END AS d2value_anomaly,
    -- Combined anomaly score
    ABS(d.dvalue) / NULLIF(ds.dvalue_median_abs, 0) +
    ABS(d.d2value) / NULLIF(ds.d2value_median_abs, 0) AS anomaly_score
FROM v_d2value d
JOIN v_derivative_stats ds USING (signal_id)
WHERE d.dvalue IS NOT NULL AND d.d2value IS NOT NULL;


-- ============================================================================
-- 005: LOCAL EXTREMA (peaks and valleys)
-- ============================================================================
-- Detected from sign change in dvalue

CREATE OR REPLACE VIEW v_local_extrema AS
SELECT
    signal_id,
    signal_0,
    value,
    dvalue,
    LAG(dvalue) OVER (PARTITION BY signal_id ORDER BY signal_0) AS dvalue_prev,
    CASE
        WHEN dvalue > 0 AND LAG(dvalue) OVER (PARTITION BY signal_id ORDER BY signal_0) < 0 THEN 'valley'
        WHEN dvalue < 0 AND LAG(dvalue) OVER (PARTITION BY signal_id ORDER BY signal_0) > 0 THEN 'peak'
        ELSE NULL
    END AS extrema_type
FROM v_dvalue
WHERE dvalue IS NOT NULL;


-- ============================================================================
-- 006: TREND DIRECTION (from calculus)
-- ============================================================================
-- Trend = sign(dvalue), persistence = consecutive same-sign dvalue

CREATE OR REPLACE VIEW v_trend AS
SELECT
    signal_id,
    signal_0,
    value,
    dvalue,
    CASE
        WHEN dvalue > 0.001 THEN 'up'
        WHEN dvalue < -0.001 THEN 'down'
        ELSE 'flat'
    END AS trend_direction,
    -- Trend strength from curvature
    CASE
        WHEN ABS(dvalue) > 0 THEN ABS(d2value) / ABS(dvalue)
        ELSE NULL
    END AS trend_acceleration
FROM v_d2value;


-- ============================================================================
-- 007: VOLATILITY PROXY (from d2value)
-- ============================================================================
-- Local volatility = |d2value| or |kappa|

CREATE OR REPLACE VIEW v_volatility_proxy AS
SELECT
    c.signal_id,
    c.signal_0,
    c.value,
    c.d2value,
    c.kappa,
    ABS(c.d2value) AS instantaneous_volatility,
    c.kappa AS curvature_volatility,
    -- Compare to signal's baseline
    ABS(c.d2value) / NULLIF(ds.d2value_median_abs, 0) AS relative_volatility
FROM v_curvature c
JOIN v_derivative_stats ds USING (signal_id);


-- ============================================================================
-- 008: AUTOCORRELATION AT SPECIFIC LAGS (KEEP: inherently needs lag)
-- ============================================================================
-- ACF at lags 1, 5, 10 - this requires comparison across samples

CREATE OR REPLACE VIEW v_autocorrelation AS
SELECT
    signal_id,
    CORR(value, value_lag1) AS acf_lag1,
    CORR(value, value_lag5) AS acf_lag5,
    CORR(value, value_lag10) AS acf_lag10
FROM (
    SELECT
        signal_id,
        value,
        LAG(value, 1) OVER (PARTITION BY signal_id ORDER BY signal_0) AS value_lag1,
        LAG(value, 5) OVER (PARTITION BY signal_id ORDER BY signal_0) AS value_lag5,
        LAG(value, 10) OVER (PARTITION BY signal_id ORDER BY signal_0) AS value_lag10
    FROM v_base
)
WHERE value_lag10 IS NOT NULL
GROUP BY signal_id;


-- ============================================================================
-- STATISTICS SUMMARY VIEW
-- ============================================================================

CREATE OR REPLACE VIEW v_statistics_complete AS
SELECT
    sg.signal_id,
    sg.n_points,
    sg.value_mean,
    sg.value_std,
    sg.value_median,
    sg.value_iqr,
    ds.dvalue_median_abs,
    ds.d2value_median_abs,
    ac.acf_lag1,
    ac.acf_lag5,
    ac.acf_lag10
FROM v_stats_global sg
LEFT JOIN v_derivative_stats ds USING (signal_id)
LEFT JOIN v_autocorrelation ac USING (signal_id);
