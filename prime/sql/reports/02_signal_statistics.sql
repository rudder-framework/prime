-- ============================================================================
-- REPORT 02: SIGNAL STATISTICS
-- "The numbers behind each signal"
-- Sources: signal_statistics, signal_primitives, signal_temporal,
--          signal_derivatives (loaded as views by runner)
-- ============================================================================

-- ============================================================================
-- 1. Basic Statistics: the foundation
-- ============================================================================
SELECT
    signal_id,
    ROUND(mean, 4) AS mean,
    ROUND(std, 4) AS std,
    ROUND(cv, 2) AS cv,
    ROUND(skewness, 3) AS skewness,
    ROUND(kurtosis, 3) AS kurtosis,
    ROUND(min_val, 4) AS min_val,
    ROUND(max_val, 4) AS max_val,
    ROUND(value_range, 4) AS range,
    ROUND(median_val, 4) AS median,
    ROUND(iqr, 4) AS iqr,
    n_obs,
    continuity
FROM signal_statistics
ORDER BY signal_id;

-- ============================================================================
-- 2. Distribution Shape: is this normal, skewed, heavy-tailed?
-- ============================================================================
SELECT
    signal_id,
    ROUND(skewness, 3) AS skewness,
    ROUND(kurtosis, 3) AS kurtosis,
    CASE
        WHEN ABS(skewness) < 0.5 THEN 'SYMMETRIC'
        WHEN skewness > 0.5 THEN 'RIGHT_SKEWED'
        WHEN skewness < -0.5 THEN 'LEFT_SKEWED'
        ELSE 'UNKNOWN'
    END AS skew_class,
    CASE
        WHEN kurtosis < 2.5 THEN 'PLATYKURTIC (light tails)'
        WHEN kurtosis BETWEEN 2.5 AND 3.5 THEN 'MESOKURTIC (normal-like)'
        WHEN kurtosis > 3.5 THEN 'LEPTOKURTIC (heavy tails)'
        ELSE 'UNKNOWN'
    END AS tail_class,
    CASE
        WHEN ABS(skewness) < 0.5 AND kurtosis BETWEEN 2.5 AND 3.5
            THEN 'APPROXIMATELY_NORMAL'
        WHEN kurtosis > 6 THEN 'EXTREME_TAILS (outlier-prone)'
        WHEN ABS(skewness) > 2 THEN 'HIGHLY_ASYMMETRIC'
        ELSE 'NON_NORMAL'
    END AS distribution_summary
FROM signal_statistics
ORDER BY kurtosis DESC;

-- ============================================================================
-- 3. Temporal Properties: memory, trend, stationarity proxies
-- ============================================================================
SELECT
    t.signal_id,
    ROUND(t.acf_lag1, 4) AS acf_lag1,
    ROUND(t.trend_slope, 6) AS trend_slope,
    ROUND(t.trend_r2, 6) AS trend_r2,
    ROUND(t.monotonicity, 4) AS monotonicity,
    ROUND(t.zero_crossing_rate, 4) AS zero_crossing_rate,
    ROUND(t.reversal_rate, 4) AS reversal_rate,
    ROUND(t.mean_stability_ratio, 4) AS mean_stability_ratio,
    CASE
        WHEN t.trend_r2 > 0.5 THEN 'STRONG_TREND'
        WHEN t.trend_r2 > 0.1 THEN 'MODERATE_TREND'
        WHEN t.trend_r2 > 0.01 THEN 'WEAK_TREND'
        ELSE 'NO_TREND'
    END AS trend_class
FROM signal_temporal t
ORDER BY t.trend_r2 DESC;

-- ============================================================================
-- 4. Advanced Primitives: Hurst, entropy, spectral properties
-- ============================================================================
SELECT
    p.signal_id,
    ROUND(p.hurst_exponent, 4) AS hurst,
    CASE
        WHEN p.hurst_exponent > 0.65 THEN 'PERSISTENT (trending)'
        WHEN p.hurst_exponent BETWEEN 0.35 AND 0.65 THEN 'RANDOM_WALK'
        WHEN p.hurst_exponent < 0.35 THEN 'ANTI_PERSISTENT (mean-reverting)'
        ELSE 'UNKNOWN'
    END AS hurst_class,
    p.sample_entropy,
    p.perm_entropy,
    ROUND(p.spectral_flatness, 4) AS spectral_flatness,
    CASE
        WHEN p.spectral_flatness > 0.5 THEN 'BROADBAND (noise-like)'
        WHEN p.spectral_flatness > 0.1 THEN 'MIXED_SPECTRUM'
        WHEN p.spectral_flatness < 0.1 THEN 'NARROWBAND (periodic)'
        ELSE 'UNKNOWN'
    END AS spectral_class,
    p.acf_half_life
FROM signal_primitives p
ORDER BY p.hurst_exponent DESC;

-- ============================================================================
-- 5. Signal Comparison Matrix: how signals relate to each other statistically
-- ============================================================================
SELECT
    s.cohort,
    s.signal_id,
    ROUND(s.std, 4) AS std,
    ROUND(s.cv, 2) AS cv,
    ROUND(t.acf_lag1, 4) AS acf_lag1,
    ROUND(p.hurst_exponent, 4) AS hurst,
    ROUND(p.spectral_flatness, 4) AS spectral_flatness,
    d.derivative_depth,
    -- Rank signals by various metrics (within cohort)
    RANK() OVER (PARTITION BY s.cohort ORDER BY s.std DESC) AS rank_by_variability,
    RANK() OVER (PARTITION BY s.cohort ORDER BY t.acf_lag1 DESC) AS rank_by_persistence,
    RANK() OVER (PARTITION BY s.cohort ORDER BY p.hurst_exponent DESC NULLS LAST) AS rank_by_hurst,
    RANK() OVER (PARTITION BY s.cohort ORDER BY d.derivative_depth DESC) AS rank_by_complexity
FROM signal_statistics s
LEFT JOIN signal_temporal t
    ON s.cohort = t.cohort AND s.signal_id = t.signal_id
LEFT JOIN signal_primitives p
    ON s.cohort = p.cohort AND s.signal_id = p.signal_id
LEFT JOIN signal_derivatives d
    ON s.cohort = d.cohort AND s.signal_id = d.signal_id
ORDER BY s.cohort, s.signal_id;

-- ============================================================================
-- 6. Anomaly Flags: signals with unusual statistical properties
-- ============================================================================
SELECT
    s.cohort,
    s.signal_id,
    CASE WHEN s.cv > 100 THEN 'HIGH_CV (>100%)' END AS cv_flag,
    CASE WHEN s.kurtosis > 10 THEN 'EXTREME_KURTOSIS (>10)' END AS kurtosis_flag,
    CASE WHEN ABS(s.skewness) > 2 THEN 'EXTREME_SKEW (|s|>2)' END AS skew_flag,
    CASE WHEN t.trend_r2 > 0.5 THEN 'STRONG_TREND (r2>0.5)' END AS trend_flag,
    CASE WHEN t.acf_lag1 > 0.99 THEN 'NEAR_UNIT_ROOT (acf>0.99)' END AS unit_root_flag,
    CASE WHEN p.hurst_exponent < 0.2 THEN 'STRONG_ANTI_PERSIST (H<0.2)' END AS hurst_flag,
    CASE WHEN s.continuity = 'BINARY' THEN 'BINARY_SIGNAL' END AS binary_flag
FROM signal_statistics s
LEFT JOIN signal_temporal t
    ON s.cohort = t.cohort AND s.signal_id = t.signal_id
LEFT JOIN signal_primitives p
    ON s.cohort = p.cohort AND s.signal_id = p.signal_id
WHERE s.cv > 100
   OR s.kurtosis > 10
   OR ABS(s.skewness) > 2
   OR t.trend_r2 > 0.5
   OR t.acf_lag1 > 0.99
   OR (p.hurst_exponent IS NOT NULL AND p.hurst_exponent < 0.2)
   OR s.continuity = 'BINARY'
ORDER BY s.cohort, s.signal_id;
