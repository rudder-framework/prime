-- ============================================================
-- SQL TYPOLOGY LAYER 2: CLASSIFICATION
-- ============================================================
-- Input:  signal_statistics.parquet
--         signal_derivatives.parquet
--         signal_temporal.parquet
-- Output: typology.parquet (full classification, manifest-ready)
-- Time:   seconds (just CASE WHEN logic on summary tables)
--
-- This is the final assembly. All measurements are done.
-- This script only classifies — no computation.
--
-- For signals that need expensive primitives (hurst, entropy, spectral),
-- those columns will be NULL here and filled in by Python/Rust in step 3.
-- After Python fills them, re-run 06b_typology_enrich.sql to update
-- classifications that depend on those columns.
-- ============================================================

CREATE OR REPLACE TABLE typology AS
SELECT 
    s.cohort,
    s.signal_id,
    
    -- ==========================================
    -- RAW STATISTICS (from 01)
    -- ==========================================
    s.n_obs,
    s.mean,
    s.std,
    s.min_val,
    s.max_val,
    s.median_val,
    s.kurtosis,
    s.skewness,
    s.cv,
    s.iqr,
    s.value_range,
    s.n_unique_values,
    s.is_constant,
    s.continuity,
    
    -- ==========================================
    -- DERIVATIVES (from 02)
    -- ==========================================
    d.d1_mean, d.d1_std, d.d1_abs_mean, d.d1_snr,
    d.d2_mean, d.d2_std, d.d2_abs_mean, d.d2_snr,
    d.d2_smooth_abs_mean, d.d2_smooth_std, d.d2_smooth_snr,
    d.d3_snr, d.d4_snr,
    d.derivative_depth,
    d.d1_late_to_early_ratio,
    d.d2_late_to_early_ratio,
    d.d2_onset_pct,
    d.d2_onset_index,
    d.d2_max_region,
    
    -- ==========================================
    -- TEMPORAL (from 03)
    -- ==========================================
    t.trend_slope,
    t.trend_r2,
    t.trend_strength_class,
    t.monotonicity,
    t.reversal_rate,
    t.trend_direction,
    t.zero_crossing_rate,
    t.acf_lag1,
    t.memory_class_acf1,
    t.mean_stability_ratio,
    t.mean_drift_range,
    
    -- ==========================================
    -- EXPENSIVE PRIMITIVES (filled by Python/Rust, NULL for now)
    -- ==========================================
    NULL::FLOAT as hurst_exponent,
    NULL::VARCHAR as hurst_class,
    NULL::FLOAT as sample_entropy,
    NULL::FLOAT as perm_entropy,
    NULL::FLOAT as spectral_slope,
    NULL::FLOAT as spectral_flatness,
    NULL::FLOAT as spectral_entropy,
    NULL::FLOAT as dominant_frequency,
    NULL::VARCHAR as spectral_class,
    NULL::FLOAT as acf_half_life,
    
    -- ==========================================
    -- CLASSIFICATIONS (SQL-computable from above)
    -- ==========================================
    
    -- Temporal pattern
    CASE
        WHEN s.continuity = 'CONSTANT' THEN 'CONSTANT'
        WHEN s.continuity = 'BINARY' THEN 'BINARY'
        WHEN s.continuity = 'DISCRETE' THEN 'DISCRETE'
        WHEN t.trend_r2 > 0.8 AND t.monotonicity > 0.8 THEN 'TRENDING'
        WHEN t.zero_crossing_rate > 0.3 AND t.reversal_rate > 0.3 THEN 'OSCILLATING'
        WHEN t.mean_stability_ratio < 0.1 AND t.acf_lag1 < 0.3 THEN 'STATIONARY_NOISE'
        WHEN t.mean_stability_ratio < 0.2 THEN 'STATIONARY'
        WHEN t.mean_stability_ratio > 1.0 THEN 'REGIME_CHANGE'
        WHEN t.trend_r2 > 0.3 THEN 'WEAK_TREND'
        ELSE 'COMPLEX'
    END as temporal_pattern,
    
    -- Memory classification (from ACF lag-1, will be refined by Hurst)
    t.memory_class_acf1 as memory_class,
    
    -- Volatility classification
    CASE 
        WHEN s.cv IS NULL OR s.is_constant THEN 'NONE'
        WHEN s.cv < 0.01 THEN 'VERY_LOW'
        WHEN s.cv < 0.05 THEN 'LOW'
        WHEN s.cv < 0.2 THEN 'MODERATE'
        WHEN s.cv < 0.5 THEN 'HIGH'
        ELSE 'VERY_HIGH'
    END as volatility_class,
    
    -- Dynamics speed (how fast does the signal change relative to its length)
    CASE 
        WHEN t.acf_lag1 > 0.95 THEN 'SLOW'
        WHEN t.acf_lag1 > 0.7 THEN 'MEDIUM'
        ELSE 'FAST'
    END as dynamics_speed,
    
    -- Distribution shape
    CASE 
        WHEN ABS(s.skewness) < 0.5 AND s.kurtosis < 4 THEN 'NORMAL_LIKE'
        WHEN s.kurtosis > 8 THEN 'HEAVY_TAILED'
        WHEN ABS(s.skewness) > 2 THEN 'HIGHLY_SKEWED'
        WHEN ABS(s.skewness) > 0.5 THEN 'SKEWED'
        ELSE 'MIXED'
    END as distribution_shape,
    
    -- Degradation indicator
    -- Signal shows systematic change over lifetime
    CASE
        WHEN s.is_constant THEN 'NONE'
        WHEN d.d2_onset_pct IS NOT NULL AND d.d1_late_to_early_ratio > 2.0 THEN 'STRONG'
        WHEN d.d1_late_to_early_ratio > 1.5 THEN 'MODERATE'
        WHEN t.trend_r2 > 0.3 AND t.monotonicity > 0.6 THEN 'WEAK'
        ELSE 'NONE'
    END as degradation_indicator,
    
    -- Complexity estimate (will be refined by entropy measures)
    CASE
        WHEN s.is_constant THEN 'TRIVIAL'
        WHEN s.continuity = 'BINARY' THEN 'LOW'
        WHEN t.trend_r2 > 0.9 THEN 'LOW'
        WHEN t.reversal_rate > 0.4 AND t.zero_crossing_rate > 0.3 THEN 'HIGH'
        WHEN d.derivative_depth >= 3 THEN 'HIGH'
        ELSE 'MODERATE'
    END as complexity_class

FROM signal_statistics s
LEFT JOIN signal_derivatives d ON s.cohort = d.cohort AND s.signal_id = d.signal_id
LEFT JOIN signal_temporal t ON s.cohort = t.cohort AND s.signal_id = t.signal_id;

-- Export
COPY typology TO '{output_dir}/typology.parquet' (FORMAT PARQUET);
