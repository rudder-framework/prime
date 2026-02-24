-- ============================================================
-- SQL TYPOLOGY LAYER 2B: ENRICHMENT (post-Python)
-- ============================================================
-- Input:  typology.parquet (from 06)
--         signal_primitives.parquet (from Python/Rust: hurst, entropy, spectral)
-- Output: typology.parquet (updated with full classifications)
-- Time:   seconds
--
-- After Python/Rust computes the expensive primitives (hurst, 
-- sample_entropy, perm_entropy, spectral features, acf_half_life),
-- this script merges those results and updates classifications
-- that depend on them.
--
-- Run AFTER Python step. This is the final typology.
-- ============================================================

-- Merge Python results into typology
CREATE OR REPLACE TABLE typology_enriched AS
SELECT 
    t.cohort,
    t.signal_id,
    
    -- All existing columns from SQL typology
    t.n_obs, t.mean, t.std, t.min_val, t.max_val, t.median_val,
    t.kurtosis, t.skewness, t.cv, t.iqr, t.value_range,
    t.n_unique_values, t.is_constant, t.continuity,
    
    t.d1_mean, t.d1_std, t.d1_abs_mean, t.d1_snr,
    t.d2_mean, t.d2_std, t.d2_abs_mean, t.d2_snr,
    t.d2_smooth_abs_mean, t.d2_smooth_std, t.d2_smooth_snr,
    t.d3_snr, t.d4_snr,
    t.derivative_depth,
    t.d1_late_to_early_ratio, t.d2_late_to_early_ratio,
    t.d2_onset_pct, t.d2_onset_index, t.d2_max_region,
    
    t.trend_slope, t.trend_r2, t.trend_strength_class,
    t.monotonicity, t.reversal_rate, t.trend_direction,
    t.zero_crossing_rate, t.mean_stability_ratio, t.mean_drift_range,
    
    -- Fill in Python-computed primitives (COALESCE keeps SQL value if Python didn't compute)
    COALESCE(p.hurst_exponent, t.hurst_exponent) as hurst_exponent,
    COALESCE(p.sample_entropy, t.sample_entropy) as sample_entropy,
    COALESCE(p.perm_entropy, t.perm_entropy) as perm_entropy,
    COALESCE(p.spectral_slope, t.spectral_slope) as spectral_slope,
    COALESCE(p.spectral_flatness, t.spectral_flatness) as spectral_flatness,
    COALESCE(p.spectral_entropy, t.spectral_entropy) as spectral_entropy,
    COALESCE(p.dominant_frequency, t.dominant_frequency) as dominant_frequency,
    COALESCE(p.acf_half_life, t.acf_half_life) as acf_half_life,
    
    -- ==========================================
    -- UPDATED CLASSIFICATIONS (using Python data)
    -- ==========================================
    
    -- Hurst classification
    CASE 
        WHEN p.hurst_exponent IS NULL THEN t.hurst_class
        WHEN p.hurst_exponent > 0.8 THEN 'PERSISTENT'
        WHEN p.hurst_exponent > 0.6 THEN 'WEAKLY_PERSISTENT'
        WHEN p.hurst_exponent > 0.4 THEN 'RANDOM_WALK'
        ELSE 'ANTI_PERSISTENT'
    END as hurst_class,
    
    -- Spectral classification (requires FFT data)
    CASE
        WHEN p.spectral_flatness IS NULL THEN t.spectral_class
        WHEN p.spectral_flatness > 0.7 THEN 'WHITE_NOISE'
        WHEN p.spectral_flatness > 0.4 THEN 'BROADBAND'
        WHEN p.dominant_frequency IS NOT NULL AND p.spectral_flatness < 0.2 THEN 'NARROWBAND'
        WHEN p.spectral_slope < -1.5 THEN 'ONE_OVER_F'
        ELSE 'MIXED'
    END as spectral_class,
    
    -- REFINED memory class (Hurst-informed, overrides ACF-only)
    CASE
        WHEN p.hurst_exponent IS NULL THEN t.memory_class
        WHEN p.hurst_exponent > 0.8 THEN 'LONG_MEMORY'
        WHEN p.hurst_exponent > 0.6 THEN 'MODERATE_MEMORY'
        WHEN p.hurst_exponent > 0.4 THEN 'SHORT_MEMORY'
        ELSE 'ANTI_PERSISTENT'
    END as memory_class,
    
    -- REFINED temporal pattern (spectral-informed)
    CASE
        WHEN t.continuity = 'CONSTANT' THEN 'CONSTANT'
        WHEN t.continuity = 'BINARY' THEN 'BINARY'
        WHEN t.continuity = 'DISCRETE' THEN 'DISCRETE'
        WHEN p.dominant_frequency IS NOT NULL 
             AND p.spectral_flatness IS NOT NULL 
             AND p.spectral_flatness < 0.2 THEN 'PERIODIC'
        WHEN p.spectral_flatness IS NOT NULL 
             AND p.spectral_flatness < 0.35
             AND t.zero_crossing_rate > 0.1 THEN 'QUASI_PERIODIC'
        WHEN t.trend_r2 > 0.8 AND t.monotonicity > 0.8 THEN 'TRENDING'
        WHEN p.perm_entropy IS NOT NULL AND p.perm_entropy > 0.9 
             AND t.mean_stability_ratio < 0.2 THEN 'CHAOTIC'
        WHEN t.mean_stability_ratio > 1.0 THEN 'REGIME_CHANGE'
        WHEN t.trend_r2 > 0.3 THEN 'WEAK_TREND'
        WHEN t.mean_stability_ratio < 0.1 THEN 'STATIONARY'
        ELSE 'COMPLEX'
    END as temporal_pattern,
    
    -- REFINED complexity (entropy-informed)
    CASE
        WHEN t.is_constant THEN 'TRIVIAL'
        WHEN t.continuity = 'BINARY' THEN 'LOW'
        WHEN p.sample_entropy IS NOT NULL AND p.sample_entropy > 2.0 THEN 'HIGH'
        WHEN p.perm_entropy IS NOT NULL AND p.perm_entropy > 0.8 THEN 'HIGH'
        WHEN p.sample_entropy IS NOT NULL AND p.sample_entropy < 0.5 THEN 'LOW'
        WHEN t.derivative_depth >= 3 THEN 'HIGH'
        ELSE 'MODERATE'
    END as complexity_class,
    
    -- Keep the SQL-only classifications unchanged
    t.volatility_class,
    t.dynamics_speed,
    t.distribution_shape,
    t.degradation_indicator,
    t.acf_lag1

FROM typology t
LEFT JOIN signal_primitives p ON t.cohort = p.cohort AND t.signal_id = p.signal_id;

-- Replace typology with enriched version
DROP TABLE IF EXISTS typology;
ALTER TABLE typology_enriched RENAME TO typology;

-- Export final typology
COPY typology TO '{output_dir}/typology.parquet' (FORMAT PARQUET);
