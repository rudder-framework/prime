-- ============================================================
-- Signal Vector Analysis
-- ============================================================

-- Column health check
SELECT
    column_name,
    COUNT(*) as total_rows,
    COUNT(*) - COUNT(column_name) as null_count,
    ROUND(100.0 * (COUNT(*) - COUNT(column_name)) / COUNT(*), 1) as null_pct
FROM signal_vector
UNPIVOT (value FOR column_name IN (COLUMNS(* EXCLUDE (signal_id, I))))
GROUP BY column_name
ORDER BY null_pct DESC;


-- Feature statistics by signal (handles NaN gracefully)
SELECT
    signal_id,
    COUNT(*) as n_windows,

    -- Crest factor
    ROUND(AVG(crest_factor), 4) as crest_mean,
    ROUND(STDDEV_POP(crest_factor) FILTER (WHERE isfinite(crest_factor)), 4) as crest_std,

    -- Kurtosis
    ROUND(AVG(kurtosis), 4) as kurt_mean,
    ROUND(STDDEV_POP(kurtosis) FILTER (WHERE isfinite(kurtosis)), 4) as kurt_std,

    -- Spectral
    ROUND(AVG(spectral_slope) FILTER (WHERE isfinite(spectral_slope)), 4) as slope_mean,
    ROUND(AVG(dominant_freq) FILTER (WHERE isfinite(dominant_freq)), 4) as dom_freq_mean,

    -- Entropy
    ROUND(AVG(permutation_entropy) FILTER (WHERE isfinite(permutation_entropy)), 4) as perm_entropy_mean,

    -- Hurst
    ROUND(AVG(hurst) FILTER (WHERE isfinite(hurst)), 4) as hurst_mean,
    ROUND(STDDEV_POP(hurst) FILTER (WHERE isfinite(hurst)), 4) as hurst_std

FROM signal_vector
GROUP BY signal_id
ORDER BY signal_id;


-- Feature evolution over time (windowed stats)
SELECT
    signal_id,
    I,
    ROUND(crest_factor, 4) as crest,
    ROUND(kurtosis, 4) as kurt,
    ROUND(spectral_slope, 4) as slope,
    ROUND(hurst, 4) as hurst,
    ROUND(permutation_entropy, 4) as perm_entropy
FROM signal_vector
ORDER BY signal_id, I;
