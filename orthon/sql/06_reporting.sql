-- ============================================================
-- Summary Reports
-- ============================================================

-- Executive summary
SELECT
    (SELECT COUNT(DISTINCT signal_id) FROM observations) as total_signals,
    (SELECT COUNT(*) FROM observations) as total_observations,
    (SELECT COUNT(*) FROM signal_vector) as total_windows,
    (SELECT COUNT(DISTINCT temporal_pattern) FROM typology) as n_patterns,
    (SELECT COUNT(DISTINCT spectral) FROM typology) as n_spectral_types;


-- Classification summary
SELECT
    temporal_pattern || ' / ' || spectral as classification,
    COUNT(*) as n_signals,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as pct
FROM typology
GROUP BY classification
ORDER BY n_signals DESC;


-- Feature ranges (for normalization reference, filters out NaN/Inf)
SELECT
    'crest_factor' as feature,
    ROUND(MIN(crest_factor) FILTER (WHERE isfinite(crest_factor)), 4) as min_val,
    ROUND(MAX(crest_factor) FILTER (WHERE isfinite(crest_factor)), 4) as max_val,
    ROUND(AVG(crest_factor) FILTER (WHERE isfinite(crest_factor)), 4) as mean_val
FROM signal_vector
UNION ALL
SELECT 'kurtosis',
    ROUND(MIN(kurtosis) FILTER (WHERE isfinite(kurtosis)), 4),
    ROUND(MAX(kurtosis) FILTER (WHERE isfinite(kurtosis)), 4),
    ROUND(AVG(kurtosis) FILTER (WHERE isfinite(kurtosis)), 4)
FROM signal_vector
UNION ALL
SELECT 'spectral_slope',
    ROUND(MIN(spectral_slope) FILTER (WHERE isfinite(spectral_slope)), 4),
    ROUND(MAX(spectral_slope) FILTER (WHERE isfinite(spectral_slope)), 4),
    ROUND(AVG(spectral_slope) FILTER (WHERE isfinite(spectral_slope)), 4)
FROM signal_vector
UNION ALL
SELECT 'hurst',
    ROUND(MIN(hurst) FILTER (WHERE isfinite(hurst)), 4),
    ROUND(MAX(hurst) FILTER (WHERE isfinite(hurst)), 4),
    ROUND(AVG(hurst) FILTER (WHERE isfinite(hurst)), 4)
FROM signal_vector
UNION ALL
SELECT 'permutation_entropy',
    ROUND(MIN(permutation_entropy) FILTER (WHERE isfinite(permutation_entropy)), 4),
    ROUND(MAX(permutation_entropy) FILTER (WHERE isfinite(permutation_entropy)), 4),
    ROUND(AVG(permutation_entropy) FILTER (WHERE isfinite(permutation_entropy)), 4)
FROM signal_vector
UNION ALL
SELECT 'sample_entropy',
    ROUND(MIN(sample_entropy) FILTER (WHERE isfinite(sample_entropy)), 4),
    ROUND(MAX(sample_entropy) FILTER (WHERE isfinite(sample_entropy)), 4),
    ROUND(AVG(sample_entropy) FILTER (WHERE isfinite(sample_entropy)), 4)
FROM signal_vector;
