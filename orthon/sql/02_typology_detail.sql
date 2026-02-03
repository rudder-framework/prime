-- ============================================================
-- Typology Analysis
-- ============================================================

-- Full classification with key metrics
SELECT
    signal_id,
    cohort,
    temporal_pattern,
    spectral,

    -- Stationarity
    ROUND(adf_pvalue, 4) as adf_p,
    ROUND(kpss_pvalue, 4) as kpss_p,
    CASE
        WHEN adf_pvalue < 0.05 AND kpss_pvalue > 0.05 THEN 'STATIONARY'
        WHEN adf_pvalue > 0.05 AND kpss_pvalue < 0.05 THEN 'UNIT_ROOT'
        ELSE 'AMBIGUOUS'
    END as stationarity_verdict,

    -- Spectral character
    ROUND(spectral_flatness, 4) as spec_flat,
    ROUND(spectral_slope, 4) as spec_slope,
    ROUND(spectral_peak_snr, 2) as peak_snr_db,

    -- Temporal character
    ROUND(perm_entropy, 4) as perm_ent,
    ROUND(hurst, 4) as hurst,
    acf_half_life,

    -- Sample info
    n_samples

FROM typology
ORDER BY cohort, signal_id;


-- Signals that might be misclassified (edge cases)
SELECT
    signal_id,
    temporal_pattern,
    spectral,
    ROUND(spectral_flatness, 4) as spec_flat,
    ROUND(perm_entropy, 4) as perm_ent,
    CASE
        WHEN temporal_pattern = 'RANDOM' AND spectral_flatness < 0.5
            THEN 'Low flatness for RANDOM'
        WHEN temporal_pattern = 'PERIODIC' AND spectral_flatness > 0.7
            THEN 'High flatness for PERIODIC'
        WHEN temporal_pattern = 'TRENDING' AND NOT is_first_bin_peak
            THEN 'No DC peak for TRENDING'
        ELSE 'OK'
    END as flag
FROM typology
WHERE flag != 'OK';
