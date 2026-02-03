-- ============================================================
-- Anomaly / Outlier Detection
-- ============================================================

-- Windows with extreme values (per signal, Z-score > 3)
WITH signal_stats AS (
    SELECT
        signal_id,
        AVG(crest_factor) FILTER (WHERE isfinite(crest_factor)) as crest_mean,
        STDDEV_POP(crest_factor) FILTER (WHERE isfinite(crest_factor)) as crest_std,
        AVG(kurtosis) FILTER (WHERE isfinite(kurtosis)) as kurt_mean,
        STDDEV_POP(kurtosis) FILTER (WHERE isfinite(kurtosis)) as kurt_std,
        AVG(hurst) FILTER (WHERE isfinite(hurst)) as hurst_mean,
        STDDEV_POP(hurst) FILTER (WHERE isfinite(hurst)) as hurst_std
    FROM signal_vector
    GROUP BY signal_id
)
SELECT
    sv.signal_id,
    sv.I,
    ROUND(sv.crest_factor, 4) as crest,
    ROUND((sv.crest_factor - s.crest_mean) / NULLIF(s.crest_std, 0), 2) as crest_z,
    ROUND(sv.kurtosis, 4) as kurt,
    ROUND((sv.kurtosis - s.kurt_mean) / NULLIF(s.kurt_std, 0), 2) as kurt_z,
    ROUND(sv.hurst, 4) as hurst,
    ROUND((sv.hurst - s.hurst_mean) / NULLIF(s.hurst_std, 0), 2) as hurst_z
FROM signal_vector sv
JOIN signal_stats s USING (signal_id)
WHERE (isfinite(sv.crest_factor) AND ABS((sv.crest_factor - s.crest_mean) / NULLIF(s.crest_std, 0)) > 3)
   OR (isfinite(sv.kurtosis) AND ABS((sv.kurtosis - s.kurt_mean) / NULLIF(s.kurt_std, 0)) > 3)
   OR (isfinite(sv.hurst) AND ABS((sv.hurst - s.hurst_mean) / NULLIF(s.hurst_std, 0)) > 3)
ORDER BY signal_id, I;


-- Sudden changes (window-to-window delta)
WITH lagged AS (
    SELECT
        signal_id,
        I,
        hurst,
        LAG(hurst) OVER (PARTITION BY signal_id ORDER BY I) as prev_hurst,
        crest_factor,
        LAG(crest_factor) OVER (PARTITION BY signal_id ORDER BY I) as prev_crest
    FROM signal_vector
)
SELECT
    signal_id,
    I,
    ROUND(hurst, 4) as hurst,
    ROUND(hurst - prev_hurst, 4) as hurst_delta,
    ROUND(crest_factor, 4) as crest,
    ROUND(crest_factor - prev_crest, 4) as crest_delta
FROM lagged
WHERE (isfinite(hurst) AND isfinite(prev_hurst) AND ABS(hurst - prev_hurst) > 0.2)
   OR (isfinite(crest_factor) AND isfinite(prev_crest) AND ABS(crest_factor - prev_crest) > 1.0)
ORDER BY signal_id, I;


-- Windows where multiple sensors spike simultaneously (correlated anomalies)
SELECT
    I,
    COUNT(*) as n_spiky_sensors,
    LIST(signal_id) as sensors
FROM signal_vector
WHERE kurtosis > 10
GROUP BY I
HAVING COUNT(*) > 3
ORDER BY n_spiky_sensors DESC;
