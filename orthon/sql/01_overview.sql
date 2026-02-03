-- ============================================================
-- Data Overview
-- ============================================================

-- Signal inventory
SELECT
    t.signal_id,
    t.cohort,
    t.temporal_pattern,
    t.spectral,
    t.n_samples,
    COUNT(DISTINCT sv.I) as n_windows
FROM typology t
LEFT JOIN signal_vector sv USING (signal_id)
GROUP BY ALL
ORDER BY t.cohort, t.signal_id;

-- Classification distribution
SELECT
    temporal_pattern,
    spectral,
    COUNT(*) as n_signals
FROM typology
GROUP BY ALL
ORDER BY n_signals DESC;

-- Window coverage
SELECT
    signal_id,
    MIN(I) as first_window,
    MAX(I) as last_window,
    COUNT(*) as n_windows,
    MAX(I) - MIN(I) as span
FROM signal_vector
GROUP BY signal_id
ORDER BY signal_id;
