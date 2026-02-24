-- ============================================================================
-- REPORT 01: TYPOLOGY SUMMARY
-- "Here's what your data looks like"
-- Sources: signal_statistics, signal_derivatives, signal_temporal,
--          signal_primitives, typology (loaded as views by runner)
-- ============================================================================

-- ============================================================================
-- 1. Signal Overview: one row per signal, the "business card"
-- ============================================================================
SELECT
    t.signal_id,
    t.continuity,
    t.memory_class,
    t.complexity_class,
    t.temporal_primary,
    d.derivative_depth,
    d.d2_onset_pct,
    d.d2_max_region,
    s.mean AS signal_mean,
    s.std AS signal_std,
    ROUND(s.cv, 2) AS cv,
    ROUND(tmp.acf_lag1, 3) AS acf_lag1,
    ROUND(tmp.trend_r2, 6) AS trend_r2
FROM typology t
LEFT JOIN signal_statistics s
    ON t.signal_id = s.signal_id
LEFT JOIN signal_derivatives d
    ON t.signal_id = d.signal_id
LEFT JOIN signal_temporal tmp
    ON t.signal_id = tmp.signal_id
ORDER BY t.signal_id;

-- ============================================================================
-- 2. Classification Distribution: how many signals in each category
-- ============================================================================
SELECT
    'Continuity' AS dimension,
    continuity AS class,
    COUNT(*) AS n_signals,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM typology
GROUP BY continuity

UNION ALL

SELECT
    'Memory' AS dimension,
    memory_class AS class,
    COUNT(*) AS n_signals,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM typology
GROUP BY memory_class

UNION ALL

SELECT
    'Temporal' AS dimension,
    temporal_primary AS class,
    COUNT(*) AS n_signals,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM typology
GROUP BY temporal_primary

UNION ALL

SELECT
    'Complexity' AS dimension,
    complexity_class AS class,
    COUNT(*) AS n_signals,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM typology
GROUP BY complexity_class

ORDER BY dimension, n_signals DESC;

-- ============================================================================
-- 3. Memory and ACF Summary: what drives windowing
-- ============================================================================
SELECT
    t.signal_id,
    t.memory_class,
    ROUND(tmp.acf_lag1, 4) AS acf_lag1,
    p.acf_half_life,
    CASE
        WHEN t.memory_class = 'LONG' THEN 'ACF-driven (4 x half-life)'
        WHEN t.memory_class = 'SHORT' THEN 'Minimum window (noise averaging)'
        ELSE 'ACF-driven'
    END AS window_rationale
FROM typology t
LEFT JOIN signal_temporal tmp
    ON t.signal_id = tmp.signal_id
LEFT JOIN signal_primitives p
    ON t.signal_id = p.signal_id
ORDER BY p.acf_half_life DESC NULLS LAST;

-- ============================================================================
-- 4. Derivative Depth Summary: how many levels of dynamics each signal has
-- ============================================================================
SELECT
    d.signal_id,
    d.derivative_depth,
    ROUND(d.d1_snr, 2) AS d1_snr,
    ROUND(d.d2_snr, 2) AS d2_snr,
    d.d2_onset_pct,
    d.d2_max_region,
    CASE
        WHEN d.derivative_depth >= 4 THEN 'DEEP_DYNAMICS (complex oscillator)'
        WHEN d.derivative_depth = 3 THEN 'MODERATE_DYNAMICS (accelerating system)'
        WHEN d.derivative_depth = 2 THEN 'SHALLOW_DYNAMICS (trending signal)'
        WHEN d.derivative_depth = 1 THEN 'MINIMAL_DYNAMICS (smooth drift)'
        WHEN d.derivative_depth = 0 THEN 'FLAT (constant or noise)'
        ELSE 'UNKNOWN'
    END AS dynamics_interpretation
FROM signal_derivatives d
ORDER BY d.derivative_depth DESC, d.d1_snr DESC;

-- ============================================================================
-- 5. Onset Detection: which signals show structural change and where
-- ============================================================================
SELECT
    d.signal_id,
    d.d2_onset_pct,
    CASE
        WHEN d.d2_onset_pct IS NULL THEN 'NO_ONSET (stationary throughout)'
        WHEN d.d2_onset_pct < 0.2 THEN 'VERY_EARLY_ONSET (degradation from start)'
        WHEN d.d2_onset_pct < 0.4 THEN 'EARLY_ONSET (early degradation)'
        WHEN d.d2_onset_pct < 0.6 THEN 'MID_LIFE_ONSET'
        WHEN d.d2_onset_pct < 0.8 THEN 'LATE_ONSET (long stable period)'
        ELSE 'VERY_LATE_ONSET (sudden end-of-life)'
    END AS onset_class,
    d.d2_max_region,
    t.complexity_class,
    t.temporal_primary
FROM signal_derivatives d
LEFT JOIN typology t
    ON d.signal_id = t.signal_id
WHERE d.d2_onset_pct IS NOT NULL
ORDER BY d.d2_onset_pct NULLS LAST;

-- ============================================================================
-- 6. System-level summary
-- ============================================================================
SELECT
    COUNT(*) AS total_signals,
    COUNT(*) FILTER (WHERE continuity = 'CONTINUOUS') AS n_continuous,
    COUNT(*) FILTER (WHERE continuity = 'BINARY') AS n_binary,
    COUNT(*) FILTER (WHERE continuity = 'DISCRETE') AS n_discrete,
    COUNT(*) FILTER (WHERE memory_class IN ('LONG', 'MODERATE')) AS n_persistent,
    COUNT(*) FILTER (WHERE memory_class = 'SHORT') AS n_short_memory,
    ROUND(AVG(d.derivative_depth), 1) AS avg_derivative_depth,
    COUNT(*) FILTER (WHERE d.d2_onset_pct IS NOT NULL) AS n_signals_with_onset
FROM typology t
LEFT JOIN signal_derivatives d
    ON t.signal_id = d.signal_id;
