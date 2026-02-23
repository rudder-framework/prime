-- ============================================================================
-- DERIVATIVE CHAIN ANALYSIS REPORT
-- ============================================================================
--
-- Analyzes recursive derivative depth, onset locations, and D2 method
-- comparison across all signals. Uses typology_raw columns computed by
-- prime/ingest/typology_raw.py (derivative chain analysis functions).
--
-- Sections:
--   1. Fleet derivative depth distribution
--   2. D2 method comparison per signal type
--   3. D1 late-to-early ratio (degradation acceleration ranking)
--   4. D2 onset locations
--   5. Per-signal derivative profile
--   6. Canary cross-reference (commented out — needs Report 13)
--   7. Adaptive windowing recommendations
--
-- Usage: Run via prime.sql.runner against a domain with typology_raw.parquet
-- ============================================================================


-- ============================================================================
-- SECTION 1: FLEET DERIVATIVE DEPTH DISTRIBUTION
-- How many derivative levels carry signal before hitting noise?
-- ============================================================================

WITH depth_counts AS (
    SELECT
        derivative_depth,
        COUNT(*) AS n_signals,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
    FROM typology_raw
    WHERE derivative_depth IS NOT NULL
    GROUP BY derivative_depth
)
SELECT
    derivative_depth,
    n_signals,
    pct AS pct_of_fleet,
    REPEAT('|', CAST(pct AS INTEGER)) AS bar,
    CASE
        WHEN derivative_depth = 0 THEN 'CONSTANT/SHORT'
        WHEN derivative_depth = 1 THEN 'LINEAR'
        WHEN derivative_depth = 2 THEN 'QUADRATIC'
        WHEN derivative_depth = 3 THEN 'CUBIC'
        WHEN derivative_depth >= 4 THEN 'DEEP_NONLINEAR'
    END AS complexity_class
FROM depth_counts
ORDER BY derivative_depth;


-- ============================================================================
-- SECTION 2: D2 METHOD COMPARISON PER SIGNAL
-- Which D2 estimation method (raw, smooth, spectral) is best per signal?
-- ============================================================================

SELECT
    signal_id,
    ROUND(d2_raw_snr, 2) AS raw_snr,
    ROUND(d2_smooth_snr, 2) AS smooth_snr,
    ROUND(d2_spectral_snr, 2) AS spectral_snr,
    d2_method_best AS best_method,
    ROUND(d2_best_snr, 2) AS best_snr,
    CASE
        WHEN d2_best_snr IS NULL THEN 'NO_D2'
        WHEN d2_best_snr < 2.0 THEN 'NOISE'
        WHEN d2_best_snr < 5.0 THEN 'WEAK'
        WHEN d2_best_snr < 20.0 THEN 'MODERATE'
        ELSE 'STRONG'
    END AS d2_quality
FROM typology_raw
WHERE d2_raw_snr IS NOT NULL
   OR d2_smooth_snr IS NOT NULL
   OR d2_spectral_snr IS NOT NULL
ORDER BY d2_best_snr DESC NULLS LAST;


-- ============================================================================
-- SECTION 3: D1 LATE-TO-EARLY RATIO (DEGRADATION ACCELERATION)
-- Signals where rate of change accelerates over time → degradation candidates
-- ============================================================================

SELECT
    signal_id,
    ROUND(d1_late_to_early_ratio, 3) AS d1_accel_ratio,
    d1_max_region,
    ROUND(d1_onset_pct, 3) AS d1_onset_pct,
    derivative_depth,
    ROUND(d1_snr, 2) AS d1_snr,
    CASE
        WHEN d1_late_to_early_ratio IS NULL THEN 'UNKNOWN'
        WHEN d1_late_to_early_ratio > 5.0 THEN 'STRONG_ACCELERATION'
        WHEN d1_late_to_early_ratio > 2.0 THEN 'MODERATE_ACCELERATION'
        WHEN d1_late_to_early_ratio > 1.2 THEN 'MILD_ACCELERATION'
        WHEN d1_late_to_early_ratio > 0.8 THEN 'STABLE_RATE'
        ELSE 'DECELERATING'
    END AS degradation_status
FROM typology_raw
WHERE d1_late_to_early_ratio IS NOT NULL
ORDER BY d1_late_to_early_ratio DESC;


-- ============================================================================
-- SECTION 4: D2 ONSET LOCATIONS
-- WHERE does curvature appear in each signal's lifetime?
-- ============================================================================

SELECT
    signal_id,
    ROUND(d2_onset_pct, 3) AS d2_onset_pct,
    d2_max_region,
    ROUND(d2_late_to_early_ratio, 3) AS d2_accel_ratio,
    derivative_depth,
    CASE
        WHEN d2_onset_pct IS NULL THEN 'NO_ONSET'
        WHEN d2_onset_pct < 0.2 THEN 'EARLY_NONLINEAR'
        WHEN d2_onset_pct < 0.5 THEN 'MID_ONSET'
        WHEN d2_onset_pct < 0.8 THEN 'LATE_ONSET'
        ELSE 'VERY_LATE_ONSET'
    END AS onset_timing,
    CASE
        WHEN d2_onset_pct IS NOT NULL AND d2_onset_pct BETWEEN 0.2 AND 0.95
            THEN 'ADAPTIVE_SPLIT'
        WHEN d2_onset_pct IS NOT NULL AND d2_onset_pct < 0.2
            THEN 'NONLINEAR_THROUGHOUT'
        ELSE 'UNIFORM_WINDOW'
    END AS windowing_recommendation
FROM typology_raw
WHERE d2_onset_pct IS NOT NULL
ORDER BY d2_onset_pct;


-- ============================================================================
-- SECTION 5: PER-SIGNAL DERIVATIVE FINGERPRINT
-- Full derivative profile for each signal
-- ============================================================================

SELECT
    signal_id,
    n_samples,
    derivative_depth,
    ROUND(d1_abs_mean, 4) AS d1_abs_mean,
    ROUND(d1_snr, 2) AS d1_snr,
    ROUND(d2_abs_mean, 4) AS d2_abs_mean,
    ROUND(d2_snr, 2) AS d2_snr,
    ROUND(d3_mean, 4) AS d3_mean,
    ROUND(d3_snr, 2) AS d3_snr,
    ROUND(d1_onset_pct, 3) AS d1_onset,
    ROUND(d2_onset_pct, 3) AS d2_onset,
    d2_method_best,
    CASE
        WHEN derivative_depth = 0 THEN 'FLAT'
        WHEN derivative_depth = 1 AND d1_max_region = 'late' THEN 'LINEAR_LATE'
        WHEN derivative_depth = 1 THEN 'LINEAR'
        WHEN derivative_depth = 2 AND d2_onset_pct IS NOT NULL THEN 'QUADRATIC_ONSET'
        WHEN derivative_depth = 2 THEN 'QUADRATIC'
        WHEN derivative_depth >= 3 THEN 'DEEP'
    END AS derivative_signature
FROM typology_raw
ORDER BY derivative_depth DESC, d2_snr DESC NULLS LAST;


-- ============================================================================
-- SECTION 6: CANARY CROSS-REFERENCE
-- Which signals show early derivative onset AND high late-to-early ratio?
-- (Commented out — requires canary_detection report 13)
-- ============================================================================

-- SELECT
--     t.signal_id,
--     t.d2_onset_pct,
--     t.d1_late_to_early_ratio,
--     t.derivative_depth,
--     c.canary_rank,
--     c.departure_pct
-- FROM typology_raw t
-- LEFT JOIN canary_signals c ON t.signal_id = c.signal_id
-- WHERE t.d2_onset_pct IS NOT NULL
--   AND t.d1_late_to_early_ratio > 2.0
-- ORDER BY t.d2_onset_pct;


-- ============================================================================
-- SECTION 7: ADAPTIVE WINDOWING RECOMMENDATIONS
-- Based on derivative analysis, recommend windowing strategy per signal
-- ============================================================================

SELECT
    signal_id,
    n_samples,
    derivative_depth,
    ROUND(d2_onset_pct, 3) AS d2_onset_pct,
    d2_max_region,
    ROUND(d1_late_to_early_ratio, 3) AS d1_accel_ratio,
    CASE
        WHEN derivative_depth = 0 AND is_constant THEN 'SKIP'
        WHEN d2_onset_pct IS NOT NULL AND d2_onset_pct BETWEEN 0.2 AND 0.95
            THEN 'ADAPTIVE_SPLIT'
        WHEN d2_onset_pct IS NOT NULL AND d2_onset_pct < 0.2
            THEN 'NARROW_UNIFORM'
        WHEN d1_late_to_early_ratio > 3.0
            THEN 'NARROW_LATE'
        ELSE 'UNIFORM'
    END AS window_strategy,
    CASE
        WHEN derivative_depth = 0 AND is_constant
            THEN 'Constant signal — skip windowed analysis'
        WHEN d2_onset_pct IS NOT NULL AND d2_onset_pct BETWEEN 0.2 AND 0.95
            THEN 'Pre-onset: wide windows; Post-onset: narrow windows'
        WHEN d2_onset_pct IS NOT NULL AND d2_onset_pct < 0.2
            THEN 'Nonlinear throughout — use narrow windows everywhere'
        WHEN d1_late_to_early_ratio > 3.0
            THEN 'Rate accelerates late — focus resolution on final third'
        ELSE 'No strong onset detected — uniform windowing sufficient'
    END AS rationale
FROM typology_raw
ORDER BY
    CASE
        WHEN d2_onset_pct IS NOT NULL AND d2_onset_pct BETWEEN 0.2 AND 0.95 THEN 0
        WHEN d2_onset_pct IS NOT NULL AND d2_onset_pct < 0.2 THEN 1
        WHEN d1_late_to_early_ratio > 3.0 THEN 2
        ELSE 3
    END,
    signal_id;
