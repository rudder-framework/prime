-- ============================================================================
-- 25_feature_relevance.sql
-- ============================================================================
-- FEATURE RELEVANCE (UNSUPERVISED)
--
-- Ranks all Manifold-computed features by:
--   1. Variance — which features vary most across the dataset?
--   2. Temporal change — which features shift from early to late?
--   3. PCA modes — which features co-move (dominant structure)?
--   4. Composite significance — combined ranking across all three
--   5. Cross-report pointers — where to look next for each top feature
--
-- No labels, no training, no ML dependencies.
-- Requires: signal_vector (loaded as view by runner)
-- Optional: v_pca_modes, v_pca_loadings, v_feature_categories
--           (created by companion feature_relevance.py preprocessor)
-- ============================================================================


-- ============================================================================
-- 1. FEATURE VARIANCE RANKING
-- ============================================================================
-- Which features vary most across the dataset?
-- Uses coefficient of variation (std/mean) for scale-invariant comparison.

CREATE OR REPLACE VIEW v_feature_variance AS
WITH feature_long AS (
    UNPIVOT signal_vector
    ON COLUMNS(* EXCLUDE (signal_id, cohort, window_index, signal_0_center, signal_0_start, signal_0_end))
    INTO NAME feature_name VALUE feature_value
),
feature_stats AS (
    SELECT
        feature_name,
        COUNT(*) AS n_obs,
        AVG(feature_value) AS mean_val,
        STDDEV(feature_value) AS std_val,
        MIN(feature_value) AS min_val,
        MAX(feature_value) AS max_val
    FROM feature_long
    WHERE feature_value IS NOT NULL
      AND NOT isnan(feature_value)
      AND isfinite(feature_value)
    GROUP BY feature_name
    HAVING COUNT(*) >= 3 AND STDDEV(feature_value) > 1e-15
)
SELECT
    ROW_NUMBER() OVER (ORDER BY std_val / NULLIF(ABS(mean_val), 0) DESC NULLS LAST) AS variance_rank,
    feature_name,
    ROUND(std_val, 6) AS std,
    ROUND(CASE WHEN ABS(mean_val) > 1e-15 THEN std_val / ABS(mean_val) ELSE NULL END, 4) AS cv,
    ROUND(min_val, 6) AS min_val,
    ROUND(max_val, 6) AS max_val,
    n_obs
FROM feature_stats;

SELECT
    variance_rank AS rank,
    v.feature_name,
    COALESCE(c.category, 'SIGNAL') AS category,
    v.std,
    v.cv,
    v.min_val,
    v.max_val
FROM v_feature_variance v
LEFT JOIN v_feature_categories c ON v.feature_name = c.feature_name
ORDER BY variance_rank
LIMIT 50;


-- ============================================================================
-- 2. TEMPORAL CHANGE (EARLY vs LATE)
-- ============================================================================
-- Which features shift most between the first and last quintile of signal_0?
-- UNPIVOT preserves non-pivoted columns (signal_0_center stays in output).

CREATE OR REPLACE VIEW v_feature_temporal AS
WITH feature_long AS (
    UNPIVOT signal_vector
    ON COLUMNS(* EXCLUDE (signal_id, cohort, window_index, signal_0_center, signal_0_start, signal_0_end))
    INTO NAME feature_name VALUE feature_value
),
with_quintile AS (
    SELECT
        feature_name,
        feature_value,
        NTILE(5) OVER (PARTITION BY feature_name ORDER BY signal_0_center) AS quintile
    FROM feature_long
    WHERE feature_value IS NOT NULL AND NOT isnan(feature_value) AND isfinite(feature_value)
),
early_late AS (
    SELECT
        feature_name,
        AVG(CASE WHEN quintile = 1 THEN feature_value END) AS early_mean,
        AVG(CASE WHEN quintile = 5 THEN feature_value END) AS late_mean
    FROM with_quintile
    GROUP BY feature_name
    HAVING COUNT(CASE WHEN quintile = 1 THEN 1 END) >= 2
       AND COUNT(CASE WHEN quintile = 5 THEN 1 END) >= 2
)
SELECT
    ROW_NUMBER() OVER (ORDER BY ABS(late_mean - early_mean) DESC) AS temporal_rank,
    feature_name,
    ROUND(early_mean, 6) AS early_mean,
    ROUND(late_mean, 6) AS late_mean,
    ROUND(ABS(late_mean - early_mean), 6) AS abs_delta,
    ROUND(
        CASE WHEN ABS(early_mean) > 1e-15
             THEN (late_mean - early_mean) / ABS(early_mean) * 100
             ELSE NULL
        END, 2
    ) AS pct_change,
    CASE WHEN late_mean > early_mean THEN 'increasing' ELSE 'decreasing' END AS direction
FROM early_late
WHERE ABS(late_mean - early_mean) > 1e-15;

SELECT
    temporal_rank AS rank,
    t.feature_name,
    COALESCE(c.category, 'SIGNAL') AS category,
    t.early_mean,
    t.late_mean,
    t.abs_delta,
    t.pct_change,
    t.direction
FROM v_feature_temporal t
LEFT JOIN v_feature_categories c ON t.feature_name = c.feature_name
ORDER BY temporal_rank
LIMIT 50;


-- ============================================================================
-- 3. DOMINANT MODES (PCA)
-- ============================================================================
-- Which features co-move? What are the principal patterns?
-- Tables created by companion feature_relevance.py preprocessor.

SELECT
    'PC' || pc AS mode,
    ROUND(variance_explained * 100, 1) AS pct_variance,
    top_loading_1,
    top_loading_2,
    top_loading_3,
    interpretation
FROM v_pca_modes
ORDER BY pc;


-- ============================================================================
-- 4. COMPOSITE SIGNIFICANCE
-- ============================================================================
-- Combined ranking: average of variance rank, temporal rank, PCA loading rank.
-- Percentile thresholds computed via window functions (no cross join).

CREATE OR REPLACE VIEW v_feature_significance AS
WITH pca_rank AS (
    SELECT
        feature_name,
        ROW_NUMBER() OVER (ORDER BY MAX(abs_loading) DESC) AS pca_rank
    FROM v_pca_loadings
    GROUP BY feature_name
),
raw_scores AS (
    SELECT
        v.feature_name,
        v.variance_rank,
        COALESCE(t.temporal_rank, 9999) AS temporal_rank,
        COALESCE(p.pca_rank, 9999) AS pca_rank,
        (v.variance_rank + COALESCE(t.temporal_rank, 9999) + COALESCE(p.pca_rank, 9999)) / 3.0 AS overall_rank
    FROM v_feature_variance v
    LEFT JOIN v_feature_temporal t ON v.feature_name = t.feature_name
    LEFT JOIN pca_rank p ON v.feature_name = p.feature_name
),
with_pctl AS (
    SELECT
        *,
        PERCENT_RANK() OVER (ORDER BY overall_rank) AS pctl
    FROM raw_scores
)
SELECT
    feature_name,
    variance_rank,
    temporal_rank,
    pca_rank,
    ROUND(overall_rank, 1) AS overall_rank,
    CASE
        WHEN pctl <= 0.10 THEN 'HIGH'
        WHEN pctl <= 0.25 THEN 'MODERATE'
        ELSE 'LOW'
    END AS significance
FROM with_pctl;

SELECT
    ROW_NUMBER() OVER (ORDER BY overall_rank) AS rank,
    s.feature_name,
    COALESCE(c.category, 'SIGNAL') AS category,
    s.variance_rank,
    s.temporal_rank,
    s.pca_rank,
    s.overall_rank,
    s.significance
FROM v_feature_significance s
LEFT JOIN v_feature_categories c ON s.feature_name = c.feature_name
ORDER BY overall_rank
LIMIT 50;


-- ============================================================================
-- 5. CROSS-REPORT POINTERS
-- ============================================================================
-- For each top feature, which other report has detailed analysis?

WITH report_map (category, report_refs) AS (
    VALUES
        ('GEOMETRY', 'Reports 12 (brittleness), 15 (geometry ranked), 17 (dimension trajectory)'),
        ('COUPLING', 'Report 16 (coupling ranked)'),
        ('DYNAMICS', 'Reports 14 (derivative analysis), 18 (FTLE/Lyapunov)'),
        ('TOPOLOGY', 'Report 20 (topology)'),
        ('SIGNAL',   'Report 02 (signal statistics)')
)
SELECT
    ROW_NUMBER() OVER (ORDER BY s.overall_rank) AS rank,
    s.feature_name,
    COALESCE(c.category, 'SIGNAL') AS category,
    s.significance,
    rm.report_refs AS detailed_in
FROM v_feature_significance s
LEFT JOIN v_feature_categories c ON s.feature_name = c.feature_name
LEFT JOIN report_map rm ON COALESCE(c.category, 'SIGNAL') = rm.category
WHERE s.significance IN ('HIGH', 'MODERATE')
ORDER BY s.overall_rank
LIMIT 30;
