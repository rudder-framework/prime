-- ============================================================
-- Cross-Signal Analysis
-- ============================================================

-- Pairwise correlation between all signals (same I alignment)
SELECT
    a.signal_id as signal_a,
    b.signal_id as signal_b,
    ROUND(CORR(a.value, b.value), 4) as correlation,
    COUNT(*) as n_pairs
FROM observations a
JOIN observations b ON a.I = b.I AND a.signal_id < b.signal_id
GROUP BY a.signal_id, b.signal_id
ORDER BY ABS(correlation) DESC;


-- Example: Are magnitude and depth correlated?
SELECT
    ROUND(CORR(m.value, d.value), 4) as mag_depth_corr,
    COUNT(*) as n_samples
FROM observations m
JOIN observations d ON m.I = d.I
WHERE m.signal_id = 'daily_max_mag'
  AND d.signal_id = 'daily_mean_depth';


-- ACF pattern interpretation
SELECT
    signal_id,
    I,
    ROUND(acf_lag1, 4) as acf_lag1,
    CASE
        WHEN acf_lag1 < -0.3 THEN 'Strong negative (mean-reverting)'
        WHEN acf_lag1 < 0 THEN 'Weak negative'
        WHEN acf_lag1 < 0.3 THEN 'Weak positive'
        WHEN acf_lag1 < 0.7 THEN 'Moderate positive (trending)'
        ELSE 'Strong positive (persistent)'
    END as acf_pattern
FROM signal_vector
WHERE isfinite(acf_lag1)
ORDER BY signal_id, I;


-- Signals with similar behavior (feature correlation)
WITH feature_vectors AS (
    SELECT
        signal_id,
        AVG(crest_factor) FILTER (WHERE isfinite(crest_factor)) as crest,
        AVG(kurtosis) FILTER (WHERE isfinite(kurtosis)) as kurt,
        AVG(skewness) FILTER (WHERE isfinite(skewness)) as skew,
        AVG(acf_lag1) FILTER (WHERE isfinite(acf_lag1)) as acf1
    FROM signal_vector
    GROUP BY signal_id
)
SELECT
    a.signal_id as signal_a,
    b.signal_id as signal_b,
    ROUND(
        (a.crest * b.crest + a.kurt * b.kurt + a.skew * b.skew + a.acf1 * b.acf1) /
        (SQRT(a.crest*a.crest + a.kurt*a.kurt + a.skew*a.skew + a.acf1*a.acf1) *
         SQRT(b.crest*b.crest + b.kurt*b.kurt + b.skew*b.skew + b.acf1*b.acf1))
    , 4) as feature_similarity
FROM feature_vectors a
CROSS JOIN feature_vectors b
WHERE a.signal_id < b.signal_id
ORDER BY feature_similarity DESC;
