-- ============================================================
-- SQL TYPOLOGY LAYER 1: PAIRWISE CORRELATIONS
-- ============================================================
-- Input:  observations.parquet + signal_statistics.parquet
-- Output: signal_correlations.parquet
-- Time:   seconds on millions of rows
--
-- Computes Pearson correlation between all signal pairs per window
-- per cohort. This replaces the pairwise correlation computation
-- currently done in Python.
--
-- Python/Rust pairwise still handles: Granger causality, 
-- transfer entropy, cointegration, mutual information
-- (these require iterative algorithms SQL can't express)
-- ============================================================

-- Step 1: Full-signal pairwise correlation
CREATE OR REPLACE TABLE pairwise_full AS
WITH active_signals AS (
    SELECT cohort, signal_id
    FROM signal_statistics
    WHERE is_constant = FALSE
),
paired AS (
    SELECT 
        a.cohort,
        a.signal_id as signal_a,
        b.signal_id as signal_b,
        a.I,
        a.value as value_a,
        b.value as value_b
    FROM observations a
    JOIN observations b
        ON a.cohort = b.cohort AND a.I = b.I
        AND a.signal_id < b.signal_id
    -- Only active signals
    WHERE (a.cohort, a.signal_id) IN (SELECT * FROM active_signals)
      AND (b.cohort, b.signal_id) IN (SELECT * FROM active_signals)
)
SELECT 
    cohort,
    signal_a,
    signal_b,
    CORR(value_a, value_b) as correlation,
    ABS(CORR(value_a, value_b)) as abs_correlation,
    COUNT(*) as n_obs
FROM paired
GROUP BY cohort, signal_a, signal_b;


-- Step 2: Windowed pairwise correlation
-- Uses system.window and system.stride from manifest
CREATE OR REPLACE TABLE pairwise_windowed AS
WITH active_signals AS (
    SELECT cohort, signal_id
    FROM signal_statistics
    WHERE is_constant = FALSE
),
paired_windowed AS (
    SELECT 
        a.cohort,
        a.signal_id as signal_a,
        b.signal_id as signal_b,
        FLOOR(a.I / {stride}) as window_id,
        a.value as value_a,
        b.value as value_b
    FROM observations a
    JOIN observations b
        ON a.cohort = b.cohort AND a.I = b.I
        AND a.signal_id < b.signal_id
    WHERE (a.cohort, a.signal_id) IN (SELECT * FROM active_signals)
      AND (b.cohort, b.signal_id) IN (SELECT * FROM active_signals)
)
SELECT 
    cohort,
    signal_a,
    signal_b,
    window_id,
    CORR(value_a, value_b) as correlation,
    ABS(CORR(value_a, value_b)) as abs_correlation,
    COUNT(*) as n_obs
FROM paired_windowed
GROUP BY cohort, signal_a, signal_b, window_id
HAVING COUNT(*) >= 10;  -- minimum samples for meaningful correlation


-- Step 3: Correlation stability (does the pair relationship change over time?)
CREATE OR REPLACE TABLE correlation_stability AS
WITH valid_windowed AS (
    SELECT cohort, signal_a, signal_b, window_id, correlation
    FROM pairwise_windowed
    WHERE correlation IS NOT NULL
      AND isfinite(correlation)
),
corr_agg AS (
    SELECT
        cohort, signal_a, signal_b,
        AVG(correlation) as mean_correlation,
        CASE WHEN COUNT(*) > 2 THEN STDDEV_POP(correlation) ELSE 0 END as std_correlation,
        CASE WHEN COUNT(*) > 2 THEN REGR_SLOPE(correlation, window_id) ELSE 0 END as correlation_trend,
        MIN(correlation) as min_correlation,
        MAX(correlation) as max_correlation,
        COUNT(*) as n_windows
    FROM valid_windowed
    GROUP BY cohort, signal_a, signal_b
)
SELECT *,
    CASE WHEN std_correlation IS NOT NULL AND std_correlation > 0.3 THEN 'UNSTABLE'
         WHEN std_correlation IS NOT NULL AND std_correlation > 0.15 THEN 'MODERATE'
         ELSE 'STABLE'
    END as coupling_stability
FROM corr_agg;


-- Step 4: Coupling progression per cohort (fleet-level)
-- How many strong correlations per window?
CREATE OR REPLACE TABLE coupling_progression AS
SELECT 
    cohort,
    window_id,
    COUNT(*) as n_pairs,
    SUM(CASE WHEN abs_correlation > 0.7 THEN 1 ELSE 0 END) as n_strong,
    SUM(CASE WHEN abs_correlation > 0.9 THEN 1 ELSE 0 END) as n_very_strong,
    AVG(abs_correlation) as avg_abs_correlation,
    MEDIAN(abs_correlation) as median_abs_correlation
FROM pairwise_windowed
GROUP BY cohort, window_id;


-- Step 5: Assemble output
CREATE OR REPLACE TABLE signal_correlations AS
SELECT 
    f.cohort,
    f.signal_a,
    f.signal_b,
    f.correlation as full_signal_correlation,
    f.abs_correlation as full_signal_abs_corr,
    cs.mean_correlation,
    cs.std_correlation,
    cs.coupling_stability,
    cs.correlation_trend,
    cs.min_correlation,
    cs.max_correlation,
    cs.n_windows
FROM pairwise_full f
LEFT JOIN correlation_stability cs 
    ON f.cohort = cs.cohort 
    AND f.signal_a = cs.signal_a 
    AND f.signal_b = cs.signal_b;

-- Export
COPY signal_correlations TO '{output_dir}/signal_correlations.parquet' (FORMAT PARQUET);
COPY pairwise_windowed TO '{output_dir}/pairwise_windowed.parquet' (FORMAT PARQUET);
COPY coupling_progression TO '{output_dir}/coupling_progression.parquet' (FORMAT PARQUET);
