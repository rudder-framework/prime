-- ============================================================
-- SQL TYPOLOGY LAYER 1: SIGNAL STATISTICS
-- ============================================================
-- Input:  observations.parquet (canonical: cohort, signal_id, I, value)
-- Output: signal_statistics.parquet
-- Time:   seconds on millions of rows
--
-- Computes per (cohort, signal_id):
--   Basic stats, distribution shape, range, coefficient of variation
--   Constant detection, continuity classification
-- ============================================================

CREATE OR REPLACE TABLE signal_statistics AS

WITH signal_base AS (
    SELECT 
        cohort,
        signal_id,
        COUNT(*) as n_obs,
        AVG(value) as mean,
        STDDEV(value) as std,
        MIN(value) as min_val,
        MAX(value) as max_val,
        MEDIAN(value) as median_val,
        KURTOSIS(value) as kurtosis,
        SKEWNESS(value) as skewness,
        QUANTILE_CONT(value, 0.25) as q25,
        QUANTILE_CONT(value, 0.75) as q75,
        MAX(value) - MIN(value) as value_range
    FROM observations
    GROUP BY cohort, signal_id
),

enriched AS (
    SELECT *,
        CASE WHEN ABS(mean) > 1e-12 
             THEN std / ABS(mean) 
             ELSE NULL 
        END as cv,
        CASE WHEN std < 1e-10 THEN TRUE ELSE FALSE
        END as is_constant,
        q75 - q25 as iqr,
        CASE WHEN (q75 - q25) > 0
             THEN 1.5 * (q75 - q25)
             ELSE NULL
        END as outlier_fence
    FROM signal_base
),

unique_counts AS (
    SELECT 
        cohort,
        signal_id,
        COUNT(DISTINCT ROUND(value, 8)) as n_unique_values
    FROM observations
    GROUP BY cohort, signal_id
)

SELECT 
    e.*,
    u.n_unique_values,
    CASE
        WHEN u.n_unique_values <= 2 THEN 'CONSTANT'
        WHEN e.is_constant THEN 'CONSTANT'
        WHEN e.value_range = 0 THEN 'CONSTANT'
        WHEN u.n_unique_values <= 10 AND u.n_unique_values < e.n_obs * 0.01 THEN 'DISCRETE'
        ELSE 'CONTINUOUS'
    END as continuity
FROM enriched e
JOIN unique_counts u ON e.cohort = u.cohort AND e.signal_id = u.signal_id;

-- Export
COPY signal_statistics TO '{output_dir}/signal_statistics.parquet' (FORMAT PARQUET);
