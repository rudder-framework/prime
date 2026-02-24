-- ============================================================
-- SQL TYPOLOGY LAYER 1: WINDOWED STATISTICS
-- ============================================================
-- Input:  observations.parquet + signal_statistics.parquet (for constant filtering)
-- Output: signal_windows.parquet
-- Time:   seconds on millions of rows
--
-- This replaces the basic statistics portion of the signal_vector 
-- computation. DuckDB computes windowed mean, std, kurtosis, skewness,
-- range, min, max for every window of every signal in seconds.
--
-- The Python/Rust signal_vector only needs to compute the EXPENSIVE
-- features per window: hurst, sample_entropy, perm_entropy, spectral.
--
-- Window size and stride come from the manifest (passed as parameters).
-- For now uses {window_size} and {stride} placeholders.
-- ============================================================

-- Note: {window_size} and {stride} are replaced at runtime from manifest
-- system.window and system.stride values.
-- Per-signal window sizes can be implemented via UNION ALL of 
-- per-signal queries, each with its own window/stride from manifest.

CREATE OR REPLACE TABLE signal_windows AS
WITH window_assignments AS (
    SELECT 
        cohort,
        signal_id,
        I,
        value,
        -- Compute which window this observation belongs to
        -- An observation can belong to multiple overlapping windows
        -- Window k starts at k * stride, ends at k * stride + window_size - 1
        FLOOR(I / {stride}) as primary_window_id,
        -- Total observations for this signal
        MAX(I) OVER (PARTITION BY cohort, signal_id) as max_I,
        MIN(I) OVER (PARTITION BY cohort, signal_id) as min_I
    FROM observations
    -- Skip constants
    WHERE (cohort, signal_id) NOT IN (
        SELECT cohort, signal_id FROM signal_statistics WHERE is_constant = TRUE
    )
),

-- Generate window IDs
-- Each window: start = window_id * stride, end = start + window_size - 1
windows AS (
    SELECT DISTINCT
        cohort,
        signal_id,
        w.window_id,
        w.window_id * {stride} as window_start,
        w.window_id * {stride} + {window_size} - 1 as window_end
    FROM window_assignments,
    LATERAL (
        SELECT UNNEST(
            GENERATE_SERIES(
                0, 
                GREATEST(0, FLOOR((max_I - min_I - {window_size}) / {stride}))::INTEGER
            )
        ) as window_id
    ) w
)

SELECT 
    w.cohort,
    w.signal_id,
    w.window_id,
    w.window_start,
    w.window_end,
    
    -- Basic statistics per window
    COUNT(o.value) as n_points,
    AVG(o.value) as window_mean,
    STDDEV(o.value) as window_std,
    MIN(o.value) as window_min,
    MAX(o.value) as window_max,
    MAX(o.value) - MIN(o.value) as window_range,
    MEDIAN(o.value) as window_median,
    KURTOSIS(o.value) as window_kurtosis,
    SKEWNESS(o.value) as window_skewness,
    
    -- Coefficient of variation per window
    CASE WHEN ABS(AVG(o.value)) > 1e-12
         THEN STDDEV(o.value) / ABS(AVG(o.value))
         ELSE NULL
    END as window_cv,
    
    -- IQR per window
    QUANTILE_CONT(o.value, 0.75) - QUANTILE_CONT(o.value, 0.25) as window_iqr,
    
    -- Window position in signal life (0.0 to 1.0)
    w.window_start * 1.0 / NULLIF(
        MAX(w.window_end) OVER (PARTITION BY w.cohort, w.signal_id), 0
    ) as window_life_pct

FROM windows w
JOIN observations o 
    ON w.cohort = o.cohort 
    AND w.signal_id = o.signal_id
    AND o.I >= w.window_start 
    AND o.I <= w.window_end
GROUP BY w.cohort, w.signal_id, w.window_id, w.window_start, w.window_end
HAVING COUNT(o.value) >= {window_size} * 0.8;  -- require 80% fill

-- Export
COPY signal_windows TO '{output_dir}/signal_windows.parquet' (FORMAT PARQUET);
