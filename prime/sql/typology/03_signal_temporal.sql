-- ============================================================
-- SQL TYPOLOGY LAYER 1: TEMPORAL FEATURES
-- ============================================================
-- Input:  observations.parquet
-- Output: signal_temporal.parquet
-- Time:   seconds on millions of rows
--
-- Computes: trend strength (linear regression slope/R²)
--           monotonicity (% of D1 sign consistency)
--           zero crossing rate
--           reversal rate (sign changes in D1)
--           autocorrelation at lag-1
--           rolling mean stability
-- ============================================================

-- Step 1: Trend via linear regression
-- DuckDB has REGR_SLOPE and REGR_R2 aggregates
CREATE OR REPLACE TABLE signal_trend AS
SELECT 
    cohort,
    signal_id,
    REGR_SLOPE(value, I) as trend_slope,
    REGR_R2(value, I) as trend_r2,
    REGR_INTERCEPT(value, I) as trend_intercept,
    -- Trend strength: R² indicates how linear the signal is
    CASE 
        WHEN REGR_R2(value, I) > 0.8 THEN 'STRONG'
        WHEN REGR_R2(value, I) > 0.3 THEN 'MODERATE'
        WHEN REGR_R2(value, I) > 0.1 THEN 'WEAK'
        ELSE 'NONE'
    END as trend_strength_class
FROM observations
GROUP BY cohort, signal_id;


-- Step 2: Monotonicity and reversal rate from D1 signs
CREATE OR REPLACE TABLE signal_monotonicity AS
WITH d1_signs AS (
    SELECT 
        cohort, signal_id, I,
        value - LAG(value) OVER (PARTITION BY cohort, signal_id ORDER BY I) as d1
    FROM observations
),
sign_stats AS (
    SELECT 
        cohort, signal_id,
        COUNT(*) as n_diffs,
        SUM(CASE WHEN d1 > 0 THEN 1 ELSE 0 END) as n_positive,
        SUM(CASE WHEN d1 < 0 THEN 1 ELSE 0 END) as n_negative,
        SUM(CASE WHEN d1 = 0 THEN 1 ELSE 0 END) as n_zero
    FROM d1_signs
    WHERE d1 IS NOT NULL
    GROUP BY cohort, signal_id
),
d1_with_lag AS (
    SELECT
        cohort, signal_id, I, d1,
        LAG(d1) OVER (PARTITION BY cohort, signal_id ORDER BY I) as d1_prev
    FROM d1_signs
),
reversals AS (
    SELECT
        cohort, signal_id,
        SUM(CASE WHEN SIGN(d1) != SIGN(d1_prev)
                  AND d1 IS NOT NULL
                  AND d1_prev IS NOT NULL
             THEN 1 ELSE 0 END) as n_reversals
    FROM d1_with_lag
    GROUP BY cohort, signal_id
)
SELECT 
    s.cohort,
    s.signal_id,
    -- Monotonicity: fraction of D1 with dominant sign direction
    -- 1.0 = perfectly monotonic, 0.5 = random
    GREATEST(s.n_positive, s.n_negative) * 1.0 / NULLIF(s.n_diffs, 0) as monotonicity,
    -- Reversal rate: fraction of consecutive D1 that change sign
    r.n_reversals * 1.0 / NULLIF(s.n_diffs - 1, 0) as reversal_rate,
    -- Direction
    CASE 
        WHEN s.n_positive > s.n_negative * 1.5 THEN 'INCREASING'
        WHEN s.n_negative > s.n_positive * 1.5 THEN 'DECREASING'
        ELSE 'MIXED'
    END as trend_direction,
    s.n_positive, s.n_negative, s.n_zero,
    r.n_reversals
FROM sign_stats s
JOIN reversals r ON s.cohort = r.cohort AND s.signal_id = r.signal_id;


-- Step 3: Zero crossing rate
CREATE OR REPLACE TABLE signal_zero_crossings AS
WITH centered AS (
    SELECT
        cohort, signal_id, I,
        value - AVG(value) OVER (PARTITION BY cohort, signal_id) as centered_value
    FROM observations
),
centered_with_lag AS (
    SELECT
        cohort, signal_id, centered_value,
        LAG(centered_value) OVER (PARTITION BY cohort, signal_id ORDER BY I) as prev_centered
    FROM centered
),
crossings AS (
    SELECT
        cohort, signal_id,
        COUNT(*) as n_obs,
        SUM(CASE
            WHEN SIGN(centered_value) != SIGN(prev_centered)
             AND centered_value IS NOT NULL
             AND prev_centered IS NOT NULL
            THEN 1 ELSE 0
        END) as n_zero_crossings
    FROM centered_with_lag
    GROUP BY cohort, signal_id
)
SELECT
    cohort, signal_id,
    n_zero_crossings,
    n_zero_crossings * 1.0 / NULLIF(n_obs - 1, 0) as zero_crossing_rate
FROM crossings;


-- Step 4: Lag-1 autocorrelation
-- ACF(1) = correlation between value(I) and value(I-1)
CREATE OR REPLACE TABLE signal_acf1 AS
WITH lagged AS (
    SELECT 
        cohort, signal_id, I, value,
        LAG(value) OVER (PARTITION BY cohort, signal_id ORDER BY I) as value_lag1
    FROM observations
)
SELECT 
    cohort, signal_id,
    CORR(value, value_lag1) as acf_lag1,
    -- Memory classification from ACF(1)
    CASE 
        WHEN CORR(value, value_lag1) > 0.95 THEN 'LONG_MEMORY'
        WHEN CORR(value, value_lag1) > 0.7 THEN 'MODERATE_MEMORY'
        WHEN CORR(value, value_lag1) > 0.3 THEN 'SHORT_MEMORY'
        ELSE 'NO_MEMORY'
    END as memory_class_acf1
FROM lagged
WHERE value_lag1 IS NOT NULL
GROUP BY cohort, signal_id;


-- Step 5: Rolling mean stability (detects regime shifts)
-- Compare rolling mean in 10 equal segments
CREATE OR REPLACE TABLE signal_rolling_stability AS
WITH segmented AS (
    SELECT 
        cohort, signal_id, value,
        NTILE(10) OVER (PARTITION BY cohort, signal_id ORDER BY I) as segment
    FROM observations
),
segment_means AS (
    SELECT 
        cohort, signal_id, segment,
        AVG(value) as segment_mean,
        STDDEV(value) as segment_std
    FROM segmented
    GROUP BY cohort, signal_id, segment
)
SELECT 
    cohort, signal_id,
    -- Stability: stddev of segment means / overall std
    -- Low = stationary, High = drifting or regime changes
    STDDEV(segment_mean) as rolling_mean_std,
    AVG(segment_std) as avg_segment_std,
    CASE WHEN AVG(segment_std) > 0
         THEN STDDEV(segment_mean) / AVG(segment_std)
         ELSE 0
    END as mean_stability_ratio,
    MAX(segment_mean) - MIN(segment_mean) as mean_drift_range
FROM segment_means
GROUP BY cohort, signal_id;


-- Step 6: Assemble signal_temporal.parquet
CREATE OR REPLACE TABLE signal_temporal AS
SELECT 
    t.cohort,
    t.signal_id,
    
    -- Trend
    t.trend_slope,
    t.trend_r2,
    t.trend_strength_class,
    
    -- Monotonicity
    m.monotonicity,
    m.reversal_rate,
    m.trend_direction,
    
    -- Zero crossings
    z.zero_crossing_rate,
    
    -- ACF lag-1
    a.acf_lag1,
    a.memory_class_acf1,
    
    -- Rolling stability
    rs.mean_stability_ratio,
    rs.mean_drift_range

FROM signal_trend t
JOIN signal_monotonicity m ON t.cohort = m.cohort AND t.signal_id = m.signal_id
JOIN signal_zero_crossings z ON t.cohort = z.cohort AND t.signal_id = z.signal_id
JOIN signal_acf1 a ON t.cohort = a.cohort AND t.signal_id = a.signal_id
JOIN signal_rolling_stability rs ON t.cohort = rs.cohort AND t.signal_id = rs.signal_id;

-- Export
COPY signal_temporal TO '{output_dir}/signal_temporal.parquet' (FORMAT PARQUET);
