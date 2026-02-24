-- ============================================================
-- SQL TYPOLOGY LAYER 1: DERIVATIVES
-- ============================================================
-- Input:  observations.parquet
-- Output: signal_derivatives.parquet (summary per cohort/signal)
-- Time:   seconds on millions of rows
--
-- Computes: D1, D2, D3, D4 via LAG window functions
--           Smoothed D1, D2 via rolling average
--           Noise floor via high-pass residual (MAD of value - smooth(value))
--           Derivative depth: how many levels have SNR > threshold vs noise floor
--           Onset detection (where D2 departs from baseline — null for chaotic)
--           Late-to-early ratios
--           Max region classification
-- ============================================================

-- Step 1: Compute raw derivatives + smoothed versions for every observation
CREATE OR REPLACE TABLE derivative_raw AS
WITH d1 AS (
    SELECT
        cohort, signal_id, I, value,
        value - LAG(value) OVER (PARTITION BY cohort, signal_id ORDER BY I) as d1
    FROM observations
),
d2 AS (
    SELECT *,
        d1 - LAG(d1) OVER (PARTITION BY cohort, signal_id ORDER BY I) as d2
    FROM d1
),
d3 AS (
    SELECT *,
        d2 - LAG(d2) OVER (PARTITION BY cohort, signal_id ORDER BY I) as d3
    FROM d2
),
d4 AS (
    SELECT *,
        d3 - LAG(d3) OVER (PARTITION BY cohort, signal_id ORDER BY I) as d4
    FROM d3
)
SELECT
    cohort, signal_id, I, value, d1, d2, d3, d4,
    -- Smoothed D1: rolling average over 11-point window
    AVG(d1) OVER (
        PARTITION BY cohort, signal_id ORDER BY I
        ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING
    ) as d1_smooth,
    -- Smoothed D2: rolling average over 11-point window
    AVG(d2) OVER (
        PARTITION BY cohort, signal_id ORDER BY I
        ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING
    ) as d2_smooth,
    -- Smoothed D3: rolling average over 11-point window
    AVG(d3) OVER (
        PARTITION BY cohort, signal_id ORDER BY I
        ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING
    ) as d3_smooth,
    -- Smoothed value: for noise floor estimation
    AVG(value) OVER (
        PARTITION BY cohort, signal_id ORDER BY I
        ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING
    ) as value_smooth,
    -- Position within signal life (0.0 to 1.0)
    (I - MIN(I) OVER (PARTITION BY cohort, signal_id)) * 1.0 /
    NULLIF(MAX(I) OVER (PARTITION BY cohort, signal_id) - MIN(I) OVER (PARTITION BY cohort, signal_id), 0) as life_pct
FROM d4;


-- Step 1b: Estimate noise floor via high-pass residual
-- noise_floor = MAD(value - smooth(value)) * 1.4826
-- This captures the noise component without destroying temporal structure
CREATE OR REPLACE TABLE signal_noise AS
SELECT
    cohort,
    signal_id,
    GREATEST(MEDIAN(ABS(value - value_smooth)) * 1.4826, 1e-12) as noise_floor,
    GREATEST(MEDIAN(ABS(d1 - d1_smooth)) * 1.4826, 1e-12) as d1_noise_floor,
    GREATEST(MEDIAN(ABS(d2 - d2_smooth)) * 1.4826, 1e-12) as d2_noise_floor,
    GREATEST(MEDIAN(ABS(d3 - d3_smooth)) * 1.4826, 1e-12) as d3_noise_floor
FROM derivative_raw
WHERE d3 IS NOT NULL
GROUP BY cohort, signal_id;


-- Step 2: Per-signal derivative statistics
CREATE OR REPLACE TABLE derivative_stats AS
SELECT
    cohort,
    signal_id,

    -- D1 statistics
    AVG(d1) as d1_mean,
    STDDEV(d1) as d1_std,
    AVG(ABS(d1)) as d1_abs_mean,

    -- D2 statistics
    AVG(d2) as d2_mean,
    STDDEV(d2) as d2_std,
    AVG(ABS(d2)) as d2_abs_mean,

    -- D2 smoothed statistics
    AVG(ABS(d2_smooth)) as d2_smooth_abs_mean,
    STDDEV(d2_smooth) as d2_smooth_std,
    CASE WHEN STDDEV(d2_smooth) > 0
         THEN AVG(ABS(d2_smooth)) / STDDEV(d2_smooth)
         ELSE 0
    END as d2_smooth_snr,

    -- D3 / D4 statistics
    AVG(ABS(d3)) as d3_abs_mean,
    AVG(ABS(d4)) as d4_abs_mean

FROM derivative_raw
GROUP BY cohort, signal_id;


-- Step 3: Derivative depth (how many levels are meaningful)
-- Uses high-pass residual noise floor: SNR = abs_mean(dk) / noise_floor_k
-- Threshold = 2.0 (same as Python typology_raw.py)
CREATE OR REPLACE TABLE derivative_depth AS
WITH snr_calc AS (
    SELECT
        ds.cohort,
        ds.signal_id,
        ds.d1_abs_mean / n.noise_floor as d1_snr,
        ds.d2_abs_mean / n.d1_noise_floor as d2_snr,
        ds.d3_abs_mean / n.d2_noise_floor as d3_snr,
        ds.d4_abs_mean / n.d3_noise_floor as d4_snr
    FROM derivative_stats ds
    JOIN signal_noise n ON ds.cohort = n.cohort AND ds.signal_id = n.signal_id
)
SELECT
    cohort,
    signal_id,
    CASE
        WHEN d1_snr < 2.0 THEN 0
        WHEN d2_snr < 2.0 THEN 1
        WHEN d3_snr < 2.0 THEN 2
        WHEN d4_snr < 2.0 THEN 3
        ELSE 4
    END as derivative_depth,
    d1_snr, d2_snr, d3_snr, d4_snr
FROM snr_calc;


-- Step 4: Late-to-early ratios (degradation acceleration)
CREATE OR REPLACE TABLE derivative_ratios AS
WITH early AS (
    SELECT cohort, signal_id,
        AVG(ABS(d1)) as d1_early,
        AVG(ABS(d2_smooth)) as d2_early
    FROM derivative_raw
    WHERE life_pct <= 0.4
    GROUP BY cohort, signal_id
),
late AS (
    SELECT cohort, signal_id,
        AVG(ABS(d1)) as d1_late,
        AVG(ABS(d2_smooth)) as d2_late
    FROM derivative_raw
    WHERE life_pct >= 0.6
    GROUP BY cohort, signal_id
)
SELECT
    e.cohort,
    e.signal_id,
    CASE WHEN e.d1_early > 1e-12
         THEN l.d1_late / e.d1_early
         ELSE NULL
    END as d1_late_to_early_ratio,
    CASE WHEN e.d2_early > 1e-12
         THEN l.d2_late / e.d2_early
         ELSE NULL
    END as d2_late_to_early_ratio
FROM early e
JOIN late l ON e.cohort = l.cohort AND e.signal_id = l.signal_id;


-- Step 5: D2 onset detection
-- Baseline = first 20% of signal life
-- Onset = first point where smoothed |D2| exceeds baseline + 3*std
-- SUPPRESSION: If D2 is already significant in the baseline window
-- (baseline mean > 50% of overall mean), there is no onset — the signal
-- was always active (chaotic/oscillating). Onset = NULL.
CREATE OR REPLACE TABLE derivative_onset AS
WITH baseline AS (
    SELECT cohort, signal_id,
        AVG(ABS(d2_smooth)) as d2_baseline_mean,
        STDDEV(d2_smooth) as d2_baseline_std
    FROM derivative_raw
    WHERE life_pct <= 0.2
    GROUP BY cohort, signal_id
),
overall AS (
    SELECT cohort, signal_id,
        AVG(ABS(d2_smooth)) as d2_overall_mean
    FROM derivative_raw
    GROUP BY cohort, signal_id
),
-- Suppress onset for signals where baseline D2 is already significant
-- (ratio > 0.5 means baseline has > 50% of overall D2 activity → no transition)
filtered_baseline AS (
    SELECT
        b.cohort, b.signal_id,
        b.d2_baseline_mean, b.d2_baseline_std,
        CASE WHEN o.d2_overall_mean > 1e-12
             THEN b.d2_baseline_mean / o.d2_overall_mean
             ELSE 0
        END as baseline_ratio
    FROM baseline b
    JOIN overall o ON b.cohort = o.cohort AND b.signal_id = o.signal_id
),
exceedances AS (
    SELECT
        d.cohort, d.signal_id, d.I, d.life_pct,
        ROW_NUMBER() OVER (PARTITION BY d.cohort, d.signal_id ORDER BY d.I) as exc_rank
    FROM derivative_raw d
    JOIN filtered_baseline b ON d.cohort = b.cohort AND d.signal_id = b.signal_id
    WHERE d.life_pct > 0.2
      AND b.baseline_ratio < 0.5
      AND ABS(d.d2_smooth) > b.d2_baseline_mean + 3 * GREATEST(b.d2_baseline_std, 1e-12)
)
SELECT
    cohort, signal_id,
    life_pct as d2_onset_pct,
    I as d2_onset_index
FROM exceedances
WHERE exc_rank = 1;


-- Step 6: D2 max region classification
CREATE OR REPLACE TABLE derivative_regions AS
WITH regional_d2 AS (
    SELECT cohort, signal_id,
        CASE
            WHEN life_pct < 0.33 THEN 'early'
            WHEN life_pct < 0.67 THEN 'mid'
            ELSE 'late'
        END as region,
        AVG(ABS(d2_smooth)) as region_d2
    FROM derivative_raw
    GROUP BY cohort, signal_id,
        CASE
            WHEN life_pct < 0.33 THEN 'early'
            WHEN life_pct < 0.67 THEN 'mid'
            ELSE 'late'
        END
),
pivoted AS (
    SELECT cohort, signal_id,
        MAX(CASE WHEN region = 'early' THEN region_d2 END) as d2_early,
        MAX(CASE WHEN region = 'mid' THEN region_d2 END) as d2_mid,
        MAX(CASE WHEN region = 'late' THEN region_d2 END) as d2_late
    FROM regional_d2
    GROUP BY cohort, signal_id
)
SELECT cohort, signal_id,
    CASE
        WHEN d2_early >= d2_mid AND d2_early >= d2_late THEN 'early'
        WHEN d2_mid >= d2_early AND d2_mid >= d2_late THEN 'mid'
        ELSE 'late'
    END as d2_max_region,
    d2_early, d2_mid, d2_late
FROM pivoted;


-- Step 7: Assemble signal_derivatives.parquet
CREATE OR REPLACE TABLE signal_derivatives AS
SELECT
    ds.cohort,
    ds.signal_id,

    -- D1
    ds.d1_mean, ds.d1_std, ds.d1_abs_mean, dd.d1_snr,

    -- D2
    ds.d2_mean, ds.d2_std, ds.d2_abs_mean, dd.d2_snr,
    ds.d2_smooth_abs_mean, ds.d2_smooth_std, ds.d2_smooth_snr,

    -- D3 / D4
    dd.d3_snr, dd.d4_snr,

    -- Derivative depth
    dd.derivative_depth,

    -- Late-to-early acceleration
    dr.d1_late_to_early_ratio,
    dr.d2_late_to_early_ratio,

    -- D2 onset
    do2.d2_onset_pct,
    do2.d2_onset_index,

    -- D2 max region
    dg.d2_max_region,
    dg.d2_early as d2_region_early,
    dg.d2_mid as d2_region_mid,
    dg.d2_late as d2_region_late

FROM derivative_stats ds
JOIN derivative_depth dd ON ds.cohort = dd.cohort AND ds.signal_id = dd.signal_id
LEFT JOIN derivative_ratios dr ON ds.cohort = dr.cohort AND ds.signal_id = dr.signal_id
LEFT JOIN derivative_onset do2 ON ds.cohort = do2.cohort AND ds.signal_id = do2.signal_id
LEFT JOIN derivative_regions dg ON ds.cohort = dg.cohort AND ds.signal_id = dg.signal_id;

-- Export
COPY signal_derivatives TO '{output_dir}/signal_derivatives.parquet' (FORMAT PARQUET);
