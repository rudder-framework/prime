-- ORTHON Typology SQL
--
-- Signal classification for engine selection.
-- ORTHON runs this BEFORE sending to PRISM.
--
-- Output columns:
--   unit_id, signal_id, signal_type, periodicity_type, tail_type,
--   stationarity_type, smoothness, memory_proxy, recommended_window
--
-- Classifications:
--   signal_type:      CONSTANT, SMOOTH, NOISY, IMPULSIVE, MIXED
--   periodicity_type: CONSTANT, PERIODIC, QUASI_PERIODIC, APERIODIC
--   tail_type:        CONSTANT, HEAVY_TAILS, MODERATE_TAILS, LIGHT_TAILS, NORMAL_TAILS
--   stationarity_type: CONSTANT, NON_STATIONARY, HIGHLY_STATIONARY, STATIONARY

WITH base AS (
    SELECT
        unit_id,
        signal_id,
        I,
        value,
        LAG(value, 1) OVER w AS lag1,
        LAG(value, 10) OVER w AS lag10,
        LAG(value, 20) OVER w AS lag20,
        LAG(value, 50) OVER w AS lag50
    FROM observations
    WINDOW w AS (PARTITION BY unit_id, signal_id ORDER BY I)
)

SELECT
    unit_id,
    signal_id,

    -- Sample size
    COUNT(*) AS n_samples,

    -- Basic stats (for diagnostics)
    STDDEV(value) AS signal_std,
    AVG(value) AS signal_mean,
    MIN(value) AS signal_min,
    MAX(value) AS signal_max,

    -- 1. SMOOTHNESS (autocorr lag-1)
    -- Range: -1 to 1, typically 0 to 1
    -- High = smooth/continuous, Low = noisy/jumpy
    CORR(value, lag1) AS smoothness,

    -- 2. PERIODICITY RATIO (autocorr lag-10 / lag-1)
    -- High = periodic, Low = aperiodic
    CASE
        WHEN ABS(CORR(value, lag1)) > 0.1
        THEN CORR(value, lag10) / NULLIF(CORR(value, lag1), 0)
        ELSE 0
    END AS periodicity_ratio,

    -- Raw autocorrelations (for advanced analysis)
    CORR(value, lag1) AS autocorr_1,
    CORR(value, lag10) AS autocorr_10,
    CORR(value, lag20) AS autocorr_20,
    CORR(value, lag50) AS autocorr_50,

    -- 3. TAIL BEHAVIOR
    KURTOSIS(value) AS kurtosis,
    SKEWNESS(value) AS skewness,

    -- 4. MEMORY PROXY (diff std / value std)
    -- Low ratio (<0.5) = high memory (smooth, trending) -> Hurst > 0.5
    -- High ratio (>1.5) = low memory (noisy, reverting) -> Hurst < 0.5
    -- ~1.0 = random walk -> Hurst ~ 0.5
    STDDEV(value - lag1) / NULLIF(STDDEV(value), 0) AS memory_proxy,

    -- ==========================================================
    -- CLASSIFICATIONS
    -- ==========================================================

    -- Is constant (skip this signal)
    CASE WHEN STDDEV(value) < 0.001 THEN TRUE ELSE FALSE END AS is_constant,

    -- Signal type classification
    CASE
        WHEN STDDEV(value) < 0.001 THEN 'CONSTANT'
        WHEN ABS(CORR(value, lag1)) > 0.95 THEN 'SMOOTH'
        WHEN ABS(CORR(value, lag1)) < 0.3 THEN 'NOISY'
        WHEN KURTOSIS(value) > 5 THEN 'IMPULSIVE'
        ELSE 'MIXED'
    END AS signal_type,

    -- Periodicity classification
    CASE
        WHEN STDDEV(value) < 0.001 THEN 'CONSTANT'
        WHEN CORR(value, lag10) / NULLIF(CORR(value, lag1), 0) > 0.7 THEN 'PERIODIC'
        WHEN CORR(value, lag10) / NULLIF(CORR(value, lag1), 0) > 0.3 THEN 'QUASI_PERIODIC'
        ELSE 'APERIODIC'
    END AS periodicity_type,

    -- Tail classification
    CASE
        WHEN STDDEV(value) < 0.001 THEN 'CONSTANT'
        WHEN KURTOSIS(value) > 6 THEN 'HEAVY_TAILS'
        WHEN KURTOSIS(value) > 4 THEN 'MODERATE_TAILS'
        WHEN KURTOSIS(value) < 2 THEN 'LIGHT_TAILS'
        ELSE 'NORMAL_TAILS'
    END AS tail_type,

    -- Stationarity classification
    CASE
        WHEN STDDEV(value) < 0.001 THEN 'CONSTANT'
        WHEN STDDEV(value - lag1) / NULLIF(STDDEV(value), 0) > 1.5 THEN 'NON_STATIONARY'
        WHEN STDDEV(value - lag1) / NULLIF(STDDEV(value), 0) < 0.3 THEN 'HIGHLY_STATIONARY'
        ELSE 'STATIONARY'
    END AS stationarity_type,

    -- Recommended window size based on autocorrelation decay
    CASE
        WHEN STDDEV(value) < 0.001 THEN NULL  -- constant, no window needed
        WHEN ABS(CORR(value, lag1)) > 0.95 THEN 10   -- very smooth, small window OK
        WHEN ABS(CORR(value, lag1)) > 0.9 THEN 15
        WHEN ABS(CORR(value, lag1)) > 0.8 THEN 20
        WHEN ABS(CORR(value, lag1)) > 0.6 THEN 30
        WHEN ABS(CORR(value, lag1)) > 0.4 THEN 50
        ELSE 100  -- noisy, need larger window
    END AS recommended_window

FROM base
WHERE lag1 IS NOT NULL
GROUP BY unit_id, signal_id
ORDER BY unit_id, signal_id
