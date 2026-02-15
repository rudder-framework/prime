-- ============================================================================
-- Engines REGIME DETECTION REPORTS
-- ============================================================================
--
-- Identifies distinct operating states/regimes in process data.
-- Detects when the system shifts between different modes of operation.
-- Per-signal detection avoids scale mixing (temperatures vs pressures vs ratios).
-- Uses percentage of signals shifted, not z-score magnitudes.
--
-- Usage: Run against observations table
-- ============================================================================


-- ============================================================================
-- REPORT 1: PER-SIGNAL REGIME IDENTIFICATION
-- Detects regime shifts per signal, then counts how many signals shift per window
-- Boundary type based on pct_signals_shifted and vol_shifts, not z-score magnitude
-- ============================================================================

WITH
windowed AS (
    SELECT
        cohort,
        signal_id,
        NTILE(20) OVER (PARTITION BY cohort, signal_id ORDER BY I) AS window_id,
        I,
        value
    FROM observations
),

window_stats AS (
    SELECT
        cohort,
        signal_id,
        window_id,
        AVG(value) AS sig_mean,
        STDDEV_POP(value) AS sig_std
    FROM windowed
    GROUP BY cohort, signal_id, window_id
),

-- Detect regime changes per signal via mean shifts relative to own variability
signal_boundaries AS (
    SELECT
        cohort,
        signal_id,
        window_id,
        sig_mean,
        sig_std,
        ABS(sig_mean - LAG(sig_mean) OVER w) / NULLIF(sig_std, 0) AS mean_change_magnitude,
        sig_std / NULLIF(LAG(sig_std) OVER w, 0) AS std_ratio
    FROM window_stats
    WINDOW w AS (PARTITION BY cohort, signal_id ORDER BY window_id)
),

-- Aggregate: how many signals shifted per cohort per window
window_regime_summary AS (
    SELECT
        cohort,
        window_id,
        COUNT(*) AS n_signals,
        SUM(CASE WHEN mean_change_magnitude > 1.5 THEN 1 ELSE 0 END) AS n_mean_shifts,
        SUM(CASE WHEN std_ratio > 1.5 OR std_ratio < 0.67 THEN 1 ELSE 0 END) AS n_vol_shifts
    FROM signal_boundaries
    WHERE mean_change_magnitude IS NOT NULL
    GROUP BY cohort, window_id
)

SELECT
    cohort,
    window_id,
    n_signals,
    n_mean_shifts,
    n_vol_shifts,
    ROUND(100.0 * n_mean_shifts / n_signals, 1) AS pct_signals_shifted,
    CASE
        WHEN 100.0 * n_mean_shifts / n_signals > 25 THEN 'REGIME_CHANGE'
        WHEN 100.0 * n_mean_shifts / n_signals > 5 THEN 'PARTIAL_SHIFT'
        WHEN n_vol_shifts > n_signals * 0.3 THEN 'VOLATILITY_CHANGE'
        WHEN n_mean_shifts BETWEEN 1 AND 3 THEN 'SIGNAL_DRIFT'
        ELSE 'STABLE'
    END AS boundary_type
FROM window_regime_summary
WHERE n_mean_shifts > 0 OR n_vol_shifts > 0
ORDER BY cohort, window_id;


-- ============================================================================
-- REPORT 2: WHICH SIGNALS SHIFT AND WHEN
-- Per-signal regime changes ranked by shift magnitude
-- ============================================================================

WITH
windowed AS (
    SELECT
        cohort,
        signal_id,
        NTILE(20) OVER (PARTITION BY cohort, signal_id ORDER BY I) AS window_id,
        value
    FROM observations
),

window_stats AS (
    SELECT cohort, signal_id, window_id,
        AVG(value) AS sig_mean, STDDEV_POP(value) AS sig_std
    FROM windowed
    GROUP BY cohort, signal_id, window_id
),

signal_boundaries AS (
    SELECT
        cohort, signal_id, window_id, sig_mean, sig_std,
        ABS(sig_mean - LAG(sig_mean) OVER w) / NULLIF(sig_std, 0) AS mean_change_magnitude,
        sig_std / NULLIF(LAG(sig_std) OVER w, 0) AS std_ratio
    FROM window_stats
    WINDOW w AS (PARTITION BY cohort, signal_id ORDER BY window_id)
)

SELECT
    cohort,
    signal_id,
    window_id,
    ROUND(sig_mean, 4) AS sig_mean,
    ROUND(mean_change_magnitude, 2) AS change_magnitude,
    ROUND(std_ratio, 2) AS volatility_ratio,
    RANK() OVER (PARTITION BY cohort ORDER BY mean_change_magnitude DESC NULLS LAST) AS shift_rank
FROM signal_boundaries
WHERE mean_change_magnitude > 1.5
ORDER BY mean_change_magnitude DESC
LIMIT 50;


-- ============================================================================
-- REPORT 3: STEADY STATE DETECTION
-- Identifies periods of stable operation
-- ============================================================================

WITH
windowed AS (
    SELECT
        cohort,
        signal_id,
        NTILE(50) OVER (PARTITION BY cohort, signal_id ORDER BY I) AS window_id,
        value
    FROM observations
),

window_stats AS (
    SELECT
        cohort,
        signal_id,
        window_id,
        AVG(value) AS window_mean,
        STDDEV_POP(value) AS window_std
    FROM windowed
    GROUP BY cohort, signal_id, window_id
),

steady_check AS (
    SELECT
        cohort,
        window_id,
        -- Count signals that are steady (low derivative, low volatility)
        SUM(CASE
            WHEN ABS(window_mean - LAG(window_mean) OVER w) / NULLIF(window_std, 0) < 0.2
            THEN 1 ELSE 0
        END) AS n_steady_signals,
        COUNT(DISTINCT signal_id) AS n_total_signals
    FROM window_stats
    WINDOW w AS (PARTITION BY cohort, signal_id ORDER BY window_id)
    GROUP BY cohort, window_id
)

SELECT
    cohort,
    window_id,
    n_steady_signals,
    n_total_signals,
    ROUND(100.0 * n_steady_signals / n_total_signals, 1) AS pct_steady,
    CASE
        WHEN 100.0 * n_steady_signals / n_total_signals > 90 THEN 'STEADY_STATE'
        WHEN 100.0 * n_steady_signals / n_total_signals > 70 THEN 'QUASI_STEADY'
        WHEN 100.0 * n_steady_signals / n_total_signals > 50 THEN 'TRANSITION'
        ELSE 'DYNAMIC'
    END AS operating_mode
FROM steady_check
ORDER BY cohort, window_id;


-- ============================================================================
-- REPORT 4: TRANSIENT DETECTION
-- Identifies rapid change events (startups, shutdowns, disturbances)
-- Uses rate_of_change percentile instead of z-score
-- ============================================================================

WITH
derivatives AS (
    SELECT
        cohort,
        signal_id,
        I,
        value,
        value - LAG(value) OVER w AS dy,
        I - LAG(I) OVER w AS dI
    FROM observations
    WINDOW w AS (PARTITION BY cohort, signal_id ORDER BY I)
),

rates AS (
    SELECT
        cohort,
        signal_id,
        I,
        dy / NULLIF(dI, 0) AS rate_of_change
    FROM derivatives
    WHERE dy IS NOT NULL
),

rate_percentiles AS (
    SELECT
        cohort,
        signal_id,
        I,
        rate_of_change,
        PERCENT_RANK() OVER (
            PARTITION BY cohort, signal_id
            ORDER BY ABS(rate_of_change)
        ) AS rate_percentile
    FROM rates
)

SELECT
    cohort,
    signal_id,
    I AS transient_time,
    ROUND(rate_of_change, 4) AS rate,
    ROUND(rate_percentile, 4) AS rate_pctl,
    RANK() OVER (
        PARTITION BY cohort
        ORDER BY rate_percentile DESC
    ) AS transient_rank
FROM rate_percentiles
WHERE rate_percentile > 0.98
ORDER BY cohort, I, rate_percentile DESC;
