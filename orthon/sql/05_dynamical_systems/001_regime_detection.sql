-- ============================================================================
-- ORTHON SQL Engine: 05_dynamical_systems/001_regime_detection.sql
-- ============================================================================
-- Detect regime changes in signals
--
-- A regime change is a significant shift in signal behavior:
-- - Mean shift (jump > 5x local volatility)
-- - Variance shift (volatility ratio > 2.5x)
--
-- Only applies to interpolatable signals (analog, periodic)
-- ============================================================================

-- Step 1: Rolling statistics (no nesting)
CREATE OR REPLACE VIEW v_rolling_stats AS
SELECT
    b.signal_id,
    b.I,
    b.y,
    -- Rolling mean (window of 51 points centered)
    AVG(b.y) OVER (
        PARTITION BY b.signal_id
        ORDER BY b.I
        ROWS BETWEEN 25 PRECEDING AND 25 FOLLOWING
    ) AS rolling_mean,
    -- Rolling std
    STDDEV(b.y) OVER (
        PARTITION BY b.signal_id
        ORDER BY b.I
        ROWS BETWEEN 25 PRECEDING AND 25 FOLLOWING
    ) AS rolling_std
FROM v_base b
JOIN v_signal_class sc USING (signal_id)
WHERE sc.interpolation_valid = TRUE;

-- Step 2: Compute lagged changes using ratios
CREATE OR REPLACE VIEW v_regime_markers AS
SELECT
    signal_id,
    I,
    y,
    rolling_mean,
    rolling_std,
    LAG(rolling_mean, 50) OVER (PARTITION BY signal_id ORDER BY I) AS prev_mean,
    LAG(rolling_std, 50) OVER (PARTITION BY signal_id ORDER BY I) AS prev_std,
    -- Mean change (absolute)
    rolling_mean - LAG(rolling_mean, 50) OVER (
        PARTITION BY signal_id ORDER BY I
    ) AS mean_change,
    -- Std ratio (how many times larger)
    CASE
        WHEN LAG(rolling_std, 50) OVER (PARTITION BY signal_id ORDER BY I) > 0.001
        THEN rolling_std / LAG(rolling_std, 50) OVER (PARTITION BY signal_id ORDER BY I)
        ELSE NULL
    END AS std_ratio
FROM v_rolling_stats;

-- Regime change detection with stricter thresholds
CREATE OR REPLACE VIEW v_regime_changes AS
SELECT
    signal_id,
    I,
    mean_change,
    std_ratio,
    prev_std,
    rolling_std,
    -- Stricter thresholds:
    -- 1. Mean shift > 5x the local std (very significant jump)
    -- 2. OR variance ratio > 2.5 (volatility more than doubles)
    -- 3. OR variance ratio < 0.4 (volatility drops by more than half)
    CASE
        WHEN prev_std IS NOT NULL AND ABS(mean_change) > 5 * prev_std THEN 'mean_shift'
        WHEN std_ratio > 2.5 THEN 'variance_increase'
        WHEN std_ratio IS NOT NULL AND std_ratio < 0.4 THEN 'variance_decrease'
        ELSE NULL
    END AS change_type,
    -- Change magnitude for ranking
    CASE
        WHEN prev_std IS NOT NULL AND prev_std > 0 THEN ABS(mean_change) / prev_std
        ELSE 0
    END AS mean_shift_magnitude,
    CASE
        WHEN std_ratio IS NOT NULL THEN GREATEST(std_ratio, 1.0/NULLIF(std_ratio, 0))
        ELSE 1
    END AS variance_shift_magnitude
FROM v_regime_markers
WHERE mean_change IS NOT NULL;

-- Step 3a: Get changes with magnitude and previous I
CREATE OR REPLACE VIEW v_regime_changes_with_lag AS
SELECT
    signal_id,
    I,
    change_type,
    mean_shift_magnitude + variance_shift_magnitude AS total_magnitude,
    LAG(I) OVER (PARTITION BY signal_id ORDER BY I) AS prev_I
FROM v_regime_changes
WHERE change_type IS NOT NULL;

-- Step 3b: Assign cluster IDs based on 50-point gap
CREATE OR REPLACE VIEW v_regime_changes_clustered AS
SELECT
    signal_id,
    I,
    change_type,
    total_magnitude,
    SUM(CASE WHEN prev_I IS NULL OR I - prev_I > 50 THEN 1 ELSE 0 END)
        OVER (PARTITION BY signal_id ORDER BY I) AS cluster_id
FROM v_regime_changes_with_lag;

-- Step 3c: Keep only the strongest change in each cluster
CREATE OR REPLACE VIEW v_regime_change_points AS
WITH best_per_cluster AS (
    SELECT
        signal_id,
        cluster_id,
        FIRST(I ORDER BY total_magnitude DESC) AS I,
        FIRST(change_type ORDER BY total_magnitude DESC) AS change_type,
        MAX(total_magnitude) AS magnitude
    FROM v_regime_changes_clustered
    GROUP BY signal_id, cluster_id
)
SELECT
    signal_id,
    I,
    change_type,
    magnitude,
    ROW_NUMBER() OVER (PARTITION BY signal_id ORDER BY I) AS change_seq
FROM best_per_cluster
WHERE magnitude > 3  -- Only keep changes with significant magnitude
ORDER BY signal_id, I;

-- Regime assignments
CREATE OR REPLACE VIEW v_regimes AS
WITH change_points AS (
    SELECT signal_id, I, change_type, change_seq
    FROM v_regime_change_points
)
SELECT
    b.signal_id,
    b.I,
    b.y,
    COALESCE(
        (SELECT MAX(change_seq) FROM change_points cp
         WHERE cp.signal_id = b.signal_id AND cp.I <= b.I),
        0
    ) + 1 AS regime_id
FROM v_base b
JOIN v_signal_class sc USING (signal_id)
WHERE sc.interpolation_valid = TRUE;

-- Regime summary
CREATE OR REPLACE VIEW v_regime_summary AS
SELECT
    signal_id,
    regime_id,
    COUNT(*) AS n_points,
    MIN(I) AS regime_start,
    MAX(I) AS regime_end,
    ROUND(AVG(y), 2) AS regime_mean,
    ROUND(STDDEV(y), 2) AS regime_std
FROM v_regimes
GROUP BY signal_id, regime_id
ORDER BY signal_id, regime_id;
