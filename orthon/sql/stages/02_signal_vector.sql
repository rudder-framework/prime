-- ============================================================================
-- ORTHON SQL: Signal Vector Stage Reports
-- ============================================================================
-- Per-signal scale-invariant features computed by PRISM
--
-- Input: signal_vector.parquet (from PRISM)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Signal Summary Statistics
-- ----------------------------------------------------------------------------
-- Basic stats per signal
SELECT
    signal_id,
    COUNT(*) as n_windows,
    MIN(I) as first_window,
    MAX(I) as last_window
FROM signal_vector
GROUP BY signal_id
ORDER BY signal_id;

-- ----------------------------------------------------------------------------
-- 2. Feature Statistics by Signal
-- ----------------------------------------------------------------------------
-- Average feature values per signal (common features)
SELECT
    signal_id,
    ROUND(AVG(kurtosis), 4) as avg_kurtosis,
    ROUND(AVG(skewness), 4) as avg_skewness,
    ROUND(AVG(crest_factor), 4) as avg_crest_factor
FROM signal_vector
GROUP BY signal_id
ORDER BY signal_id;

-- ----------------------------------------------------------------------------
-- 3. Recent Windows (last 10 per signal)
-- ----------------------------------------------------------------------------
SELECT *
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY signal_id ORDER BY I DESC) as rn
    FROM signal_vector
) ranked
WHERE rn <= 10
ORDER BY signal_id, I DESC;

-- ----------------------------------------------------------------------------
-- 4. Anomaly Detection: High Kurtosis Windows
-- ----------------------------------------------------------------------------
SELECT
    signal_id,
    I,
    kurtosis,
    skewness,
    crest_factor
FROM signal_vector
WHERE kurtosis > 10
ORDER BY kurtosis DESC
LIMIT 50;

-- ----------------------------------------------------------------------------
-- 5. Full Signal Vector (sample)
-- ----------------------------------------------------------------------------
SELECT * FROM signal_vector LIMIT 100;
