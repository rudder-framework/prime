-- ============================================================================
-- Typology Stage Reports
-- ============================================================================
-- Signal classification results from typology analysis
--
-- Input: typology.parquet (from Prime)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Classification Summary
-- ----------------------------------------------------------------------------
-- Count signals by temporal pattern
SELECT
    temporal_pattern,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
FROM typology
GROUP BY temporal_pattern
ORDER BY count DESC;

-- ----------------------------------------------------------------------------
-- 2. Full Typology View
-- ----------------------------------------------------------------------------
-- All classification dimensions per signal
SELECT
    signal_id,
    temporal_pattern,
    spectral,
    stationarity,
    memory,
    complexity,
    continuity,
    determinism,
    distribution,
    amplitude,
    volatility
FROM typology
ORDER BY signal_id;

-- ----------------------------------------------------------------------------
-- 3. Spectral Distribution
-- ----------------------------------------------------------------------------
SELECT
    spectral,
    COUNT(*) as count
FROM typology
GROUP BY spectral
ORDER BY count DESC;

-- ----------------------------------------------------------------------------
-- 4. Memory Classification
-- ----------------------------------------------------------------------------
SELECT
    memory,
    COUNT(*) as count
FROM typology
GROUP BY memory
ORDER BY count DESC;

-- ----------------------------------------------------------------------------
-- 5. Cross-tabulation: Temporal vs Spectral
-- ----------------------------------------------------------------------------
SELECT
    temporal_pattern,
    spectral,
    COUNT(*) as count
FROM typology
GROUP BY temporal_pattern, spectral
ORDER BY count DESC
LIMIT 20;
