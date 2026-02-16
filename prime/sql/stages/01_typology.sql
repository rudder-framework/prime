-- ============================================================================
-- Typology Stage Reports
-- ============================================================================
-- Signal classification results from typology analysis
--
-- Input: typology.parquet (from Prime)
-- Dual classification: temporal_primary is the main label,
-- temporal_secondary is non-null for boundary signals.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Classification Summary
-- ----------------------------------------------------------------------------
-- Count signals by temporal pattern (primary)
SELECT
    temporal_primary,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
FROM typology
GROUP BY temporal_primary
ORDER BY count DESC;

-- ----------------------------------------------------------------------------
-- 2. Full Typology View
-- ----------------------------------------------------------------------------
-- All classification dimensions per signal
SELECT
    signal_id,
    temporal_primary,
    temporal_secondary,
    classification_confidence,
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
    temporal_primary,
    spectral,
    COUNT(*) as count
FROM typology
GROUP BY temporal_primary, spectral
ORDER BY count DESC
LIMIT 20;

-- ----------------------------------------------------------------------------
-- 6. Dual Classification Summary
-- ----------------------------------------------------------------------------
SELECT
    temporal_primary,
    temporal_secondary,
    classification_confidence,
    COUNT(*) as count
FROM typology
WHERE temporal_secondary IS NOT NULL
GROUP BY temporal_primary, temporal_secondary, classification_confidence
ORDER BY count DESC;
