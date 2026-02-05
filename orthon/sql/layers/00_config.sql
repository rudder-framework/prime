-- ============================================================================
-- ORTHON SQL: 00_config.sql
-- ============================================================================
-- CONFIGURABLE THRESHOLDS AND METHODOLOGY PARAMETERS
--
-- Adjust these values based on your data characteristics and analysis needs.
-- Import this file before running other analysis scripts.
--
-- Usage:
--   .read 00_config.sql
--   .read 12_load_physics.sql
--   -- etc.
-- ============================================================================

-- ============================================================================
-- ENGINE MINIMUM DATA REQUIREMENTS
-- ============================================================================
-- Different engines require different amounts of data to produce meaningful
-- results. Below these thresholds, results should be filtered or flagged.
--
-- References:
--   - Lyapunov: Wolf et al. (1985) - attractor reconstruction requires ~10^4 pts
--   - Hurst: Mandelbrot & Wallis (1969) - R/S rescaling needs ~256 pts
--   - Spectral: Nyquist + frequency resolution requires 2× longest cycle
--   - Recurrence: Varies by embedding dimension and threshold selection
-- ============================================================================

CREATE OR REPLACE TABLE config_engine_requirements AS
SELECT
    -- Engine name, minimum observations, and rationale
    'lyapunov'    AS engine, 10000 AS min_observations, 'Attractor reconstruction'           AS rationale UNION ALL
SELECT 'hurst',               256,                        'R/S rescaling statistics'         UNION ALL
SELECT 'spectral',            NULL,                       '2× longest cycle (Nyquist)'       UNION ALL  -- Dynamic, depends on cycle
SELECT 'recurrence',          500,                        'Sufficient recurrence density'    UNION ALL
SELECT 'entropy',             100,                        'Symbol sequence statistics'       UNION ALL
SELECT 'garch',               250,                        'Volatility clustering detection'  UNION ALL
SELECT 'granger',             100,                        'VAR model estimation'             UNION ALL
SELECT 'transfer_entropy',    500,                        'Information transfer statistics'  UNION ALL
SELECT 'cointegration',       250,                        'Long-run equilibrium detection'   UNION ALL
SELECT 'dmd',                 100,                        'Dynamic mode estimation'          UNION ALL
SELECT 'attractor',           1000,                       'Strange attractor reconstruction' UNION ALL
SELECT 'basin',               1000,                       'Basin of attraction estimation';

-- ============================================================================
-- INTERPRETATION THRESHOLDS
-- ============================================================================
-- These thresholds define how metrics are classified into categories.
-- Adjust based on domain knowledge and system characteristics.
-- ============================================================================

CREATE OR REPLACE TABLE config_thresholds AS
SELECT
    -- -------------------------------------------------------------------------
    -- Lyapunov exponent interpretation
    -- -------------------------------------------------------------------------
    10000 AS lyapunov_min_observations,     -- Minimum observations for reliable Lyapunov
    0.1   AS lyapunov_chaotic_threshold,    -- λ > this = chaotic
    0.0   AS lyapunov_unstable_threshold,   -- λ > this = unstable (weakly chaotic)
    -0.1  AS lyapunov_stable_threshold,     -- λ < this = strongly stable

    -- -------------------------------------------------------------------------
    -- Hurst exponent interpretation
    -- -------------------------------------------------------------------------
    256   AS hurst_min_observations,        -- Minimum for reliable Hurst
    0.65  AS hurst_persistent_threshold,    -- H > this = persistent (trending)
    0.35  AS hurst_antipersistent_threshold,-- H < this = anti-persistent (mean-reverting)

    -- -------------------------------------------------------------------------
    -- Coherence interpretation (eigenvalue-based)
    -- -------------------------------------------------------------------------
    0.7   AS coherence_strongly_coupled,    -- coherence > this = strongly coupled
    0.4   AS coherence_weakly_coupled,      -- coherence > this = weakly coupled
    0.001 AS coherence_velocity_threshold,  -- |dC/dt| > this = significant change

    -- -------------------------------------------------------------------------
    -- Effective dimension interpretation
    -- -------------------------------------------------------------------------
    1.5   AS effective_dim_unified,         -- dim < this = unified mode
    0.5   AS effective_dim_fragmented_ratio,-- dim > N * this = fragmented

    -- -------------------------------------------------------------------------
    -- State interpretation
    -- -------------------------------------------------------------------------
    3.0   AS state_distance_significant,    -- distance > this σ = significantly different
    2.0   AS state_distance_notable,        -- distance > this σ = notable
    0.01  AS state_velocity_diverging,      -- velocity > this = diverging
    -0.01 AS state_velocity_converging,     -- velocity < this = converging

    -- -------------------------------------------------------------------------
    -- Energy interpretation
    -- -------------------------------------------------------------------------
    0.001 AS energy_velocity_threshold,     -- |dE/dt| > this = significant change

    -- -------------------------------------------------------------------------
    -- Entropy interpretation
    -- -------------------------------------------------------------------------
    0.8   AS entropy_high_complexity,       -- entropy > this = high complexity
    0.3   AS entropy_low_complexity         -- entropy < this = low complexity (predictable)
;

-- ============================================================================
-- HELPER VIEW: Data sufficiency per entity per engine
-- ============================================================================

CREATE OR REPLACE VIEW v_data_sufficiency AS
WITH entity_counts AS (
    SELECT
        entity_id,
        COUNT(*) AS n_observations,
        MAX(I) - MIN(I) AS I_range
    FROM physics
    GROUP BY entity_id
)
SELECT
    e.entity_id,
    e.n_observations,
    e.I_range,

    -- Lyapunov
    e.n_observations >= 10000 AS lyapunov_reliable,
    CASE
        WHEN e.n_observations >= 10000 THEN 'reliable'
        WHEN e.n_observations >= 5000 THEN 'marginal'
        ELSE 'insufficient'
    END AS lyapunov_quality,

    -- Hurst
    e.n_observations >= 256 AS hurst_reliable,

    -- Attractor/Basin
    e.n_observations >= 1000 AS attractor_reliable,

    -- Transfer Entropy
    e.n_observations >= 500 AS transfer_entropy_reliable,

    -- General
    CASE
        WHEN e.n_observations >= 10000 THEN 'full_analysis'
        WHEN e.n_observations >= 1000 THEN 'standard_analysis'
        WHEN e.n_observations >= 256 THEN 'basic_analysis'
        ELSE 'limited_analysis'
    END AS analysis_tier

FROM entity_counts e;

-- ============================================================================
-- HELPER VIEW: Lyapunov reliability (for backward compatibility)
-- ============================================================================

CREATE OR REPLACE VIEW v_lyapunov_reliability AS
SELECT
    entity_id,
    n_observations,
    I_range,
    10000 AS lyapunov_min_observations,
    lyapunov_reliable AS lyapunov_is_reliable,
    lyapunov_quality AS lyapunov_reliability_class
FROM v_data_sufficiency;

-- ============================================================================
-- PRINT CONFIG SUMMARY
-- ============================================================================

.print ''
.print '=============================================='
.print 'ORTHON CONFIGURATION'
.print '=============================================='
.print ''
.print 'Engine Minimum Data Requirements:'
.print '──────────────────────────────────────────────'

SELECT
    printf('  %-20s %6d pts   %s', engine, min_observations, rationale) AS requirement
FROM config_engine_requirements
WHERE min_observations IS NOT NULL
ORDER BY min_observations DESC;

.print ''
.print 'Interpretation Thresholds:'
.print '──────────────────────────────────────────────'

SELECT '  Lyapunov chaotic:     λ > ' || lyapunov_chaotic_threshold FROM config_thresholds
UNION ALL SELECT '  Lyapunov stable:      λ < ' || lyapunov_stable_threshold FROM config_thresholds
UNION ALL SELECT '  Coherence coupled:    > ' || coherence_strongly_coupled FROM config_thresholds
UNION ALL SELECT '  State significant:    > ' || state_distance_significant || 'σ' FROM config_thresholds
UNION ALL SELECT '  Hurst persistent:     H > ' || hurst_persistent_threshold FROM config_thresholds;

.print ''
.print 'Adjust values in config_thresholds table as needed.'
.print '=============================================='
.print ''
