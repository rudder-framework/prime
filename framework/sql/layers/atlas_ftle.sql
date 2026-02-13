-- ============================================================
-- Atlas FTLE Classification
-- Classifies chaos boundary from ftle_rolling.parquet
--
-- Engines computes rolling FTLE. Rudder classifies stability.
-- ============================================================

-- ------------------------------------------------------------
-- FTLE EVOLUTION
-- Classifies trajectory stability from rolling finite-time
-- Lyapunov exponent and its derivatives.
--
-- FTLE thresholds (consistent with classification.sql):
--   ftle > 0.1   : chaotic (exponential divergence)
--   ftle > 0.01  : marginal (edge of chaos)
--   ftle <= 0.01 : stable (converging/periodic)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_ftle_evolution AS
SELECT
    I,
    cohort,
    signal_id,
    ftle,
    ftle_std,
    ftle_velocity,
    ftle_acceleration,
    confidence,
    embedding_dim,
    embedding_tau,
    window_start,
    window_end,
    direction,

    -- Chaos classification from FTLE sign/magnitude
    CASE
        WHEN ftle IS NULL THEN 'unknown'
        WHEN ftle > 0.1 THEN 'chaotic'
        WHEN ftle > 0.01 THEN 'marginal'
        WHEN ftle > -0.01 THEN 'neutral'
        ELSE 'stable'
    END AS chaos_class,

    -- Evolution trend: approaching or leaving chaos
    CASE
        WHEN ftle_velocity IS NULL THEN 'unknown'
        WHEN ftle_velocity > 0.01 THEN 'destabilizing'
        WHEN ftle_velocity < -0.01 THEN 'stabilizing'
        ELSE 'steady'
    END AS stability_trend,

    -- Classification confidence
    CASE
        WHEN confidence IS NULL THEN 'unknown'
        WHEN confidence > 0.8 THEN 'high'
        WHEN confidence > 0.5 THEN 'medium'
        ELSE 'low'
    END AS classification_confidence,

    -- FTLE uncertainty relative to value
    CASE
        WHEN ftle IS NULL OR ftle_std IS NULL THEN 'unknown'
        WHEN ABS(ftle) < ftle_std THEN 'indeterminate'
        WHEN ftle_std < 0.1 * ABS(ftle) THEN 'precise'
        ELSE 'approximate'
    END AS estimate_quality

FROM ftle_rolling;
