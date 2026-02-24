-- ============================================================================
-- LAYER 18: CONFIDENCE INTERVAL BREACH DETECTION
-- When effective_dim moves outside its jackknife CI, geometry is changing
-- faster than measurement uncertainty can explain. That's a canary.
--
-- One view. Three columns on top of state_geometry.
-- Source: state_geometry (eff_dim_ci_low, eff_dim_ci_high from jackknife)
-- ============================================================================

CREATE OR REPLACE VIEW v_ci_breach AS
SELECT
    *,
    -- Breach: effective_dim outside [ci_low, ci_high]
    (effective_dim < eff_dim_ci_low OR effective_dim > eff_dim_ci_high) AS ci_breach,
    -- Direction: which way did it escape?
    CASE
        WHEN effective_dim < eff_dim_ci_low THEN 'below'
        WHEN effective_dim > eff_dim_ci_high THEN 'above'
        ELSE NULL
    END AS ci_breach_direction,
    -- Magnitude normalized by CI width.
    -- 5.0 means geometry moved 5x the width of the CI. That's not noise.
    CASE
        WHEN (eff_dim_ci_high - eff_dim_ci_low) > 0 THEN
            CASE
                WHEN effective_dim < eff_dim_ci_low
                    THEN (eff_dim_ci_low - effective_dim) / (eff_dim_ci_high - eff_dim_ci_low)
                WHEN effective_dim > eff_dim_ci_high
                    THEN (effective_dim - eff_dim_ci_high) / (eff_dim_ci_high - eff_dim_ci_low)
                ELSE 0.0
            END
        ELSE NULL
    END AS ci_breach_magnitude
FROM state_geometry
WHERE eff_dim_ci_low IS NOT NULL
  AND eff_dim_ci_high IS NOT NULL;
