-- ============================================================================
-- ORTHON SQL Engine: 02_signal_class/003_classify.sql
-- ============================================================================
-- Final signal classification combining units + data properties
--
-- Classification hierarchy:
--   1. Digital/Event from units → definitive
--   2. Digital/Event from data properties → definitive
--   3. Periodic from oscillation detection → subclass of analog
--   4. Default → analog
--
-- Output: signal_class (analog | digital | periodic | event)
-- ============================================================================

CREATE OR REPLACE VIEW v_signal_class AS
SELECT
    m.signal_id,
    m.value_unit,

    -- Final signal class
    CASE
        -- Digital from unit
        WHEN u.base_class = 'digital' THEN 'digital'
        -- Event from unit
        WHEN u.base_class = 'event' THEN 'event'
        -- Event from high sparsity (>90% zeros)
        WHEN d.sparsity > 0.9 THEN 'event'
        -- Digital from data (integer + few unique values)
        WHEN d.is_integer AND d.unique_ratio < 0.05 THEN 'digital'
        -- Periodic: low sign change rate (clean oscillation)
        WHEN p.changes_per_1000 < 100 AND p.est_period IS NOT NULL THEN 'periodic'
        -- Periodic: consistent curvature (CV < 0.8)
        WHEN c.kappa_cv IS NOT NULL AND c.kappa_cv < 0.8 THEN 'periodic'
        -- Default: analog
        ELSE 'analog'
    END AS signal_class,

    -- Interpolation validity
    CASE
        WHEN u.base_class = 'digital' THEN FALSE
        WHEN u.base_class = 'event' THEN FALSE
        WHEN d.sparsity > 0.9 THEN FALSE
        WHEN d.is_integer AND d.unique_ratio < 0.05 THEN FALSE
        ELSE TRUE
    END AS interpolation_valid,

    -- Classification source
    CASE
        WHEN u.base_class IN ('digital', 'event') THEN 'unit'
        WHEN d.sparsity > 0.9 THEN 'sparsity'
        WHEN d.is_integer AND d.unique_ratio < 0.05 THEN 'discrete_values'
        WHEN p.changes_per_1000 < 100 AND p.est_period IS NOT NULL THEN 'oscillation'
        WHEN c.kappa_cv IS NOT NULL AND c.kappa_cv < 0.8 THEN 'curvature'
        ELSE 'default'
    END AS class_source,

    -- Estimated period (for periodic signals)
    ROUND(p.est_period, 1) AS est_period,

    -- Curvature consistency
    ROUND(c.kappa_cv, 3) AS kappa_cv,

    -- Data properties for debugging
    d.unique_ratio,
    d.is_integer,
    d.sparsity

FROM v_signal_meta m
JOIN v_class_from_units u USING (signal_id)
JOIN v_data_props d USING (signal_id)
JOIN v_periodicity p USING (signal_id)
LEFT JOIN v_curvature_stats c USING (signal_id);

-- Summary view
CREATE OR REPLACE VIEW v_signal_class_summary AS
SELECT
    signal_class,
    COUNT(*) AS n_signals,
    ARRAY_AGG(signal_id ORDER BY signal_id) AS signals
FROM v_signal_class
GROUP BY signal_class
ORDER BY signal_class;
