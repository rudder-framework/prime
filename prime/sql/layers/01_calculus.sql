-- ============================================================================
-- ENGINES: CALCULUS
-- ============================================================================
-- These engines compute derivatives, curvature, and related calculus operations.
-- All use LAG/LEAD window functions - the foundation of SQL-based calculus.
-- ============================================================================

-- ============================================================================
-- 001: FIRST DERIVATIVE (dvalue/dI)
-- ============================================================================
-- Central difference: (value[i+1] - value[i-1]) / 2
-- More accurate than forward/backward difference

CREATE OR REPLACE VIEW v_dvalue AS
SELECT
    signal_id,
    signal_0,
    value,
    index_dimension,
    signal_class,
    (LEAD(value) OVER (PARTITION BY signal_id ORDER BY signal_0) -
     LAG(value) OVER (PARTITION BY signal_id ORDER BY signal_0)) / 2.0 AS dvalue
FROM v_base;


-- ============================================================================
-- 002: SECOND DERIVATIVE (d²value/dI²)
-- ============================================================================
-- Second central difference: value[i+1] - 2*value[i] + value[i-1]

CREATE OR REPLACE VIEW v_d2value AS
SELECT
    signal_id,
    signal_0,
    value,
    index_dimension,
    signal_class,
    dvalue,
    LEAD(value) OVER (PARTITION BY signal_id ORDER BY signal_0) - 2*value +
    LAG(value) OVER (PARTITION BY signal_id ORDER BY signal_0) AS d2value
FROM v_dvalue;


-- ============================================================================
-- 003: THIRD DERIVATIVE (d³value/dI³) - JERK
-- ============================================================================
-- For detecting sudden changes in acceleration

CREATE OR REPLACE VIEW v_d3value AS
SELECT
    signal_id,
    signal_0,
    value,
    index_dimension,
    signal_class,
    dvalue,
    d2value,
    (LEAD(d2value) OVER (PARTITION BY signal_id ORDER BY signal_0) -
     LAG(d2value) OVER (PARTITION BY signal_id ORDER BY signal_0)) / 2.0 AS d3value
FROM v_d2value;


-- ============================================================================
-- 004: CURVATURE (κ)
-- ============================================================================
-- κ = |d²value| / (1 + dvalue²)^(3/2)
-- Measures how fast direction changes along the curve

CREATE OR REPLACE VIEW v_curvature AS
SELECT
    signal_id,
    signal_0,
    value,
    index_dimension,
    signal_class,
    dvalue,
    d2value,
    ABS(d2value) / POWER(1 + dvalue*dvalue + 1e-10, 1.5) AS kappa
FROM v_d2value
WHERE dvalue IS NOT NULL AND d2value IS NOT NULL;


-- ============================================================================
-- 005: LAPLACIAN (∇²value) - For Spatial Fields
-- ============================================================================
-- Same as second derivative but semantic difference

CREATE OR REPLACE VIEW v_laplacian AS
SELECT
    signal_id,
    signal_0,
    value,
    index_dimension,
    signal_class,
    LEAD(value) OVER (PARTITION BY signal_id ORDER BY signal_0) - 2*value +
    LAG(value) OVER (PARTITION BY signal_id ORDER BY signal_0) AS laplacian
FROM v_base;


-- ============================================================================
-- 006: GRADIENT MAGNITUDE (|∇value|)
-- ============================================================================
-- For 1D: just |dvalue|

CREATE OR REPLACE VIEW v_gradient AS
SELECT
    signal_id,
    signal_0,
    value,
    index_dimension,
    signal_class,
    dvalue,
    ABS(dvalue) AS gradient_magnitude
FROM v_dvalue;


-- ============================================================================
-- 007: ARC LENGTH (cumulative)
-- ============================================================================
-- s = ∫√(1 + dvalue²) dI

CREATE OR REPLACE VIEW v_arc_length AS
SELECT
    signal_id,
    signal_0,
    value,
    dvalue,
    SQRT(1 + dvalue*dvalue) AS segment_length,
    SUM(SQRT(1 + COALESCE(dvalue*dvalue, 0))) OVER (
        PARTITION BY signal_id
        ORDER BY signal_0
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS arc_length_cumulative
FROM v_dvalue;


-- ============================================================================
-- 008: VELOCITY MAGNITUDE (for phase space)
-- ============================================================================

CREATE OR REPLACE VIEW v_velocity AS
SELECT
    signal_id,
    signal_0,
    value,
    dvalue AS velocity,
    ABS(dvalue) AS speed,
    d2value AS acceleration,
    ABS(d2value) AS acceleration_magnitude
FROM v_d2value;


-- ============================================================================
-- 009: DIVERGENCE PROXY (for multi-signal systems)
-- ============================================================================

CREATE OR REPLACE VIEW v_divergence AS
SELECT
    signal_id,
    signal_0,
    d2value AS divergence,
    CASE
        WHEN d2value > 0.1 THEN 'expanding'
        WHEN d2value < -0.1 THEN 'contracting'
        ELSE 'stable'
    END AS divergence_state
FROM v_d2value;


-- ============================================================================
-- 010: SMOOTHNESS INDEX
-- ============================================================================

CREATE OR REPLACE VIEW v_smoothness AS
SELECT
    signal_id,
    signal_0,
    dvalue,
    d2value,
    CASE
        WHEN ABS(dvalue) > 1e-10
        THEN ABS(d2value) / ABS(dvalue)
        ELSE NULL
    END AS roughness_index,
    CASE
        WHEN ABS(dvalue) > 1e-10
        THEN ABS(dvalue) / (ABS(d2value) + 1e-10)
        ELSE NULL
    END AS smoothness_index
FROM v_d2value;


-- ============================================================================
-- CALCULUS SUMMARY VIEW
-- ============================================================================

CREATE OR REPLACE VIEW v_calculus_complete AS
SELECT
    c.signal_id,
    c.signal_0,
    c.value,
    c.index_dimension,
    c.signal_class,
    c.dvalue,
    c.d2value,
    d.d3value,
    c.kappa,
    l.laplacian,
    ABS(c.dvalue) AS gradient_magnitude,
    a.arc_length_cumulative,
    s.roughness_index,
    s.smoothness_index,
    div.divergence,
    div.divergence_state
FROM v_curvature c
LEFT JOIN v_d3value d USING (signal_id, signal_0)
LEFT JOIN v_laplacian l USING (signal_id, signal_0)
LEFT JOIN v_arc_length a USING (signal_id, signal_0)
LEFT JOIN v_smoothness s USING (signal_id, signal_0)
LEFT JOIN v_divergence div USING (signal_id, signal_0);
