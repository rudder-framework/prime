-- ============================================================================
-- ENGINES: DYNAMICAL SYSTEMS (CALCULUS-BASED)
-- ============================================================================
-- Regime detection, transitions, stability, basins, and attractors.
-- Uses calculus (dvalue, d2value) instead of rolling windows.
-- ============================================================================

-- ============================================================================
-- 001: REGIME CHANGE DETECTION (from derivative discontinuities)
-- ============================================================================
-- Regime change = sudden jump in dvalue or d2value

CREATE OR REPLACE VIEW v_regime_changes AS
SELECT
    d.signal_id,
    d.signal_0,
    d.dvalue,
    d.d2value,
    LAG(d.dvalue) OVER (PARTITION BY d.signal_id ORDER BY d.signal_0) AS dvalue_prev,
    LAG(d.d2value) OVER (PARTITION BY d.signal_id ORDER BY d.signal_0) AS d2value_prev,
    -- Change in derivative
    ABS(d.dvalue - LAG(d.dvalue) OVER (PARTITION BY d.signal_id ORDER BY d.signal_0)) AS dvalue_jump,
    ABS(d.d2value - LAG(d.d2value) OVER (PARTITION BY d.signal_id ORDER BY d.signal_0)) AS d2value_jump,
    -- Normalized change score
    ABS(d.dvalue - LAG(d.dvalue) OVER (PARTITION BY d.signal_id ORDER BY d.signal_0)) / NULLIF(ds.dvalue_median_abs, 0) +
    ABS(d.d2value - LAG(d.d2value) OVER (PARTITION BY d.signal_id ORDER BY d.signal_0)) / NULLIF(ds.d2value_median_abs, 0) AS change_score,
    -- Is this a regime change?
    CASE
        WHEN ABS(d.dvalue - LAG(d.dvalue) OVER (PARTITION BY d.signal_id ORDER BY d.signal_0)) > 3 * ds.dvalue_median_abs
          OR ABS(d.d2value - LAG(d.d2value) OVER (PARTITION BY d.signal_id ORDER BY d.signal_0)) > 3 * ds.d2value_median_abs
        THEN TRUE
        ELSE FALSE
    END AS is_regime_change
FROM v_d2value d
JOIN v_derivative_stats ds USING (signal_id)
WHERE d.dvalue IS NOT NULL AND d.d2value IS NOT NULL;


-- ============================================================================
-- 002: REGIME BOUNDARIES
-- ============================================================================
-- Filter to only regime change points, assign regime numbers

CREATE OR REPLACE VIEW v_regime_boundaries AS
SELECT
    signal_id,
    signal_0 AS regime_boundary,
    change_score,
    ROW_NUMBER() OVER (PARTITION BY signal_id ORDER BY signal_0) AS regime_number
FROM v_regime_changes
WHERE is_regime_change;


-- ============================================================================
-- 003: REGIME ASSIGNMENT
-- ============================================================================
-- Assign each point to a regime

CREATE OR REPLACE VIEW v_regime_assignment AS
SELECT
    b.signal_id,
    b.signal_0,
    COALESCE(
        (SELECT MAX(regime_number)
         FROM v_regime_boundaries rb
         WHERE rb.signal_id = b.signal_id AND rb.regime_boundary <= b.signal_0),
        0
    ) AS regime_id
FROM v_base b;


-- ============================================================================
-- 004: REGIME STATISTICS
-- ============================================================================
-- Stats per regime

CREATE OR REPLACE VIEW v_regime_stats AS
SELECT
    r.signal_id,
    r.regime_id,
    COUNT(*) AS regime_length,
    MIN(b.signal_0) AS regime_start,
    MAX(b.signal_0) AS regime_end,
    AVG(b.value) AS regime_mean,
    STDDEV(b.value) AS regime_std,
    MIN(b.value) AS regime_min,
    MAX(b.value) AS regime_max,
    AVG(d.dvalue) AS regime_avg_velocity,
    AVG(c.kappa) AS regime_avg_curvature
FROM v_regime_assignment r
JOIN v_base b ON r.signal_id = b.signal_id AND r.signal_0 = b.signal_0
LEFT JOIN v_dvalue d ON r.signal_id = d.signal_id AND r.signal_0 = d.signal_0
LEFT JOIN v_curvature c ON r.signal_id = c.signal_id AND r.signal_0 = c.signal_0
GROUP BY r.signal_id, r.regime_id;


-- ============================================================================
-- 005: REGIME TRANSITIONS
-- ============================================================================

CREATE OR REPLACE VIEW v_regime_transitions AS
SELECT
    signal_id,
    regime_id AS from_regime,
    LEAD(regime_id) OVER (PARTITION BY signal_id ORDER BY regime_start) AS to_regime,
    regime_end AS transition_point,
    regime_mean,
    LEAD(regime_mean) OVER (PARTITION BY signal_id ORDER BY regime_start) - regime_mean AS mean_jump,
    LEAD(regime_std) OVER (PARTITION BY signal_id ORDER BY regime_start) / NULLIF(regime_std, 0) AS volatility_ratio,
    CASE
        WHEN LEAD(regime_std) OVER (PARTITION BY signal_id ORDER BY regime_start) > regime_std * 1.5
            THEN 'volatility_increase'
        WHEN LEAD(regime_std) OVER (PARTITION BY signal_id ORDER BY regime_start) < regime_std * 0.67
            THEN 'volatility_decrease'
        WHEN LEAD(regime_mean) OVER (PARTITION BY signal_id ORDER BY regime_start) > regime_mean + regime_std
            THEN 'upward_shift'
        WHEN LEAD(regime_mean) OVER (PARTITION BY signal_id ORDER BY regime_start) < regime_mean - regime_std
            THEN 'downward_shift'
        ELSE 'lateral'
    END AS transition_type
FROM v_regime_stats;


-- ============================================================================
-- 006: STABILITY ANALYSIS (from calculus)
-- ============================================================================
-- Local stability from dvalue and d2value

CREATE OR REPLACE VIEW v_stability AS
SELECT
    signal_id,
    signal_0,
    dvalue,
    d2value,
    -- Local expansion rate (Lyapunov-like)
    ABS(dvalue) AS local_expansion_rate,
    -- Convergence indicator
    CASE
        WHEN dvalue > 0 AND d2value < 0 THEN 'converging_up'
        WHEN dvalue < 0 AND d2value > 0 THEN 'converging_down'
        WHEN dvalue > 0 AND d2value > 0 THEN 'diverging_up'
        WHEN dvalue < 0 AND d2value < 0 THEN 'diverging_down'
        ELSE 'neutral'
    END AS stability_state,
    -- Is locally stable? (converging)
    CASE
        WHEN (dvalue > 0 AND d2value < 0) OR (dvalue < 0 AND d2value > 0) THEN TRUE
        ELSE FALSE
    END AS is_locally_stable
FROM v_d2value
WHERE dvalue IS NOT NULL AND d2value IS NOT NULL;


-- ============================================================================
-- 007: BASIN DETECTION (from local minima)
-- ============================================================================
-- Basins of attraction around valleys

CREATE OR REPLACE VIEW v_basins AS
SELECT
    e.signal_id,
    e.signal_0 AS basin_center,
    e.value AS basin_depth,
    b.signal_0,
    b.value,
    ABS(b.signal_0 - e.signal_0) AS distance_to_center
FROM v_local_extrema e
JOIN v_base b ON e.signal_id = b.signal_id
WHERE e.extrema_type = 'valley'
  AND ABS(b.signal_0 - e.signal_0) < 50;


-- ============================================================================
-- 008: ATTRACTOR DETECTION
-- ============================================================================
-- Values the signal frequently visits
-- NOTE: Two-step view to avoid DuckDB window function filter issue

CREATE OR REPLACE VIEW v_attractors_all AS
WITH binned AS (
    SELECT
        signal_id,
        ROUND(value, 1) AS attractor_value,
        AVG(value) AS bin_center,
        COUNT(*) AS visit_count
    FROM v_base
    GROUP BY signal_id, ROUND(value, 1)
)
SELECT
    signal_id,
    attractor_value,
    bin_center,
    visit_count,
    visit_count::FLOAT / SUM(visit_count) OVER (PARTITION BY signal_id) AS visit_frequency,
    ROW_NUMBER() OVER (PARTITION BY signal_id ORDER BY visit_count DESC) AS rank_by_visits
FROM binned;

CREATE OR REPLACE VIEW v_attractors AS
SELECT
    signal_id,
    attractor_value,
    bin_center,
    visit_count,
    visit_frequency,
    CASE
        WHEN rank_by_visits = 1 THEN 'primary'
        WHEN rank_by_visits <= 3 THEN 'secondary'
        ELSE 'minor'
    END AS attractor_type
FROM v_attractors_all
WHERE visit_frequency > 0.05;


-- ============================================================================
-- 009: PHASE VELOCITY (from calculus)
-- ============================================================================
-- Speed of movement through phase space

CREATE OR REPLACE VIEW v_phase_velocity AS
SELECT
    signal_id,
    signal_0,
    SQRT(1 + dvalue*dvalue) AS phase_velocity,
    SQRT(dvalue*dvalue + d2value*d2value) AS tangent_magnitude
FROM v_d2value
WHERE dvalue IS NOT NULL AND d2value IS NOT NULL;


-- ============================================================================
-- 010: BIFURCATION DETECTION (from curvature changes)
-- ============================================================================
-- Qualitative changes in behavior detected from kappa
-- NOTE: Two-step view to avoid DuckDB window function filter issue

-- First: compute kappa changes and thresholds
CREATE OR REPLACE VIEW v_bifurcation_base AS
SELECT
    c.signal_id,
    c.signal_0 AS bifurcation_point,
    c.kappa,
    LAG(c.kappa) OVER (PARTITION BY c.signal_id ORDER BY c.signal_0) AS kappa_prev,
    ABS(c.kappa - LAG(c.kappa) OVER (PARTITION BY c.signal_id ORDER BY c.signal_0)) AS kappa_jump,
    CASE
        WHEN c.kappa > LAG(c.kappa) OVER (PARTITION BY c.signal_id ORDER BY c.signal_0) * 2 THEN 'complexity_increase'
        WHEN c.kappa < LAG(c.kappa) OVER (PARTITION BY c.signal_id ORDER BY c.signal_0) * 0.5 THEN 'complexity_decrease'
        ELSE NULL
    END AS bifurcation_type
FROM v_curvature c
WHERE c.kappa IS NOT NULL;

-- Compute per-signal thresholds
CREATE OR REPLACE VIEW v_kappa_thresholds AS
SELECT
    signal_id,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS(kappa)) AS kappa_p95
FROM v_curvature
WHERE kappa IS NOT NULL
GROUP BY signal_id;

-- Second: filter by threshold
CREATE OR REPLACE VIEW v_bifurcation_candidates AS
SELECT
    b.signal_id,
    b.bifurcation_point,
    b.kappa,
    b.kappa_prev,
    b.kappa_jump,
    b.bifurcation_type
FROM v_bifurcation_base b
JOIN v_kappa_thresholds t USING (signal_id)
WHERE b.kappa_prev IS NOT NULL
  AND b.kappa_jump > t.kappa_p95;


-- ============================================================================
-- DYNAMICS SUMMARY VIEW
-- ============================================================================

CREATE OR REPLACE VIEW v_dynamics_complete AS
SELECT
    ra.signal_id,
    ra.signal_0,
    ra.regime_id,
    rs.regime_mean,
    rs.regime_std,
    rs.regime_length,
    s.stability_state,
    s.is_locally_stable,
    pv.phase_velocity
FROM v_regime_assignment ra
LEFT JOIN v_regime_stats rs ON ra.signal_id = rs.signal_id AND ra.regime_id = rs.regime_id
LEFT JOIN v_stability s ON ra.signal_id = s.signal_id AND ra.signal_0 = s.signal_0
LEFT JOIN v_phase_velocity pv ON ra.signal_id = pv.signal_id AND ra.signal_0 = pv.signal_0;


-- ============================================================================
-- SYSTEM-LEVEL REGIME DETECTION
-- ============================================================================
-- When multiple signals change regime together

CREATE OR REPLACE VIEW v_system_regime AS
SELECT
    signal_0 AS system_regime_boundary,
    COUNT(DISTINCT signal_id) AS n_signals_changing,
    CASE
        WHEN COUNT(DISTINCT signal_id) > (SELECT COUNT(DISTINCT signal_id) FROM v_base) * 0.5
            THEN 'major_system_change'
        WHEN COUNT(DISTINCT signal_id) > 2 THEN 'moderate_system_change'
        ELSE 'isolated_change'
    END AS change_magnitude
FROM v_regime_changes
WHERE is_regime_change
GROUP BY signal_0
HAVING COUNT(DISTINCT signal_id) > 1
ORDER BY signal_0;
