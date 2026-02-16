-- ============================================================
-- Atlas Analytics
-- Derived analytical views built on atlas classification views.
--
-- Engines computes. Atlas views classify. This layer analyzes.
-- ============================================================

-- ------------------------------------------------------------
-- GEOMETRY SHAPE (windowed)
-- Classifies attractor geometry from geometry_dynamics (shape engine).
-- Replaces the deprecated v_geometry_cumulative (geometry_full.parquet)
-- which saturated on long time series.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_geometry_shape AS
SELECT
    signal_0_center,
    cohort,
    effective_dim,
    effective_dim_velocity,

    CASE
        WHEN effective_dim < 1.5 THEN 'line_like'
        WHEN effective_dim < 2.5 THEN 'surface_like'
        WHEN effective_dim < 3.5 THEN 'volume_like'
        ELSE 'high_dimensional'
    END AS attractor_type,

    CASE
        WHEN effective_dim_velocity < -0.1 THEN 'collapsing'
        WHEN effective_dim_velocity > 0.1 THEN 'expanding'
        WHEN ABS(effective_dim_velocity) < 0.01 THEN 'stable'
        ELSE 'drifting'
    END AS geometry_status,

    CASE
        WHEN effective_dim < 1.2 THEN 'collapsed'
        WHEN effective_dim < 2.0 THEN 'stretched'
        WHEN effective_dim < 3.0 THEN 'elongated'
        ELSE 'compact'
    END AS shape_character,

    CASE
        WHEN ABS(effective_dim_velocity) < 0.01 THEN 'well_conditioned'
        WHEN ABS(effective_dim_velocity) < 0.1 THEN 'moderately_conditioned'
        ELSE 'ill_conditioned'
    END AS numerical_stability

FROM geometry_dynamics
WHERE engine = 'shape'
  AND effective_dim IS NOT NULL
  AND NOT isnan(effective_dim);


-- ------------------------------------------------------------
-- GEOMETRIC TRANSITIONS (smoothed)
-- Detects when the attractor changes shape or stability class.
-- Single-window flickers are smoothed: if a window differs from
-- both its neighbors and the neighbors agree, use the neighbor
-- consensus instead.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_geometric_transitions AS
WITH raw AS (
    SELECT
        signal_0_center,
        cohort,
        attractor_type,
        geometry_status,
        shape_character,
        numerical_stability,
        LAG(attractor_type) OVER (PARTITION BY cohort ORDER BY signal_0_center) AS prev_attractor_type,
        LEAD(attractor_type) OVER (PARTITION BY cohort ORDER BY signal_0_center) AS next_attractor_type,
        LAG(geometry_status) OVER (PARTITION BY cohort ORDER BY signal_0_center) AS prev_geometry_status,
        LEAD(geometry_status) OVER (PARTITION BY cohort ORDER BY signal_0_center) AS next_geometry_status,
        LAG(shape_character) OVER (PARTITION BY cohort ORDER BY signal_0_center) AS prev_shape_character
    FROM v_geometry_shape
),
smoothed AS (
    SELECT
        signal_0_center,
        cohort,
        -- Smooth single-window outliers: if this window differs from both
        -- neighbors and neighbors agree, use neighbor consensus
        CASE
            WHEN attractor_type != prev_attractor_type
                 AND attractor_type != next_attractor_type
                 AND prev_attractor_type = next_attractor_type
            THEN prev_attractor_type
            ELSE attractor_type
        END AS attractor_type,
        CASE
            WHEN geometry_status != prev_geometry_status
                 AND geometry_status != next_geometry_status
                 AND prev_geometry_status = next_geometry_status
            THEN prev_geometry_status
            ELSE geometry_status
        END AS geometry_status,
        shape_character,
        numerical_stability,
        -- Keep raw values for reference
        attractor_type AS attractor_type_raw,
        geometry_status AS geometry_status_raw
    FROM raw
    WHERE prev_attractor_type IS NOT NULL
)
SELECT
    signal_0_center,
    cohort,
    attractor_type,
    LAG(attractor_type) OVER (PARTITION BY cohort ORDER BY signal_0_center) AS prev_attractor_type,
    shape_character,
    LAG(shape_character) OVER (PARTITION BY cohort ORDER BY signal_0_center) AS prev_shape_character,
    geometry_status,
    numerical_stability,
    attractor_type_raw,
    geometry_status_raw,

    CASE WHEN attractor_type != LAG(attractor_type) OVER (PARTITION BY cohort ORDER BY signal_0_center)
         THEN 1 ELSE 0
    END AS is_geometry_transition,

    CASE WHEN geometry_status != LAG(geometry_status) OVER (PARTITION BY cohort ORDER BY signal_0_center)
         THEN 1 ELSE 0
    END AS is_stability_transition

FROM smoothed;


-- ------------------------------------------------------------
-- STABILITY PERIODS
-- Groups consecutive windows with the same geometry_status
-- into contiguous periods, measuring duration.
-- Uses smoothed geometry_status from v_geometric_transitions.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_stability_periods AS
SELECT
    cohort,
    geometry_status,
    MIN(signal_0_center) AS period_start,
    MAX(signal_0_center) AS period_end,
    COUNT(*) AS period_length

FROM (
    SELECT
        signal_0_center, cohort, geometry_status,
        signal_0_center - ROW_NUMBER() OVER (PARTITION BY cohort, geometry_status ORDER BY signal_0_center) AS grp
    FROM v_geometric_transitions
) grouped
GROUP BY cohort, geometry_status, grp
ORDER BY period_start;


-- ------------------------------------------------------------
-- CHAOS-GEOMETRY ALIGNMENT
-- Joins FTLE chaos classification with windowed geometry
-- at matching windows to detect mismatches (e.g., geometry says
-- stable but FTLE says chaotic).
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_chaos_geometry_alignment AS
SELECT
    f.signal_0_center,
    f.cohort,
    f.signal_id,
    f.chaos_class,
    f.stability_trend,
    g.attractor_type,
    g.geometry_status,

    CASE
        WHEN f.chaos_class = 'chaotic' AND g.geometry_status = 'stable' THEN 'mismatch_chaotic_stable'
        WHEN f.chaos_class = 'stable' AND g.geometry_status = 'collapsing' THEN 'mismatch_stable_collapsing'
        WHEN f.chaos_class = 'chaotic' AND g.geometry_status = 'collapsing' THEN 'consistent_degrading'
        WHEN f.chaos_class = 'stable' AND g.geometry_status = 'stable' THEN 'consistent_healthy'
        WHEN f.chaos_class = 'marginal' AND g.geometry_status = 'stable' THEN 'marginal_but_stable'
        WHEN f.chaos_class = 'neutral' AND g.geometry_status = 'stable' THEN 'neutral_stable'
        ELSE 'mixed'
    END AS alignment

FROM v_ftle_evolution f
LEFT JOIN v_geometry_shape g ON f.signal_0_center = g.signal_0_center AND f.cohort = g.cohort;


-- ------------------------------------------------------------
-- DATA QUALITY CHECK
-- Per-view record counts and coverage statistics for validation.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_data_quality AS
SELECT * FROM (
    SELECT 'v_geometry_shape' AS view_name, COUNT(*) AS total_rows,
           COUNT(DISTINCT signal_0_center) AS unique_windows, COUNT(DISTINCT cohort) AS cohorts
    FROM v_geometry_shape
    UNION ALL
    SELECT 'v_motion_class', COUNT(*), COUNT(DISTINCT signal_0_center), COUNT(DISTINCT cohort)
    FROM v_motion_class
    UNION ALL
    SELECT 'v_ftle_evolution', COUNT(*), COUNT(DISTINCT signal_0_center), COUNT(DISTINCT cohort)
    FROM v_ftle_evolution
    UNION ALL
    SELECT 'v_urgency_class', COUNT(*), COUNT(DISTINCT signal_0_center), COUNT(DISTINCT cohort)
    FROM v_urgency_class
    UNION ALL
    SELECT 'v_break_cascade', COUNT(*), 0, COUNT(DISTINCT cohort)
    FROM v_break_cascade
    UNION ALL
    SELECT 'v_network_class', COUNT(*), COUNT(DISTINCT signal_0_center), COUNT(DISTINCT cohort)
    FROM v_network_class
);
