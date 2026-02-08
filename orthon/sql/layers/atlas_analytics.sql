-- ============================================================
-- Atlas Analytics
-- Derived analytical views built on atlas classification views.
--
-- Engines computes. Atlas views classify. This layer analyzes.
-- ============================================================

-- ------------------------------------------------------------
-- GEOMETRIC TRANSITIONS
-- Detects when the attractor changes shape or stability class.
-- Each row where is_transition=1 marks a regime boundary.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_geometric_transitions AS
SELECT
    I,
    cohort,
    attractor_type,
    LAG(attractor_type) OVER (PARTITION BY cohort ORDER BY I) AS prev_attractor_type,
    shape_character,
    LAG(shape_character) OVER (PARTITION BY cohort ORDER BY I) AS prev_shape_character,
    geometry_status,
    numerical_stability,

    CASE WHEN attractor_type != LAG(attractor_type) OVER (PARTITION BY cohort ORDER BY I)
         THEN 1 ELSE 0
    END AS is_geometry_transition,

    CASE WHEN geometry_status != LAG(geometry_status) OVER (PARTITION BY cohort ORDER BY I)
         THEN 1 ELSE 0
    END AS is_stability_transition

FROM v_attractor_geometry;


-- ------------------------------------------------------------
-- STABILITY PERIODS
-- Groups consecutive windows with the same geometry_status
-- into contiguous periods, measuring duration.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_stability_periods AS
SELECT
    cohort,
    geometry_status,
    MIN(I) AS period_start,
    MAX(I) AS period_end,
    COUNT(*) AS period_length

FROM (
    SELECT
        I, cohort, geometry_status,
        I - ROW_NUMBER() OVER (PARTITION BY cohort, geometry_status ORDER BY I) AS grp
    FROM v_attractor_geometry
) grouped
GROUP BY cohort, geometry_status, grp
ORDER BY period_start;


-- ------------------------------------------------------------
-- CHAOS-GEOMETRY ALIGNMENT
-- Joins FTLE chaos classification with geometry classification
-- at matching windows to detect mismatches (e.g., geometry says
-- stable but FTLE says chaotic).
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_chaos_geometry_alignment AS
SELECT
    f.I,
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
        ELSE 'mixed'
    END AS alignment

FROM v_ftle_evolution f
LEFT JOIN v_attractor_geometry g ON f.I = g.I AND f.cohort = g.cohort;


-- ------------------------------------------------------------
-- DATA QUALITY CHECK
-- Per-view record counts and coverage statistics for validation.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_data_quality AS
SELECT * FROM (
    SELECT 'v_attractor_geometry' AS view_name, COUNT(*) AS total_rows,
           COUNT(DISTINCT I) AS unique_windows, COUNT(DISTINCT cohort) AS cohorts
    FROM v_attractor_geometry
    UNION ALL
    SELECT 'v_motion_class', COUNT(*), COUNT(DISTINCT I), COUNT(DISTINCT cohort)
    FROM v_motion_class
    UNION ALL
    SELECT 'v_ftle_evolution', COUNT(*), COUNT(DISTINCT I), COUNT(DISTINCT cohort)
    FROM v_ftle_evolution
    UNION ALL
    SELECT 'v_urgency_class', COUNT(*), COUNT(DISTINCT I), COUNT(DISTINCT cohort)
    FROM v_urgency_class
    UNION ALL
    SELECT 'v_break_cascade', COUNT(*), 0, COUNT(DISTINCT cohort)
    FROM v_break_cascade
    UNION ALL
    SELECT 'v_network_class', COUNT(*), COUNT(DISTINCT I), COUNT(DISTINCT cohort)
    FROM v_network_class
);
