-- ============================================================
-- Atlas Geometry Classification
-- Classifies attractor shape from geometry_full.parquet
--
-- Engines computes eigenvalues. ORTHON classifies geometry.
-- ============================================================

-- ------------------------------------------------------------
-- ATTRACTOR GEOMETRY
-- Classifies the shape and evolution of the state-space attractor
-- based on eigenvalue structure and effective dimensionality.
--
-- effective_dim thresholds:
--   < 1.5  : line_like (1D attractor, single dominant mode)
--   < 2.5  : surface_like (2D attractor, two dominant modes)
--   < 3.5  : volume_like (3D attractor, three dominant modes)
--   >= 3.5 : high_dimensional (complex attractor)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_attractor_geometry AS
SELECT
    I,
    cohort,
    effective_dim,
    effective_dim_velocity,
    effective_dim_acceleration,
    eigenvalue_1,
    eigenvalue_2,
    eigenvalue_3,
    ratio_2_1,
    condition_number,
    explained_1,
    total_variance,

    -- Attractor type from effective dimensionality
    CASE
        WHEN effective_dim < 1.5 THEN 'line_like'
        WHEN effective_dim < 2.5 THEN 'surface_like'
        WHEN effective_dim < 3.5 THEN 'volume_like'
        ELSE 'high_dimensional'
    END AS attractor_type,

    -- Shape character from eigenvalue ratios
    CASE
        WHEN ratio_2_1 IS NULL THEN 'unknown'
        WHEN ratio_2_1 < 0.01 THEN 'collapsed'
        WHEN ratio_2_1 < 0.1 THEN 'stretched'
        WHEN ratio_2_1 < 0.5 THEN 'elongated'
        ELSE 'compact'
    END AS shape_character,

    -- Geometry evolution status
    CASE
        WHEN effective_dim_velocity < -0.1 THEN 'collapsing'
        WHEN effective_dim_velocity > 0.1 THEN 'expanding'
        WHEN ABS(effective_dim_velocity) < 0.01 THEN 'stable'
        ELSE 'drifting'
    END AS geometry_status,

    -- Numerical conditioning
    CASE
        WHEN condition_number IS NULL THEN 'unknown'
        WHEN condition_number < 100 THEN 'well_conditioned'
        WHEN condition_number < 10000 THEN 'poorly_conditioned'
        ELSE 'ill_conditioned'
    END AS numerical_stability,

    -- Variance concentration (how much the first mode explains)
    CASE
        WHEN explained_1 IS NULL THEN 'unknown'
        WHEN explained_1 > 0.95 THEN 'single_mode'
        WHEN explained_1 > 0.8 THEN 'dominant_mode'
        WHEN explained_1 > 0.5 THEN 'distributed'
        ELSE 'highly_distributed'
    END AS variance_structure

FROM geometry_full;
