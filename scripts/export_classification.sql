-- ============================================================
-- Classification Export Script
-- Loads engine outputs, creates views, exports atlas + core CSVs
--
-- Usage: duckdb :memory: -c ".read scripts/export_classification.sql"
-- Requires: DOMAIN_DIR variable set before loading
-- ============================================================

-- Load atlas views
.read framework/sql/layers/atlas_velocity_field.sql
.read framework/sql/layers/atlas_ftle.sql
.read framework/sql/layers/atlas_ridge_proximity.sql
.read framework/sql/layers/atlas_breaks.sql
.read framework/sql/layers/atlas_topology.sql

-- Adapted core views (actual engine schemas)
CREATE OR REPLACE VIEW v_coupling_strength AS
SELECT I, signal_a, signal_b, cohort, correlation, distance, cosine_similarity,
    CASE WHEN ABS(correlation) > 0.9 THEN 'strongly_coupled'
         WHEN ABS(correlation) > 0.7 THEN 'moderately_coupled'
         WHEN ABS(correlation) > 0.4 THEN 'weakly_coupled'
         ELSE 'uncoupled' END AS coupling_strength,
    CASE WHEN correlation > 0.7 THEN 'positive'
         WHEN correlation < -0.7 THEN 'negative'
         ELSE 'neutral' END AS coupling_direction
FROM signal_pairwise WHERE engine = 'shape';

-- Geometry windowed (NaN-filtered)
CREATE OR REPLACE VIEW v_geometry_windowed AS
SELECT I, cohort, effective_dim, effective_dim_velocity,
    collapse_onset_idx, collapse_onset_fraction,
    CASE WHEN effective_dim_velocity < -0.1 THEN 'collapsing'
         WHEN effective_dim_velocity > 0.1 THEN 'expanding'
         WHEN ABS(effective_dim_velocity) < 0.01 THEN 'stable'
         ELSE 'drifting' END AS geometry_status,
    CASE WHEN collapse_onset_fraction IS NULL THEN 'none_detected'
         WHEN collapse_onset_fraction < 0.2 THEN 'early_warning'
         WHEN collapse_onset_fraction < 0.5 THEN 'mid_life'
         WHEN collapse_onset_fraction < 0.8 THEN 'late_stage'
         ELSE 'imminent' END AS collapse_stage,
    CASE WHEN collapse_onset_fraction IS NULL THEN NULL
         ELSE 1.0 - collapse_onset_fraction END AS remaining_fraction
FROM geometry_dynamics
WHERE effective_dim IS NOT NULL AND NOT isnan(effective_dim);

-- Stability (aligned thresholds)
CREATE OR REPLACE VIEW v_stability_class AS
SELECT signal_id, ftle AS ftle_value,
    CASE WHEN ftle IS NULL THEN 'unknown'
         WHEN ftle > 0.1 THEN 'unstable'
         WHEN ftle > 0.01 THEN 'quasi_periodic'
         WHEN ftle > -0.01 THEN 'marginally_stable'
         WHEN ftle > -0.1 THEN 'stable'
         ELSE 'strongly_stable' END AS stability_class,
    CASE WHEN ftle IS NULL THEN 0
         ELSE LEAST(1.0, GREATEST(-1.0, ftle * 10)) END AS stability_score,
    confidence, is_deterministic
FROM dynamics;

-- Analytics (smoothed transitions, alignment, quality)
.read framework/sql/layers/atlas_analytics.sql
