-- ============================================================================
-- ORTHON SQL: 03_load_prism_results.sql
-- ============================================================================
-- Load PRISM results from parquet files for visualization
--
-- PRISM creates these files:
--   - signal_typology.parquet    (Layer 1: WHAT)
--   - behavioral_geometry.parquet (Layer 2: HOW)
--   - dynamical_systems.parquet   (Layer 3: WHEN/HOW)
--   - causal_mechanics.parquet    (Layer 4: WHY)
--
-- ORTHON queries these for visualization. No compute, just queries.
-- ============================================================================

-- =========================================================================
-- LAYER 1: SIGNAL TYPOLOGY (WHAT)
-- =========================================================================
CREATE OR REPLACE TABLE signal_typology AS
SELECT * FROM read_parquet('{prism_output}/signal_typology.parquet');

CREATE OR REPLACE VIEW v_typology AS
SELECT
    signal_id,
    -- Persistence
    hurst,
    CASE
        WHEN hurst > 0.65 THEN 'persistent'
        WHEN hurst < 0.35 THEN 'anti-persistent'
        ELSE 'random-walk'
    END AS persistence_class,
    -- Entropy
    entropy,
    CASE
        WHEN entropy > 0.8 THEN 'high-entropy'
        WHEN entropy < 0.3 THEN 'low-entropy'
        ELSE 'moderate-entropy'
    END AS entropy_class,
    -- Stationarity
    stationarity_pvalue,
    CASE
        WHEN stationarity_pvalue < 0.05 THEN 'stationary'
        ELSE 'non-stationary'
    END AS stationarity_class
FROM signal_typology;

-- =========================================================================
-- LAYER 2: BEHAVIORAL GEOMETRY (HOW)
-- =========================================================================
CREATE OR REPLACE TABLE behavioral_geometry AS
SELECT * FROM read_parquet('{prism_output}/behavioral_geometry.parquet');

CREATE OR REPLACE VIEW v_geometry AS
SELECT
    signal_a,
    signal_b,
    -- Correlation
    instant_correlation,
    CASE
        WHEN ABS(instant_correlation) > 0.7 THEN 'strong'
        WHEN ABS(instant_correlation) > 0.4 THEN 'moderate'
        ELSE 'weak'
    END AS correlation_strength,
    CASE
        WHEN instant_correlation > 0 THEN 'positive'
        ELSE 'negative'
    END AS correlation_direction,
    -- DTW
    dtw_distance,
    -- Optimal lag
    optimal_lag
FROM behavioral_geometry;

-- =========================================================================
-- LAYER 3: DYNAMICAL SYSTEMS (WHEN/HOW)
-- =========================================================================
CREATE OR REPLACE TABLE dynamical_systems AS
SELECT * FROM read_parquet('{prism_output}/dynamical_systems.parquet');

CREATE OR REPLACE VIEW v_dynamics AS
SELECT
    signal_id,
    -- Regime
    regime_id,
    regime_start_idx,
    regime_end_idx,
    -- Lyapunov
    lyapunov,
    CASE
        WHEN lyapunov > 0 THEN 'chaotic'
        WHEN lyapunov < 0 THEN 'stable'
        ELSE 'edge-of-chaos'
    END AS stability_class,
    -- Attractor
    attractor_dimension
FROM dynamical_systems;

-- =========================================================================
-- LAYER 4: CAUSAL MECHANICS (WHY)
-- =========================================================================
CREATE OR REPLACE TABLE causal_mechanics AS
SELECT * FROM read_parquet('{prism_output}/causal_mechanics.parquet');

CREATE OR REPLACE VIEW v_causality AS
SELECT
    signal_id,
    -- Causal role
    causal_role,
    n_drives,
    n_driven_by,
    -- Causal pairs (if available)
    COALESCE(drives_signals, ARRAY[]::VARCHAR[]) AS drives_signals,
    COALESCE(driven_by_signals, ARRAY[]::VARCHAR[]) AS driven_by_signals
FROM causal_mechanics;

-- =========================================================================
-- COMBINED VIEW: All layers joined
-- =========================================================================
CREATE OR REPLACE VIEW v_signal_analysis AS
SELECT
    t.signal_id,
    -- Layer 1: Typology
    t.hurst,
    t.persistence_class,
    t.entropy,
    t.entropy_class,
    t.stationarity_class,
    -- Layer 3: Dynamics
    d.regime_id,
    d.lyapunov,
    d.stability_class,
    d.attractor_dimension,
    -- Layer 4: Causality
    c.causal_role,
    c.n_drives,
    c.n_driven_by
FROM v_typology t
LEFT JOIN v_dynamics d USING (signal_id)
LEFT JOIN v_causality c USING (signal_id);

-- Verify loads
SELECT
    (SELECT COUNT(*) FROM signal_typology) AS typology_rows,
    (SELECT COUNT(*) FROM behavioral_geometry) AS geometry_rows,
    (SELECT COUNT(*) FROM dynamical_systems) AS dynamics_rows,
    (SELECT COUNT(*) FROM causal_mechanics) AS causality_rows;
