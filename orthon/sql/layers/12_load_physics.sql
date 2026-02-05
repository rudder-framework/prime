-- ============================================================================
-- ORTHON SQL: 12_load_physics.sql
-- ============================================================================
-- Load physics.parquet from PRISM
-- This is the first script to run for physics analysis
-- ============================================================================

-- Load physics data
CREATE OR REPLACE TABLE physics AS
SELECT * FROM read_parquet('{prism_output}/physics.parquet');

-- Basic info
CREATE OR REPLACE VIEW v_physics_info AS
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT entity_id) AS n_entities,
    MIN(I) AS min_I,
    MAX(I) AS max_I,
    MAX(n_signals) AS max_signals,
    MAX(n_pairs) AS max_pairs
FROM physics;

-- Column availability check
CREATE OR REPLACE VIEW v_physics_columns AS
SELECT
    -- L4: Thermodynamics
    COUNT(energy_proxy) > 0 AS has_energy_proxy,
    COUNT(energy_velocity) > 0 AS has_energy_velocity,
    COUNT(dissipation_rate) > 0 AS has_dissipation_rate,
    COUNT(entropy_production) > 0 AS has_entropy_production,

    -- L2: Coherence (eigenvalue-based)
    COUNT(coherence) > 0 AS has_coherence,
    COUNT(coherence_velocity) > 0 AS has_coherence_velocity,
    COUNT(effective_dim) > 0 AS has_effective_dim,
    COUNT(eigenvalue_entropy) > 0 AS has_eigenvalue_entropy,

    -- L1: State
    COUNT(state_distance) > 0 AS has_state_distance,
    COUNT(state_velocity) > 0 AS has_state_velocity,
    COUNT(state_acceleration) > 0 AS has_state_acceleration
FROM physics;

-- Verify load
SELECT * FROM v_physics_info;
