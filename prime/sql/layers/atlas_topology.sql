-- ============================================================
-- Atlas Topology Classification
-- Classifies network structure from topology.parquet
--
-- Engines computes graph metrics. Prime classifies networks.
-- ============================================================

-- ------------------------------------------------------------
-- NETWORK CLASSIFICATION
-- Classifies the correlation network structure of signal
-- interactions based on density, degree, and coupling fraction.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_network_class AS
SELECT
    cohort,
    signal_0_center,
    topology_computed,
    n_signals,
    n_edges,
    density,
    mean_degree,
    max_degree,
    threshold,

    -- Network type from density
    CASE
        WHEN NOT topology_computed THEN 'not_computed'
        WHEN density < 0.1 THEN 'sparse'
        WHEN density < 0.4 THEN 'moderate'
        WHEN density < 0.7 THEN 'dense'
        ELSE 'fully_connected'
    END AS network_type,

    -- Connectivity class from mean degree
    CASE
        WHEN NOT topology_computed THEN 'not_computed'
        WHEN mean_degree < 1.0 THEN 'isolated'
        WHEN mean_degree < 3.0 THEN 'low_connectivity'
        WHEN mean_degree < 6.0 THEN 'moderate_connectivity'
        ELSE 'high_connectivity'
    END AS connectivity_class,

    -- Coupling fraction (edges / possible edges)
    CASE
        WHEN n_signals IS NULL OR n_signals < 2 THEN NULL
        ELSE CAST(n_edges AS DOUBLE) / (CAST(n_signals AS DOUBLE) * (n_signals - 1) / 2.0)
    END AS coupling_fraction,

    -- Hub presence from max_degree vs mean_degree
    CASE
        WHEN NOT topology_computed THEN 'not_computed'
        WHEN mean_degree < 0.01 THEN 'no_connections'
        WHEN max_degree > 3.0 * mean_degree THEN 'hub_dominated'
        WHEN max_degree > 1.5 * mean_degree THEN 'weak_hubs'
        ELSE 'uniform'
    END AS hub_structure

FROM topology;
