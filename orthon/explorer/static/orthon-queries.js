// ORTHON SQL Query Library
// These queries run client-side via DuckDB-WASM against user-uploaded parquet files
// Assumes PRISM has already computed raw engine outputs

const ORTHON_QUERIES = {

  // ============================================================
  // SIGNAL TYPOLOGY - Per-signal characterization
  // ============================================================
  typology: {
    label: "Sensor Typology",
    queries: [
      {
        id: "typology_summary",
        name: "Sensor Summary",
        description: "Overview of all sensors with key metrics",
        sql: `
SELECT
    signal_id,
    COUNT(*) AS n_windows,
    ROUND(AVG(hurst), 3) AS avg_hurst,
    ROUND(AVG(entropy), 3) AS avg_entropy,
    ROUND(AVG(trend_slope), 6) AS avg_trend,
    ROUND(AVG(adf_pvalue), 4) AS avg_adf_pvalue
FROM signal_typology
GROUP BY signal_id
ORDER BY signal_id
        `
      },
      {
        id: "typology_characterization",
        name: "Sensor Characterization",
        description: "Classify sensors by memory, complexity, stationarity",
        sql: `
SELECT
    signal_id,
    window_idx,
    hurst,
    CASE
        WHEN hurst > 0.6 THEN 'persistent'
        WHEN hurst < 0.4 THEN 'anti-persistent'
        ELSE 'random_walk'
    END AS memory_type,
    entropy,
    CASE
        WHEN entropy > 2.0 THEN 'complex'
        WHEN entropy < 0.5 THEN 'regular'
        ELSE 'moderate'
    END AS complexity_class,
    adf_pvalue,
    CASE
        WHEN adf_pvalue < 0.05 THEN 'stationary'
        ELSE 'non_stationary'
    END AS stationarity
FROM signal_typology
ORDER BY signal_id, window_idx
        `
      },
      {
        id: "typology_evolution",
        name: "Metric Evolution Over Time",
        description: "Track how sensor properties change across windows",
        sql: `
SELECT
    window_idx,
    signal_id,
    hurst,
    entropy,
    trend_slope,
    hurst - LAG(hurst) OVER (PARTITION BY signal_id ORDER BY window_idx) AS hurst_delta,
    entropy - LAG(entropy) OVER (PARTITION BY signal_id ORDER BY window_idx) AS entropy_delta
FROM signal_typology
ORDER BY signal_id, window_idx
        `
      },
      {
        id: "typology_anomalies",
        name: "Anomalous Windows",
        description: "Windows with unusual metric values (>2 std from mean)",
        sql: `
WITH stats AS (
    SELECT
        signal_id,
        AVG(hurst) AS mean_hurst,
        STDDEV(hurst) AS std_hurst,
        AVG(entropy) AS mean_entropy,
        STDDEV(entropy) AS std_entropy
    FROM signal_typology
    GROUP BY signal_id
)
SELECT
    t.signal_id,
    t.window_idx,
    t.hurst,
    t.entropy,
    CASE
        WHEN ABS(t.hurst - s.mean_hurst) > 2 * s.std_hurst THEN 'hurst_anomaly'
        WHEN ABS(t.entropy - s.mean_entropy) > 2 * s.std_entropy THEN 'entropy_anomaly'
        ELSE 'normal'
    END AS anomaly_flag
FROM signal_typology t
JOIN stats s ON t.signal_id = s.signal_id
WHERE ABS(t.hurst - s.mean_hurst) > 2 * s.std_hurst
   OR ABS(t.entropy - s.mean_entropy) > 2 * s.std_entropy
ORDER BY t.window_idx
        `
      }
    ]
  },

  // ============================================================
  // BEHAVIORAL GEOMETRY - Cross-signal relationships
  // ============================================================
  geometry: {
    label: "Behavioral Geometry",
    queries: [
      {
        id: "geometry_correlation_matrix",
        name: "Correlation Matrix",
        description: "Pairwise correlations between all sensors",
        sql: `
SELECT
    signal_a,
    signal_b,
    ROUND(correlation, 4) AS correlation,
    ROUND(coherence, 4) AS coherence
FROM behavioral_geometry
WHERE signal_a < signal_b
ORDER BY ABS(correlation) DESC
        `
      },
      {
        id: "geometry_coupling_strength",
        name: "Coupling Strength Over Time",
        description: "How sensor relationships evolve across windows",
        sql: `
SELECT
    window_idx,
    signal_a,
    signal_b,
    correlation,
    CASE
        WHEN correlation > 0.7 THEN 'strong_positive'
        WHEN correlation < -0.7 THEN 'strong_negative'
        WHEN ABS(correlation) < 0.3 THEN 'decoupled'
        ELSE 'moderate'
    END AS coupling_regime
FROM behavioral_geometry
ORDER BY window_idx, signal_a, signal_b
        `
      },
      {
        id: "geometry_cluster_candidates",
        name: "Cluster Candidates",
        description: "Sensor groups with high mutual correlation",
        sql: `
WITH strong_pairs AS (
    SELECT signal_a, signal_b, correlation
    FROM behavioral_geometry
    WHERE ABS(correlation) > 0.7
)
SELECT
    signal_a AS signal,
    COUNT(*) AS n_strong_connections,
    ROUND(AVG(correlation), 3) AS avg_correlation,
    STRING_AGG(signal_b, ', ') AS connected_to
FROM strong_pairs
GROUP BY signal_a
HAVING COUNT(*) >= 2
ORDER BY n_strong_connections DESC
        `
      },
      {
        id: "geometry_decoupling_events",
        name: "Decoupling Events",
        description: "Windows where previously coupled sensors diverge",
        sql: `
WITH lagged AS (
    SELECT
        window_idx,
        signal_a,
        signal_b,
        correlation,
        LAG(correlation) OVER (
            PARTITION BY signal_a, signal_b
            ORDER BY window_idx
        ) AS prev_correlation
    FROM behavioral_geometry
)
SELECT
    window_idx,
    signal_a,
    signal_b,
    ROUND(prev_correlation, 3) AS prev_correlation,
    ROUND(correlation, 3) AS correlation,
    ROUND(correlation - prev_correlation, 3) AS correlation_change
FROM lagged
WHERE ABS(correlation - prev_correlation) > 0.4
ORDER BY ABS(correlation - prev_correlation) DESC
        `
      }
    ]
  },

  // ============================================================
  // DYNAMICAL SYSTEMS - Phase space & system evolution
  // ============================================================
  dynamics: {
    label: "Dynamical Systems",
    queries: [
      {
        id: "dynamics_stability",
        name: "System Stability",
        description: "Lyapunov exponents and stability classification",
        sql: `
SELECT
    window_idx,
    lyapunov_exponent,
    CASE
        WHEN lyapunov_exponent > 0.1 THEN 'chaotic'
        WHEN lyapunov_exponent > 0 THEN 'weakly_chaotic'
        WHEN lyapunov_exponent > -0.1 THEN 'marginally_stable'
        ELSE 'stable'
    END AS stability_class,
    correlation_dimension,
    embedding_dimension
FROM dynamical_systems
ORDER BY window_idx
        `
      },
      {
        id: "dynamics_energy",
        name: "System Energy Evolution",
        description: "Hamiltonian energy across windows",
        sql: `
SELECT
    window_idx,
    ROUND(energy, 4) AS energy,
    ROUND(kinetic_energy, 4) AS kinetic,
    ROUND(potential_energy, 4) AS potential,
    ROUND(energy - LAG(energy) OVER (ORDER BY window_idx), 4) AS energy_delta,
    CASE
        WHEN energy > LAG(energy) OVER (ORDER BY window_idx) THEN 'increasing'
        WHEN energy < LAG(energy) OVER (ORDER BY window_idx) THEN 'decreasing'
        ELSE 'stable'
    END AS energy_trend
FROM dynamical_systems
ORDER BY window_idx
        `
      },
      {
        id: "dynamics_bifurcation",
        name: "Bifurcation Detection",
        description: "Windows where system behavior fundamentally changes",
        sql: `
WITH stability_changes AS (
    SELECT
        window_idx,
        lyapunov_exponent,
        SIGN(lyapunov_exponent) AS stability_sign,
        LAG(SIGN(lyapunov_exponent)) OVER (ORDER BY window_idx) AS prev_sign
    FROM dynamical_systems
)
SELECT
    window_idx,
    lyapunov_exponent,
    CASE
        WHEN prev_sign <= 0 AND stability_sign > 0 THEN 'onset_of_chaos'
        WHEN prev_sign > 0 AND stability_sign <= 0 THEN 'return_to_stability'
        ELSE 'no_transition'
    END AS bifurcation_type
FROM stability_changes
WHERE stability_sign != prev_sign
ORDER BY window_idx
        `
      },
      {
        id: "dynamics_attractor",
        name: "Attractor Properties",
        description: "Phase space attractor characteristics",
        sql: `
SELECT
    window_idx,
    correlation_dimension AS fractal_dim,
    CASE
        WHEN correlation_dimension < 1.5 THEN 'point_attractor'
        WHEN correlation_dimension < 2.5 THEN 'limit_cycle'
        WHEN correlation_dimension < 3.5 THEN 'torus'
        ELSE 'strange_attractor'
    END AS attractor_type,
    embedding_dimension,
    lyapunov_exponent
FROM dynamical_systems
ORDER BY window_idx
        `
      }
    ]
  },

  // ============================================================
  // CAUSAL MECHANICS - Information flow & causality
  // ============================================================
  mechanics: {
    label: "Causal Mechanics",
    queries: [
      {
        id: "mechanics_transfer_entropy",
        name: "Information Flow",
        description: "Transfer entropy between sensor pairs",
        sql: `
SELECT
    source_signal,
    target_signal,
    ROUND(transfer_entropy, 4) AS transfer_entropy,
    ROUND(transfer_entropy - reverse_transfer_entropy, 4) AS net_flow,
    CASE
        WHEN transfer_entropy > reverse_transfer_entropy THEN 'forward'
        WHEN transfer_entropy < reverse_transfer_entropy THEN 'backward'
        ELSE 'bidirectional'
    END AS flow_direction
FROM causal_mechanics
ORDER BY transfer_entropy DESC
        `
      },
      {
        id: "mechanics_granger",
        name: "Granger Causality",
        description: "Statistical causality tests",
        sql: `
SELECT
    source_signal,
    target_signal,
    granger_fstat,
    granger_pvalue,
    CASE
        WHEN granger_pvalue < 0.01 THEN 'strong_causality'
        WHEN granger_pvalue < 0.05 THEN 'significant'
        WHEN granger_pvalue < 0.10 THEN 'marginal'
        ELSE 'no_causality'
    END AS causality_strength,
    optimal_lag
FROM causal_mechanics
WHERE granger_pvalue < 0.05
ORDER BY granger_pvalue
        `
      },
      {
        id: "mechanics_causal_graph",
        name: "Causal Graph Edges",
        description: "Edges for causal network visualization",
        sql: `
SELECT
    source_signal AS source,
    target_signal AS target,
    transfer_entropy AS weight,
    granger_pvalue,
    optimal_lag AS lag,
    CASE WHEN granger_pvalue < 0.05 THEN 1 ELSE 0 END AS significant
FROM causal_mechanics
WHERE transfer_entropy > 0.1 OR granger_pvalue < 0.05
ORDER BY weight DESC
        `
      },
      {
        id: "mechanics_drivers",
        name: "System Drivers",
        description: "Sensors that drive others vs respond to others",
        sql: `
WITH outflow AS (
    SELECT
        source_signal AS signal,
        SUM(transfer_entropy) AS total_outflow,
        COUNT(*) AS n_targets
    FROM causal_mechanics
    GROUP BY source_signal
),
inflow AS (
    SELECT
        target_signal AS signal,
        SUM(transfer_entropy) AS total_inflow,
        COUNT(*) AS n_sources
    FROM causal_mechanics
    GROUP BY target_signal
)
SELECT
    COALESCE(o.signal, i.signal) AS signal,
    ROUND(COALESCE(o.total_outflow, 0), 4) AS total_outflow,
    ROUND(COALESCE(i.total_inflow, 0), 4) AS total_inflow,
    ROUND(COALESCE(o.total_outflow, 0) - COALESCE(i.total_inflow, 0), 4) AS net_influence,
    CASE
        WHEN COALESCE(o.total_outflow, 0) > COALESCE(i.total_inflow, 0) * 1.5 THEN 'driver'
        WHEN COALESCE(i.total_inflow, 0) > COALESCE(o.total_outflow, 0) * 1.5 THEN 'responder'
        ELSE 'mixed'
    END AS signal_role
FROM outflow o
FULL OUTER JOIN inflow i ON o.signal = i.signal
ORDER BY net_influence DESC
        `
      }
    ]
  },

  // ============================================================
  // VECTOR (PRISM output) - Raw engine results
  // ============================================================
  vector: {
    label: "Vector (Raw)",
    queries: [
      {
        id: "vector_all",
        name: "All Vector Metrics",
        description: "Full vector.parquet contents",
        sql: `
SELECT *
FROM vector
ORDER BY entity_id, signal_id, window_idx
LIMIT 1000
        `
      },
      {
        id: "vector_summary",
        name: "Vector Summary by Sensor",
        description: "Aggregate metrics per sensor",
        sql: `
SELECT
    entity_id,
    signal_id,
    COUNT(*) AS n_windows,
    ROUND(AVG(hurst_dfa), 3) AS avg_hurst,
    ROUND(AVG(sample_entropy), 3) AS avg_entropy,
    ROUND(AVG(trend_slope), 6) AS avg_trend
FROM vector
GROUP BY entity_id, signal_id
ORDER BY entity_id, signal_id
        `
      },
      {
        id: "vector_degrading",
        name: "Degrading Entities",
        description: "Entities with positive trend slope (degrading)",
        sql: `
SELECT
    entity_id,
    signal_id,
    window_idx,
    trend_slope,
    hurst_dfa,
    sample_entropy
FROM vector
WHERE trend_slope > 0
ORDER BY trend_slope DESC
LIMIT 100
        `
      }
    ]
  },

  // ============================================================
  // INDEX & WINDOWS - Core infrastructure
  // ============================================================
  infrastructure: {
    label: "Infrastructure",
    queries: [
      {
        id: "infra_windows",
        name: "Window Summary",
        description: "Overview of all analysis windows",
        sql: `
SELECT
    window_idx,
    index_start,
    index_end,
    index_center,
    n_observations,
    span,
    density,
    is_complete,
    boundary_type
FROM orthon_windows
ORDER BY window_idx
        `
      },
      {
        id: "infra_coverage",
        name: "Data Coverage",
        description: "Index range and window distribution",
        sql: `
SELECT
    MIN(index_start) AS index_min,
    MAX(index_end) AS index_max,
    MAX(index_end) - MIN(index_start) AS total_span,
    COUNT(*) AS n_windows,
    SUM(n_observations) AS total_observations,
    AVG(density) AS avg_density,
    SUM(CASE WHEN is_complete THEN 1 ELSE 0 END) AS complete_windows,
    SUM(CASE WHEN NOT is_complete THEN 1 ELSE 0 END) AS incomplete_windows
FROM orthon_windows
        `
      },
      {
        id: "infra_signals",
        name: "Sensor Inventory",
        description: "List of all sensors with basic stats",
        sql: `
SELECT DISTINCT
    signal_id,
    COUNT(*) AS n_windows,
    SUM(CASE WHEN hurst IS NOT NULL THEN 1 ELSE 0 END) AS hurst_computed,
    SUM(CASE WHEN entropy IS NOT NULL THEN 1 ELSE 0 END) AS entropy_computed,
    SUM(CASE WHEN trend_slope IS NOT NULL THEN 1 ELSE 0 END) AS trend_computed
FROM signal_typology
GROUP BY signal_id
ORDER BY signal_id
        `
      },
      {
        id: "infra_tables",
        name: "Available Tables",
        description: "List all loaded tables",
        sql: `
SHOW TABLES
        `
      }
    ]
  },

  // ============================================================
  // VALIDATION - Data quality & consistency checks
  // ============================================================
  validation: {
    label: "Validation",
    queries: [
      {
        id: "validation_unit_consistency",
        name: "Unit Consistency",
        description: "Check sensor units match expected engine requirements",
        sql: `
SELECT
    signal_id,
    unit,
    CASE
        WHEN engine = 'kinetic_energy' AND unit NOT IN ('m/s', 'ft/s', 'km/h')
        THEN 'WARN: velocity unit mismatch'
        WHEN engine = 'pressure_drop' AND unit NOT IN ('Pa', 'psi', 'bar', 'kPa')
        THEN 'WARN: pressure unit mismatch'
        WHEN engine = 'reynolds' AND unit NOT IN ('m/s', 'ft/s')
        THEN 'WARN: velocity unit for Reynolds'
        ELSE 'OK'
    END AS validation
FROM signal_metadata
WHERE validation != 'OK'
ORDER BY signal_id
        `
      },
      {
        id: "validation_null_check",
        name: "Null Value Check",
        description: "Find sensors with excessive null values",
        sql: `
SELECT
    signal_id,
    COUNT(*) AS total_windows,
    SUM(CASE WHEN hurst IS NULL THEN 1 ELSE 0 END) AS null_hurst,
    SUM(CASE WHEN entropy IS NULL THEN 1 ELSE 0 END) AS null_entropy,
    ROUND(100.0 * SUM(CASE WHEN hurst IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_null_hurst
FROM signal_typology
GROUP BY signal_id
HAVING pct_null_hurst > 10
ORDER BY pct_null_hurst DESC
        `
      },
      {
        id: "validation_range_check",
        name: "Value Range Check",
        description: "Detect out-of-range metric values",
        sql: `
SELECT
    signal_id,
    window_idx,
    hurst,
    entropy,
    CASE
        WHEN hurst < 0 OR hurst > 1 THEN 'WARN: hurst out of [0,1]'
        WHEN entropy < 0 THEN 'WARN: negative entropy'
        ELSE 'OK'
    END AS validation
FROM signal_typology
WHERE hurst < 0 OR hurst > 1 OR entropy < 0
ORDER BY signal_id, window_idx
        `
      },
      {
        id: "validation_window_gaps",
        name: "Window Gap Check",
        description: "Detect gaps or overlaps in window coverage",
        sql: `
WITH window_pairs AS (
    SELECT
        window_idx,
        index_start,
        index_end,
        LEAD(index_start) OVER (ORDER BY window_idx) AS next_start
    FROM orthon_windows
)
SELECT
    window_idx,
    index_end,
    next_start,
    next_start - index_end AS gap_size,
    CASE
        WHEN next_start - index_end > 0 THEN 'GAP'
        WHEN next_start - index_end < 0 THEN 'OVERLAP'
        ELSE 'CONTIGUOUS'
    END AS status
FROM window_pairs
WHERE next_start IS NOT NULL
  AND next_start - index_end != 0
ORDER BY window_idx
        `
      }
    ]
  }
};

// Export for use in HTML
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ORTHON_QUERIES;
}
