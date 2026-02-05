-- ============================================================================
-- ORTHON SQL: 12_physics_layer.sql
-- ============================================================================
-- Load and interpret PRISM physics.parquet output
--
-- The Physics Layer (L4→L1) contains:
--   L4: Thermodynamics - energy_proxy, dissipation_rate, entropy_production
--   L3: Mechanics      - energy_velocity, energy_flow_asymmetry
--   L2: Coherence      - coherence, effective_dim, eigenvalue_entropy
--   L1: State          - state_distance, state_velocity, state_acceleration
--
-- The Ørthon Signal: dissipating + decoupling + diverging
-- ============================================================================

-- ============================================================================
-- LOAD PHYSICS DATA
-- ============================================================================

CREATE OR REPLACE TABLE physics AS
SELECT * FROM read_parquet('{prism_output}/physics.parquet');

-- ============================================================================
-- L4: THERMODYNAMICS - Is energy conserved?
-- ============================================================================

CREATE OR REPLACE VIEW v_l4_thermodynamics AS
SELECT
    entity_id,
    I,
    -- Energy metrics
    energy_proxy,
    energy_velocity,
    dissipation_rate,
    entropy_production,

    -- Energy trend classification
    CASE
        WHEN energy_velocity < -0.001 THEN 'dissipating'
        WHEN energy_velocity > 0.001 THEN 'accumulating'
        ELSE 'stable'
    END AS energy_trend,

    -- Entropy trend
    CASE
        WHEN entropy_production > 0.001 THEN 'increasing'
        WHEN entropy_production < -0.001 THEN 'decreasing'
        ELSE 'stable'
    END AS entropy_trend,

    -- Conservation check (rolling coefficient of variation)
    STDDEV(energy_proxy) OVER w / NULLIF(AVG(energy_proxy) OVER w, 0) AS energy_cv,

    -- Is energy approximately conserved?
    CASE
        WHEN STDDEV(energy_proxy) OVER w / NULLIF(AVG(energy_proxy) OVER w, 0) < 0.1
         AND ABS(AVG(energy_velocity) OVER w) < 0.01 * AVG(energy_proxy) OVER w
        THEN TRUE
        ELSE FALSE
    END AS energy_conserved

FROM physics
WINDOW w AS (PARTITION BY entity_id ORDER BY I ROWS BETWEEN 10 PRECEDING AND CURRENT ROW);


-- ============================================================================
-- L3: MECHANICS - Where is energy flowing?
-- ============================================================================

CREATE OR REPLACE VIEW v_l3_mechanics AS
SELECT
    entity_id,
    I,
    energy_velocity,
    energy_flow_asymmetry,

    -- Energy distribution classification
    CASE
        WHEN energy_flow_asymmetry > 0.6 THEN 'concentrated'
        WHEN energy_flow_asymmetry > 0.3 THEN 'uneven'
        ELSE 'distributed'
    END AS energy_distribution,

    -- Transfer event detection (high velocity periods)
    CASE
        WHEN ABS(energy_velocity) > 2 * STDDEV(energy_velocity) OVER (PARTITION BY entity_id)
        THEN TRUE
        ELSE FALSE
    END AS is_transfer_event

FROM physics
WHERE energy_velocity IS NOT NULL;


-- ============================================================================
-- L2: COHERENCE - Eigenvalue-Based Symplectic Structure
-- ============================================================================
-- coherence = λ₁/Σλ (spectral coherence)
-- effective_dim = participation ratio
-- eigenvalue_entropy = spectral disorder

CREATE OR REPLACE VIEW v_l2_coherence AS
SELECT
    entity_id,
    I,
    -- Raw metrics
    coherence,
    coherence_velocity,
    effective_dim,
    eigenvalue_entropy,
    n_signals,
    n_pairs,

    -- Coupling state (eigenvalue-based thresholds)
    CASE
        WHEN coherence > 0.7 THEN 'strongly_coupled'
        WHEN coherence > 0.4 THEN 'weakly_coupled'
        ELSE 'decoupled'
    END AS coupling_state,

    -- Structure state (how fragmented)
    CASE
        WHEN effective_dim < 1.5 THEN 'unified'
        WHEN effective_dim < n_signals / 2.0 THEN 'clustered'
        ELSE 'fragmented'
    END AS structure_state,

    -- Eigenvalue entropy interpretation
    CASE
        WHEN eigenvalue_entropy < 0.3 THEN 'highly_ordered'
        WHEN eigenvalue_entropy < 0.6 THEN 'partially_ordered'
        ELSE 'disordered'
    END AS spectral_order,

    -- Baseline comparison (first 10% as baseline)
    coherence / NULLIF(
        FIRST_VALUE(coherence) OVER (PARTITION BY entity_id ORDER BY I), 0
    ) AS coherence_vs_baseline,

    effective_dim / NULLIF(
        FIRST_VALUE(effective_dim) OVER (PARTITION BY entity_id ORDER BY I), 0
    ) AS effective_dim_vs_baseline

FROM physics
WHERE coherence IS NOT NULL;


-- Decoupling detection
CREATE OR REPLACE VIEW v_decoupling_detection AS
WITH baseline AS (
    SELECT
        entity_id,
        AVG(coherence) AS baseline_coherence,
        AVG(effective_dim) AS baseline_effective_dim
    FROM (
        SELECT entity_id, coherence, effective_dim,
               ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY I) AS rn,
               COUNT(*) OVER (PARTITION BY entity_id) AS total
        FROM physics
        WHERE coherence IS NOT NULL
    ) sub
    WHERE rn <= total * 0.1  -- First 10%
    GROUP BY entity_id
),
trends AS (
    SELECT
        entity_id,
        -- Linear regression slope for coherence
        REGR_SLOPE(coherence, I) AS coherence_trend,
        REGR_SLOPE(effective_dim, I) AS effective_dim_trend
    FROM physics
    WHERE coherence IS NOT NULL
    GROUP BY entity_id
)
SELECT
    p.entity_id,
    p.I,
    p.coherence,
    p.effective_dim,
    p.eigenvalue_entropy,
    b.baseline_coherence,
    b.baseline_effective_dim,
    t.coherence_trend,
    t.effective_dim_trend,

    -- Decoupling detection
    CASE
        WHEN t.coherence_trend < -0.001 THEN TRUE
        WHEN p.coherence < b.baseline_coherence * 0.8 THEN TRUE
        WHEN p.effective_dim > b.baseline_effective_dim * 1.5 THEN TRUE
        ELSE FALSE
    END AS is_decoupling,

    -- Fragmentation detection
    CASE
        WHEN t.effective_dim_trend > 0.01 AND p.eigenvalue_entropy > 0.5 THEN TRUE
        ELSE FALSE
    END AS is_fragmenting

FROM physics p
JOIN baseline b USING (entity_id)
JOIN trends t USING (entity_id)
WHERE p.coherence IS NOT NULL;


-- ============================================================================
-- L1: STATE - Phase Space Position
-- ============================================================================
-- state_distance = Mahalanobis distance from baseline
-- state_velocity = generalized hd_slope (using ALL metrics)

CREATE OR REPLACE VIEW v_l1_state AS
SELECT
    entity_id,
    I,
    -- Raw metrics
    state_distance,
    state_velocity,
    state_acceleration,
    n_metrics_used,

    -- Stability classification
    CASE
        WHEN ABS(state_velocity) < 0.01 AND state_distance < 2.0 THEN TRUE
        ELSE FALSE
    END AS is_stable,

    -- Trend classification
    CASE
        WHEN state_velocity > 0.01 THEN 'diverging'
        WHEN state_velocity < -0.01 THEN 'converging'
        ELSE 'stable'
    END AS state_trend,

    -- Distance severity
    CASE
        WHEN state_distance > 3.0 THEN 'critical'
        WHEN state_distance > 2.0 THEN 'warning'
        WHEN state_distance > 1.0 THEN 'watch'
        ELSE 'normal'
    END AS distance_severity,

    -- Acceleration (is degradation speeding up?)
    CASE
        WHEN state_velocity > 0 AND state_acceleration > 0 THEN 'accelerating_away'
        WHEN state_velocity > 0 AND state_acceleration < 0 THEN 'decelerating_away'
        WHEN state_velocity < 0 AND state_acceleration < 0 THEN 'accelerating_return'
        WHEN state_velocity < 0 AND state_acceleration > 0 THEN 'decelerating_return'
        ELSE 'stable'
    END AS motion_state

FROM physics
WHERE state_distance IS NOT NULL;


-- ============================================================================
-- THE ØRTHON SIGNAL
-- ============================================================================
-- dissipating + decoupling + diverging = symplectic structure loss

CREATE OR REPLACE VIEW v_orthon_signal AS
SELECT
    p.entity_id,
    p.I,

    -- L4: Thermodynamics
    l4.energy_trend,
    l4.entropy_trend,
    l4.energy_conserved,

    -- L2: Coherence
    l2.coupling_state,
    l2.structure_state,
    dd.is_decoupling,
    dd.is_fragmenting,

    -- L1: State
    l1.state_trend,
    l1.is_stable,
    l1.distance_severity,

    -- The Ørthon Signal
    CASE
        WHEN l4.energy_trend = 'dissipating'
         AND (dd.is_decoupling OR dd.is_fragmenting OR l2.coupling_state = 'decoupled')
         AND l1.state_trend = 'diverging'
        THEN TRUE
        ELSE FALSE
    END AS orthon_signal,

    -- Severity scoring
    (CASE WHEN l4.energy_trend = 'dissipating' THEN 1 ELSE 0 END) +
    (CASE WHEN dd.is_decoupling THEN 1 ELSE 0 END) +
    (CASE WHEN dd.is_fragmenting THEN 1 ELSE 0 END) +
    (CASE WHEN l2.coupling_state = 'decoupled' THEN 2 ELSE 0 END) +
    (CASE WHEN l1.state_trend = 'diverging' THEN 1 ELSE 0 END) +
    (CASE WHEN p.state_distance > 3 THEN 1 ELSE 0 END) AS severity_score,

    -- Overall severity
    CASE
        WHEN l4.energy_trend = 'dissipating'
         AND (dd.is_decoupling OR l2.coupling_state = 'decoupled')
         AND l1.state_trend = 'diverging'
        THEN 'critical'
        WHEN l4.energy_trend = 'dissipating' AND l1.state_trend = 'diverging'
        THEN 'warning'
        WHEN dd.is_decoupling OR l1.state_trend = 'diverging'
        THEN 'watch'
        ELSE 'normal'
    END AS severity

FROM physics p
LEFT JOIN v_l4_thermodynamics l4 ON p.entity_id = l4.entity_id AND p.I = l4.I
LEFT JOIN v_l2_coherence l2 ON p.entity_id = l2.entity_id AND p.I = l2.I
LEFT JOIN v_decoupling_detection dd ON p.entity_id = dd.entity_id AND p.I = dd.I
LEFT JOIN v_l1_state l1 ON p.entity_id = l1.entity_id AND p.I = l1.I;


-- ============================================================================
-- ENTITY SUMMARY (Current State)
-- ============================================================================

CREATE OR REPLACE VIEW v_entity_physics_summary AS
WITH latest AS (
    SELECT
        entity_id,
        MAX(I) AS max_I
    FROM physics
    GROUP BY entity_id
)
SELECT
    p.entity_id,

    -- L4: Thermodynamics
    p.energy_proxy AS current_energy,
    p.dissipation_rate AS current_dissipation,
    l4.energy_trend,
    l4.energy_conserved,

    -- L2: Coherence (eigenvalue-based)
    p.coherence AS current_coherence,
    p.effective_dim AS current_effective_dim,
    p.eigenvalue_entropy AS current_eigenvalue_entropy,
    l2.coupling_state,
    l2.structure_state,
    dd.is_decoupling,
    dd.is_fragmenting,
    dd.baseline_coherence,
    dd.coherence_trend,

    -- L1: State
    p.state_distance AS current_state_distance,
    p.state_velocity AS current_state_velocity,
    l1.state_trend,
    l1.is_stable,
    l1.distance_severity,

    -- Ørthon Signal
    os.orthon_signal,
    os.severity_score,
    os.severity,

    -- Counts
    p.n_signals,
    p.n_pairs

FROM physics p
JOIN latest ON p.entity_id = latest.entity_id AND p.I = latest.max_I
LEFT JOIN v_l4_thermodynamics l4 ON p.entity_id = l4.entity_id AND p.I = l4.I
LEFT JOIN v_l2_coherence l2 ON p.entity_id = l2.entity_id AND p.I = l2.I
LEFT JOIN v_decoupling_detection dd ON p.entity_id = dd.entity_id AND p.I = dd.I
LEFT JOIN v_l1_state l1 ON p.entity_id = l1.entity_id AND p.I = l1.I
LEFT JOIN v_orthon_signal os ON p.entity_id = os.entity_id AND p.I = os.I;


-- ============================================================================
-- FLEET SUMMARY
-- ============================================================================

CREATE OR REPLACE VIEW v_fleet_physics_summary AS
SELECT
    COUNT(DISTINCT entity_id) AS n_entities,

    -- Severity distribution
    SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) AS n_critical,
    SUM(CASE WHEN severity = 'warning' THEN 1 ELSE 0 END) AS n_warning,
    SUM(CASE WHEN severity = 'watch' THEN 1 ELSE 0 END) AS n_watch,
    SUM(CASE WHEN severity = 'normal' THEN 1 ELSE 0 END) AS n_normal,

    -- Ørthon signals
    SUM(CASE WHEN orthon_signal THEN 1 ELSE 0 END) AS n_orthon_signals,

    -- Percent healthy
    100.0 * SUM(CASE WHEN severity = 'normal' THEN 1 ELSE 0 END) / COUNT(*) AS pct_healthy,

    -- Coupling state distribution
    SUM(CASE WHEN coupling_state = 'strongly_coupled' THEN 1 ELSE 0 END) AS n_strongly_coupled,
    SUM(CASE WHEN coupling_state = 'weakly_coupled' THEN 1 ELSE 0 END) AS n_weakly_coupled,
    SUM(CASE WHEN coupling_state = 'decoupled' THEN 1 ELSE 0 END) AS n_decoupled,

    -- Average metrics
    AVG(current_coherence) AS avg_coherence,
    AVG(current_effective_dim) AS avg_effective_dim,
    AVG(current_state_distance) AS avg_state_distance

FROM v_entity_physics_summary;


-- ============================================================================
-- COHERENCE INTERPRETATION (Human Readable)
-- ============================================================================

CREATE OR REPLACE VIEW v_coherence_interpretation AS
SELECT
    entity_id,
    I,
    coherence,
    effective_dim,
    eigenvalue_entropy,
    n_signals,
    coupling_state,
    structure_state,

    -- Human readable interpretation
    CASE coupling_state
        WHEN 'strongly_coupled' THEN 'Signals are strongly coupled (coherence ' || ROUND(coherence, 2) || ')'
        WHEN 'weakly_coupled' THEN 'Signals are weakly coupled (coherence ' || ROUND(coherence, 2) || ')'
        ELSE 'Signals are decoupled (coherence ' || ROUND(coherence, 2) || ')'
    END || '. ' ||
    CASE structure_state
        WHEN 'unified' THEN 'System moving as one mode (effective dim ' || ROUND(effective_dim, 1) || ' of ' || n_signals || ')'
        WHEN 'clustered' THEN 'System has ~' || ROUND(effective_dim) || ' independent clusters'
        ELSE 'System fragmented into ~' || ROUND(effective_dim) || ' independent modes'
    END AS interpretation

FROM v_l2_coherence;


-- ============================================================================
-- ALERTS
-- ============================================================================

CREATE OR REPLACE VIEW v_physics_alerts AS
SELECT
    entity_id,
    I,
    'CRITICAL' AS alert_level,
    'ØRTHON SIGNAL: Symplectic structure loss detected' AS alert_message,
    severity_score
FROM v_orthon_signal
WHERE orthon_signal = TRUE

UNION ALL

SELECT
    entity_id,
    I,
    'WARNING' AS alert_level,
    'Energy dissipating while state diverging' AS alert_message,
    severity_score
FROM v_orthon_signal
WHERE severity = 'warning' AND orthon_signal = FALSE

UNION ALL

SELECT
    entity_id,
    I,
    'WATCH' AS alert_level,
    CASE
        WHEN is_decoupling THEN 'Signals decoupling'
        WHEN is_fragmenting THEN 'Mode fragmentation in progress'
        WHEN state_trend = 'diverging' THEN 'State diverging from baseline'
        ELSE 'Anomaly detected'
    END AS alert_message,
    severity_score
FROM v_orthon_signal
WHERE severity = 'watch';


-- ============================================================================
-- VERIFICATION
-- ============================================================================

SELECT
    'physics' AS table_name,
    COUNT(*) AS row_count,
    COUNT(DISTINCT entity_id) AS n_entities
FROM physics;
