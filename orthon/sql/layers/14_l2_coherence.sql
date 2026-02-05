-- ============================================================================
-- ORTHON SQL: 14_l2_coherence.sql
-- ============================================================================
-- L2: COHERENCE - Eigenvalue-Based Symplectic Structure
--
-- Third question in physics stack: Is the symplectic structure intact?
--
-- Eigenvalue-based coherence captures STRUCTURE, not just average correlation.
--   coherence = λ₁/Σλ (spectral coherence) - fraction of variance in dominant mode
--   effective_dim = participation ratio - how many independent modes
--   eigenvalue_entropy = spectral disorder - 0 (ordered) to 1 (disordered)
--
-- When first eigenvalue dominates, signals move as one.
-- When eigenvalues spread out, system is fragmenting into independent modes.
-- ============================================================================

-- Basic coherence metrics
CREATE OR REPLACE VIEW v_l2_coherence AS
SELECT
    entity_id,
    I,
    -- Raw eigenvalue-based metrics
    coherence,                  -- λ₁/Σλ
    coherence_velocity,
    effective_dim,              -- Participation ratio
    eigenvalue_entropy,         -- Spectral disorder
    n_signals,
    n_pairs,

    -- Coupling state (based on spectral coherence)
    CASE
        WHEN coherence > 0.7 THEN 'strongly_coupled'
        WHEN coherence > 0.4 THEN 'weakly_coupled'
        ELSE 'decoupled'
    END AS coupling_state,

    -- Structure state (based on effective dimensionality)
    CASE
        WHEN effective_dim < 1.5 THEN 'unified'
        WHEN effective_dim < n_signals / 2.0 THEN 'clustered'
        ELSE 'fragmented'
    END AS structure_state,

    -- Spectral order (based on eigenvalue entropy)
    CASE
        WHEN eigenvalue_entropy < 0.3 THEN 'highly_ordered'
        WHEN eigenvalue_entropy < 0.6 THEN 'partially_ordered'
        ELSE 'disordered'
    END AS spectral_order

FROM physics
WHERE coherence IS NOT NULL;


-- Coherence baseline per entity
CREATE OR REPLACE VIEW v_l2_baseline AS
SELECT
    entity_id,
    AVG(coherence) AS baseline_coherence,
    AVG(effective_dim) AS baseline_effective_dim,
    AVG(eigenvalue_entropy) AS baseline_entropy
FROM (
    SELECT
        entity_id,
        coherence,
        effective_dim,
        eigenvalue_entropy,
        ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY I) AS rn,
        COUNT(*) OVER (PARTITION BY entity_id) AS total
    FROM physics
    WHERE coherence IS NOT NULL
) sub
WHERE rn <= GREATEST(total * 0.1, 5)  -- First 10% or at least 5 points
GROUP BY entity_id;


-- Coherence trends per entity
CREATE OR REPLACE VIEW v_l2_trends AS
SELECT
    entity_id,
    REGR_SLOPE(coherence, I) AS coherence_trend,
    REGR_SLOPE(effective_dim, I) AS effective_dim_trend,
    REGR_SLOPE(eigenvalue_entropy, I) AS entropy_trend
FROM physics
WHERE coherence IS NOT NULL
GROUP BY entity_id;


-- Decoupling detection
CREATE OR REPLACE VIEW v_l2_decoupling AS
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

    -- Is decoupling? (coherence dropping or below baseline)
    CASE
        WHEN t.coherence_trend < -0.001 THEN TRUE
        WHEN p.coherence < b.baseline_coherence * 0.8 AND b.baseline_coherence > 0.3 THEN TRUE
        WHEN p.effective_dim > b.baseline_effective_dim * 1.5 AND b.baseline_effective_dim > 1 THEN TRUE
        ELSE FALSE
    END AS is_decoupling,

    -- Is fragmenting? (dimensions increasing with high disorder)
    CASE
        WHEN t.effective_dim_trend > 0.01 AND p.eigenvalue_entropy > 0.5 THEN TRUE
        ELSE FALSE
    END AS is_fragmenting,

    -- Coherence ratio vs baseline
    p.coherence / NULLIF(b.baseline_coherence, 0) AS coherence_ratio,

    -- Effective dim ratio vs baseline
    p.effective_dim / NULLIF(b.baseline_effective_dim, 0) AS effective_dim_ratio

FROM physics p
JOIN v_l2_baseline b USING (entity_id)
JOIN v_l2_trends t USING (entity_id)
WHERE p.coherence IS NOT NULL;


-- Entity-level coherence summary
CREATE OR REPLACE VIEW v_l2_entity_summary AS
WITH latest AS (
    SELECT entity_id, MAX(I) AS max_I
    FROM physics WHERE coherence IS NOT NULL
    GROUP BY entity_id
)
SELECT
    p.entity_id,
    p.n_signals,
    p.n_pairs,

    -- Current state
    p.coherence AS current_coherence,
    p.effective_dim AS current_effective_dim,
    p.eigenvalue_entropy AS current_eigenvalue_entropy,

    -- Classifications
    CASE
        WHEN p.coherence > 0.7 THEN 'strongly_coupled'
        WHEN p.coherence > 0.4 THEN 'weakly_coupled'
        ELSE 'decoupled'
    END AS coupling_state,

    CASE
        WHEN p.effective_dim < 1.5 THEN 'unified'
        WHEN p.effective_dim < p.n_signals / 2.0 THEN 'clustered'
        ELSE 'fragmented'
    END AS structure_state,

    -- Baseline comparison
    b.baseline_coherence,
    b.baseline_effective_dim,
    p.coherence / NULLIF(b.baseline_coherence, 0) AS coherence_vs_baseline,

    -- Trends
    t.coherence_trend,
    t.effective_dim_trend,

    -- Decoupling status
    d.is_decoupling,
    d.is_fragmenting

FROM physics p
JOIN latest l ON p.entity_id = l.entity_id AND p.I = l.max_I
JOIN v_l2_baseline b USING (entity_id)
JOIN v_l2_trends t USING (entity_id)
JOIN v_l2_decoupling d ON p.entity_id = d.entity_id AND p.I = d.I;


-- Human readable interpretation
CREATE OR REPLACE VIEW v_l2_interpretation AS
SELECT
    entity_id,
    I,
    coherence,
    effective_dim,
    n_signals,
    coupling_state,
    structure_state,

    -- Build interpretation string
    CASE coupling_state
        WHEN 'strongly_coupled' THEN 'Signals are strongly coupled (coherence ' || ROUND(coherence::DECIMAL, 2) || ')'
        WHEN 'weakly_coupled' THEN 'Signals are weakly coupled (coherence ' || ROUND(coherence::DECIMAL, 2) || ')'
        ELSE 'Signals are decoupled (coherence ' || ROUND(coherence::DECIMAL, 2) || ')'
    END || '. ' ||
    CASE structure_state
        WHEN 'unified' THEN 'System moving as one mode (effective dim ' || ROUND(effective_dim::DECIMAL, 1) || ' of ' || n_signals || ')'
        WHEN 'clustered' THEN 'System has ~' || ROUND(effective_dim)::INT || ' independent clusters'
        ELSE 'System fragmented into ~' || ROUND(effective_dim)::INT || ' independent modes'
    END AS interpretation

FROM v_l2_coherence;


-- Coherence change events
CREATE OR REPLACE VIEW v_l2_change_events AS
SELECT
    entity_id,
    I,
    coherence,
    coherence_velocity,
    effective_dim,
    LAG(coupling_state) OVER w AS prev_coupling_state,
    coupling_state,
    CASE
        WHEN LAG(coupling_state) OVER w != coupling_state THEN 'coupling_state_change'
        WHEN ABS(coherence_velocity) > 0.05 THEN 'rapid_coherence_change'
        ELSE NULL
    END AS event_type
FROM v_l2_coherence
WINDOW w AS (PARTITION BY entity_id ORDER BY I)
HAVING event_type IS NOT NULL;


-- Verify
SELECT COUNT(*) AS coherence_rows FROM v_l2_coherence;
