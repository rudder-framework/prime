-- ============================================================================
-- ORTHON SQL: 30_dynamics_stability.sql
-- ============================================================================
-- DYNAMICS STABILITY: Lyapunov + RQA-based stability classification
--
-- Interprets dynamical systems metrics from PRISM dynamics.parquet:
--   - Lyapunov exponent: trajectory divergence rate
--   - Determinism (DET): predictability of dynamics
--   - Laminarity (LAM): state trapping behavior
--   - RQA entropy: complexity of recurrence structure
--   - Correlation dimension: attractor complexity
--
-- Usage:
--   .read 30_dynamics_stability.sql
-- ============================================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║                    DYNAMICAL STABILITY ANALYSIS                             ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'

-- ============================================================================
-- SECTION 1: LOAD DYNAMICS (window-level Lyapunov + RQA)
-- ============================================================================

.print ''
.print '=== SECTION 1: Window-Level Dynamics ==='

CREATE OR REPLACE TABLE dynamics_windows AS
SELECT
    entity_id,
    I,
    lyapunov_max,
    determinism,
    laminarity,
    rqa_entropy,
    trapping_time,
    correlation_dim,
    embedding_dim,

    -- Classify each window
    CASE
        WHEN lyapunov_max > 0.1 THEN 'CHAOTIC'
        WHEN lyapunov_max > 0.02 THEN 'WEAKLY_UNSTABLE'
        WHEN lyapunov_max > 0 THEN 'MARGINAL'
        WHEN lyapunov_max IS NOT NULL THEN 'STABLE'
        ELSE 'UNKNOWN'
    END as lyapunov_class,

    -- RQA-based dynamics classification
    CASE
        WHEN determinism > 0.5 AND laminarity < 0.2 THEN 'PERIODIC'
        WHEN determinism > 0.3 AND laminarity > 0.3 THEN 'INTERMITTENT'
        WHEN determinism < 0.1 AND laminarity < 0.1 THEN 'STOCHASTIC'
        WHEN determinism < 0.1 AND laminarity > 0.2 THEN 'TRAPPED_STOCHASTIC'
        ELSE 'MIXED'
    END as rqa_class

FROM read_parquet('{prism_output}/dynamics.parquet');

SELECT
    lyapunov_class,
    COUNT(*) as n_windows,
    ROUND(AVG(lyapunov_max), 4) as avg_lyapunov,
    ROUND(AVG(determinism), 3) as avg_det,
    ROUND(AVG(laminarity), 3) as avg_lam
FROM dynamics_windows
GROUP BY lyapunov_class
ORDER BY avg_lyapunov DESC;


-- ============================================================================
-- SECTION 2: ENTITY-LEVEL STABILITY
-- ============================================================================

.print ''
.print '=== SECTION 2: Entity-Level Stability ==='

CREATE OR REPLACE TABLE dynamics_entities AS
SELECT
    entity_id,
    COUNT(*) as n_windows,

    -- Lyapunov statistics
    ROUND(AVG(lyapunov_max), 4) as mean_lyapunov,
    ROUND(MAX(lyapunov_max), 4) as max_lyapunov,
    ROUND(STDDEV(lyapunov_max), 4) as std_lyapunov,

    -- RQA statistics
    ROUND(AVG(determinism), 3) as mean_determinism,
    ROUND(AVG(laminarity), 3) as mean_laminarity,
    ROUND(AVG(rqa_entropy), 3) as mean_rqa_entropy,
    ROUND(AVG(trapping_time), 2) as mean_trapping_time,
    ROUND(AVG(correlation_dim), 2) as mean_corr_dim,

    -- Count windows in each state
    SUM(CASE WHEN lyapunov_class = 'CHAOTIC' THEN 1 ELSE 0 END) as n_chaotic,
    SUM(CASE WHEN lyapunov_class = 'WEAKLY_UNSTABLE' THEN 1 ELSE 0 END) as n_weak_unstable,
    SUM(CASE WHEN rqa_class = 'STOCHASTIC' THEN 1 ELSE 0 END) as n_stochastic,
    SUM(CASE WHEN rqa_class = 'INTERMITTENT' THEN 1 ELSE 0 END) as n_intermittent,

    -- Entity stability classification (combined Lyapunov + RQA)
    CASE
        WHEN MAX(lyapunov_max) > 0.1 THEN 'CHAOTIC'
        WHEN AVG(lyapunov_max) > 0.02 AND AVG(determinism) < 0.1 THEN 'UNSTABLE'
        WHEN AVG(lyapunov_max) > 0 THEN 'WEAKLY_UNSTABLE'
        WHEN AVG(lyapunov_max) > -0.02 AND AVG(determinism) < 0.2 THEN 'MARGINAL'
        ELSE 'STABLE'
    END as entity_stability,

    -- Dynamics type (from RQA)
    CASE
        WHEN AVG(determinism) > 0.5 THEN 'DETERMINISTIC'
        WHEN AVG(determinism) > 0.2 THEN 'MIXED'
        ELSE 'STOCHASTIC'
    END as dynamics_type,

    -- Stability score: 0 (chaotic) to 1 (stable)
    -- Combines Lyapunov (70%) and RQA determinism (30%)
    ROUND(
        0.7 * (1.0 / (1.0 + EXP(AVG(lyapunov_max) * 20))) +
        0.3 * AVG(determinism)
    , 3) as stability_score

FROM dynamics_windows
GROUP BY entity_id;

SELECT
    entity_stability,
    dynamics_type,
    COUNT(*) as n_entities,
    ROUND(AVG(mean_lyapunov), 4) as avg_lyapunov,
    ROUND(AVG(mean_determinism), 3) as avg_det,
    ROUND(AVG(stability_score), 3) as avg_score
FROM dynamics_entities
GROUP BY entity_stability, dynamics_type
ORDER BY avg_score ASC;


-- ============================================================================
-- SECTION 3: RQA DYNAMICS BREAKDOWN
-- ============================================================================

.print ''
.print '=== SECTION 3: RQA Dynamics Breakdown ==='

SELECT
    rqa_class,
    COUNT(*) as n_windows,
    COUNT(DISTINCT entity_id) as n_entities,
    ROUND(AVG(determinism), 3) as avg_det,
    ROUND(AVG(laminarity), 3) as avg_lam,
    ROUND(AVG(trapping_time), 2) as avg_trap,
    ROUND(AVG(lyapunov_max), 4) as avg_lyap
FROM dynamics_windows
GROUP BY rqa_class
ORDER BY n_windows DESC;


-- ============================================================================
-- SECTION 4: STABILITY vs ORTHON SIGNAL
-- ============================================================================

.print ''
.print '=== SECTION 4: Stability vs ORTHON Signal ==='

SELECT
    d.entity_stability,
    COUNT(*) as n_entities,
    ROUND(AVG(CASE WHEN p.dissipation_rate > 0.01 AND p.coherence < 0.5
        AND p.state_velocity > 0.05 THEN 1.0 ELSE 0.0 END) * 100, 1) as pct_orthon_signal,
    ROUND(AVG(p.coherence), 3) as avg_coherence,
    ROUND(AVG(p.state_velocity), 3) as avg_velocity,
    ROUND(AVG(d.mean_determinism), 3) as avg_determinism
FROM dynamics_entities d
LEFT JOIN (
    SELECT
        entity_id,
        AVG(dissipation_rate) as dissipation_rate,
        AVG(coherence) as coherence,
        AVG(state_velocity) as state_velocity
    FROM read_parquet('{prism_output}/physics.parquet')
    GROUP BY entity_id
) p ON d.entity_id = p.entity_id
GROUP BY d.entity_stability
ORDER BY
    CASE d.entity_stability
        WHEN 'CHAOTIC' THEN 1
        WHEN 'UNSTABLE' THEN 2
        WHEN 'WEAKLY_UNSTABLE' THEN 3
        WHEN 'MARGINAL' THEN 4
        ELSE 5
    END;


-- ============================================================================
-- SECTION 5: STABILITY RANKING
-- ============================================================================

.print ''
.print '=== SECTION 5: Stability Ranking (most unstable first) ==='

SELECT
    entity_id,
    entity_stability,
    dynamics_type,
    stability_score,
    mean_lyapunov,
    mean_determinism as det,
    mean_laminarity as lam,
    n_windows
FROM dynamics_entities
ORDER BY stability_score ASC
LIMIT 15;


-- ============================================================================
-- CREATE VIEWS
-- ============================================================================

CREATE OR REPLACE VIEW v_dynamics_summary AS
SELECT
    d.entity_id,
    d.entity_stability,
    d.dynamics_type,
    d.stability_score,
    d.mean_lyapunov,
    d.max_lyapunov,
    d.mean_determinism,
    d.mean_laminarity,
    d.mean_rqa_entropy,
    d.mean_trapping_time,
    d.mean_corr_dim,
    d.n_windows,
    p.avg_coherence,
    p.avg_velocity
FROM dynamics_entities d
LEFT JOIN (
    SELECT
        entity_id,
        AVG(coherence) as avg_coherence,
        AVG(state_velocity) as avg_velocity
    FROM read_parquet('{prism_output}/physics.parquet')
    GROUP BY entity_id
) p ON d.entity_id = p.entity_id;

CREATE OR REPLACE VIEW v_dynamics_alerts AS
SELECT
    entity_id,
    CASE entity_stability
        WHEN 'CHAOTIC' THEN 'CRITICAL'
        WHEN 'UNSTABLE' THEN 'CRITICAL'
        WHEN 'WEAKLY_UNSTABLE' THEN 'WARNING'
        WHEN 'MARGINAL' THEN 'WATCH'
        ELSE 'NORMAL'
    END as alert_level,
    entity_stability || ': λ=' || ROUND(mean_lyapunov, 3) ||
    ', DET=' || ROUND(mean_determinism, 2) ||
    ', ' || dynamics_type as alert_message,
    1.0 - stability_score as severity_score
FROM dynamics_entities
WHERE entity_stability IN ('CHAOTIC', 'UNSTABLE', 'WEAKLY_UNSTABLE', 'MARGINAL');

-- Temporal dynamics: track Lyapunov over lifecycle
CREATE OR REPLACE VIEW v_dynamics_temporal AS
SELECT
    entity_id,
    I,
    lyapunov_max,
    determinism,
    laminarity,
    lyapunov_class,
    rqa_class,
    -- Rolling trend (simplified)
    lyapunov_max - LAG(lyapunov_max, 1) OVER (PARTITION BY entity_id ORDER BY I) as lyap_delta
FROM dynamics_windows;


.print ''
.print '=== DYNAMICS ANALYSIS COMPLETE ==='
.print ''
.print 'Views created:'
.print '  v_dynamics_summary   - Entity stability with Lyapunov + RQA'
.print '  v_dynamics_alerts    - Entities requiring attention'
.print '  v_dynamics_temporal  - Window-level dynamics over time'
.print ''
.print 'INTERPRETATION:'
.print '  Lyapunov > 0.1:    CHAOTIC - trajectories diverging exponentially'
.print '  Lyapunov > 0.02:   WEAKLY_UNSTABLE - mild instability'
.print '  DET < 0.1:         STOCHASTIC - unpredictable dynamics'
.print '  LAM > 0.2:         TRAPPED - system getting stuck in states'
.print ''
