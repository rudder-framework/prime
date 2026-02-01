-- ============================================================================
-- ORTHON SQL: ff_stable_periods.sql
-- ============================================================================
-- Find most stable market geometry periods (Fama-French specific)
--
-- PURPOSE:
--   Financial markets have no "healthy baseline" - they're always in flux.
--   This query finds periods of STRUCTURAL STABILITY:
--   - Low chaos (negative Lyapunov)
--   - High determinism (predictable dynamics)
--   - Differentiated sectors (low cross-correlation = healthy diversification)
--
-- KEY INSIGHT:
--   In markets, HIGH correlation across sectors = CRISIS (contagion)
--   LOW correlation = HEALTHY (sectors move independently, diversification works)
--
-- Usage:
--   duckdb < ff_stable_periods.sql
-- ============================================================================

-- Load data
CREATE OR REPLACE VIEW dynamics AS
SELECT * FROM read_parquet('/Users/jasonrudder/prism/data/dynamics.parquet');

CREATE OR REPLACE VIEW geometry AS
SELECT * FROM read_parquet('/Users/jasonrudder/prism/data/geometry.parquet');

CREATE OR REPLACE VIEW physics AS
SELECT * FROM read_parquet('/Users/jasonrudder/prism/data/physics.parquet');

-- ============================================================================
-- STEP 1: Market-wide stability per time window
-- ============================================================================

CREATE OR REPLACE VIEW market_stability AS
SELECT
    d.I as window_idx,

    -- Average stability across all industries/portfolios
    AVG(d.lyapunov_max) AS avg_lyapunov,
    AVG(d.determinism) AS avg_determinism,
    AVG(d.recurrence_rate) AS avg_recurrence,
    AVG(d.entropy_rate) AS avg_entropy,

    -- Count of entities in this window
    COUNT(DISTINCT d.entity_id) AS n_entities,

    -- Cross-sector coherence from geometry (average pairwise correlation)
    -- HIGH coherence = correlated = CRISIS
    -- LOW coherence = differentiated = STABLE (healthy diversification)
    AVG(g.correlation_mean) AS avg_cross_correlation,
    AVG(g.coherence_mean) AS avg_coherence,

    -- Effective dimension (higher = more independent factors)
    AVG(p.effective_dimension) AS avg_eff_dim,

    -- STABILITY SCORE:
    -- + low chaos (negate Lyapunov)
    -- + high determinism
    -- - high cross-correlation (penalize crisis-like periods)
    -- + high effective dimension (reward diversification)
    (
        -1 * AVG(d.lyapunov_max)
        + AVG(d.determinism)
        - 0.5 * AVG(g.correlation_mean)  -- Penalize high correlation
        + 0.3 * AVG(p.effective_dimension)  -- Reward high dimension
    ) AS stability_score

FROM dynamics d
LEFT JOIN geometry g ON d.entity_id = g.entity_id AND d.I = g.I
LEFT JOIN physics p ON d.entity_id = p.entity_id AND d.I = p.I
GROUP BY d.I;

-- ============================================================================
-- STEP 2: Rank market periods by stability
-- ============================================================================

CREATE OR REPLACE VIEW v_market_stability_ranked AS
SELECT
    window_idx,
    stability_score,
    avg_lyapunov,
    avg_determinism,
    avg_recurrence,
    avg_entropy,
    avg_cross_correlation,
    avg_coherence,
    avg_eff_dim,
    n_entities,
    RANK() OVER (ORDER BY stability_score DESC) AS stability_rank,

    -- Classify periods
    CASE
        WHEN stability_score > PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY stability_score) OVER ()
        THEN 'VERY_STABLE'
        WHEN stability_score > PERCENTILE_CONT(0.7) WITHIN GROUP (ORDER BY stability_score) OVER ()
        THEN 'STABLE'
        WHEN stability_score < PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY stability_score) OVER ()
        THEN 'CRISIS'
        WHEN stability_score < PERCENTILE_CONT(0.3) WITHIN GROUP (ORDER BY stability_score) OVER ()
        THEN 'UNSTABLE'
        ELSE 'NORMAL'
    END AS market_regime

FROM market_stability;

-- ============================================================================
-- STEP 3: Top stable periods (for baseline)
-- ============================================================================

CREATE OR REPLACE VIEW v_ff_stable_periods AS
SELECT
    window_idx,
    stability_score,
    avg_lyapunov,
    avg_determinism,
    avg_cross_correlation,
    avg_eff_dim,
    stability_rank,
    market_regime
FROM v_market_stability_ranked
WHERE stability_rank <= 100  -- Top 100 most stable market periods
ORDER BY stability_rank;

-- ============================================================================
-- STEP 4: Crisis periods (for comparison)
-- ============================================================================

CREATE OR REPLACE VIEW v_ff_crisis_periods AS
SELECT
    window_idx,
    stability_score,
    avg_lyapunov,
    avg_determinism,
    avg_cross_correlation,
    avg_eff_dim,
    stability_rank,
    market_regime
FROM v_market_stability_ranked
WHERE market_regime IN ('CRISIS', 'UNSTABLE')
ORDER BY stability_score ASC;

-- ============================================================================
-- STEP 5: Regime transition detection
-- ============================================================================

CREATE OR REPLACE VIEW v_ff_regime_transitions AS
WITH lagged AS (
    SELECT
        window_idx,
        market_regime,
        stability_score,
        LAG(market_regime) OVER (ORDER BY window_idx) AS prev_regime,
        LAG(stability_score) OVER (ORDER BY window_idx) AS prev_score
    FROM v_market_stability_ranked
)
SELECT
    window_idx,
    prev_regime || ' -> ' || market_regime AS transition,
    stability_score,
    prev_score,
    stability_score - prev_score AS score_change
FROM lagged
WHERE market_regime != prev_regime
ORDER BY window_idx;

-- ============================================================================
-- STEP 6: Baseline from stable periods
-- ============================================================================

CREATE OR REPLACE VIEW v_ff_baseline AS
SELECT
    'MARKET_BASELINE' AS baseline_type,

    -- Average metrics across top 100 stable periods
    AVG(avg_lyapunov) AS baseline_lyapunov,
    AVG(avg_determinism) AS baseline_determinism,
    AVG(avg_cross_correlation) AS baseline_correlation,
    AVG(avg_coherence) AS baseline_coherence,
    AVG(avg_eff_dim) AS baseline_eff_dim,
    AVG(stability_score) AS baseline_stability,

    -- Standard deviation (for z-score calculation)
    STDDEV(avg_lyapunov) AS lyapunov_std,
    STDDEV(avg_determinism) AS determinism_std,
    STDDEV(avg_cross_correlation) AS correlation_std,
    STDDEV(stability_score) AS stability_std,

    COUNT(*) AS n_baseline_periods,
    MIN(window_idx) AS earliest_stable,
    MAX(window_idx) AS latest_stable

FROM v_ff_stable_periods;

-- ============================================================================
-- OUTPUT
-- ============================================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║           FAMA-FRENCH MARKET STABILITY ANALYSIS                             ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'
.print ''

.print '=== MARKET REGIME DISTRIBUTION ==='
SELECT
    market_regime,
    COUNT(*) AS n_periods,
    ROUND(AVG(stability_score), 3) AS avg_stability,
    ROUND(AVG(avg_cross_correlation), 3) AS avg_correlation,
    ROUND(AVG(avg_eff_dim), 2) AS avg_dimension
FROM v_market_stability_ranked
GROUP BY market_regime
ORDER BY avg_stability DESC;

.print ''
.print '=== TOP 20 MOST STABLE MARKET PERIODS ==='
SELECT
    window_idx,
    ROUND(stability_score, 3) AS stability,
    ROUND(avg_lyapunov, 4) AS lyapunov,
    ROUND(avg_determinism, 3) AS determ,
    ROUND(avg_cross_correlation, 3) AS x_corr,
    ROUND(avg_eff_dim, 2) AS eff_dim,
    market_regime
FROM v_ff_stable_periods
LIMIT 20;

.print ''
.print '=== CRISIS PERIODS (high correlation, chaos) ==='
SELECT
    window_idx,
    ROUND(stability_score, 3) AS stability,
    ROUND(avg_lyapunov, 4) AS lyapunov,
    ROUND(avg_cross_correlation, 3) AS x_corr,
    market_regime
FROM v_ff_crisis_periods
LIMIT 20;

.print ''
.print '=== REGIME TRANSITIONS ==='
SELECT
    window_idx,
    transition,
    ROUND(score_change, 3) AS score_change
FROM v_ff_regime_transitions
ORDER BY ABS(score_change) DESC
LIMIT 20;

.print ''
.print '=== MARKET BASELINE (from stable periods) ==='
SELECT
    baseline_type,
    ROUND(baseline_lyapunov, 4) AS lyapunov,
    ROUND(baseline_determinism, 3) AS determinism,
    ROUND(baseline_correlation, 3) AS correlation,
    ROUND(baseline_eff_dim, 2) AS eff_dim,
    ROUND(baseline_stability, 3) AS stability,
    n_baseline_periods
FROM v_ff_baseline;

.print ''
.print '=== INTERPRETATION ==='
.print ''
.print 'STABLE periods: Low chaos + high determinism + differentiated sectors'
.print 'CRISIS periods: High chaos + low determinism + high correlation (contagion)'
.print ''
.print 'Use stable periods as baseline for anomaly detection.'
.print 'Current state deviation from baseline indicates market stress.'
