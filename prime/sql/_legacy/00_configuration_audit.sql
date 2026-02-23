-- ============================================================
-- CONFIGURATION AUDIT
-- ============================================================
-- Detects heterogeneous sensor configurations that confound
-- fingerprint comparisons. MUST run before cross-entity analysis.
--
-- This report should be the FIRST thing run on any dataset.
-- It prevents false conclusions about "different behavior" when
-- the real difference is sensor count/configuration.
-- ============================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║                    SIGNAL CONFIGURATION AUDIT                                ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'

-- ============================================================
-- SECTION 1: FLEET CONFIGURATION SUMMARY
-- ============================================================

.print ''
.print '=== SECTION 1: FLEET CONFIGURATION SUMMARY ==='

WITH config_summary AS (
    SELECT
        cohort,
        MIN(n_signals) as min_signals,
        MAX(n_signals) as max_signals,
        MODE(n_signals) as typical_signals,
        COUNT(DISTINCT n_signals) as n_configs
    FROM read_parquet('{manifold_output}/physics.parquet')
    GROUP BY cohort
),
fleet_summary AS (
    SELECT
        COUNT(DISTINCT cohort) as n_entities,
        COUNT(DISTINCT typical_signals) as n_unique_configs,
        MIN(typical_signals) as min_fleet_signals,
        MAX(typical_signals) as max_fleet_signals
    FROM config_summary
)
SELECT
    n_entities,
    n_unique_configs,
    min_fleet_signals,
    max_fleet_signals,
    CASE
        WHEN n_unique_configs = 1 THEN '✓ HOMOGENEOUS - Safe to compare'
        ELSE '⚠️ HETEROGENEOUS - Normalization required'
    END as configuration_status
FROM fleet_summary;

-- Store result for downstream use
CREATE OR REPLACE TABLE config_audit_result AS
WITH config_summary AS (
    SELECT
        cohort,
        MODE(n_signals) as typical_signals
    FROM read_parquet('{manifold_output}/physics.parquet')
    GROUP BY cohort
)
SELECT
    COUNT(DISTINCT cohort) as n_entities,
    COUNT(DISTINCT typical_signals) as n_unique_configs,
    MIN(typical_signals) as min_signals,
    MAX(typical_signals) as max_signals,
    COUNT(DISTINCT typical_signals) > 1 as is_heterogeneous
FROM config_summary;


-- ============================================================
-- SECTION 2: PER-ENTITY CONFIGURATION
-- ============================================================

.print ''
.print '=== SECTION 2: PER-ENTITY CONFIGURATION ==='

SELECT
    cohort,
    MIN(n_signals) as min_signals,
    MAX(n_signals) as max_signals,
    ROUND(AVG(n_signals), 1) as avg_signals,
    COUNT(*) as observations,
    CASE
        WHEN MIN(n_signals) != MAX(n_signals) THEN '⚠️ VARIABLE within entity'
        ELSE '✓ STABLE'
    END as within_entity_status
FROM read_parquet('{manifold_output}/physics.parquet')
GROUP BY cohort
ORDER BY MIN(n_signals), cohort;


-- ============================================================
-- SECTION 3: CONFIGURATION GROUPS
-- ============================================================

.print ''
.print '=== SECTION 3: CONFIGURATION GROUPS ==='

WITH entity_config AS (
    SELECT
        cohort,
        MODE(n_signals) as n_signals
    FROM read_parquet('{manifold_output}/physics.parquet')
    GROUP BY cohort
)
SELECT
    n_signals as signal_count,
    COUNT(*) as n_entities,
    STRING_AGG(cohort, ', ' ORDER BY cohort) as entities,
    CASE
        WHEN COUNT(*) = 1 THEN '⚠️ SINGLETON - Cannot compare within group'
        ELSE '✓ Group size OK (' || COUNT(*) || ' entities)'
    END as group_status
FROM entity_config
GROUP BY n_signals
ORDER BY n_signals;


-- ============================================================
-- SECTION 4: RAW vs NORMALIZED METRICS
-- ============================================================

.print ''
.print '=== SECTION 4: RAW vs NORMALIZED METRICS ==='
.print '(Shows why normalization matters)'

SELECT
    cohort,
    n_signals,
    ROUND(AVG(coherence), 3) as coherence,
    ROUND(AVG(effective_dim), 2) as raw_eff_dim,
    ROUND(AVG(effective_dim) / n_signals, 3) as norm_eff_dim,
    ROUND(AVG(eigenvalue_entropy), 3) as raw_entropy,
    -- Max entropy for n signals = ln(n)/ln(2) in bits
    ROUND(LN(n_signals) / LN(2), 3) as max_entropy,
    ROUND(AVG(eigenvalue_entropy) / (LN(n_signals) / LN(2)), 3) as norm_entropy
FROM read_parquet('{manifold_output}/physics.parquet')
GROUP BY cohort, n_signals
ORDER BY n_signals, cohort;


-- ============================================================
-- SECTION 5: CROSS-GROUP COMPARISON VALIDITY
-- ============================================================

.print ''
.print '=== SECTION 5: CROSS-GROUP COMPARISON VALIDITY ==='

WITH entity_config AS (
    SELECT
        cohort,
        MODE(n_signals) as n_signals,
        AVG(coherence) as coherence,
        AVG(effective_dim) as eff_dim
    FROM read_parquet('{manifold_output}/physics.parquet')
    GROUP BY cohort
),
config_groups AS (
    SELECT
        n_signals,
        AVG(coherence) as group_coherence,
        AVG(eff_dim) as group_eff_dim,
        STDDEV(coherence) as group_std,
        COUNT(*) as n_entities
    FROM entity_config
    GROUP BY n_signals
)
SELECT
    a.n_signals as config_a,
    b.n_signals as config_b,
    a.n_entities as entities_a,
    b.n_entities as entities_b,
    ROUND(ABS(a.group_coherence - b.group_coherence), 3) as coherence_diff,
    ROUND(ABS(a.group_eff_dim - b.group_eff_dim), 2) as eff_dim_diff,
    CASE
        WHEN ABS(a.group_eff_dim - b.group_eff_dim) > 0.5
        THEN '⚠️ HIGH - Likely confounded by signal count'
        ELSE '✓ Acceptable'
    END as comparison_validity
FROM config_groups a
CROSS JOIN config_groups b
WHERE a.n_signals < b.n_signals;


-- ============================================================
-- SECTION 6: NORMALIZED FINGERPRINT RANKING
-- ============================================================

.print ''
.print '=== SECTION 6: NORMALIZED FINGERPRINT RANKING ==='
.print '(Fair comparison across different configurations)'

WITH normalized_stats AS (
    SELECT
        cohort,
        n_signals,
        AVG(coherence) as coherence,
        AVG(effective_dim / n_signals) as norm_dim,
        AVG(state_distance) as state_dist,
        SUM(CASE WHEN dissipation_rate > 0.01 AND coherence < 0.6
            THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as signal_rate
    FROM read_parquet('{manifold_output}/physics.parquet')
    GROUP BY cohort, n_signals
)
SELECT
    cohort,
    n_signals,
    ROUND(coherence, 3) as coherence,
    ROUND(norm_dim, 3) as norm_dim,
    ROUND(state_dist, 1) as state_dist,
    ROUND(signal_rate, 1) as signal_pct,
    CASE
        WHEN norm_dim < 0.7 THEN 'UNIFIED'
        WHEN norm_dim < 0.85 THEN 'MODERATE'
        ELSE 'FRAGMENTED'
    END as structure,
    CASE
        WHEN coherence > 0.6 AND norm_dim < 0.85 THEN 'TYPE_A (coupled/unified)'
        WHEN coherence < 0.5 AND norm_dim > 0.85 THEN 'TYPE_B (decoupled/fragmented)'
        ELSE 'MIXED'
    END as fingerprint_class
FROM normalized_stats
ORDER BY fingerprint_class, coherence DESC;


-- ============================================================
-- SECTION 7: WITHIN-GROUP OUTLIER DETECTION
-- ============================================================

.print ''
.print '=== SECTION 7: WITHIN-GROUP OUTLIER DETECTION ==='

WITH entity_stats AS (
    SELECT
        cohort,
        n_signals,
        AVG(coherence) as coherence,
        AVG(effective_dim / n_signals) as norm_dim
    FROM read_parquet('{manifold_output}/physics.parquet')
    GROUP BY cohort, n_signals
),
group_stats AS (
    SELECT
        n_signals,
        AVG(coherence) as group_mean_coh,
        STDDEV(coherence) as group_std_coh,
        AVG(norm_dim) as group_mean_dim,
        STDDEV(norm_dim) as group_std_dim,
        COUNT(*) as group_size
    FROM entity_stats
    GROUP BY n_signals
)
SELECT
    e.cohort,
    e.n_signals,
    g.group_size,
    ROUND(e.coherence, 3) as coherence,
    ROUND(PERCENT_RANK() OVER (PARTITION BY e.n_signals ORDER BY e.coherence), 2) as coherence_pctile,
    ROUND(e.norm_dim, 3) as norm_dim,
    ROUND(PERCENT_RANK() OVER (PARTITION BY e.n_signals ORDER BY e.norm_dim), 2) as norm_dim_pctile,
    CASE
        WHEN g.group_size = 1 THEN '⚠️ SINGLETON - No comparison possible'
        WHEN PERCENT_RANK() OVER (PARTITION BY e.n_signals ORDER BY e.coherence) > 0.975
          OR PERCENT_RANK() OVER (PARTITION BY e.n_signals ORDER BY e.coherence) < 0.025
          OR PERCENT_RANK() OVER (PARTITION BY e.n_signals ORDER BY e.norm_dim) > 0.975
          OR PERCENT_RANK() OVER (PARTITION BY e.n_signals ORDER BY e.norm_dim) < 0.025
        THEN '⚠️ OUTLIER within config group'
        ELSE '✓ Typical for config group'
    END as status
FROM entity_stats e
JOIN group_stats g ON e.n_signals = g.n_signals
ORDER BY e.n_signals, e.cohort;


-- ============================================================
-- SECTION 8: CONFIGURATION IMPACT SUMMARY
-- ============================================================

.print ''
.print '=== SECTION 8: CONFIGURATION IMPACT SUMMARY ==='

WITH metrics_by_config AS (
    SELECT
        n_signals,
        COUNT(DISTINCT cohort) as n_entities,
        ROUND(AVG(coherence), 3) as avg_coherence,
        ROUND(AVG(effective_dim), 2) as avg_raw_dim,
        ROUND(AVG(effective_dim / n_signals), 3) as avg_norm_dim,
        ROUND(AVG(eigenvalue_entropy), 3) as avg_entropy,
        ROUND(SUM(CASE WHEN dissipation_rate > 0.01 AND coherence < 0.5
            AND state_velocity > 0.05 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as signal_rate
    FROM read_parquet('{manifold_output}/physics.parquet')
    GROUP BY n_signals
)
SELECT
    n_signals || '-signal' as config,
    n_entities,
    avg_coherence,
    avg_raw_dim,
    avg_norm_dim,
    avg_entropy,
    signal_rate as pct_signal,
    CASE
        WHEN signal_rate > 5 THEN '⚠️ ACTIVE FAILURES'
        WHEN signal_rate > 0 THEN '⚡ WATCH'
        ELSE '✓ HEALTHY'
    END as fleet_status
FROM metrics_by_config
ORDER BY n_signals;


-- ============================================================
-- FINAL SUMMARY & RECOMMENDATIONS
-- ============================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║                    CONFIGURATION AUDIT COMPLETE                              ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'

-- Print recommendations based on audit results
.print ''
.print 'INTERPRETATION GUIDE:'
.print ''
.print '  HOMOGENEOUS fleet (all same n_signals):'
.print '    → Direct comparison of all metrics is valid'
.print '    → No normalization required'
.print ''
.print '  HETEROGENEOUS fleet (mixed n_signals):'
.print '    → effective_dim must be normalized: norm_dim = eff_dim / n_signals'
.print '    → entropy may need normalization: norm_entropy = entropy / log2(n)'
.print '    → coherence is ALREADY normalized (λ₁/Σλ) - no adjustment needed'
.print '    → Compare within config groups when possible'
.print '    → Cross-group comparisons require normalized metrics'
.print ''
.print '  SINGLETON config groups:'
.print '    → Cannot compute within-group percentile ranks'
.print '    → Entity may appear as outlier simply due to different config'
.print '    → Report this in narratives'
.print ''
.print 'KEY METRICS AFFECTED BY SIGNAL COUNT:'
.print '  ✗ effective_dim (max = n_signals)'
.print '  ✗ eigenvalue_entropy (max = log2(n_signals))'
.print '  ✗ n_pairs for geometry (n*(n-1)/2)'
.print ''
.print 'KEY METRICS NOT AFFECTED:'
.print '  ✓ coherence (already a ratio)'
.print '  ✓ state_distance (normalized by covariance)'
.print '  ✓ dissipation_rate (energy flow rate)'
.print ''


-- ============================================================
-- CREATE VIEWS FOR DOWNSTREAM USE
-- ============================================================

-- Normalized physics view for heterogeneous fleets
CREATE OR REPLACE VIEW v_physics_normalized AS
SELECT
    p.*,
    p.effective_dim / p.n_signals as norm_effective_dim,
    p.eigenvalue_entropy / (LN(p.n_signals) / LN(2)) as norm_entropy
FROM read_parquet('{manifold_output}/physics.parquet') p;

-- Configuration summary view
CREATE OR REPLACE VIEW v_entity_config AS
SELECT
    cohort,
    MODE(n_signals) as n_signals,
    COUNT(*) as n_observations
FROM read_parquet('{manifold_output}/physics.parquet')
GROUP BY cohort;

-- Config group membership
CREATE OR REPLACE VIEW v_config_groups AS
WITH entity_config AS (
    SELECT
        cohort,
        MODE(n_signals) as n_signals
    FROM read_parquet('{manifold_output}/physics.parquet')
    GROUP BY cohort
)
SELECT
    n_signals as config_group,
    COUNT(*) as n_entities,
    LIST(cohort ORDER BY cohort) as entities
FROM entity_config
GROUP BY n_signals;
