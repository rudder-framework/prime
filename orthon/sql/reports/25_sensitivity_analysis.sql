-- ============================================================
-- ORTHON SENSITIVITY ANALYSIS
-- ============================================================
-- Testing robustness of findings across:
-- 1. Threshold sensitivity
-- 2. Baseline window sensitivity
-- 3. Parameter sensitivity
-- 4. Temporal stability
-- 5. Cross-entity consistency
-- ============================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║                     ORTHON SENSITIVITY ANALYSIS                              ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'


-- ============================================================
-- REPORT 1: COHERENCE THRESHOLD SENSITIVITY
-- How do classifications change with different thresholds?
-- ============================================================

.print ''
.print '=============================================='
.print 'REPORT 1: COHERENCE THRESHOLD SENSITIVITY'
.print '=============================================='

WITH coherence_classifications AS (
    SELECT
        entity_id,
        -- Current thresholds (0.7 / 0.4)
        SUM(CASE WHEN coherence > 0.7 THEN 1 ELSE 0 END) as strong_70,
        SUM(CASE WHEN coherence > 0.4 AND coherence <= 0.7 THEN 1 ELSE 0 END) as weak_70_40,
        SUM(CASE WHEN coherence <= 0.4 THEN 1 ELSE 0 END) as decoupled_40,

        -- Tighter thresholds (0.8 / 0.5)
        SUM(CASE WHEN coherence > 0.8 THEN 1 ELSE 0 END) as strong_80,
        SUM(CASE WHEN coherence > 0.5 AND coherence <= 0.8 THEN 1 ELSE 0 END) as weak_80_50,
        SUM(CASE WHEN coherence <= 0.5 THEN 1 ELSE 0 END) as decoupled_50,

        -- Looser thresholds (0.6 / 0.3)
        SUM(CASE WHEN coherence > 0.6 THEN 1 ELSE 0 END) as strong_60,
        SUM(CASE WHEN coherence > 0.3 AND coherence <= 0.6 THEN 1 ELSE 0 END) as weak_60_30,
        SUM(CASE WHEN coherence <= 0.3 THEN 1 ELSE 0 END) as decoupled_30,

        COUNT(*) as total
    FROM read_parquet('{prism_output}/physics.parquet')
    GROUP BY entity_id
)
SELECT
    entity_id,
    '0.7/0.4' as thresholds,
    ROUND(strong_70 * 100.0 / total, 1) as pct_strong,
    ROUND(weak_70_40 * 100.0 / total, 1) as pct_weak,
    ROUND(decoupled_40 * 100.0 / total, 1) as pct_decoupled
FROM coherence_classifications
UNION ALL
SELECT
    entity_id,
    '0.8/0.5' as thresholds,
    ROUND(strong_80 * 100.0 / total, 1) as pct_strong,
    ROUND(weak_80_50 * 100.0 / total, 1) as pct_weak,
    ROUND(decoupled_50 * 100.0 / total, 1) as pct_decoupled
FROM coherence_classifications
UNION ALL
SELECT
    entity_id,
    '0.6/0.3' as thresholds,
    ROUND(strong_60 * 100.0 / total, 1) as pct_strong,
    ROUND(weak_60_30 * 100.0 / total, 1) as pct_weak,
    ROUND(decoupled_30 * 100.0 / total, 1) as pct_decoupled
FROM coherence_classifications
ORDER BY entity_id, thresholds;


-- ============================================================
-- REPORT 2: ORTHON SIGNAL THRESHOLD SENSITIVITY
-- How does the Ørthon signal count change with thresholds?
-- ============================================================

.print ''
.print '=============================================='
.print 'REPORT 2: ORTHON SIGNAL THRESHOLD SENSITIVITY'
.print '=============================================='

WITH orthon_sensitivity AS (
    SELECT
        entity_id,

        -- Strict thresholds
        SUM(CASE WHEN dissipation_rate > 0.05 AND coherence < 0.4 AND state_velocity > 0.1
            THEN 1 ELSE 0 END) as orthon_strict,

        -- Current thresholds
        SUM(CASE WHEN dissipation_rate > 0.01 AND coherence < 0.5 AND state_velocity > 0.05
            THEN 1 ELSE 0 END) as orthon_current,

        -- Loose thresholds
        SUM(CASE WHEN dissipation_rate > 0.005 AND coherence < 0.6 AND state_velocity > 0.02
            THEN 1 ELSE 0 END) as orthon_loose,

        -- Very loose
        SUM(CASE WHEN dissipation_rate > 0.001 AND coherence < 0.7 AND state_velocity > 0.01
            THEN 1 ELSE 0 END) as orthon_very_loose,

        COUNT(*) as total
    FROM read_parquet('{prism_output}/physics.parquet')
    GROUP BY entity_id
)
SELECT
    entity_id,
    orthon_strict as strict,
    orthon_current as current,
    orthon_loose as loose,
    orthon_very_loose as very_loose,
    total,
    ROUND(orthon_current * 100.0 / total, 1) as pct_current,
    CASE
        WHEN orthon_strict > 0 THEN 'ROBUST (detected at strict)'
        WHEN orthon_current > 0 THEN 'MODERATE (detected at current)'
        WHEN orthon_loose > 0 THEN 'WEAK (only at loose)'
        ELSE 'NONE'
    END as signal_robustness
FROM orthon_sensitivity
ORDER BY orthon_current DESC;


-- ============================================================
-- REPORT 3: BASELINE WINDOW SENSITIVITY
-- How do results change with different baseline periods?
-- ============================================================

.print ''
.print '=============================================='
.print 'REPORT 3: BASELINE WINDOW SENSITIVITY'
.print '=============================================='

WITH baseline_50 AS (
    SELECT entity_id,
           AVG(total_energy) as energy_bl,
           AVG(coherence) as coherence_bl,
           AVG(state_distance) as state_bl
    FROM read_parquet('{prism_output}/physics.parquet')
    WHERE I <= 50
    GROUP BY entity_id
),
baseline_100 AS (
    SELECT entity_id,
           AVG(total_energy) as energy_bl,
           AVG(coherence) as coherence_bl,
           AVG(state_distance) as state_bl
    FROM read_parquet('{prism_output}/physics.parquet')
    WHERE I <= 100
    GROUP BY entity_id
),
baseline_200 AS (
    SELECT entity_id,
           AVG(total_energy) as energy_bl,
           AVG(coherence) as coherence_bl,
           AVG(state_distance) as state_bl
    FROM read_parquet('{prism_output}/physics.parquet')
    WHERE I <= 200
    GROUP BY entity_id
),
current_state AS (
    SELECT DISTINCT ON (entity_id)
           entity_id,
           total_energy as energy_now,
           coherence as coherence_now,
           state_distance as state_now
    FROM read_parquet('{prism_output}/physics.parquet')
    ORDER BY entity_id, I DESC
)
SELECT
    c.entity_id,
    'BL=50' as baseline,
    ROUND((c.energy_now / NULLIF(b50.energy_bl, 0) - 1) * 100, 1) as energy_pct_change,
    ROUND((c.coherence_now / NULLIF(b50.coherence_bl, 0) - 1) * 100, 1) as coherence_pct_change,
    ROUND(c.state_now - b50.state_bl, 1) as state_change
FROM current_state c
JOIN baseline_50 b50 ON c.entity_id = b50.entity_id
UNION ALL
SELECT
    c.entity_id,
    'BL=100' as baseline,
    ROUND((c.energy_now / NULLIF(b100.energy_bl, 0) - 1) * 100, 1) as energy_pct_change,
    ROUND((c.coherence_now / NULLIF(b100.coherence_bl, 0) - 1) * 100, 1) as coherence_pct_change,
    ROUND(c.state_now - b100.state_bl, 1) as state_change
FROM current_state c
JOIN baseline_100 b100 ON c.entity_id = b100.entity_id
UNION ALL
SELECT
    c.entity_id,
    'BL=200' as baseline,
    ROUND((c.energy_now / NULLIF(b200.energy_bl, 0) - 1) * 100, 1) as energy_pct_change,
    ROUND((c.coherence_now / NULLIF(b200.coherence_bl, 0) - 1) * 100, 1) as coherence_pct_change,
    ROUND(c.state_now - b200.state_bl, 1) as state_change
FROM current_state c
JOIN baseline_200 b200 ON c.entity_id = b200.entity_id
ORDER BY entity_id, baseline;


-- ============================================================
-- REPORT 4: TEMPORAL STABILITY - ROLLING WINDOW ANALYSIS
-- Are findings consistent across different time periods?
-- ============================================================

.print ''
.print '=============================================='
.print 'REPORT 4: TEMPORAL STABILITY'
.print '=============================================='

WITH time_windows AS (
    SELECT
        entity_id,
        CASE
            WHEN I <= 250 THEN 'Q1 (0-250)'
            WHEN I <= 500 THEN 'Q2 (250-500)'
            WHEN I <= 750 THEN 'Q3 (500-750)'
            ELSE 'Q4 (750-1000)'
        END as quarter,
        AVG(coherence) as avg_coherence,
        AVG(state_distance) as avg_state,
        AVG(total_energy) as avg_energy,
        AVG(dissipation_rate) as avg_dissipation,
        STDDEV(coherence) as std_coherence,
        STDDEV(state_distance) as std_state
    FROM read_parquet('{prism_output}/physics.parquet')
    GROUP BY entity_id, CASE
            WHEN I <= 250 THEN 'Q1 (0-250)'
            WHEN I <= 500 THEN 'Q2 (250-500)'
            WHEN I <= 750 THEN 'Q3 (500-750)'
            ELSE 'Q4 (750-1000)'
        END
)
SELECT
    entity_id,
    quarter,
    ROUND(avg_coherence, 3) as coherence,
    ROUND(std_coherence, 3) as coherence_std,
    ROUND(avg_state, 1) as state_dist,
    ROUND(std_state, 1) as state_std,
    ROUND(avg_energy, 4) as energy,
    ROUND(avg_dissipation, 4) as dissipation
FROM time_windows
ORDER BY entity_id, quarter;

-- Trend consistency check
.print ''
.print '--- Trend Consistency Across Quarters ---'

WITH quarterly_trends AS (
    SELECT
        entity_id,
        CASE
            WHEN I <= 250 THEN 1
            WHEN I <= 500 THEN 2
            WHEN I <= 750 THEN 3
            ELSE 4
        END as q,
        AVG(coherence) as coherence,
        AVG(state_distance) as state
    FROM read_parquet('{prism_output}/physics.parquet')
    GROUP BY entity_id, CASE
            WHEN I <= 250 THEN 1
            WHEN I <= 500 THEN 2
            WHEN I <= 750 THEN 3
            ELSE 4
        END
),
trend_analysis AS (
    SELECT
        entity_id,
        REGR_SLOPE(coherence, q) as coherence_trend,
        REGR_SLOPE(state, q) as state_trend,
        REGR_R2(coherence, q) as coherence_r2,
        REGR_R2(state, q) as state_r2
    FROM quarterly_trends
    GROUP BY entity_id
)
SELECT
    entity_id,
    ROUND(coherence_trend, 4) as coherence_trend,
    ROUND(coherence_r2, 3) as coherence_r2,
    CASE WHEN coherence_r2 > 0.8 THEN 'CONSISTENT'
         WHEN coherence_r2 > 0.5 THEN 'MODERATE'
         ELSE 'VARIABLE' END as coherence_stability,
    ROUND(state_trend, 2) as state_trend,
    ROUND(state_r2, 3) as state_r2,
    CASE WHEN state_r2 > 0.8 THEN 'CONSISTENT'
         WHEN state_r2 > 0.5 THEN 'MODERATE'
         ELSE 'VARIABLE' END as state_stability
FROM trend_analysis
ORDER BY entity_id;


-- ============================================================
-- REPORT 5: CROSS-ENTITY CONSISTENCY
-- Do similar conditions produce similar classifications?
-- ============================================================

.print ''
.print '=============================================='
.print 'REPORT 5: CROSS-ENTITY CONSISTENCY'
.print '=============================================='

WITH entity_profiles AS (
    SELECT
        entity_id,
        AVG(coherence) as avg_coherence,
        AVG(state_distance) as avg_state,
        AVG(total_energy) as avg_energy,
        AVG(effective_dim) as avg_dim,
        STDDEV(coherence) as std_coherence,
        STDDEV(state_distance) as std_state
    FROM read_parquet('{prism_output}/physics.parquet')
    GROUP BY entity_id
),
pairwise_comparison AS (
    SELECT
        a.entity_id as entity_a,
        b.entity_id as entity_b,
        ABS(a.avg_coherence - b.avg_coherence) as coherence_diff,
        ABS(a.avg_state - b.avg_state) as state_diff,
        ABS(a.avg_energy - b.avg_energy) as energy_diff,
        ABS(a.avg_dim - b.avg_dim) as dim_diff
    FROM entity_profiles a
    CROSS JOIN entity_profiles b
    WHERE a.entity_id < b.entity_id
)
SELECT
    entity_a,
    entity_b,
    ROUND(coherence_diff, 3) as coherence_diff,
    ROUND(state_diff, 1) as state_diff,
    ROUND(energy_diff, 4) as energy_diff,
    ROUND(dim_diff, 2) as dim_diff,
    CASE
        WHEN coherence_diff < 0.1 AND state_diff < 10 THEN 'SIMILAR'
        WHEN coherence_diff < 0.2 AND state_diff < 30 THEN 'MODERATE'
        ELSE 'DIFFERENT'
    END as similarity
FROM pairwise_comparison
ORDER BY coherence_diff + state_diff / 100;


-- ============================================================
-- REPORT 6: METRIC CORRELATION SENSITIVITY
-- Are the layer correlations stable?
-- ============================================================

.print ''
.print '=============================================='
.print 'REPORT 6: METRIC CORRELATION STABILITY'
.print '=============================================='

-- First half vs second half correlations
WITH first_half AS (
    SELECT
        entity_id,
        CORR(total_energy, state_distance) as energy_state,
        CORR(coherence, state_distance) as coherence_state,
        CORR(total_energy, coherence) as energy_coherence,
        CORR(dissipation_rate, state_velocity) as dissipation_velocity
    FROM read_parquet('{prism_output}/physics.parquet')
    WHERE I <= 500
    GROUP BY entity_id
),
second_half AS (
    SELECT
        entity_id,
        CORR(total_energy, state_distance) as energy_state,
        CORR(coherence, state_distance) as coherence_state,
        CORR(total_energy, coherence) as energy_coherence,
        CORR(dissipation_rate, state_velocity) as dissipation_velocity
    FROM read_parquet('{prism_output}/physics.parquet')
    WHERE I > 500
    GROUP BY entity_id
)
SELECT
    f.entity_id,
    'energy↔state' as correlation,
    ROUND(f.energy_state, 3) as first_half,
    ROUND(s.energy_state, 3) as second_half,
    ROUND(ABS(f.energy_state - s.energy_state), 3) as change,
    CASE WHEN ABS(f.energy_state - s.energy_state) < 0.2 THEN 'STABLE' ELSE 'SHIFTED' END as stability
FROM first_half f
JOIN second_half s ON f.entity_id = s.entity_id
UNION ALL
SELECT
    f.entity_id,
    'coherence↔state' as correlation,
    ROUND(f.coherence_state, 3) as first_half,
    ROUND(s.coherence_state, 3) as second_half,
    ROUND(ABS(f.coherence_state - s.coherence_state), 3) as change,
    CASE WHEN ABS(f.coherence_state - s.coherence_state) < 0.2 THEN 'STABLE' ELSE 'SHIFTED' END as stability
FROM first_half f
JOIN second_half s ON f.entity_id = s.entity_id
UNION ALL
SELECT
    f.entity_id,
    'energy↔coherence' as correlation,
    ROUND(f.energy_coherence, 3) as first_half,
    ROUND(s.energy_coherence, 3) as second_half,
    ROUND(ABS(f.energy_coherence - s.energy_coherence), 3) as change,
    CASE WHEN ABS(f.energy_coherence - s.energy_coherence) < 0.2 THEN 'STABLE' ELSE 'SHIFTED' END as stability
FROM first_half f
JOIN second_half s ON f.entity_id = s.entity_id
ORDER BY entity_id, correlation;


-- ============================================================
-- REPORT 7: EFFECTIVE DIMENSIONALITY SENSITIVITY
-- How stable is the mode fragmentation finding?
-- ============================================================

.print ''
.print '=============================================='
.print 'REPORT 7: EFFECTIVE DIM SENSITIVITY'
.print '=============================================='

WITH dim_analysis AS (
    SELECT
        entity_id,
        n_signals,
        AVG(effective_dim) as avg_dim,
        STDDEV(effective_dim) as std_dim,
        MIN(effective_dim) as min_dim,
        MAX(effective_dim) as max_dim,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY effective_dim) as p25_dim,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY effective_dim) as p75_dim
    FROM read_parquet('{prism_output}/physics.parquet')
    GROUP BY entity_id, n_signals
)
SELECT
    entity_id,
    n_signals,
    ROUND(avg_dim, 2) as avg_dim,
    ROUND(std_dim, 2) as std_dim,
    ROUND(min_dim, 2) as min_dim,
    ROUND(max_dim, 2) as max_dim,
    ROUND(p25_dim, 2) as p25,
    ROUND(p75_dim, 2) as p75,
    ROUND((p75_dim - p25_dim), 2) as iqr,
    CASE
        WHEN std_dim / NULLIF(avg_dim, 0) < 0.1 THEN 'VERY STABLE'
        WHEN std_dim / NULLIF(avg_dim, 0) < 0.2 THEN 'STABLE'
        WHEN std_dim / NULLIF(avg_dim, 0) < 0.3 THEN 'MODERATE'
        ELSE 'VARIABLE'
    END as stability,
    CASE
        WHEN avg_dim < 1.5 THEN 'UNIFIED'
        WHEN avg_dim < n_signals * 0.5 THEN 'CLUSTERED'
        ELSE 'FRAGMENTED'
    END as structure_class
FROM dim_analysis
ORDER BY entity_id;


-- ============================================================
-- REPORT 8: ENDOGENOUS VS EXOGENOUS ROBUSTNESS
-- Is the force attribution finding robust?
-- ============================================================

.print ''
.print '=============================================='
.print 'REPORT 8: FORCE ATTRIBUTION ROBUSTNESS'
.print '=============================================='

WITH force_analysis AS (
    SELECT
        entity_id,
        I,
        state_velocity,
        state_acceleration,
        coherence_velocity,
        energy_velocity,

        -- Endogenous: internal dynamics (acceleration opposes velocity = restoring)
        CASE WHEN state_velocity * state_acceleration < 0 THEN 1 ELSE 0 END as endogenous_indicator,

        -- Exogenous: external forcing (acceleration same direction as velocity = driving)
        CASE WHEN state_velocity * state_acceleration > 0 THEN 1 ELSE 0 END as exogenous_indicator

    FROM read_parquet('{prism_output}/physics.parquet')
),
force_summary AS (
    SELECT
        entity_id,
        SUM(endogenous_indicator) as endogenous_count,
        SUM(exogenous_indicator) as exogenous_count,
        COUNT(*) as total
    FROM force_analysis
    GROUP BY entity_id
)
SELECT
    entity_id,
    endogenous_count,
    exogenous_count,
    total - endogenous_count - exogenous_count as neutral_count,
    ROUND(endogenous_count * 100.0 / total, 1) as pct_endogenous,
    ROUND(exogenous_count * 100.0 / total, 1) as pct_exogenous,
    CASE
        WHEN endogenous_count > exogenous_count * 1.5 THEN 'STRONGLY ENDOGENOUS'
        WHEN endogenous_count > exogenous_count THEN 'ENDOGENOUS'
        WHEN exogenous_count > endogenous_count * 1.5 THEN 'STRONGLY EXOGENOUS'
        WHEN exogenous_count > endogenous_count THEN 'EXOGENOUS'
        ELSE 'MIXED'
    END as force_classification,
    ROUND(ABS(endogenous_count - exogenous_count) * 1.0 / total, 3) as confidence
FROM force_summary
ORDER BY confidence DESC;


-- ============================================================
-- REPORT 9: NOISE FLOOR ANALYSIS
-- Are we above the noise floor?
-- ============================================================

.print ''
.print '=============================================='
.print 'REPORT 9: SIGNAL VS NOISE'
.print '=============================================='

WITH signal_noise AS (
    SELECT
        entity_id,

        -- Coherence signal to noise
        AVG(coherence) as coherence_mean,
        STDDEV(coherence) as coherence_std,
        AVG(coherence) / NULLIF(STDDEV(coherence), 0) as coherence_snr,

        -- State signal to noise
        AVG(state_distance) as state_mean,
        STDDEV(state_distance) as state_std,
        AVG(state_distance) / NULLIF(STDDEV(state_distance), 0) as state_snr,

        -- Energy signal to noise
        AVG(total_energy) as energy_mean,
        STDDEV(total_energy) as energy_std,
        AVG(total_energy) / NULLIF(STDDEV(total_energy), 0) as energy_snr

    FROM read_parquet('{prism_output}/physics.parquet')
    GROUP BY entity_id
)
SELECT
    entity_id,
    ROUND(coherence_snr, 2) as coherence_snr,
    CASE WHEN coherence_snr > 5 THEN 'STRONG'
         WHEN coherence_snr > 2 THEN 'MODERATE'
         ELSE 'WEAK' END as coherence_signal,
    ROUND(state_snr, 2) as state_snr,
    CASE WHEN state_snr > 5 THEN 'STRONG'
         WHEN state_snr > 2 THEN 'MODERATE'
         ELSE 'WEAK' END as state_signal,
    ROUND(energy_snr, 2) as energy_snr,
    CASE WHEN energy_snr > 5 THEN 'STRONG'
         WHEN energy_snr > 2 THEN 'MODERATE'
         ELSE 'WEAK' END as energy_signal
FROM signal_noise
ORDER BY entity_id;


-- ============================================================
-- REPORT 10: BOOTSTRAP CONFIDENCE - SUBSAMPLE CONSISTENCY
-- Do findings hold across random subsamples?
-- ============================================================

.print ''
.print '=============================================='
.print 'REPORT 10: SUBSAMPLE CONSISTENCY'
.print '=============================================='

-- Odd vs Even index comparison (poor man's bootstrap)
WITH odd_sample AS (
    SELECT
        entity_id,
        AVG(coherence) as coherence,
        AVG(state_distance) as state,
        AVG(total_energy) as energy
    FROM read_parquet('{prism_output}/physics.parquet')
    WHERE I % 2 = 1
    GROUP BY entity_id
),
even_sample AS (
    SELECT
        entity_id,
        AVG(coherence) as coherence,
        AVG(state_distance) as state,
        AVG(total_energy) as energy
    FROM read_parquet('{prism_output}/physics.parquet')
    WHERE I % 2 = 0
    GROUP BY entity_id
)
SELECT
    o.entity_id as entity_id,
    ROUND(o.coherence, 4) as odd_coherence,
    ROUND(e.coherence, 4) as even_coherence,
    ROUND(ABS(o.coherence - e.coherence) / NULLIF((o.coherence + e.coherence) / 2, 0) * 100, 2) as coherence_pct_diff,
    ROUND(o.state, 2) as odd_state,
    ROUND(e.state, 2) as even_state,
    ROUND(ABS(o.state - e.state) / NULLIF((o.state + e.state) / 2, 0) * 100, 2) as state_pct_diff,
    CASE
        WHEN ABS(o.coherence - e.coherence) < 0.01 AND ABS(o.state - e.state) < 1
        THEN 'HIGHLY CONSISTENT'
        WHEN ABS(o.coherence - e.coherence) < 0.05 AND ABS(o.state - e.state) < 5
        THEN 'CONSISTENT'
        ELSE 'VARIABLE'
    END as subsample_stability
FROM odd_sample o
JOIN even_sample e ON o.entity_id = e.entity_id
ORDER BY o.entity_id;


-- ============================================================
-- REPORT 11: CLASSIFICATION STABILITY MATRIX
-- Final robustness summary
-- ============================================================

.print ''
.print '=============================================='
.print 'REPORT 11: ROBUSTNESS SUMMARY MATRIX'
.print '=============================================='

WITH robustness_factors AS (
    SELECT
        entity_id,

        -- Threshold robustness: Ørthon signal detected at current thresholds with >5% frequency
        CASE WHEN SUM(CASE WHEN dissipation_rate > 0.01 AND coherence < 0.5 AND state_velocity > 0.05
                           THEN 1 ELSE 0 END) * 100.0 / COUNT(*) > 5 THEN 1 ELSE 0 END as threshold_robust,

        -- Coherence signal strength (coherence SNR > 5 is strong)
        CASE WHEN AVG(coherence) / NULLIF(STDDEV(coherence), 0) > 5 THEN 1 ELSE 0 END as signal_robust,

        -- Temporal consistency: coefficient of variation < 0.3
        CASE WHEN STDDEV(coherence) / NULLIF(AVG(coherence), 0) < 0.3 THEN 1 ELSE 0 END as temporal_robust,

        -- Structure clarity: effective_dim variance is low (CV < 0.2)
        CASE WHEN STDDEV(effective_dim) / NULLIF(AVG(effective_dim), 0) < 0.2 THEN 1 ELSE 0 END as structure_robust,

        -- State trend clarity: state shows clear movement (>10σ max distance)
        CASE WHEN MAX(state_distance) > 10 THEN 1 ELSE 0 END as state_robust

    FROM read_parquet('{prism_output}/physics.parquet')
    GROUP BY entity_id
)
SELECT
    entity_id,
    threshold_robust as threshold,
    signal_robust as signal,
    temporal_robust as temporal,
    structure_robust as structure,
    state_robust as state,
    threshold_robust + signal_robust + temporal_robust + structure_robust + state_robust as robustness_score,
    CASE
        WHEN threshold_robust + signal_robust + temporal_robust + structure_robust + state_robust >= 4 THEN '★★★★ VERY HIGH'
        WHEN threshold_robust + signal_robust + temporal_robust + structure_robust + state_robust >= 3 THEN '★★★ HIGH'
        WHEN threshold_robust + signal_robust + temporal_robust + structure_robust + state_robust >= 2 THEN '★★ MODERATE'
        WHEN threshold_robust + signal_robust + temporal_robust + structure_robust + state_robust >= 1 THEN '★ LOW'
        ELSE '⚠️ UNRELIABLE'
    END as confidence_level
FROM robustness_factors
ORDER BY robustness_score DESC;


-- ============================================================
-- FINAL SUMMARY
-- ============================================================

.print ''
.print '╔══════════════════════════════════════════════════════════════════════════════╗'
.print '║                     SENSITIVITY ANALYSIS COMPLETE                            ║'
.print '╚══════════════════════════════════════════════════════════════════════════════╝'
.print ''
.print 'Key Questions Answered:'
.print '  1. Are coherence classifications threshold-dependent? → See Report 1'
.print '  2. Is the Ørthon signal robust to threshold changes? → See Report 2'
.print '  3. Does baseline window choice matter? → See Report 3'
.print '  4. Are trends consistent over time? → See Report 4'
.print '  5. Are entities truly different from each other? → See Report 5'
.print '  6. Are layer correlations stable? → See Report 6'
.print '  7. Is mode fragmentation a stable finding? → See Report 7'
.print '  8. Is endogenous/exogenous attribution reliable? → See Report 8'
.print '  9. Are signals above noise floor? → See Report 9'
.print ' 10. Do subsamples give consistent results? → See Report 10'
.print ' 11. Overall robustness score → See Report 11'
.print ''
