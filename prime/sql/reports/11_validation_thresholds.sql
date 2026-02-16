-- ============================================================================
-- Engines VALIDATION THRESHOLDS
-- ============================================================================
--
-- Validates data sufficiency, baseline quality, and finding significance.
-- Use these reports to prevent false positives and overstated findings.
--
-- Based on: INTERPRETATION_THRESHOLDS.md v2.0
-- Purpose: "A finding is not significant just because the math runs."
--
-- Updated: Replaced z-score thresholds with trajectory-based and
-- fleet-percentile-based thresholds.
--
-- Usage: Run against observations table and engine outputs
-- ============================================================================


-- ============================================================================
-- REPORT 1: DATA SUFFICIENCY CHECK (Updated thresholds)
-- ============================================================================
-- Validates whether each entity has enough data for reliable engine results.
-- Hard minimums: Below these, results are invalid.
-- Soft minimums: Below these, results are unreliable.

WITH entity_stats AS (
    SELECT
        cohort,
        COUNT(*) AS total_observations,
        COUNT(DISTINCT signal_id) AS n_signals,
        MAX(I) - MIN(I) AS time_span,
        COUNT(DISTINCT FLOOR(I / 100)) AS approx_windows
    FROM observations
    GROUP BY cohort
)
SELECT
    cohort,
    total_observations,
    n_signals,
    approx_windows,

    -- ========== HARD MINIMUMS (Updated) ==========

    -- Lyapunov: 3,000 hard minimum (Rosenstein/Kantz work with shorter series)
    CASE WHEN total_observations >= 3000 THEN 'WITHIN_BASELINE' ELSE 'INSUFFICIENT' END AS lyapunov_status,
    total_observations || '/3000' AS lyapunov_detail,

    -- Correlation Dimension: Depends on embedding dimension
    -- Low-dim (d<=5): 1,000; Med (d=5-10): 2,000-5,000; High (d>10): 5,000+
    CASE WHEN total_observations >= 1000 THEN 'WITHIN_BASELINE' ELSE 'INSUFFICIENT' END AS corr_dim_low_status,
    CASE WHEN total_observations >= 5000 THEN 'WITHIN_BASELINE' ELSE 'INSUFFICIENT' END AS corr_dim_high_status,
    total_observations || '/1000 (low-d) or /5000 (high-d)' AS corr_dim_detail,

    -- Transfer Entropy: Needs 1,000+ observations AND 3+ signals
    CASE WHEN total_observations >= 1000 AND n_signals >= 3 THEN 'WITHIN_BASELINE' ELSE 'INSUFFICIENT' END AS te_status,
    'obs:' || total_observations || '/1000, signals:' || n_signals || '/3' AS te_detail,

    -- Granger Causality: Needs 500+ observations AND 2+ signals
    CASE WHEN total_observations >= 500 AND n_signals >= 2 THEN 'WITHIN_BASELINE' ELSE 'INSUFFICIENT' END AS granger_status,
    'obs:' || total_observations || '/500, signals:' || n_signals || '/2' AS granger_detail,

    -- Topology (Betti): Needs 500+ observations
    CASE WHEN total_observations >= 500 THEN 'WITHIN_BASELINE' ELSE 'INSUFFICIENT' END AS topology_status,
    total_observations || '/500' AS topology_detail,

    -- RQA: Needs 1,000+ observations
    CASE WHEN total_observations >= 1000 THEN 'WITHIN_BASELINE' ELSE 'INSUFFICIENT' END AS rqa_status,
    total_observations || '/1000' AS rqa_detail,

    -- Coherence: Needs 3+ signals
    CASE WHEN n_signals >= 3 THEN 'WITHIN_BASELINE' ELSE 'INSUFFICIENT' END AS coherence_status,
    n_signals || '/3 signals' AS coherence_detail,

    -- PID (synergy/redundancy): Needs 500+ observations AND 3+ signals
    CASE WHEN total_observations >= 500 AND n_signals >= 3 THEN 'WITHIN_BASELINE' ELSE 'INSUFFICIENT' END AS pid_status,

    -- ========== SOFT MINIMUMS (Recommended for reliability) ==========

    -- Lyapunov reliability tier (updated: 10k soft minimum)
    CASE
        WHEN total_observations >= 10000 THEN 'RELIABLE'
        WHEN total_observations >= 3000 THEN 'MARGINAL'
        ELSE 'UNRELIABLE'
    END AS lyapunov_confidence,

    -- Transfer Entropy reliability tier
    CASE
        WHEN total_observations >= 5000 AND n_signals >= 5 THEN 'RELIABLE'
        WHEN total_observations >= 1000 AND n_signals >= 3 THEN 'MARGINAL'
        ELSE 'UNRELIABLE'
    END AS te_confidence,

    -- ========== OVERALL ASSESSMENT ==========

    CASE
        WHEN total_observations >= 5000 AND n_signals >= 3 THEN 'FULL_ANALYSIS'
        WHEN total_observations >= 1000 AND n_signals >= 2 THEN 'PARTIAL_ANALYSIS'
        WHEN total_observations >= 200 THEN 'LIMITED_ANALYSIS'
        ELSE 'INSUFFICIENT_DATA'
    END AS analysis_capability,

    -- Engines that CAN run reliably
    CASE
        WHEN total_observations >= 10000 AND n_signals >= 3
        THEN 'All engines: Lyapunov (reliable), TE, RQA, Topology, Coherence, Granger'
        WHEN total_observations >= 5000 AND n_signals >= 3
        THEN 'Most engines: TE, RQA, Topology, Coherence, Granger, Lyapunov (marginal)'
        WHEN total_observations >= 3000 AND n_signals >= 3
        THEN 'Core engines: TE, RQA, Topology, Coherence, Lyapunov (marginal)'
        WHEN total_observations >= 1000 AND n_signals >= 3
        THEN 'Basic engines: TE, RQA, Topology, Coherence'
        WHEN total_observations >= 500 AND n_signals >= 2
        THEN 'Limited: RQA, Topology, Granger'
        ELSE 'Very limited: Basic statistics only'
    END AS recommended_engines

FROM entity_stats
ORDER BY total_observations DESC;


-- ============================================================================
-- REPORT 2: BASELINE VALIDITY CHECK
-- ============================================================================
-- Validates the first 20% of data as a baseline period.
-- Checks for sufficient samples, stability, and outliers.

WITH time_bounds AS (
    SELECT
        cohort,
        MIN(I) AS min_I,
        MAX(I) AS max_I,
        MIN(I) + 0.20 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations
    GROUP BY cohort
),

baseline_stats AS (
    SELECT
        o.cohort,
        o.signal_id,
        COUNT(*) AS baseline_obs,
        AVG(o.value) AS baseline_mean,
        STDDEV_POP(o.value) AS baseline_std,
        MIN(o.value) AS baseline_min,
        MAX(o.value) AS baseline_max,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY o.value) AS p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY o.value) AS p50,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY o.value) AS p75
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort
    WHERE o.I <= t.baseline_end
    GROUP BY o.cohort, o.signal_id
)

SELECT
    cohort,
    signal_id,
    baseline_obs,
    ROUND(baseline_mean, 4) AS baseline_mean,
    ROUND(baseline_std, 4) AS baseline_std,
    ROUND(p50, 4) AS baseline_median,
    ROUND(p75 - p25, 4) AS baseline_iqr,

    -- Validity checks
    CASE
        WHEN baseline_obs >= 100 THEN 'SUFFICIENT'
        WHEN baseline_obs >= 50 THEN 'MARGINAL'
        ELSE 'TOO_FEW'
    END AS sample_size_check,

    CASE
        WHEN baseline_std = 0 THEN 'CONSTANT'
        WHEN baseline_std / NULLIF(ABS(baseline_mean), 0.0001) < 0.2 THEN 'VERY_STABLE'
        WHEN baseline_std / NULLIF(ABS(baseline_mean), 0.0001) < 0.5 THEN 'STABLE'
        ELSE 'VOLATILE'
    END AS stability_check,

    CASE
        WHEN baseline_std = 0 THEN 'N/A'
        WHEN (baseline_max - baseline_min) / NULLIF(baseline_std, 0.0001) < 4 THEN 'NO_OUTLIERS'
        WHEN (baseline_max - baseline_min) / NULLIF(baseline_std, 0.0001) < 6 THEN 'MINOR_OUTLIERS'
        ELSE 'SIGNIFICANT_OUTLIERS'
    END AS outlier_check,

    -- Coefficient of variation
    ROUND(100 * baseline_std / NULLIF(ABS(baseline_mean), 0.0001), 1) AS cv_percent,

    -- Overall baseline validity
    CASE
        WHEN baseline_obs >= 50
             AND (baseline_std = 0 OR baseline_std / NULLIF(ABS(baseline_mean), 0.0001) < 0.5)
             AND (baseline_std = 0 OR (baseline_max - baseline_min) / NULLIF(baseline_std, 0.0001) < 6)
        THEN 'VALID'
        WHEN baseline_obs >= 30
        THEN 'MARGINAL'
        ELSE 'QUESTIONABLE'
    END AS baseline_validity

FROM baseline_stats
ORDER BY cohort, signal_id;


-- ============================================================================
-- REPORT 3: BASELINE SUMMARY BY ENTITY
-- ============================================================================
-- Aggregated baseline validity per entity

WITH time_bounds AS (
    SELECT
        cohort,
        MIN(I) + 0.20 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations
    GROUP BY cohort
),

baseline_stats AS (
    SELECT
        o.cohort,
        o.signal_id,
        COUNT(*) AS baseline_obs,
        AVG(o.value) AS baseline_mean,
        STDDEV_POP(o.value) AS baseline_std
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort
    WHERE o.I <= t.baseline_end
    GROUP BY o.cohort, o.signal_id
),

signal_validity AS (
    SELECT
        cohort,
        signal_id,
        CASE
            WHEN baseline_obs >= 50
                 AND (baseline_std = 0 OR baseline_std / NULLIF(ABS(baseline_mean), 0.0001) < 0.5)
            THEN 1 ELSE 0
        END AS is_valid
    FROM baseline_stats
)

SELECT
    cohort,
    COUNT(*) AS total_signals,
    SUM(is_valid) AS valid_baselines,
    COUNT(*) - SUM(is_valid) AS questionable_baselines,
    ROUND(100.0 * SUM(is_valid) / COUNT(*), 1) AS pct_valid,
    CASE
        WHEN 100.0 * SUM(is_valid) / COUNT(*) >= 90 THEN 'GOOD'
        WHEN 100.0 * SUM(is_valid) / COUNT(*) >= 70 THEN 'ACCEPTABLE'
        ELSE 'POOR'
    END AS baseline_quality
FROM signal_validity
GROUP BY cohort
ORDER BY pct_valid DESC;


-- ============================================================================
-- REPORT 4: FLEET SIZE VALIDITY (Updated — trajectory-based guidance)
-- ============================================================================
-- Checks if fleet is large enough for fleet-level analytics

SELECT
    COUNT(DISTINCT cohort) AS fleet_size,

    -- Fleet guidance (trajectory-based, not z-score dependent)
    CASE
        WHEN COUNT(DISTINCT cohort) >= 30 THEN 'RELIABLE - Full fleet analytics, percentile-based thresholds'
        WHEN COUNT(DISTINCT cohort) >= 10 THEN 'MARGINAL - Fleet percentiles OK, use trajectory metrics per-entity'
        WHEN COUNT(DISTINCT cohort) >= 5 THEN 'LIMITED - Robust stats only (median, IQR), per-entity trajectory analysis'
        ELSE 'MINIMAL - Individual baselines only, no fleet statistics'
    END AS fleet_validity,

    -- Clustering validity
    CASE
        WHEN COUNT(DISTINCT cohort) >= 50 THEN 'Full clustering recommended'
        WHEN COUNT(DISTINCT cohort) >= 20 THEN 'Basic clustering OK'
        WHEN COUNT(DISTINCT cohort) >= 10 THEN 'Limited clustering'
        WHEN COUNT(DISTINCT cohort) >= 5 THEN 'Simple grouping only'
        ELSE 'Clustering not recommended'
    END AS clustering_validity,

    -- Fleet analytics recommendation
    CASE
        WHEN COUNT(DISTINCT cohort) >= 50 THEN 'Full fleet analytics'
        WHEN COUNT(DISTINCT cohort) >= 30 THEN 'Standard fleet analytics'
        WHEN COUNT(DISTINCT cohort) >= 10 THEN 'Basic fleet analytics'
        WHEN COUNT(DISTINCT cohort) >= 5 THEN 'Minimal fleet comparison'
        ELSE 'Individual entity analysis only'
    END AS recommendation,

    -- What's valid at this fleet size
    CASE
        WHEN COUNT(DISTINCT cohort) >= 30
        THEN 'Fleet percentiles, slope_ratio thresholds, clustering, fleet departure, deviation detection'
        WHEN COUNT(DISTINCT cohort) >= 10
        THEN 'Percentile ranks, per-entity trajectory analysis, basic comparisons'
        WHEN COUNT(DISTINCT cohort) >= 5
        THEN 'Median, IQR, percentile ranks, per-entity slope analysis only'
        ELSE 'Compare to own baseline slope only, no fleet statistics'
    END AS valid_analyses

FROM observations;


-- ============================================================================
-- REPORT 5: ENGINE-SPECIFIC THRESHOLD REFERENCE (Updated)
-- ============================================================================
-- Static reference table for actionable thresholds by engine/metric
-- Z-score deviation thresholds replaced with trajectory-based thresholds

SELECT * FROM (VALUES
    -- Lyapunov thresholds
    ('Lyapunov', 'lambda_max', '< -0.1', 'Strongly stable', 'BASELINE'),
    ('Lyapunov', 'lambda_max', '-0.1 to 0', 'Weakly stable', 'OBSERVE'),
    ('Lyapunov', 'lambda_max', '0 (+/- 0.01)', 'Marginal', 'Inconclusive'),
    ('Lyapunov', 'lambda_max', '0 to 0.05', 'Weakly chaotic', 'EXAMINE'),
    ('Lyapunov', 'lambda_max', '> 0.05', 'Strongly chaotic', 'DEPARTED'),

    -- RQA thresholds (updated with DIV)
    ('RQA', 'DET', '> 0.8', 'STABLE', 'BASELINE'),
    ('RQA', 'DET', '0.6 to 0.8', 'SHIFTED', 'OBSERVE'),
    ('RQA', 'DET', '0.5 to 0.6', 'SHIFTED', 'EXAMINE'),
    ('RQA', 'DET', '< 0.5', 'DEPARTED', 'DEPARTED'),
    ('RQA', 'LAM', '0.7 to 0.95', 'STABLE', 'BASELINE'),
    ('RQA', 'LAM', '> 0.98', 'RIGIDIFIED', 'DEPARTED'),
    ('RQA', 'DIV', '< 0.05', 'STABLE', 'BASELINE'),
    ('RQA', 'DIV', '0.05 to 0.1', 'SHIFTED', 'OBSERVE'),
    ('RQA', 'DIV', '0.1 to 0.3', 'High divergence', 'EXAMINE'),
    ('RQA', 'DIV', '> 0.3', 'DEPARTED', 'DEPARTED'),

    -- Coherence thresholds (absolute)
    ('Coherence', 'ratio', '0.3 to 0.7', 'STABLE', 'BASELINE'),
    ('Coherence', 'ratio', '> 0.85', 'Over-coupling', 'EXAMINE'),
    ('Coherence', 'ratio', '> 0.95', 'Rigidification', 'DEPARTED'),
    ('Coherence', 'ratio', '< 0.15', 'Decoupling', 'EXAMINE'),
    ('Coherence', 'ratio', '< 0.05', 'Fragmentation', 'DEPARTED'),

    -- Coherence VELOCITY thresholds (rate of change)
    ('Coherence', 'delta/window', '< 0.02', 'Stable', 'Normal'),
    ('Coherence', 'delta/window', '0.02 to 0.05', 'Drifting', 'OBSERVE'),
    ('Coherence', 'delta/window', '0.05 to 0.10', 'Rapid change', 'EXAMINE'),
    ('Coherence', 'delta/window', '> 0.10', 'ALARM - Rapid transition', 'DEPARTED'),

    -- Transfer Entropy thresholds
    ('Transfer Entropy', 'TE (8 bins)', '< 0.01', 'Noise floor', 'Ignore'),
    ('Transfer Entropy', 'TE (16 bins)', '< 0.02', 'Noise floor', 'Ignore'),
    ('Transfer Entropy', 'TE', '0.01 to 0.05', 'Weak', 'Note'),
    ('Transfer Entropy', 'TE', '0.05 to 0.15', 'Moderate', 'EXAMINE'),
    ('Transfer Entropy', 'TE', '> 0.15', 'Strong', 'DEPARTED'),

    -- Topology thresholds
    ('Topology', 'beta_0', '= 1', 'STABLE', 'BASELINE'),
    ('Topology', 'beta_0', '= 2', 'SHIFTED', 'EXAMINE'),
    ('Topology', 'beta_0', '> 2', 'Fragmentation', 'DEPARTED'),
    ('Topology', 'wasserstein', '< 0.2', 'Stable', 'BASELINE'),
    ('Topology', 'wasserstein', '0.2 to 0.5', 'Shifting', 'EXAMINE'),
    ('Topology', 'wasserstein', '> 0.5', 'Structural change', 'DEPARTED'),

    -- Departure Score thresholds
    ('Departure', 'score', '85-100', 'STABLE', 'BASELINE'),
    ('Departure', 'score', '70-84', 'Good', 'Normal'),
    ('Departure', 'score', '55-69', 'Fair', 'OBSERVE'),
    ('Departure', 'score', '40-54', 'Poor', 'SHIFTED'),
    ('Departure', 'score', '25-39', 'At Risk', 'DEPARTED'),
    ('Departure', 'score', '< 25', 'DEPARTED', 'DEPARTED'),

    -- Trajectory-based deviation thresholds (replaces z-score thresholds)
    ('Trajectory', 'slope_ratio', '0.5 to 1.5', 'Normal', 'Expected'),
    ('Trajectory', 'slope_ratio', '1.5 to 2.0 or 0.3 to 0.5', 'Elevated', 'OBSERVE'),
    ('Trajectory', 'slope_ratio', '2.0 to 3.0 or < 0.3', 'SHIFTED', 'EXAMINE'),
    ('Trajectory', 'slope_ratio', '> 3.0 or sign reversed', 'DEPARTED', 'DEPARTED'),

    -- Volatility ratio thresholds
    ('Trajectory', 'vol_ratio', '0.8 to 1.2', 'Stable', 'Expected'),
    ('Trajectory', 'vol_ratio', '1.2 to 1.5', 'Elevated', 'OBSERVE'),
    ('Trajectory', 'vol_ratio', '1.5 to 2.0', 'SHIFTED', 'EXAMINE'),
    ('Trajectory', 'vol_ratio', '> 2.0', 'DEPARTED', 'DEPARTED'),

    -- Slope departure (canary detection)
    ('Trajectory', 'slope_departure', '< 2x baseline_std', 'Normal', 'Expected'),
    ('Trajectory', 'slope_departure', '2-3x baseline_std', 'Elevated', 'OBSERVE'),
    ('Trajectory', 'slope_departure', '3-5x baseline_std', 'SHIFTED', 'EXAMINE'),
    ('Trajectory', 'slope_departure', '> 5x baseline_std', 'DEPARTED', 'DEPARTED')

) AS t(engine, metric, threshold_range, classification, action);


-- ============================================================================
-- REPORT 6: MINIMUM DATA REQUIREMENTS REFERENCE (Updated)
-- ============================================================================
-- Static reference table for hard and soft minimums

SELECT * FROM (VALUES
    -- Hard minimums (updated with revised Lyapunov)
    ('Lyapunov (max)', 3000, NULL, 1, 'Hard', 'Rosenstein/Kantz work with shorter series'),
    ('Lyapunov (spectrum)', 20000, NULL, 1, 'Hard', 'Full spectrum needs longer trajectories'),
    ('Correlation dim (d<=5)', 1000, NULL, 1, 'Hard', 'Low-dimensional systems'),
    ('Correlation dim (d=5-10)', 2000, NULL, 1, 'Hard', 'Medium embedding dimension'),
    ('Correlation dim (d>10)', 5000, NULL, 1, 'Hard', 'High-dimensional requires more data'),
    ('RQA (DET, LAM, DIV)', 1000, NULL, 1, 'Hard', 'Sufficient recurrence matrix'),
    ('Transfer entropy', 1000, NULL, 3, 'Hard', 'Directional causality'),
    ('Granger causality', 500, NULL, 2, 'Hard', 'VAR model fitting'),
    ('Betti numbers', 500, NULL, 1, 'Hard', 'Persistent homology'),
    ('Persistence entropy', 1000, NULL, 1, 'Hard', 'Enough topological features'),
    ('Coherence', 100, NULL, 3, 'Hard', 'Covariance matrix stability'),
    ('PID (synergy)', 500, NULL, 3, 'Hard', 'Triplet information'),
    ('Hurst exponent', 256, NULL, 1, 'Hard', 'DFA scaling regime'),
    ('Sample entropy', 200, NULL, 1, 'Hard', 'Pattern matching'),

    -- Soft minimums (recommended for reliability)
    ('Lyapunov (max)', 10000, NULL, 1, 'Soft', 'Stable estimate with good convergence'),
    ('Correlation dimension', 5000, NULL, 1, 'Soft', 'Depends on embedding dimension'),
    ('Transfer entropy', 5000, NULL, 3, 'Soft', 'Robust directional inference'),
    ('Granger causality', 2000, NULL, 2, 'Soft', 'Significant lags'),
    ('Betti numbers', 2000, NULL, 1, 'Soft', 'Stable topology'),
    ('Fleet trajectory analysis', NULL, 10, NULL, 'Soft', 'N >= 10 entities for fleet percentile thresholds')

) AS t(engine_metric, min_observations, min_entities, min_signals, minimum_type, notes);


-- ============================================================================
-- REPORT 7: WINDOW SIZE GUIDANCE (NEW)
-- ============================================================================
-- Minimum window sizes by analysis type

SELECT * FROM (VALUES
    ('Basic statistics', 50, 100, 'Mean, std, percentiles'),
    ('Spectral analysis', NULL, NULL, '2x longest period minimum, 4x recommended'),
    ('RQA', 200, 500, 'Sufficient recurrence structure'),
    ('Lyapunov', 500, 1000, 'Trajectory divergence'),
    ('Topology (Betti)', 300, 500, 'Persistent homology'),
    ('Transfer entropy', 100, 200, 'Per-window causality')
) AS t(analysis_type, min_window_samples, recommended_samples, notes);


-- ============================================================================
-- REPORT 8: EFFECT SIZE CLASSIFICATION (Updated — trajectory-based)
-- ============================================================================
-- Reference for what constitutes meaningful vs negligible effect sizes

SELECT * FROM (VALUES
    ('Slope ratio', '0.8 to 1.2', 'Negligible', 'Within normal trajectory variation'),
    ('Slope ratio', '1.2 to 2.0 or 0.5 to 0.8', 'Small', 'Notable trajectory change'),
    ('Slope ratio', '2.0 to 3.0 or 0.3 to 0.5', 'Medium', 'Significant, investigate'),
    ('Slope ratio', '> 3.0 or < 0.3 or reversed', 'Large', 'ACTIONABLE'),

    ('Vol ratio', '0.9 to 1.1', 'Negligible', 'Within normal variation'),
    ('Vol ratio', '1.1 to 1.3 or 0.7 to 0.9', 'Small', 'Minor volatility shift'),
    ('Vol ratio', '1.3 to 1.5 or 0.5 to 0.7', 'Medium', 'Meaningful change'),
    ('Vol ratio', '> 1.5 or < 0.5', 'Large', 'ACTIONABLE'),

    ('Percent change', '< 5%', 'Negligible', 'Noise level'),
    ('Percent change', '5% - 15%', 'Small', 'Minor shift'),
    ('Percent change', '15% - 30%', 'Medium', 'Meaningful change'),
    ('Percent change', '> 30%', 'Large', 'ACTIONABLE'),

    ('Correlation change', '< 0.1', 'Negligible', 'Statistical noise'),
    ('Correlation change', '0.1 - 0.2', 'Small', 'Weak change'),
    ('Correlation change', '0.2 - 0.4', 'Medium', 'Moderate restructuring'),
    ('Correlation change', '> 0.4', 'Large', 'ACTIONABLE'),

    ('Entropy change', '< 0.2', 'Negligible', 'Normal fluctuation'),
    ('Entropy change', '0.2 - 0.5', 'Small', 'Minor complexity shift'),
    ('Entropy change', '0.5 - 1.0', 'Medium', 'Significant change'),
    ('Entropy change', '> 1.0', 'Large', 'ACTIONABLE'),

    ('Departure score drop', '< 5 pts', 'Negligible', 'Normal variation'),
    ('Departure score drop', '5 - 10 pts', 'Small', 'Watch'),
    ('Departure score drop', '10 - 15 pts', 'Medium', 'Investigate'),
    ('Departure score drop', '> 15 pts', 'Large', 'ACTIONABLE'),

    ('Coherence velocity', '< 0.02/window', 'Negligible', 'Stable'),
    ('Coherence velocity', '0.02 - 0.05/window', 'Small', 'Drifting'),
    ('Coherence velocity', '0.05 - 0.10/window', 'Medium', 'Investigate'),
    ('Coherence velocity', '> 0.10/window', 'Large', 'ALARM')

) AS t(metric_type, range, effect_size, interpretation);


-- ============================================================================
-- REPORT 9: MULTI-PILLAR AGREEMENT REFERENCE
-- ============================================================================
-- Reference for confidence levels based on pillar agreement

SELECT * FROM (VALUES
    (1, 4, 25, 'Low', 'LOW_CONFIDENCE'),
    (2, 4, 50, 'Moderate', 'MODERATE_CONFIDENCE'),
    (3, 4, 75, 'High', 'HIGH_CONFIDENCE'),
    (4, 4, 95, 'Very High', 'VERY_HIGH_CONFIDENCE')
) AS t(pillars_agreeing, total_pillars, confidence_pct, confidence_level, recommendation);


-- ============================================================================
-- REPORT 10: COHERENCE VELOCITY ANALYSIS (NEW)
-- ============================================================================
-- Monitor rate of change in coherence - often more predictive than absolute value

WITH coherence_with_lag AS (
    SELECT
        cohort,
        window_id,
        coherence_ratio,
        LAG(coherence_ratio) OVER (PARTITION BY cohort ORDER BY window_id) AS prev_coherence
    FROM geometry
)
SELECT
    cohort,
    window_id,
    ROUND(coherence_ratio, 4) AS coherence,
    ROUND(coherence_ratio - prev_coherence, 4) AS delta_coherence,
    CASE
        WHEN ABS(coherence_ratio - prev_coherence) >= 0.10 THEN 'ALARM - Rapid transition'
        WHEN ABS(coherence_ratio - prev_coherence) >= 0.05 THEN 'WARNING - Fast change'
        WHEN ABS(coherence_ratio - prev_coherence) >= 0.02 THEN 'WATCH - Drifting'
        ELSE 'STABLE'
    END AS velocity_status,
    CASE
        WHEN (coherence_ratio - prev_coherence) > 0.05 THEN 'Rapid COUPLING'
        WHEN (coherence_ratio - prev_coherence) < -0.05 THEN 'Rapid DECOUPLING'
        ELSE 'Stable'
    END AS direction
FROM coherence_with_lag
WHERE prev_coherence IS NOT NULL
ORDER BY ABS(coherence_ratio - prev_coherence) DESC;


-- ============================================================================
-- REPORT 11: CURRENT DATA VALIDATION SUMMARY
-- ============================================================================
-- Quick overview of data quality for the current dataset

WITH entity_stats AS (
    SELECT
        COUNT(DISTINCT cohort) AS n_entities,
        COUNT(DISTINCT signal_id) AS n_signals,
        COUNT(*) AS total_obs,
        MIN(total_obs_per_entity) AS min_obs_entity,
        MAX(total_obs_per_entity) AS max_obs_entity,
        AVG(total_obs_per_entity) AS avg_obs_entity
    FROM (
        SELECT cohort, COUNT(*) AS total_obs_per_entity
        FROM observations
        GROUP BY cohort
    ) sub
),

validation_counts AS (
    SELECT
        SUM(CASE WHEN total_obs >= 3000 THEN 1 ELSE 0 END) AS lyapunov_ready,
        SUM(CASE WHEN total_obs >= 10000 THEN 1 ELSE 0 END) AS lyapunov_reliable,
        SUM(CASE WHEN total_obs >= 1000 THEN 1 ELSE 0 END) AS rqa_ready,
        SUM(CASE WHEN total_obs >= 500 THEN 1 ELSE 0 END) AS topology_ready,
        COUNT(*) AS total_entities
    FROM (
        SELECT cohort, COUNT(*) AS total_obs
        FROM observations
        GROUP BY cohort
    ) sub
)

SELECT
    '=== DATASET VALIDATION SUMMARY ===' AS header,
    NULL AS value
UNION ALL
SELECT 'Total entities', CAST(n_entities AS VARCHAR) FROM entity_stats
UNION ALL
SELECT 'Total signals', CAST(n_signals AS VARCHAR) FROM entity_stats
UNION ALL
SELECT 'Total observations', CAST(total_obs AS VARCHAR) FROM entity_stats
UNION ALL
SELECT 'Min obs/entity', CAST(ROUND(min_obs_entity) AS VARCHAR) FROM entity_stats
UNION ALL
SELECT 'Max obs/entity', CAST(ROUND(max_obs_entity) AS VARCHAR) FROM entity_stats
UNION ALL
SELECT 'Avg obs/entity', CAST(ROUND(avg_obs_entity) AS VARCHAR) FROM entity_stats
UNION ALL
SELECT '--- Engine Readiness ---', NULL
UNION ALL
SELECT 'Lyapunov ready (>=3k obs)', lyapunov_ready || '/' || total_entities || ' entities' FROM validation_counts
UNION ALL
SELECT 'Lyapunov reliable (>=10k obs)', lyapunov_reliable || '/' || total_entities || ' entities' FROM validation_counts
UNION ALL
SELECT 'RQA ready (>=1k obs)', rqa_ready || '/' || total_entities || ' entities' FROM validation_counts
UNION ALL
SELECT 'Topology ready (>=500 obs)', topology_ready || '/' || total_entities || ' entities' FROM validation_counts
UNION ALL
SELECT '--- Fleet Analytics ---', NULL
UNION ALL
SELECT 'Fleet trajectory analysis validity',
    CASE
        WHEN n_entities >= 30 THEN 'RELIABLE (N=' || n_entities || ') - Full fleet analytics with percentile thresholds'
        WHEN n_entities >= 10 THEN 'MARGINAL (N=' || n_entities || ') - Fleet percentiles OK, use per-entity trajectory metrics'
        WHEN n_entities >= 5 THEN 'LIMITED (N=' || n_entities || ') - Robust stats only, per-entity slope analysis'
        ELSE 'MINIMAL (N=' || n_entities || ') - Individual baseline analysis only'
    END
FROM entity_stats;


-- ============================================================================
-- REPORT 12: FINDING VALIDATION CHECKLIST
-- ============================================================================
-- Use this before reporting any finding as significant

SELECT * FROM (VALUES
    (1, 'Data Sufficiency', 'Does observation count exceed hard minimum for the engine?'),
    (2, 'Sampling Adequacy', 'Does data capture >=10 cycles of the phenomenon of interest?'),
    (3, 'Baseline Validity', 'Was baseline period representative (stable, sufficient samples)?'),
    (4, 'Trajectory Departure', 'Has the slope ratio changed beyond 2x or reversed sign?'),
    (5, 'Temporal Persistence', 'Does finding persist over 5+ consecutive windows?'),
    (6, 'Cross-Pillar Agreement', 'Do at least 2 of 4 pillars show consistent evidence?'),
    (7, 'Physical Plausibility', 'Does the finding make physical/engineering sense?'),
    (8, 'Actionability', 'What specific action would you take based on this finding?')
) AS t(check_order, check_name, question);
