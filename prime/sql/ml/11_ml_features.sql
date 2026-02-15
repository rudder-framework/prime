-- ============================================================================
-- MACHINE LEARNING FEATURE EXTRACTION
-- ============================================================================
-- Pivots Engines outputs into ML-ready feature matrices.
-- One row per entity with all signal metrics as columns.
--
-- Output: Ready for sklearn, XGBoost, PyTorch, etc.
-- ============================================================================

-- ============================================================================
-- 001: SIGNAL-LEVEL FEATURES (wide format)
-- ============================================================================
-- Pivot signal metrics so each signal becomes columns
-- Result: cohort, signal1_hurst, signal1_entropy, signal2_hurst, ...

CREATE OR REPLACE VIEW v_ml_signal_features_long AS
SELECT
    cohort,
    signal_id,
    -- Typology features
    COALESCE(hurst_rs, 0.5) AS hurst,
    COALESCE(sample_entropy, 0) AS entropy,
    COALESCE(lyapunov, 0) AS lyapunov,
    COALESCE(spectral_slope, 0) AS spectral_slope,
    COALESCE(dominant_freq, 0) AS dominant_freq,
    COALESCE(spectral_centroid, 0) AS spectral_centroid,
    -- GARCH volatility
    COALESCE(omega, 0) AS garch_omega,
    COALESCE(alpha, 0) AS garch_alpha,
    COALESCE(beta, 0) AS garch_beta,
    -- RQA features
    COALESCE(recurrence_rate, 0) AS rqa_recurrence,
    COALESCE(determinism, 0) AS rqa_determinism,
    -- Basic stats
    COALESCE(mean, 0) AS mean,
    COALESCE(std, 1) AS std,
    COALESCE(skewness, 0) AS skewness,
    COALESCE(kurtosis, 0) AS kurtosis,
    -- Classification
    behavioral_class
FROM primitives;


-- ============================================================================
-- 002: ENTITY-LEVEL AGGREGATED FEATURES
-- ============================================================================
-- Aggregate across all signals for each entity
-- Good for entity-level classification (e.g., fault detection)

CREATE OR REPLACE VIEW v_ml_entity_features AS
SELECT
    cohort,
    COUNT(DISTINCT signal_id) AS n_signals,

    -- Hurst statistics
    AVG(hurst) AS hurst_mean,
    STDDEV(hurst) AS hurst_std,
    MIN(hurst) AS hurst_min,
    MAX(hurst) AS hurst_max,

    -- Entropy statistics
    AVG(entropy) AS entropy_mean,
    STDDEV(entropy) AS entropy_std,
    MAX(entropy) AS entropy_max,

    -- Lyapunov (chaos indicator)
    AVG(lyapunov) AS lyapunov_mean,
    MAX(lyapunov) AS lyapunov_max,
    SUM(CASE WHEN lyapunov > 0.01 THEN 1 ELSE 0 END) AS n_chaotic_signals,

    -- Spectral features
    AVG(spectral_slope) AS spectral_slope_mean,
    AVG(dominant_freq) AS dominant_freq_mean,
    STDDEV(dominant_freq) AS dominant_freq_std,

    -- GARCH volatility
    AVG(garch_alpha) AS garch_alpha_mean,
    AVG(garch_beta) AS garch_beta_mean,
    AVG(garch_alpha + garch_beta) AS garch_persistence_mean,

    -- RQA
    AVG(rqa_recurrence) AS rqa_recurrence_mean,
    AVG(rqa_determinism) AS rqa_determinism_mean,

    -- Basic stats
    AVG(std) AS signal_std_mean,
    AVG(ABS(skewness)) AS abs_skewness_mean,
    AVG(kurtosis) AS kurtosis_mean,

    -- Behavioral class distribution
    SUM(CASE WHEN behavioral_class = 'trending' THEN 1 ELSE 0 END) AS n_trending,
    SUM(CASE WHEN behavioral_class = 'mean_reverting' THEN 1 ELSE 0 END) AS n_mean_reverting,
    SUM(CASE WHEN behavioral_class = 'random_walk' THEN 1 ELSE 0 END) AS n_random_walk

FROM v_ml_signal_features_long
GROUP BY cohort;


-- ============================================================================
-- 003: PAIRWISE RELATIONSHIP FEATURES
-- ============================================================================
-- Aggregate correlation and causality features per entity

CREATE OR REPLACE VIEW v_ml_pairwise_features AS
SELECT
    cohort,

    -- Correlation statistics
    COUNT(*) AS n_pairs,
    AVG(ABS(correlation)) AS abs_corr_mean,
    MAX(ABS(correlation)) AS abs_corr_max,
    SUM(CASE WHEN ABS(correlation) > 0.7 THEN 1 ELSE 0 END) AS n_high_corr_pairs,

    -- Mutual information
    AVG(mutual_information) AS mi_mean,
    MAX(mutual_information) AS mi_max,

    -- DTW distance
    AVG(dtw_distance) AS dtw_mean,
    STDDEV(dtw_distance) AS dtw_std,

    -- Copula tail dependence
    AVG(copula_lower_tail) AS copula_lower_mean,
    AVG(copula_upper_tail) AS copula_upper_mean

FROM geometry
GROUP BY cohort;


-- ============================================================================
-- 004: DYNAMICS FEATURES
-- ============================================================================
-- Regime and stability features per entity

CREATE OR REPLACE VIEW v_ml_dynamics_features AS
SELECT
    cohort,

    -- Regime statistics
    AVG(n_regimes) AS n_regimes_mean,
    MAX(n_regimes) AS n_regimes_max,
    SUM(CASE WHEN n_regimes > 1 THEN 1 ELSE 0 END) AS n_signals_multiregime,

    -- Attractor types
    SUM(CASE WHEN attractor_type = 'fixed_point' THEN 1 ELSE 0 END) AS n_fixed_point,
    SUM(CASE WHEN attractor_type = 'limit_cycle' THEN 1 ELSE 0 END) AS n_limit_cycle,
    SUM(CASE WHEN attractor_type = 'strange' THEN 1 ELSE 0 END) AS n_strange_attractor,

    -- Basin stability
    AVG(basin_stability) AS basin_stability_mean,
    MIN(basin_stability) AS basin_stability_min,

    -- Chaos
    SUM(CASE WHEN is_chaotic THEN 1 ELSE 0 END) AS n_chaotic

FROM dynamics
GROUP BY cohort;


-- ============================================================================
-- 005: CAUSALITY FEATURES
-- ============================================================================
-- Causal structure features per entity

CREATE OR REPLACE VIEW v_ml_causality_features AS
SELECT
    cohort,

    -- Granger causality
    COUNT(*) AS n_directed_pairs,
    SUM(CASE WHEN granger_significant THEN 1 ELSE 0 END) AS n_granger_edges,
    AVG(CASE WHEN granger_significant THEN granger_f ELSE NULL END) AS granger_f_mean,

    -- Transfer entropy
    AVG(transfer_entropy) AS te_mean,
    MAX(transfer_entropy) AS te_max,
    SUM(CASE WHEN transfer_entropy > 0.1 THEN 1 ELSE 0 END) AS n_high_te_edges,

    -- Cointegration
    SUM(CASE WHEN is_cointegrated THEN 1 ELSE 0 END) AS n_cointegrated_pairs,

    -- Network density
    SUM(CASE WHEN granger_significant OR transfer_entropy > 0.1 THEN 1 ELSE 0 END) * 1.0 /
        NULLIF(COUNT(*), 0) AS causal_density

FROM mechanics
GROUP BY cohort;


-- ============================================================================
-- 006: CLUSTER/OUTLIER FEATURES
-- ============================================================================
-- Clustering and outlier features per entity

CREATE OR REPLACE VIEW v_ml_cluster_features AS
SELECT
    cohort,

    -- Cluster distribution
    COUNT(DISTINCT cluster_id) AS n_clusters,

    -- Outlier statistics
    SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END) AS n_outliers,
    AVG(lof_score) AS lof_mean,
    MAX(lof_score) AS lof_max,

    -- Silhouette (cluster quality)
    AVG(silhouette_score) AS silhouette_mean

FROM clusters
GROUP BY cohort;


-- ============================================================================
-- 007: COMPLETE ML FEATURE MATRIX
-- ============================================================================
-- Join all feature views into one ML-ready table

CREATE OR REPLACE VIEW v_ml_features AS
SELECT
    e.cohort,

    -- Entity-level signal features
    e.n_signals,
    e.hurst_mean, e.hurst_std, e.hurst_min, e.hurst_max,
    e.entropy_mean, e.entropy_std, e.entropy_max,
    e.lyapunov_mean, e.lyapunov_max, e.n_chaotic_signals,
    e.spectral_slope_mean,
    e.dominant_freq_mean, e.dominant_freq_std,
    e.garch_alpha_mean, e.garch_beta_mean, e.garch_persistence_mean,
    e.rqa_recurrence_mean, e.rqa_determinism_mean,
    e.signal_std_mean, e.abs_skewness_mean, e.kurtosis_mean,
    e.n_trending, e.n_mean_reverting, e.n_random_walk,

    -- Pairwise features
    COALESCE(p.n_pairs, 0) AS n_pairs,
    COALESCE(p.abs_corr_mean, 0) AS abs_corr_mean,
    COALESCE(p.abs_corr_max, 0) AS abs_corr_max,
    COALESCE(p.n_high_corr_pairs, 0) AS n_high_corr_pairs,
    COALESCE(p.mi_mean, 0) AS mi_mean,
    COALESCE(p.dtw_mean, 0) AS dtw_mean,

    -- Dynamics features
    COALESCE(d.n_regimes_mean, 1) AS n_regimes_mean,
    COALESCE(d.n_regimes_max, 1) AS n_regimes_max,
    COALESCE(d.n_signals_multiregime, 0) AS n_signals_multiregime,
    COALESCE(d.n_fixed_point, 0) AS n_fixed_point,
    COALESCE(d.n_limit_cycle, 0) AS n_limit_cycle,
    COALESCE(d.n_strange_attractor, 0) AS n_strange_attractor,
    COALESCE(d.basin_stability_mean, 1) AS basin_stability_mean,
    COALESCE(d.n_chaotic, 0) AS n_chaotic,

    -- Causality features
    COALESCE(c.n_granger_edges, 0) AS n_granger_edges,
    COALESCE(c.granger_f_mean, 0) AS granger_f_mean,
    COALESCE(c.te_mean, 0) AS te_mean,
    COALESCE(c.te_max, 0) AS te_max,
    COALESCE(c.n_cointegrated_pairs, 0) AS n_cointegrated_pairs,
    COALESCE(c.causal_density, 0) AS causal_density,

    -- Cluster features
    COALESCE(cl.n_clusters, 1) AS n_clusters,
    COALESCE(cl.n_outliers, 0) AS n_outliers,
    COALESCE(cl.lof_mean, 1) AS lof_mean,
    COALESCE(cl.lof_max, 1) AS lof_max

FROM v_ml_entity_features e
LEFT JOIN v_ml_pairwise_features p ON e.cohort = p.cohort
LEFT JOIN v_ml_dynamics_features d ON e.cohort = d.cohort
LEFT JOIN v_ml_causality_features c ON e.cohort = c.cohort
LEFT JOIN v_ml_cluster_features cl ON e.cohort = cl.cohort;


-- ============================================================================
-- 008: EXPORT TO PARQUET
-- ============================================================================
-- Use: COPY v_ml_features TO 'ml_features.parquet' (FORMAT PARQUET);

CREATE OR REPLACE VIEW v_ml_feature_names AS
SELECT
    column_name,
    ordinal_position
FROM information_schema.columns
WHERE table_name = 'v_ml_features'
ORDER BY ordinal_position;


-- ============================================================================
-- 009: SIGNAL-LEVEL FEATURES (for per-signal classification)
-- ============================================================================
-- One row per (entity, signal) - for signal-level ML tasks

CREATE OR REPLACE VIEW v_ml_signal_features AS
SELECT
    s.cohort,
    s.signal_id,

    -- Core metrics
    s.hurst,
    s.entropy,
    s.lyapunov,
    s.spectral_slope,
    s.dominant_freq,
    s.spectral_centroid,
    s.garch_omega,
    s.garch_alpha,
    s.garch_beta,
    s.rqa_recurrence,
    s.rqa_determinism,
    s.mean,
    s.std,
    s.skewness,
    s.kurtosis,

    -- Dynamics
    COALESCE(d.n_regimes, 1) AS n_regimes,
    COALESCE(d.basin_stability, 1) AS basin_stability,
    CASE WHEN d.is_chaotic THEN 1 ELSE 0 END AS is_chaotic,
    CASE
        WHEN d.attractor_type = 'fixed_point' THEN 0
        WHEN d.attractor_type = 'limit_cycle' THEN 1
        WHEN d.attractor_type = 'strange' THEN 2
        ELSE -1
    END AS attractor_type_code,

    -- Cluster membership
    COALESCE(cl.cluster_id, 0) AS cluster_id,
    COALESCE(cl.lof_score, 1) AS lof_score,
    CASE WHEN cl.is_outlier THEN 1 ELSE 0 END AS is_outlier,

    -- Behavioral class (target for classification)
    s.behavioral_class,
    CASE
        WHEN s.behavioral_class = 'trending' THEN 0
        WHEN s.behavioral_class = 'mean_reverting' THEN 1
        WHEN s.behavioral_class = 'random_walk' THEN 2
        ELSE -1
    END AS behavioral_class_code

FROM v_ml_signal_features_long s
LEFT JOIN dynamics d ON s.cohort = d.cohort AND s.signal_id = d.signal_id
LEFT JOIN clusters cl ON s.cohort = cl.cohort AND s.signal_id = cl.signal_id;


-- ============================================================================
-- 010: FEATURE STATISTICS (for normalization)
-- ============================================================================
-- Provides percentile bounds and summary stats for each feature

CREATE OR REPLACE VIEW v_ml_feature_stats AS
SELECT
    'hurst' AS feature,
    AVG(hurst) AS mean,
    MIN(hurst) AS min,
    MAX(hurst) AS max,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY hurst) AS p05,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY hurst) AS median,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY hurst) AS p95
FROM v_ml_signal_features_long
UNION ALL
SELECT 'entropy', AVG(entropy), MIN(entropy), MAX(entropy),
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY entropy),
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY entropy),
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY entropy)
FROM v_ml_signal_features_long
UNION ALL
SELECT 'lyapunov', AVG(lyapunov), MIN(lyapunov), MAX(lyapunov),
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY lyapunov),
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY lyapunov),
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY lyapunov)
FROM v_ml_signal_features_long
UNION ALL
SELECT 'spectral_slope', AVG(spectral_slope), MIN(spectral_slope), MAX(spectral_slope),
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY spectral_slope),
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY spectral_slope),
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY spectral_slope)
FROM v_ml_signal_features_long
UNION ALL
SELECT 'garch_alpha', AVG(garch_alpha), MIN(garch_alpha), MAX(garch_alpha),
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY garch_alpha),
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY garch_alpha),
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY garch_alpha)
FROM v_ml_signal_features_long
UNION ALL
SELECT 'garch_beta', AVG(garch_beta), MIN(garch_beta), MAX(garch_beta),
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY garch_beta),
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY garch_beta),
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY garch_beta)
FROM v_ml_signal_features_long;
