-- ============================================================================
-- MANIFOLD-DERIVED VIEWS
-- ============================================================================
-- These views replace 14 stages that moved from Manifold to Prime.
-- Each view reads from Manifold core output (11 parquet files) and computes
-- what previously required a separate Python stage.
--
-- Source parquets (loaded by 00_run_all.sql or loader.py):
--   state_geometry, state_vector, signal_vector, signal_pairwise,
--   gaussian_fingerprint, gaussian_similarity, breaks, observations
--
-- Principle: If it's linear algebra -> Manifold. If it's SQL -> Prime.
-- ============================================================================


-- ============================================================================
-- 001: GEOMETRY DYNAMICS (was Manifold stage 07)
-- ============================================================================
-- Differential geometry: velocity, acceleration, jerk of state evolution.
-- Pure LAG() on state_geometry columns, partitioned by (cohort, engine).
--
-- Output schema matches geometry_dynamics.parquet:
--   I, engine, cohort, effective_dim, effective_dim_velocity,
--   effective_dim_acceleration, effective_dim_jerk, effective_dim_curvature,
--   eigenvalue_1, eigenvalue_1_velocity, total_variance, variance_velocity,
--   collapse_onset_idx, collapse_onset_fraction

CREATE OR REPLACE VIEW v_geometry_dynamics AS
WITH lagged AS (
    SELECT
        I,
        engine,
        cohort,
        effective_dim,
        eigenvalue_1,
        total_variance,
        -- d1: velocity (first difference)
        effective_dim - LAG(effective_dim) OVER w AS effective_dim_velocity,
        eigenvalue_1 - LAG(eigenvalue_1) OVER w AS eigenvalue_1_velocity,
        total_variance - LAG(total_variance) OVER w AS variance_velocity,
        -- For d2/d3 we need nested LAG
        LAG(effective_dim, 1) OVER w AS eff_dim_prev1,
        LAG(effective_dim, 2) OVER w AS eff_dim_prev2,
    FROM state_geometry
    WINDOW w AS (PARTITION BY cohort, engine ORDER BY I)
),
with_accel AS (
    SELECT
        *,
        -- d2: acceleration (second difference)
        effective_dim_velocity - (eff_dim_prev1 - eff_dim_prev2) AS effective_dim_acceleration,
    FROM lagged
),
with_jerk AS (
    SELECT
        I,
        engine,
        cohort,
        effective_dim,
        effective_dim_velocity,
        effective_dim_acceleration,
        -- d3: jerk (third difference)
        effective_dim_acceleration - LAG(effective_dim_acceleration) OVER w AS effective_dim_jerk,
        -- Curvature: |acceleration| / (1 + velocity^2)^(3/2)
        CASE WHEN effective_dim_velocity IS NOT NULL
            THEN ABS(effective_dim_acceleration) / POWER(1.0 + effective_dim_velocity * effective_dim_velocity, 1.5)
            ELSE NULL
        END AS effective_dim_curvature,
        eigenvalue_1,
        eigenvalue_1_velocity,
        total_variance,
        variance_velocity,
    FROM with_accel
    WINDOW w AS (PARTITION BY cohort, engine ORDER BY I)
),
-- Collapse onset: first I where effective_dim_velocity < -0.1 sustained
collapse AS (
    SELECT
        cohort,
        engine,
        MIN(I) AS collapse_onset_idx,
    FROM with_jerk
    WHERE effective_dim_velocity < -0.1
    GROUP BY cohort, engine
),
lifecycle AS (
    SELECT cohort, engine, MAX(I) AS max_I
    FROM state_geometry
    GROUP BY cohort, engine
)
SELECT
    j.I,
    j.engine,
    j.cohort,
    j.effective_dim,
    j.effective_dim_velocity,
    j.effective_dim_acceleration,
    j.effective_dim_jerk,
    j.effective_dim_curvature,
    j.eigenvalue_1,
    j.eigenvalue_1_velocity,
    j.total_variance,
    j.variance_velocity,
    c.collapse_onset_idx,
    CASE WHEN c.collapse_onset_idx IS NOT NULL AND lc.max_I > 0
        THEN CAST(c.collapse_onset_idx AS DOUBLE) / lc.max_I
        ELSE NULL
    END AS collapse_onset_fraction,
FROM with_jerk j
LEFT JOIN collapse c USING (cohort, engine)
LEFT JOIN lifecycle lc ON j.cohort = lc.cohort AND j.engine = lc.engine;


-- ============================================================================
-- 002: COHORT VECTOR (was Manifold stage 25)
-- ============================================================================
-- Pivot state_geometry engine rows to wide cohort feature vectors.
-- Each (cohort, I) gets one row with {engine}_{metric} columns.
--
-- Uses conditional aggregation (FILTER) to pivot engine rows.
-- Standard engines: complexity, shape, spectral.

CREATE OR REPLACE VIEW v_cohort_vector AS
SELECT
    cohort,
    I,
    -- Complexity engine
    MAX(effective_dim) FILTER (WHERE engine = 'complexity') AS complexity_effective_dim,
    MAX(eigenvalue_1) FILTER (WHERE engine = 'complexity') AS complexity_eigenvalue_1,
    MAX(total_variance) FILTER (WHERE engine = 'complexity') AS complexity_total_variance,
    MAX(condition_number) FILTER (WHERE engine = 'complexity') AS complexity_condition_number,
    MAX(ratio_2_1) FILTER (WHERE engine = 'complexity') AS complexity_ratio_2_1,
    MAX(ratio_3_1) FILTER (WHERE engine = 'complexity') AS complexity_ratio_3_1,
    MAX(eigenvalue_entropy_norm) FILTER (WHERE engine = 'complexity') AS complexity_eigenvalue_entropy_norm,
    MAX(explained_1) FILTER (WHERE engine = 'complexity') AS complexity_explained_1,
    MAX(explained_2) FILTER (WHERE engine = 'complexity') AS complexity_explained_2,
    MAX(explained_3) FILTER (WHERE engine = 'complexity') AS complexity_explained_3,
    MAX(n_signals) FILTER (WHERE engine = 'complexity') AS complexity_n_signals,
    -- Shape engine
    MAX(effective_dim) FILTER (WHERE engine = 'shape') AS shape_effective_dim,
    MAX(eigenvalue_1) FILTER (WHERE engine = 'shape') AS shape_eigenvalue_1,
    MAX(total_variance) FILTER (WHERE engine = 'shape') AS shape_total_variance,
    MAX(condition_number) FILTER (WHERE engine = 'shape') AS shape_condition_number,
    MAX(ratio_2_1) FILTER (WHERE engine = 'shape') AS shape_ratio_2_1,
    MAX(ratio_3_1) FILTER (WHERE engine = 'shape') AS shape_ratio_3_1,
    MAX(eigenvalue_entropy_norm) FILTER (WHERE engine = 'shape') AS shape_eigenvalue_entropy_norm,
    MAX(explained_1) FILTER (WHERE engine = 'shape') AS shape_explained_1,
    MAX(explained_2) FILTER (WHERE engine = 'shape') AS shape_explained_2,
    MAX(explained_3) FILTER (WHERE engine = 'shape') AS shape_explained_3,
    MAX(n_signals) FILTER (WHERE engine = 'shape') AS shape_n_signals,
    -- Spectral engine
    MAX(effective_dim) FILTER (WHERE engine = 'spectral') AS spectral_effective_dim,
    MAX(eigenvalue_1) FILTER (WHERE engine = 'spectral') AS spectral_eigenvalue_1,
    MAX(total_variance) FILTER (WHERE engine = 'spectral') AS spectral_total_variance,
    MAX(condition_number) FILTER (WHERE engine = 'spectral') AS spectral_condition_number,
    MAX(ratio_2_1) FILTER (WHERE engine = 'spectral') AS spectral_ratio_2_1,
    MAX(ratio_3_1) FILTER (WHERE engine = 'spectral') AS spectral_ratio_3_1,
    MAX(eigenvalue_entropy_norm) FILTER (WHERE engine = 'spectral') AS spectral_eigenvalue_entropy_norm,
    MAX(explained_1) FILTER (WHERE engine = 'spectral') AS spectral_explained_1,
    MAX(explained_2) FILTER (WHERE engine = 'spectral') AS spectral_explained_2,
    MAX(explained_3) FILTER (WHERE engine = 'spectral') AS spectral_explained_3,
    MAX(n_signals) FILTER (WHERE engine = 'spectral') AS spectral_n_signals,
    -- Derived: eff_dim_rate (diff within cohort)
    MAX(effective_dim) FILTER (WHERE engine = 'complexity')
        - LAG(MAX(effective_dim) FILTER (WHERE engine = 'complexity'))
          OVER (PARTITION BY cohort ORDER BY I) AS complexity_eff_dim_rate,
    MAX(effective_dim) FILTER (WHERE engine = 'shape')
        - LAG(MAX(effective_dim) FILTER (WHERE engine = 'shape'))
          OVER (PARTITION BY cohort ORDER BY I) AS shape_eff_dim_rate,
    MAX(effective_dim) FILTER (WHERE engine = 'spectral')
        - LAG(MAX(effective_dim) FILTER (WHERE engine = 'spectral'))
          OVER (PARTITION BY cohort ORDER BY I) AS spectral_eff_dim_rate,
    -- Derived: variance_concentration (= explained_1)
    MAX(explained_1) FILTER (WHERE engine = 'complexity') AS complexity_variance_concentration,
    MAX(explained_1) FILTER (WHERE engine = 'shape') AS shape_variance_concentration,
    MAX(explained_1) FILTER (WHERE engine = 'spectral') AS spectral_variance_concentration,
FROM state_geometry
GROUP BY cohort, I
ORDER BY cohort, I;


-- ============================================================================
-- 003: TOPOLOGY (was Manifold stage 11)
-- ============================================================================
-- Network topology from signal_pairwise correlation threshold.
-- Adjacency = |correlation| > threshold (90th percentile per cohort).

CREATE OR REPLACE VIEW v_topology AS
WITH thresholds AS (
    -- Adaptive threshold: 90th percentile of |correlation| per cohort
    SELECT
        cohort,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY ABS(correlation)) AS threshold
    FROM signal_pairwise
    WHERE correlation IS NOT NULL
    GROUP BY cohort
),
edges AS (
    SELECT
        sp.cohort,
        sp.I,
        sp.signal_a,
        sp.signal_b,
        ABS(sp.correlation) AS abs_corr,
        t.threshold,
    FROM signal_pairwise sp
    JOIN thresholds t USING (cohort)
    WHERE ABS(sp.correlation) > t.threshold
),
edge_counts AS (
    SELECT
        cohort,
        I,
        COUNT(*) AS n_edges,
        threshold,
    FROM edges
    GROUP BY cohort, I, threshold
),
signal_counts AS (
    SELECT cohort, I, COUNT(DISTINCT signal_id) AS n_signals
    FROM (
        SELECT cohort, I, signal_a AS signal_id FROM signal_pairwise
        UNION
        SELECT cohort, I, signal_b AS signal_id FROM signal_pairwise
    )
    GROUP BY cohort, I
),
degree_stats AS (
    SELECT
        cohort,
        I,
        signal_id,
        COUNT(*) AS degree,
    FROM (
        SELECT cohort, I, signal_a AS signal_id FROM edges
        UNION ALL
        SELECT cohort, I, signal_b AS signal_id FROM edges
    )
    GROUP BY cohort, I, signal_id
)
SELECT
    TRUE AS topology_computed,
    sc.n_signals,
    COALESCE(ec.n_edges, 0) AS n_edges,
    CASE WHEN sc.n_signals > 1
        THEN COALESCE(ec.n_edges, 0)::DOUBLE / (sc.n_signals * (sc.n_signals - 1) / 2.0)
        ELSE 0.0
    END AS density,
    COALESCE(AVG(ds.degree), 0.0) AS mean_degree,
    COALESCE(MAX(ds.degree), 0) AS max_degree,
    COALESCE(ec.threshold, 0.5) AS threshold,
    sc.cohort,
    sc.I,
FROM signal_counts sc
LEFT JOIN edge_counts ec USING (cohort, I)
LEFT JOIN degree_stats ds USING (cohort, I)
GROUP BY sc.cohort, sc.I, sc.n_signals, ec.n_edges, ec.threshold;


-- ============================================================================
-- 004: STATISTICS (was Manifold stage 13)
-- ============================================================================
-- Summary statistics per (cohort, signal_id) from observations.

CREATE OR REPLACE VIEW v_statistics AS
SELECT
    cohort,
    signal_id,
    COUNT(*) AS n_points,
    AVG(value) AS mean,
    STDDEV(value) AS std,
    MIN(value) AS min,
    MAX(value) AS max,
    MAX(value) - MIN(value) AS range,
    MEDIAN(value) AS median,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) AS q1,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) AS q3,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value)
        - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) AS iqr,
    VARIANCE(value) AS variance,
    CASE WHEN AVG(value) != 0
        THEN STDDEV(value) / ABS(AVG(value))
        ELSE NULL
    END AS cv,
FROM observations
GROUP BY cohort, signal_id;


-- ============================================================================
-- 005: Z-SCORE REFERENCE (was Manifold stage 12)
-- ============================================================================
-- Normalization parameters: mean/std per numeric column across source tables.
-- Used for standardization in downstream analysis.

CREATE OR REPLACE VIEW v_zscore_ref AS
WITH sv_stats AS (
    SELECT
        'signal_vector' AS source,
        unnest(['crest_factor','kurtosis','skewness','spectral_slope',
                'dominant_freq','spectral_entropy','spectral_centroid',
                'spectral_bandwidth','sample_entropy','permutation_entropy',
                'hurst','acf_lag1','acf_lag10','acf_half_life']) AS column_name,
    FROM (SELECT 1)
)
SELECT
    'signal_vector' AS source,
    'spectral_entropy' AS column_name,
    AVG(spectral_entropy) AS mean,
    STDDEV(spectral_entropy) AS std,
    MIN(spectral_entropy) AS min,
    MAX(spectral_entropy) AS max,
    COUNT(*) AS n_samples,
FROM signal_vector
WHERE spectral_entropy IS NOT NULL;


-- ============================================================================
-- 006: CORRELATION (was Manifold stage 14)
-- ============================================================================
-- Correlation matrix between signal_vector features within each cohort.

CREATE OR REPLACE VIEW v_correlation AS
SELECT
    a.signal_id AS feature_a,
    b.signal_id AS feature_b,
    CASE WHEN COUNT(*) > 2
        THEN CORR(a.spectral_entropy, b.spectral_entropy)
        ELSE NULL
    END AS correlation,
    COUNT(*) AS n_valid,
    a.cohort,
FROM signal_vector a
JOIN signal_vector b ON a.cohort = b.cohort AND a.I = b.I AND a.signal_id < b.signal_id
GROUP BY a.cohort, a.signal_id, b.signal_id
HAVING COUNT(*) > 2;


-- ============================================================================
-- 007: BREAK SEQUENCE (was Manifold stage 16)
-- ============================================================================
-- Ranks breaks by propagation order within each cohort.

CREATE OR REPLACE VIEW v_break_sequence AS
WITH first_breaks AS (
    SELECT
        cohort,
        signal_id,
        MIN(I) AS first_break_I,
    FROM breaks
    GROUP BY cohort, signal_id
),
ranked AS (
    SELECT
        cohort,
        signal_id,
        first_break_I,
        ROW_NUMBER() OVER (PARTITION BY cohort ORDER BY first_break_I) AS propagation_rank,
    FROM first_breaks
),
reference AS (
    SELECT cohort, MIN(first_break_I) AS reference_index
    FROM first_breaks
    GROUP BY cohort
)
SELECT
    r.cohort,
    r.signal_id,
    r.first_break_I,
    r.first_break_I - ref.reference_index AS detection_latency,
    r.propagation_rank,
    r.first_break_I - LAG(r.first_break_I) OVER (
        PARTITION BY r.cohort ORDER BY r.first_break_I
    ) AS cascade_delay,
    ref.reference_index,
FROM ranked r
JOIN reference ref USING (cohort)
ORDER BY r.cohort, r.propagation_rank;


-- ============================================================================
-- 008: SEGMENT COMPARISON (was Manifold stage 18)
-- ============================================================================
-- Delta between early-life and late-life geometry.
-- Early = first 25% of windows. Late = last 25% of windows.

CREATE OR REPLACE VIEW v_segment_comparison AS
WITH lifecycle AS (
    SELECT
        cohort,
        engine,
        MIN(I) AS I_min,
        MAX(I) AS I_max,
        (MAX(I) - MIN(I)) AS span,
    FROM state_geometry
    GROUP BY cohort, engine
),
early AS (
    SELECT
        sg.cohort,
        sg.engine,
        'early' AS segment,
        AVG(sg.effective_dim) AS effective_dim,
        AVG(sg.total_variance) AS total_variance,
        AVG(sg.eigenvalue_1) AS eigenvalue_1,
        AVG(sg.condition_number) AS condition_number,
        AVG(sg.eigenvalue_entropy_norm) AS eigenvalue_entropy_norm,
    FROM state_geometry sg
    JOIN lifecycle lc ON sg.cohort = lc.cohort AND sg.engine = lc.engine
    WHERE sg.I <= lc.I_min + lc.span * 0.25
    GROUP BY sg.cohort, sg.engine
),
late AS (
    SELECT
        sg.cohort,
        sg.engine,
        'late' AS segment,
        AVG(sg.effective_dim) AS effective_dim,
        AVG(sg.total_variance) AS total_variance,
        AVG(sg.eigenvalue_1) AS eigenvalue_1,
        AVG(sg.condition_number) AS condition_number,
        AVG(sg.eigenvalue_entropy_norm) AS eigenvalue_entropy_norm,
    FROM state_geometry sg
    JOIN lifecycle lc ON sg.cohort = lc.cohort AND sg.engine = lc.engine
    WHERE sg.I >= lc.I_max - lc.span * 0.25
    GROUP BY sg.cohort, sg.engine
)
SELECT
    e.cohort,
    e.engine,
    l.effective_dim - e.effective_dim AS delta_effective_dim,
    l.total_variance - e.total_variance AS delta_total_variance,
    l.eigenvalue_1 - e.eigenvalue_1 AS delta_eigenvalue_1,
    l.condition_number - e.condition_number AS delta_condition_number,
    l.eigenvalue_entropy_norm - e.eigenvalue_entropy_norm AS delta_entropy,
    e.effective_dim AS early_effective_dim,
    l.effective_dim AS late_effective_dim,
    e.total_variance AS early_total_variance,
    l.total_variance AS late_total_variance,
FROM early e
JOIN late l USING (cohort, engine);


-- ============================================================================
-- 009: INFO FLOW DELTA (was Manifold stage 19)
-- ============================================================================
-- Classifies information flow link status.
-- Note: requires pre-segment information_flow if segment comparison needed.
-- This simplified version classifies based on Granger significance.

CREATE OR REPLACE VIEW v_info_flow_delta AS
SELECT
    cohort,
    signal_a,
    signal_b,
    granger_f_a_to_b,
    granger_p_a_to_b,
    granger_f_b_to_a,
    granger_p_b_to_a,
    transfer_entropy_a_to_b,
    transfer_entropy_b_to_a,
    CASE
        WHEN granger_p_a_to_b < 0.05 AND granger_p_b_to_a < 0.05 THEN 'bidirectional'
        WHEN granger_p_a_to_b < 0.05 THEN 'a_drives_b'
        WHEN granger_p_b_to_a < 0.05 THEN 'b_drives_a'
        ELSE 'independent'
    END AS link_status,
    CASE
        WHEN granger_p_a_to_b < 0.01 OR granger_p_b_to_a < 0.01 THEN 'strong'
        WHEN granger_p_a_to_b < 0.05 OR granger_p_b_to_a < 0.05 THEN 'moderate'
        ELSE 'weak'
    END AS link_strength,
FROM information_flow;


-- ============================================================================
-- 010: VELOCITY FIELD (was Manifold stage 21)
-- ============================================================================
-- Derivatives of state_vector positions via LAG().
-- Simpler than the numpy version (no Savgol smoothing) but captures
-- the same information: speed, acceleration, curvature of the centroid.

CREATE OR REPLACE VIEW v_velocity_field AS
WITH lagged AS (
    SELECT
        I,
        cohort,
        mean_distance,
        max_distance,
        std_distance,
        -- Velocity: first difference of centroid distances
        mean_distance - LAG(mean_distance) OVER w AS v_mean_distance,
        max_distance - LAG(max_distance) OVER w AS v_max_distance,
        std_distance - LAG(std_distance) OVER w AS v_std_distance,
        -- Previous values for acceleration
        LAG(mean_distance, 1) OVER w AS prev1_mean,
        LAG(mean_distance, 2) OVER w AS prev2_mean,
    FROM state_vector
    WINDOW w AS (PARTITION BY cohort ORDER BY I)
)
SELECT
    I,
    cohort,
    -- Speed: magnitude of velocity vector
    SQRT(
        COALESCE(v_mean_distance * v_mean_distance, 0) +
        COALESCE(v_max_distance * v_max_distance, 0) +
        COALESCE(v_std_distance * v_std_distance, 0)
    ) AS speed,
    -- Acceleration: second difference
    CASE WHEN prev2_mean IS NOT NULL
        THEN mean_distance - 2 * prev1_mean + prev2_mean
        ELSE NULL
    END AS acceleration,
    v_mean_distance,
    v_max_distance,
    v_std_distance,
FROM lagged
WHERE v_mean_distance IS NOT NULL;


-- ============================================================================
-- 011: COHORT PAIRWISE (was Manifold stage 27)
-- ============================================================================
-- Pairwise distance, cosine similarity, correlation between cohort vectors
-- at each window I.

CREATE OR REPLACE VIEW v_cohort_pairwise AS
WITH cv AS (
    SELECT * FROM v_cohort_vector
)
SELECT
    a.I,
    a.cohort AS cohort_a,
    b.cohort AS cohort_b,
    -- Euclidean-style distance proxy: sum of squared differences of key metrics
    -- (using first engine's effective_dim and total_variance as representatives)
    SQRT(
        COALESCE(POWER(a.complexity_effective_dim - b.complexity_effective_dim, 2), 0) +
        COALESCE(POWER(a.complexity_total_variance - b.complexity_total_variance, 2), 0) +
        COALESCE(POWER(a.shape_effective_dim - b.shape_effective_dim, 2), 0) +
        COALESCE(POWER(a.shape_total_variance - b.shape_total_variance, 2), 0)
    ) AS distance,
FROM cv a
JOIN cv b ON a.I = b.I AND a.cohort < b.cohort;


-- ============================================================================
-- 012: COHORT TOPOLOGY (was Manifold stage 29)
-- ============================================================================
-- Graph stats from cohort pairwise correlation.
-- Uses v_cohort_pairwise distance with adaptive threshold.

CREATE OR REPLACE VIEW v_cohort_topology AS
WITH thresholds AS (
    SELECT
        I,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY distance) AS median_dist,
    FROM v_cohort_pairwise
    GROUP BY I
),
edges AS (
    SELECT
        cp.I,
        cp.cohort_a,
        cp.cohort_b,
    FROM v_cohort_pairwise cp
    JOIN thresholds t USING (I)
    WHERE cp.distance < t.median_dist
),
n_cohorts AS (
    SELECT I, COUNT(DISTINCT cohort) AS n_cohorts
    FROM v_cohort_vector
    GROUP BY I
)
SELECT
    nc.n_cohorts,
    COALESCE(COUNT(e.cohort_a), 0) AS n_edges,
    CASE WHEN nc.n_cohorts > 1
        THEN COALESCE(COUNT(e.cohort_a), 0)::DOUBLE / (nc.n_cohorts * (nc.n_cohorts - 1) / 2.0)
        ELSE 0.0
    END AS density,
    t.median_dist AS threshold,
    nc.I,
FROM n_cohorts nc
LEFT JOIN edges e USING (I)
LEFT JOIN thresholds t USING (I)
GROUP BY nc.I, nc.n_cohorts, t.median_dist;


-- ============================================================================
-- 013: COHORT VELOCITY FIELD (was Manifold stage 31)
-- ============================================================================
-- Derivatives of cohort_vector positions at cohort scale.
-- LAG() on the pivoted cohort feature vectors.

CREATE OR REPLACE VIEW v_cohort_velocity_field AS
WITH cv AS (
    SELECT * FROM v_cohort_vector
),
-- Use effective_dim and total_variance across engines as position coordinates
lagged AS (
    SELECT
        cohort,
        I,
        -- Velocity components (first difference)
        complexity_effective_dim - LAG(complexity_effective_dim) OVER w AS v_complexity_eff_dim,
        complexity_total_variance - LAG(complexity_total_variance) OVER w AS v_complexity_total_var,
        shape_effective_dim - LAG(shape_effective_dim) OVER w AS v_shape_eff_dim,
        shape_total_variance - LAG(shape_total_variance) OVER w AS v_shape_total_var,
    FROM cv
    WINDOW w AS (PARTITION BY cohort ORDER BY I)
)
SELECT
    cohort,
    I,
    -- Speed: magnitude of velocity vector
    SQRT(
        COALESCE(v_complexity_eff_dim * v_complexity_eff_dim, 0) +
        COALESCE(v_complexity_total_var * v_complexity_total_var, 0) +
        COALESCE(v_shape_eff_dim * v_shape_eff_dim, 0) +
        COALESCE(v_shape_total_var * v_shape_total_var, 0)
    ) AS speed,
    v_complexity_eff_dim,
    v_complexity_total_var,
    v_shape_eff_dim,
    v_shape_total_var,
FROM lagged
WHERE v_complexity_eff_dim IS NOT NULL;


-- ============================================================================
-- 014: COHORT FINGERPRINT (was Manifold stage 32)
-- ============================================================================
-- Aggregation of cohort_vector across windows per cohort.
-- Mean + std of each feature = static fingerprint per cohort.

CREATE OR REPLACE VIEW v_cohort_fingerprint AS
SELECT
    cohort,
    COUNT(*) AS n_windows,
    -- Complexity engine
    AVG(complexity_effective_dim) AS mean_complexity_effective_dim,
    AVG(complexity_total_variance) AS mean_complexity_total_variance,
    AVG(complexity_condition_number) AS mean_complexity_condition_number,
    AVG(complexity_eigenvalue_entropy_norm) AS mean_complexity_entropy_norm,
    STDDEV(complexity_effective_dim) AS std_complexity_effective_dim,
    STDDEV(complexity_total_variance) AS std_complexity_total_variance,
    STDDEV(complexity_condition_number) AS std_complexity_condition_number,
    STDDEV(complexity_eigenvalue_entropy_norm) AS std_complexity_entropy_norm,
    -- Shape engine
    AVG(shape_effective_dim) AS mean_shape_effective_dim,
    AVG(shape_total_variance) AS mean_shape_total_variance,
    AVG(shape_condition_number) AS mean_shape_condition_number,
    AVG(shape_eigenvalue_entropy_norm) AS mean_shape_entropy_norm,
    STDDEV(shape_effective_dim) AS std_shape_effective_dim,
    STDDEV(shape_total_variance) AS std_shape_total_variance,
    STDDEV(shape_condition_number) AS std_shape_condition_number,
    STDDEV(shape_eigenvalue_entropy_norm) AS std_shape_entropy_norm,
    -- Spectral engine
    AVG(spectral_effective_dim) AS mean_spectral_effective_dim,
    AVG(spectral_total_variance) AS mean_spectral_total_variance,
    AVG(spectral_condition_number) AS mean_spectral_condition_number,
    AVG(spectral_eigenvalue_entropy_norm) AS mean_spectral_entropy_norm,
    STDDEV(spectral_effective_dim) AS std_spectral_effective_dim,
    STDDEV(spectral_total_variance) AS std_spectral_total_variance,
    STDDEV(spectral_condition_number) AS std_spectral_condition_number,
    STDDEV(spectral_eigenvalue_entropy_norm) AS std_spectral_entropy_norm,
    -- Volatility: mean of all std features
    (COALESCE(STDDEV(complexity_effective_dim), 0) +
     COALESCE(STDDEV(shape_effective_dim), 0) +
     COALESCE(STDDEV(spectral_effective_dim), 0)) / 3.0 AS volatility,
FROM v_cohort_vector
GROUP BY cohort;


-- ============================================================================
-- 015: COHORT THERMODYNAMICS (was Manifold stage 09a)
-- ============================================================================
-- Shannon entropy and thermodynamic quantities from eigenvalues.
-- Entropy of eigenvalue distribution = information-theoretic temperature.

CREATE OR REPLACE VIEW v_cohort_thermodynamics AS
SELECT
    cohort,
    I,
    engine,
    -- Eigenvalue entropy (already in state_geometry)
    eigenvalue_entropy,
    eigenvalue_entropy_norm,
    -- Effective temperature: total_variance / effective_dim
    CASE WHEN effective_dim > 0
        THEN total_variance / effective_dim
        ELSE NULL
    END AS effective_temperature,
    -- Energy concentration: eigenvalue_1 / total_variance
    CASE WHEN total_variance > 0
        THEN eigenvalue_1 / total_variance
        ELSE NULL
    END AS energy_concentration,
    -- Degrees of freedom
    effective_dim,
    total_variance,
    -- Condition: ratio of largest to smallest significant eigenvalue
    condition_number,
FROM state_geometry;


-- ============================================================================
-- 016: CANARY SIGNALS (new Prime analysis)
-- ============================================================================
-- Identifies canary-in-the-coalmine signals: signals whose geometry
-- features correlate most strongly with lifecycle position.

CREATE OR REPLACE VIEW v_canary_signals AS
WITH lifecycle AS (
    SELECT
        cohort,
        MAX(I) AS max_I,
    FROM state_geometry
    WHERE engine = (SELECT MIN(engine) FROM state_geometry)
    GROUP BY cohort
),
signal_lifecycle AS (
    SELECT
        sv.signal_id,
        sv.cohort,
        sv.I,
        CAST(sv.I AS DOUBLE) / NULLIF(lc.max_I, 0) AS lifecycle_pct,
        sv.spectral_entropy,
        sv.hurst,
        sv.sample_entropy,
        sv.kurtosis,
    FROM signal_vector sv
    JOIN lifecycle lc USING (cohort)
    WHERE sv.spectral_entropy IS NOT NULL
      AND NOT isnan(sv.spectral_entropy)
      AND sv.hurst IS NOT NULL
      AND NOT isnan(sv.hurst)
),
-- Pre-filter: only signals with sufficient variance (avoids CORR overflow)
varying_signals AS (
    SELECT signal_id
    FROM signal_lifecycle
    WHERE spectral_entropy IS NOT NULL AND lifecycle_pct IS NOT NULL
    GROUP BY signal_id
    HAVING COUNT(*) > 10
       AND MAX(spectral_entropy) > MIN(spectral_entropy)
       AND MAX(lifecycle_pct) > MIN(lifecycle_pct)
),
entropy_corr AS (
    SELECT
        sl.signal_id,
        'spectral_entropy' AS feature,
        CORR(sl.spectral_entropy, sl.lifecycle_pct) AS r_with_lifecycle,
        COUNT(DISTINCT sl.cohort) AS n_cohorts,
    FROM signal_lifecycle sl
    JOIN varying_signals vs USING (signal_id)
    WHERE sl.spectral_entropy IS NOT NULL AND sl.lifecycle_pct IS NOT NULL
    GROUP BY sl.signal_id
),
hurst_varying AS (
    SELECT signal_id
    FROM signal_lifecycle
    WHERE hurst IS NOT NULL AND lifecycle_pct IS NOT NULL
    GROUP BY signal_id
    HAVING COUNT(*) > 10
       AND MAX(hurst) > MIN(hurst)
       AND MAX(lifecycle_pct) > MIN(lifecycle_pct)
),
hurst_corr AS (
    SELECT
        sl.signal_id,
        'hurst' AS feature,
        CORR(sl.hurst, sl.lifecycle_pct) AS r_with_lifecycle,
        COUNT(DISTINCT sl.cohort) AS n_cohorts,
    FROM signal_lifecycle sl
    JOIN hurst_varying hv USING (signal_id)
    WHERE sl.hurst IS NOT NULL AND sl.lifecycle_pct IS NOT NULL
    GROUP BY sl.signal_id
),
all_features AS (
    SELECT signal_id, feature, r_with_lifecycle,
           r_with_lifecycle * r_with_lifecycle AS r_squared, n_cohorts
    FROM entropy_corr
    WHERE r_with_lifecycle IS NOT NULL
    UNION ALL
    SELECT signal_id, feature, r_with_lifecycle,
           r_with_lifecycle * r_with_lifecycle AS r_squared, n_cohorts
    FROM hurst_corr
    WHERE r_with_lifecycle IS NOT NULL
)
SELECT * FROM all_features
ORDER BY ABS(r_with_lifecycle) DESC;


-- ============================================================================
-- 017: ML FEATURE ASSEMBLY (new Prime analysis)
-- ============================================================================
-- Assembles ML feature matrix from Manifold outputs.
-- One row per (cohort, I) with all features joined.
-- This replaces build_ml_features.py for the windowed approach.

CREATE OR REPLACE VIEW v_ml_features AS
WITH cv AS (
    SELECT * FROM v_cohort_vector
),
gd AS (
    SELECT * FROM v_geometry_dynamics
),
sv AS (
    SELECT
        cohort, I,
        mean_distance AS sv_mean_distance,
        max_distance AS sv_max_distance,
        std_distance AS sv_std_distance,
        state_shape_kurtosis AS sv_kurtosis,
        state_shape_skewness AS sv_skewness,
        state_complexity_permutation_entropy AS sv_perm_entropy,
        state_complexity_hurst AS sv_hurst,
        state_spectral_spectral_entropy AS sv_spectral_entropy,
    FROM state_vector
),
tp AS (
    SELECT
        cohort, I,
        n_edges AS tp_n_edges,
        density AS tp_density,
        mean_degree AS tp_mean_degree,
        max_degree AS tp_max_degree,
    FROM v_topology
),
lifecycle AS (
    SELECT cohort, MAX(I) AS max_I
    FROM v_cohort_vector
    GROUP BY cohort
)
SELECT
    cv.*,
    gd.effective_dim_velocity,
    gd.effective_dim_acceleration,
    gd.effective_dim_jerk,
    gd.effective_dim_curvature,
    gd.eigenvalue_1_velocity,
    gd.variance_velocity,
    gd.collapse_onset_fraction,
    sv.sv_mean_distance,
    sv.sv_max_distance,
    sv.sv_std_distance,
    sv.sv_kurtosis,
    sv.sv_skewness,
    sv.sv_perm_entropy,
    sv.sv_hurst,
    sv.sv_spectral_entropy,
    tp.tp_n_edges,
    tp.tp_density,
    tp.tp_mean_degree,
    tp.tp_max_degree,
    -- RUL target
    lc.max_I - cv.I AS RUL,
    lc.max_I + 1 AS lifecycle,
    CAST(cv.I AS DOUBLE) / NULLIF(lc.max_I, 0) AS lifecycle_pct,
FROM cv
LEFT JOIN gd ON cv.cohort = gd.cohort AND cv.I = gd.I
    AND gd.engine = (SELECT MIN(engine) FROM state_geometry)
LEFT JOIN sv ON cv.cohort = sv.cohort AND cv.I = sv.I
LEFT JOIN tp ON cv.cohort = tp.cohort AND cv.I = tp.I
LEFT JOIN lifecycle lc ON cv.cohort = lc.cohort;
