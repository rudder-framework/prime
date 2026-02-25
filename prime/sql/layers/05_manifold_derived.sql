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
-- Pure LAG() on state_geometry columns, partitioned by cohort.
--
-- Output schema matches geometry_dynamics.parquet:
--   signal_0_center, cohort, effective_dim, effective_dim_velocity,
--   effective_dim_acceleration, effective_dim_jerk, effective_dim_curvature,
--   eigenvalue_0, eigenvalue_0_velocity, total_variance, variance_velocity,
--   collapse_onset_idx, collapse_onset_fraction

CREATE OR REPLACE VIEW v_geometry_dynamics AS
WITH lagged AS (
    SELECT
        signal_0_center,
        cohort,
        effective_dim,
        eigenvalue_0,
        total_variance,
        condition_number,
        eigenvalue_entropy_normalized AS eigenvalue_entropy,
        -- Ratio and spectral gap from D0
        CASE WHEN eigenvalue_0 IS NOT NULL AND eigenvalue_1 IS NOT NULL AND eigenvalue_0 > 0
            THEN eigenvalue_1 / eigenvalue_0 ELSE NULL END AS ratio_2_1,
        CASE WHEN eigenvalue_0 IS NOT NULL AND total_variance IS NOT NULL AND total_variance > 0
            THEN eigenvalue_0 / total_variance ELSE NULL END AS spectral_gap,
        eigenvalue_1,
        -- d1: velocity (first difference)
        effective_dim - LAG(effective_dim) OVER w AS effective_dim_velocity,
        eigenvalue_0 - LAG(eigenvalue_0) OVER w AS eigenvalue_0_velocity,
        eigenvalue_1 - LAG(eigenvalue_1) OVER w AS eigenvalue_1_velocity,
        total_variance - LAG(total_variance) OVER w AS variance_velocity,
        condition_number - LAG(condition_number) OVER w AS condition_number_velocity,
        -- For d2/d3 we need nested LAG
        LAG(effective_dim, 1) OVER w AS eff_dim_prev1,
        LAG(effective_dim, 2) OVER w AS eff_dim_prev2,
        LAG(eigenvalue_1, 1) OVER w AS eig1_prev1,
        LAG(eigenvalue_1, 2) OVER w AS eig1_prev2,
        LAG(condition_number, 1) OVER w AS cond_prev1,
        LAG(condition_number, 2) OVER w AS cond_prev2,
        LAG(total_variance, 1) OVER w AS tvar_prev1,
        LAG(total_variance, 2) OVER w AS tvar_prev2,
    FROM state_geometry
    WINDOW w AS (PARTITION BY cohort ORDER BY signal_0_center)
),
with_accel AS (
    SELECT
        *,
        -- d2: acceleration (second difference)
        effective_dim_velocity - (eff_dim_prev1 - eff_dim_prev2) AS effective_dim_acceleration,
        CASE WHEN eig1_prev2 IS NOT NULL
            THEN eigenvalue_1_velocity - (eig1_prev1 - eig1_prev2) ELSE NULL
        END AS eigenvalue_1_acceleration,
        CASE WHEN cond_prev2 IS NOT NULL
            THEN condition_number_velocity - (cond_prev1 - cond_prev2) ELSE NULL
        END AS condition_number_acceleration,
        CASE WHEN tvar_prev2 IS NOT NULL
            THEN variance_velocity - (tvar_prev1 - tvar_prev2) ELSE NULL
        END AS total_variance_acceleration,
    FROM lagged
),
with_jerk AS (
    SELECT
        signal_0_center,
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
        eigenvalue_0,
        eigenvalue_0_velocity,
        eigenvalue_1,
        eigenvalue_1_velocity,
        eigenvalue_1_acceleration,
        eigenvalue_1_acceleration - LAG(eigenvalue_1_acceleration) OVER w AS eigenvalue_1_jerk,
        CASE WHEN eigenvalue_1_velocity IS NOT NULL
            THEN ABS(eigenvalue_1_acceleration) / POWER(1.0 + eigenvalue_1_velocity * eigenvalue_1_velocity, 1.5)
            ELSE NULL
        END AS eigenvalue_1_curvature,
        total_variance,
        variance_velocity,
        total_variance_acceleration,
        condition_number,
        condition_number_velocity,
        condition_number_acceleration,
        condition_number_acceleration - LAG(condition_number_acceleration) OVER w AS condition_number_jerk,
        CASE WHEN condition_number_velocity IS NOT NULL
            THEN ABS(condition_number_acceleration) / POWER(1.0 + condition_number_velocity * condition_number_velocity, 1.5)
            ELSE NULL
        END AS condition_number_curvature,
        ratio_2_1,
        spectral_gap,
        eigenvalue_entropy,
    FROM with_accel
    WINDOW w AS (PARTITION BY cohort ORDER BY signal_0_center)
),
-- Collapse onset: first I where effective_dim_velocity < -0.1 sustained
collapse AS (
    SELECT
        cohort,
        MIN(signal_0_center) AS collapse_onset_idx,
    FROM with_jerk
    WHERE effective_dim_velocity < -0.1
    GROUP BY cohort
),
lifecycle AS (
    SELECT cohort, MAX(signal_0_center) AS max_I
    FROM state_geometry
    GROUP BY cohort
)
SELECT
    j.signal_0_center,
    j.cohort,
    j.effective_dim,
    j.effective_dim_velocity,
    j.effective_dim_acceleration,
    j.effective_dim_jerk,
    j.effective_dim_curvature,
    j.eigenvalue_0,
    j.eigenvalue_0_velocity,
    j.eigenvalue_1,
    j.eigenvalue_1_velocity,
    j.eigenvalue_1_acceleration,
    j.eigenvalue_1_jerk,
    j.eigenvalue_1_curvature,
    j.total_variance,
    j.variance_velocity,
    j.total_variance_acceleration,
    j.condition_number,
    j.condition_number_velocity,
    j.condition_number_acceleration,
    j.condition_number_jerk,
    j.condition_number_curvature,
    j.ratio_2_1,
    j.spectral_gap,
    j.eigenvalue_entropy,
    c.collapse_onset_idx,
    CASE WHEN c.collapse_onset_idx IS NOT NULL AND lc.max_I > 0
        THEN CAST(c.collapse_onset_idx AS DOUBLE) / lc.max_I
        ELSE NULL
    END AS collapse_onset_fraction,
FROM with_jerk j
LEFT JOIN collapse c USING (cohort)
LEFT JOIN lifecycle lc ON j.cohort = lc.cohort;


-- ============================================================================
-- 002: COHORT VECTOR (was Manifold stage 25)
-- ============================================================================
-- TODO: This view pivoted state_geometry rows by engine (complexity/shape/spectral)
-- into wide cohort feature vectors. The engine column no longer exists in
-- cohort_geometry.parquet — there is one geometry row per (cohort, window).
-- This view needs redesigning to work with the single-geometry-per-window schema.
-- Dependents: v_cohort_pairwise, v_cohort_topology, v_cohort_velocity_field,
--             v_cohort_fingerprint, v_ml_features
-- ============================================================================


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
        sp.signal_0_end,
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
        signal_0_end,
        COUNT(*) AS n_edges,
        threshold,
    FROM edges
    GROUP BY cohort, signal_0_end, threshold
),
signal_counts AS (
    SELECT cohort, signal_0_end, COUNT(DISTINCT signal_id) AS n_signals
    FROM (
        SELECT cohort, signal_0_end, signal_a AS signal_id FROM signal_pairwise
        UNION
        SELECT cohort, signal_0_end, signal_b AS signal_id FROM signal_pairwise
    )
    GROUP BY cohort, signal_0_end
),
degree_stats AS (
    SELECT
        cohort,
        signal_0_end,
        signal_id,
        COUNT(*) AS degree,
    FROM (
        SELECT cohort, signal_0_end, signal_a AS signal_id FROM edges
        UNION ALL
        SELECT cohort, signal_0_end, signal_b AS signal_id FROM edges
    )
    GROUP BY cohort, signal_0_end, signal_id
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
    sc.signal_0_end,
FROM signal_counts sc
LEFT JOIN edge_counts ec USING (cohort, signal_0_end)
LEFT JOIN degree_stats ds USING (cohort, signal_0_end)
GROUP BY sc.cohort, sc.signal_0_end, sc.n_signals, ec.n_edges, ec.threshold;


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
-- 005: FEATURE STATS REFERENCE (was Manifold stage 12)
-- ============================================================================
-- Normalization parameters: percentile bounds per numeric column.
-- Used for range-based normalization in downstream analysis.

CREATE OR REPLACE VIEW v_feature_stats_ref AS
SELECT
    'signal_vector' AS source,
    'spectral_entropy' AS column_name,
    AVG(spectral_entropy) AS mean,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY spectral_entropy) AS p05,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY spectral_entropy) AS p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY spectral_entropy) AS median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY spectral_entropy) AS p75,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY spectral_entropy) AS p95,
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
JOIN signal_vector b ON a.cohort = b.cohort AND a.signal_0_center = b.signal_0_center AND a.signal_id < b.signal_id
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
        MIN(signal_0_center) AS first_break_I,
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
        MIN(signal_0_center) AS I_min,
        MAX(signal_0_center) AS I_max,
        (MAX(signal_0_center) - MIN(signal_0_center)) AS span,
    FROM state_geometry
    GROUP BY cohort
),
early AS (
    SELECT
        sg.cohort,
        'early' AS segment,
        AVG(sg.effective_dim) AS effective_dim,
        AVG(sg.total_variance) AS total_variance,
        AVG(sg.eigenvalue_0) AS eigenvalue_0,
        AVG(sg.condition_number) AS condition_number,
        AVG(sg.eigenvalue_entropy_normalized) AS eigenvalue_entropy_normalized,
    FROM state_geometry sg
    JOIN lifecycle lc ON sg.cohort = lc.cohort
    WHERE sg.signal_0_center <= lc.I_min + lc.span * 0.25
    GROUP BY sg.cohort
),
late AS (
    SELECT
        sg.cohort,
        'late' AS segment,
        AVG(sg.effective_dim) AS effective_dim,
        AVG(sg.total_variance) AS total_variance,
        AVG(sg.eigenvalue_0) AS eigenvalue_0,
        AVG(sg.condition_number) AS condition_number,
        AVG(sg.eigenvalue_entropy_normalized) AS eigenvalue_entropy_normalized,
    FROM state_geometry sg
    JOIN lifecycle lc ON sg.cohort = lc.cohort
    WHERE sg.signal_0_center >= lc.I_max - lc.span * 0.25
    GROUP BY sg.cohort
)
SELECT
    e.cohort,
    l.effective_dim - e.effective_dim AS delta_effective_dim,
    l.total_variance - e.total_variance AS delta_total_variance,
    l.eigenvalue_0 - e.eigenvalue_0 AS delta_eigenvalue_0,
    l.condition_number - e.condition_number AS delta_condition_number,
    l.eigenvalue_entropy_normalized - e.eigenvalue_entropy_normalized AS delta_entropy,
    e.effective_dim AS early_effective_dim,
    l.effective_dim AS late_effective_dim,
    e.total_variance AS early_total_variance,
    l.total_variance AS late_total_variance,
FROM early e
JOIN late l USING (cohort);


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
        signal_0_center,
        cohort,
        dispersion_mean,
        dispersion_max,
        dispersion_std,
        -- Velocity: first difference of centroid distances
        dispersion_mean - LAG(dispersion_mean) OVER w AS v_mean_distance,
        dispersion_max - LAG(dispersion_max) OVER w AS v_max_distance,
        dispersion_std - LAG(dispersion_std) OVER w AS v_std_distance,
        -- Previous values for acceleration
        LAG(dispersion_mean, 1) OVER w AS prev1_mean,
        LAG(dispersion_mean, 2) OVER w AS prev2_mean,
    FROM state_vector
    WINDOW w AS (PARTITION BY cohort ORDER BY signal_0_center)
)
SELECT
    signal_0_center,
    cohort,
    -- Speed: magnitude of velocity vector
    SQRT(
        COALESCE(v_mean_distance * v_mean_distance, 0) +
        COALESCE(v_max_distance * v_max_distance, 0) +
        COALESCE(v_std_distance * v_std_distance, 0)
    ) AS speed,
    -- Acceleration: second difference
    CASE WHEN prev2_mean IS NOT NULL
        THEN dispersion_mean - 2 * prev1_mean + prev2_mean
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
-- TODO: Depends on v_cohort_vector which is disabled (engine column removed).
-- Needs redesign to compute pairwise distances from single-geometry schema.
-- ============================================================================


-- ============================================================================
-- 012: COHORT TOPOLOGY (was Manifold stage 29)
-- ============================================================================
-- TODO: Depends on v_cohort_pairwise / v_cohort_vector (disabled).
-- Needs redesign for single-geometry schema.
-- ============================================================================


-- ============================================================================
-- 013: COHORT VELOCITY FIELD (was Manifold stage 31)
-- ============================================================================
-- TODO: Depends on v_cohort_vector (disabled).
-- Needs redesign: compute velocity from single-geometry schema columns
-- (effective_dim, total_variance) directly, without engine pivoting.
-- ============================================================================


-- ============================================================================
-- 014: COHORT FINGERPRINT (was Manifold stage 32)
-- ============================================================================
-- TODO: Depends on v_cohort_vector (disabled).
-- Needs redesign: aggregate geometry metrics directly from state_geometry
-- instead of from engine-pivoted cohort vector.
-- ============================================================================


-- ============================================================================
-- 015: COHORT THERMODYNAMICS (was Manifold stage 09a)
-- ============================================================================
-- Shannon entropy and thermodynamic quantities from eigenvalues.
-- Entropy of eigenvalue distribution = information-theoretic temperature.

CREATE OR REPLACE VIEW v_cohort_thermodynamics AS
SELECT
    cohort,
    signal_0_center,
    -- Eigenvalue entropy (already in state_geometry)
    eigenvalue_entropy,
    eigenvalue_entropy_normalized,
    -- Effective temperature: total_variance / effective_dim
    CASE WHEN effective_dim > 0
        THEN total_variance / effective_dim
        ELSE NULL
    END AS effective_temperature,
    -- Energy concentration: eigenvalue_0 / total_variance
    CASE WHEN total_variance > 0
        THEN eigenvalue_0 / total_variance
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
        MAX(signal_0_center) AS max_I,
    FROM state_geometry
    GROUP BY cohort
),
signal_lifecycle AS (
    SELECT
        sv.signal_id,
        sv.cohort,
        sv.signal_0_center,
        CAST(sv.signal_0_center AS DOUBLE) / NULLIF(lc.max_I, 0) AS lifecycle_pct,
        sv.spectral_entropy,
        sv.hurst_exponent,
        sv.complexity_sample_entropy,
        sv.statistics_kurtosis,
    FROM signal_vector sv
    JOIN lifecycle lc USING (cohort)
    WHERE sv.spectral_entropy IS NOT NULL
      AND NOT isnan(sv.spectral_entropy)
      AND sv.hurst_exponent IS NOT NULL
      AND NOT isnan(sv.hurst_exponent)
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
    WHERE hurst_exponent IS NOT NULL AND lifecycle_pct IS NOT NULL
    Group BY signal_id
    HAVING COUNT(*) > 10
       AND MAX(hurst_exponent) > MIN(hurst_exponent)
       AND MAX(lifecycle_pct) > MIN(lifecycle_pct)
),
hurst_corr AS (
    SELECT
        sl.signal_id,
        'hurst_exponent' AS feature,
        CORR(sl.hurst_exponent, sl.lifecycle_pct) AS r_with_lifecycle,
        COUNT(DISTINCT sl.cohort) AS n_cohorts,
    FROM signal_lifecycle sl
    JOIN hurst_varying hv USING (signal_id)
    WHERE sl.hurst_exponent IS NOT NULL AND sl.lifecycle_pct IS NOT NULL
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
-- TODO: Depends on v_cohort_vector (disabled) and used engine filter on
-- v_geometry_dynamics. Needs redesign to join geometry, state_vector,
-- and topology directly without engine pivoting.
-- ============================================================================
