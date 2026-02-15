-- ============================================================
-- Rudder Classification SQL — Ranked Views
-- ============================================================
-- Every metric gets a rank per cohort per time window.
-- No gates, no filtering, just ordered lists.
--
-- The analyst queries WHERE rank = 1 to see what's most extreme.
-- The threshold is a query parameter, not baked into the view.
--
-- Source tables (from manifold parquet output):
--   ftle_rolling    — per-signal FTLE (Lyapunov) over rolling windows
--   geometry_dynamics — per-engine effective dim derivatives
--   signal_vector   — per-signal statistical features
--   signal_pairwise — pairwise correlation/distance per window
-- ============================================================

-- ------------------------------------------------------------
-- TRAJECTORY RANKED
-- Ranks signals by FTLE (Lyapunov exponent) magnitude
-- Higher |FTLE| = more dynamically interesting
-- Source: ftle_rolling.parquet (per-signal rolling Lyapunov)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_trajectory_ranked AS
SELECT
    I,
    signal_id,
    cohort,

    -- Raw computed values
    ftle AS lyapunov_max,
    ftle_velocity,
    ftle_acceleration,
    confidence,

    ABS(ftle) AS lyapunov_magnitude,

    -- Rank by Lyapunov magnitude within cohort at each timestep
    RANK() OVER (
        PARTITION BY cohort, I
        ORDER BY ABS(ftle) DESC NULLS LAST
    ) AS lyapunov_rank,

    -- Fleet-wide rank at each timestep
    RANK() OVER (
        PARTITION BY I
        ORDER BY ABS(ftle) DESC NULLS LAST
    ) AS fleet_lyapunov_rank,

    -- Percentile within cohort history
    PERCENT_RANK() OVER (
        PARTITION BY cohort
        ORDER BY ABS(ftle) NULLS FIRST
    ) AS lyapunov_percentile,

    -- Classification confidence based on data quality
    CASE
        WHEN confidence > 0.8 THEN 'high'
        WHEN confidence > 0.5 THEN 'medium'
        ELSE 'low'
    END AS classification_confidence

FROM ftle_rolling
WHERE direction = 'forward';


-- ------------------------------------------------------------
-- STABILITY RANKED
-- Ranks signals by FTLE (positive = unstable)
-- Source: ftle_rolling.parquet
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_stability_ranked AS
SELECT
    I,
    signal_id,
    cohort,
    ftle AS lyapunov_max,

    -- Numeric stability score (-1 to +1, negative is stable)
    CASE
        WHEN ftle IS NULL THEN 0
        ELSE LEAST(1.0, GREATEST(-1.0, ftle * 10))
    END AS stability_score,

    -- Rank by instability (most positive FTLE first)
    RANK() OVER (
        PARTITION BY cohort
        ORDER BY ftle DESC NULLS LAST
    ) AS instability_rank,

    -- Fleet-wide instability rank
    RANK() OVER (
        ORDER BY ftle DESC NULLS LAST
    ) AS fleet_instability_rank,

    -- Percentile within cohort (how unusual is this stability level)
    PERCENT_RANK() OVER (
        PARTITION BY cohort
        ORDER BY ftle NULLS FIRST
    ) AS instability_percentile

FROM ftle_rolling
WHERE direction = 'forward';


-- ------------------------------------------------------------
-- GEOMETRY RANKED (sliding-window derivative analysis)
-- From geometry_dynamics.parquet (per-engine windowed derivatives)
-- Note: geometry_dynamics is per-engine, not per-signal
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_geometry_ranked AS
SELECT
    I,
    engine,
    cohort,
    effective_dim,
    effective_dim_velocity,
    collapse_onset_idx,
    collapse_onset_fraction,

    ABS(effective_dim_velocity) AS velocity_magnitude,

    -- Rank by velocity magnitude within each cohort at each time window
    RANK() OVER (
        PARTITION BY cohort, I
        ORDER BY ABS(effective_dim_velocity) DESC
    ) AS velocity_rank,

    -- Rank across all cohorts at each time window (fleet-wide)
    RANK() OVER (
        PARTITION BY I
        ORDER BY ABS(effective_dim_velocity) DESC
    ) AS fleet_velocity_rank,

    -- Percentile within cohort history (how unusual is this for THIS engine)
    PERCENT_RANK() OVER (
        PARTITION BY cohort, engine
        ORDER BY ABS(effective_dim_velocity)
    ) AS velocity_percentile,

    -- Collapse onset rank: rank cohorts by how early collapse was detected
    RANK() OVER (
        ORDER BY collapse_onset_fraction ASC NULLS LAST
    ) AS collapse_onset_rank,

    -- Time remaining estimate (as fraction of lifecycle)
    CASE
        WHEN collapse_onset_fraction IS NULL THEN NULL
        ELSE 1.0 - collapse_onset_fraction
    END AS remaining_fraction

FROM geometry_dynamics
WHERE effective_dim IS NOT NULL
  AND NOT isnan(effective_dim);


-- ------------------------------------------------------------
-- SIGNAL RANKED
-- Ranks each signal_vector metric within the cohort
-- Source: signal_vector.parquet
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_signal_ranked AS
SELECT
    signal_id,
    I,
    cohort,
    spectral_entropy,
    kurtosis,
    skewness,
    hurst,
    acf_lag1,
    sample_entropy,
    permutation_entropy,
    crest_factor,

    -- Within-cohort ranks
    RANK() OVER (PARTITION BY cohort ORDER BY spectral_entropy DESC) AS entropy_rank,
    RANK() OVER (PARTITION BY cohort ORDER BY kurtosis DESC) AS kurtosis_rank,
    RANK() OVER (PARTITION BY cohort ORDER BY ABS(skewness) DESC) AS skewness_rank,
    RANK() OVER (PARTITION BY cohort ORDER BY hurst DESC) AS hurst_rank,
    RANK() OVER (PARTITION BY cohort ORDER BY acf_lag1 DESC) AS memory_rank,

    -- Fleet-wide ranks
    RANK() OVER (ORDER BY spectral_entropy DESC) AS fleet_entropy_rank,
    RANK() OVER (ORDER BY kurtosis DESC) AS fleet_kurtosis_rank,
    RANK() OVER (ORDER BY hurst DESC) AS fleet_hurst_rank,

    -- Percentiles within cohort
    PERCENT_RANK() OVER (PARTITION BY cohort ORDER BY spectral_entropy) AS entropy_percentile,
    PERCENT_RANK() OVER (PARTITION BY cohort ORDER BY kurtosis) AS kurtosis_percentile,
    PERCENT_RANK() OVER (PARTITION BY cohort ORDER BY hurst) AS hurst_percentile

FROM signal_vector
WHERE spectral_entropy IS NOT NULL AND NOT isnan(spectral_entropy);


-- ------------------------------------------------------------
-- ANOMALY RANKED
-- Ranks signal_vector features by deviation from per-signal distribution
-- Uses PERCENT_RANK — no z-scores, no Gaussian assumption
-- Source: signal_vector.parquet
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_anomaly_ranked AS
WITH ranked_signals AS (
    SELECT
        sv.signal_id,
        sv.cohort,
        sv.I,
        sv.spectral_entropy AS value,

        -- Percentile within this signal's history
        PERCENT_RANK() OVER (
            PARTITION BY sv.cohort, sv.signal_id
            ORDER BY sv.spectral_entropy
        ) AS signal_percentile

    FROM signal_vector sv
    WHERE sv.spectral_entropy IS NOT NULL
      AND NOT isnan(sv.spectral_entropy)
)
SELECT
    I,
    signal_id,
    cohort,
    value,

    -- Distance from median in percentile space (0.5 = median, 0/1 = extremes)
    ABS(signal_percentile - 0.5) * 2.0 AS deviation_magnitude,

    signal_percentile,

    -- Rank within this timestep (what's deviating most right now)
    RANK() OVER (
        PARTITION BY cohort, I
        ORDER BY ABS(signal_percentile - 0.5) DESC
    ) AS deviation_rank,

    -- Fleet rank at this timestep
    RANK() OVER (
        PARTITION BY I
        ORDER BY ABS(signal_percentile - 0.5) DESC
    ) AS fleet_deviation_rank

FROM ranked_signals;


-- ------------------------------------------------------------
-- COUPLING RANKED
-- Ranks signal pairs by correlation magnitude
-- signal_pairwise has multiple engine rows per pair per window;
-- use engine='shape' for correlation (others duplicate it)
-- Source: signal_pairwise.parquet
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_coupling_ranked AS
WITH base AS (
    SELECT
        I,
        signal_a,
        signal_b,
        cohort,
        correlation,
        distance,
        cosine_similarity,
        ABS(correlation) AS coupling_magnitude,
        ABS(correlation - LAG(correlation) OVER (
            PARTITION BY cohort, signal_a, signal_b ORDER BY I
        )) AS coupling_delta
    FROM signal_pairwise
    WHERE engine = 'shape'
)
SELECT
    I,
    signal_a,
    signal_b,
    cohort,
    correlation,
    distance,
    cosine_similarity,
    coupling_magnitude,
    coupling_delta,

    -- Rank pairs by coupling strength at each window
    RANK() OVER (
        PARTITION BY cohort, I
        ORDER BY coupling_magnitude DESC
    ) AS coupling_rank,

    -- Rank by how much coupling changed from previous window
    RANK() OVER (
        PARTITION BY cohort, I
        ORDER BY coupling_delta DESC NULLS LAST
    ) AS decoupling_rank,

    -- Fleet-wide coupling rank at each window
    RANK() OVER (
        PARTITION BY I
        ORDER BY coupling_magnitude DESC
    ) AS fleet_coupling_rank,

    -- Percentile within this pair's history
    PERCENT_RANK() OVER (
        PARTITION BY cohort, signal_a, signal_b
        ORDER BY coupling_magnitude
    ) AS coupling_percentile

FROM base;


-- ------------------------------------------------------------
-- UNIFIED HEALTH VIEW (ranked, no categorical gates)
-- Combines FTLE stability + geometry velocity into composite
-- Source: ftle_rolling + geometry_dynamics (aggregated to cohort level)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_system_health AS
WITH ftle_agg AS (
    SELECT
        cohort,
        I,
        AVG(ftle) AS mean_ftle,
        MAX(ftle) AS max_ftle,
        MAX(ABS(ftle)) AS max_lyapunov_magnitude,
        AVG(CASE WHEN ftle IS NULL THEN 0
            ELSE LEAST(1.0, GREATEST(-1.0, ftle * 10))
        END) AS stability_score
    FROM ftle_rolling
    WHERE direction = 'forward'
    GROUP BY cohort, I
),
geo AS (
    SELECT
        cohort,
        I,
        effective_dim,
        effective_dim_velocity,
        collapse_onset_fraction
    FROM geometry_dynamics
    WHERE engine = (SELECT MIN(engine) FROM geometry_dynamics)
),
combined AS (
    SELECT
        geo.I,
        geo.cohort,
        geo.effective_dim,
        geo.effective_dim_velocity,
        fa.mean_ftle AS mean_lyapunov,
        fa.max_ftle AS max_lyapunov,
        fa.max_lyapunov_magnitude,
        fa.stability_score,
        geo.collapse_onset_fraction,
        PERCENT_RANK() OVER (
            PARTITION BY geo.cohort
            ORDER BY ABS(geo.effective_dim_velocity)
        ) AS velocity_percentile,
        PERCENT_RANK() OVER (
            PARTITION BY geo.cohort
            ORDER BY fa.max_lyapunov_magnitude NULLS FIRST
        ) AS lyapunov_percentile
    FROM geo
    LEFT JOIN ftle_agg fa ON geo.cohort = fa.cohort AND geo.I = fa.I
)
SELECT
    *,
    (COALESCE(lyapunov_percentile, 0) + COALESCE(velocity_percentile, 0)) / 2.0 AS composite_instability,
    RANK() OVER (
        ORDER BY (COALESCE(lyapunov_percentile, 0) + COALESCE(velocity_percentile, 0)) DESC
    ) AS composite_rank
FROM combined;


-- ------------------------------------------------------------
-- SUMMARY REPORT VIEW
-- Aggregates rankings across all cohorts
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_health_summary AS
SELECT
    cohort,

    COUNT(*) AS total_observations,

    -- Rank distribution summaries
    AVG(composite_instability) AS mean_composite_instability,
    MAX(composite_instability) AS max_composite_instability,

    -- Count of high-percentile observations (top 5%)
    SUM(CASE WHEN composite_instability > 0.95 THEN 1 ELSE 0 END) AS n_extreme,
    SUM(CASE WHEN composite_instability > 0.90 THEN 1 ELSE 0 END) AS n_high,

    -- Stability score stats
    AVG(stability_score) AS mean_stability_score,
    MIN(stability_score) AS min_stability_score,

    -- Worst ranks (lowest rank number = most extreme)
    MIN(composite_rank) AS best_composite_rank

FROM v_system_health
GROUP BY cohort;
