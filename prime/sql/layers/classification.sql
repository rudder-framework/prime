-- ============================================================
-- Rudder Classification SQL — Ranked Views
-- ============================================================
-- Every metric gets a rank per cohort per time window.
-- No gates, no filtering, just ordered lists.
--
-- The analyst queries WHERE rank = 1 to see what's most extreme.
-- The threshold is a query parameter, not baked into the view.
-- ============================================================

-- ------------------------------------------------------------
-- TRAJECTORY RANKED
-- Ranks signals by Lyapunov exponent magnitude
-- Higher |λ| = more dynamically interesting
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_trajectory_ranked AS
SELECT
    gd.I,
    gd.signal_id,
    gd.cohort,

    -- Raw computed values from Engines
    d.lyapunov_max,
    gd.effective_dim_velocity,
    gd.effective_dim_acceleration,

    ABS(d.lyapunov_max) AS lyapunov_magnitude,

    -- Rank by Lyapunov magnitude within cohort at each timestep
    RANK() OVER (
        PARTITION BY gd.cohort, gd.I
        ORDER BY ABS(d.lyapunov_max) DESC NULLS LAST
    ) AS lyapunov_rank,

    -- Fleet-wide rank at each timestep
    RANK() OVER (
        PARTITION BY gd.I
        ORDER BY ABS(d.lyapunov_max) DESC NULLS LAST
    ) AS fleet_lyapunov_rank,

    -- Percentile within cohort history
    PERCENT_RANK() OVER (
        PARTITION BY gd.cohort
        ORDER BY ABS(d.lyapunov_max) NULLS FIRST
    ) AS lyapunov_percentile,

    -- Classification confidence based on data quality
    CASE
        WHEN d.lyapunov_max IS NOT NULL THEN 'high'
        WHEN gd.effective_dim IS NOT NULL THEN 'medium'
        ELSE 'low'
    END AS classification_confidence

FROM geometry_dynamics gd
LEFT JOIN dynamics d ON gd.I = d.I AND gd.signal_id = d.signal_id;


-- ------------------------------------------------------------
-- STABILITY RANKED
-- Ranks signals by Lyapunov exponent (positive = unstable)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_stability_ranked AS
SELECT
    I,
    signal_id,
    cohort,
    lyapunov_max,

    -- Numeric stability score (-1 to +1, negative is stable)
    CASE
        WHEN lyapunov_max IS NULL THEN 0
        ELSE LEAST(1.0, GREATEST(-1.0, lyapunov_max * 10))
    END AS stability_score,

    -- Rank by instability (most positive Lyapunov first)
    RANK() OVER (
        PARTITION BY cohort
        ORDER BY lyapunov_max DESC NULLS LAST
    ) AS instability_rank,

    -- Fleet-wide instability rank
    RANK() OVER (
        ORDER BY lyapunov_max DESC NULLS LAST
    ) AS fleet_instability_rank,

    -- Percentile within cohort (how unusual is this stability level)
    PERCENT_RANK() OVER (
        PARTITION BY cohort
        ORDER BY lyapunov_max NULLS FIRST
    ) AS instability_percentile

FROM dynamics;


-- ------------------------------------------------------------
-- GEOMETRY RANKED (sliding-window derivative analysis)
-- From geometry_dynamics.parquet (per-engine windowed derivatives)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_geometry_ranked AS
SELECT
    I,
    signal_id,
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
        PARTITION BY cohort
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
-- Ranks each typology metric within the cohort
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_signal_ranked AS
SELECT
    signal_id,
    cohort,
    smoothness,
    periodicity_ratio,
    kurtosis,
    skewness,
    memory_proxy,

    -- Within-cohort ranks
    RANK() OVER (PARTITION BY cohort ORDER BY smoothness DESC) AS smoothness_rank,
    RANK() OVER (PARTITION BY cohort ORDER BY ABS(periodicity_ratio) DESC) AS periodicity_rank,
    RANK() OVER (PARTITION BY cohort ORDER BY kurtosis DESC) AS kurtosis_rank,
    RANK() OVER (PARTITION BY cohort ORDER BY ABS(skewness) DESC) AS skewness_rank,
    RANK() OVER (PARTITION BY cohort ORDER BY memory_proxy DESC) AS memory_rank,

    -- Fleet-wide ranks
    RANK() OVER (ORDER BY smoothness DESC) AS fleet_smoothness_rank,
    RANK() OVER (ORDER BY kurtosis DESC) AS fleet_kurtosis_rank,
    RANK() OVER (ORDER BY memory_proxy DESC) AS fleet_memory_rank,

    -- Percentiles within cohort
    PERCENT_RANK() OVER (PARTITION BY cohort ORDER BY smoothness) AS smoothness_percentile,
    PERCENT_RANK() OVER (PARTITION BY cohort ORDER BY kurtosis) AS kurtosis_percentile,
    PERCENT_RANK() OVER (PARTITION BY cohort ORDER BY memory_proxy) AS memory_percentile

FROM typology;


-- ------------------------------------------------------------
-- ANOMALY RANKED
-- Ranks by z_score magnitude
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_anomaly_ranked AS
SELECT
    I,
    signal_id,
    cohort,
    value,
    z_score,
    is_anomaly,
    ABS(z_score) AS z_magnitude,

    -- Rank within this timestep (what's deviating most right now)
    RANK() OVER (
        PARTITION BY cohort, I
        ORDER BY ABS(z_score) DESC
    ) AS deviation_rank,

    -- Rank within this signal's history (how unusual is this for THIS signal)
    PERCENT_RANK() OVER (
        PARTITION BY cohort, signal_id
        ORDER BY ABS(z_score)
    ) AS signal_percentile,

    -- Fleet rank at this timestep
    RANK() OVER (
        PARTITION BY I
        ORDER BY ABS(z_score) DESC
    ) AS fleet_deviation_rank

FROM zscore;


-- ------------------------------------------------------------
-- COUPLING RANKED
-- Ranks signal pairs by correlation magnitude
-- signal_pairwise has multiple engine rows per pair per window;
-- use engine='shape' for correlation (others duplicate it)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_coupling_ranked AS
SELECT
    I,
    signal_a,
    signal_b,
    cohort,
    correlation,
    distance,
    cosine_similarity,
    ABS(correlation) AS coupling_magnitude,

    -- Rank pairs by coupling strength at each window
    RANK() OVER (
        PARTITION BY cohort, I
        ORDER BY ABS(correlation) DESC
    ) AS coupling_rank,

    -- Rank by how much coupling changed from previous window
    ABS(correlation - LAG(correlation) OVER (
        PARTITION BY cohort, signal_a, signal_b ORDER BY I
    )) AS coupling_delta,

    RANK() OVER (
        PARTITION BY cohort, I
        ORDER BY ABS(correlation - LAG(correlation) OVER (
            PARTITION BY cohort, signal_a, signal_b ORDER BY I
        )) DESC NULLS LAST
    ) AS decoupling_rank,

    -- Fleet-wide coupling rank at each window
    RANK() OVER (
        PARTITION BY I
        ORDER BY ABS(correlation) DESC
    ) AS fleet_coupling_rank,

    -- Percentile within this pair's history
    PERCENT_RANK() OVER (
        PARTITION BY cohort, signal_a, signal_b
        ORDER BY ABS(correlation)
    ) AS coupling_percentile

FROM signal_pairwise
WHERE engine = 'shape';


-- ------------------------------------------------------------
-- UNIFIED HEALTH VIEW (ranked, no categorical gates)
-- Combines all rankings into single health assessment
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_system_health AS
SELECT
    gd.I,
    gd.signal_id,
    gd.cohort,

    -- Computed values from Engines
    gd.effective_dim,
    gd.effective_dim_velocity,
    d.lyapunov_max,

    -- Raw ranks from component views
    t.lyapunov_rank,
    t.lyapunov_percentile,
    t.classification_confidence,
    s.instability_rank,
    s.instability_percentile,
    s.stability_score,
    c.velocity_rank,
    c.velocity_percentile,
    c.collapse_onset_rank,

    -- Composite instability signal: average percentile across dimensions
    (COALESCE(t.lyapunov_percentile, 0)
     + COALESCE(s.instability_percentile, 0)
     + COALESCE(c.velocity_percentile, 0)) / 3.0 AS composite_instability,

    -- Fleet-wide composite rank
    RANK() OVER (
        ORDER BY (
            COALESCE(t.lyapunov_percentile, 0)
            + COALESCE(s.instability_percentile, 0)
            + COALESCE(c.velocity_percentile, 0)
        ) DESC
    ) AS composite_rank

FROM geometry_dynamics gd
LEFT JOIN dynamics d ON gd.I = d.I AND gd.signal_id = d.signal_id
LEFT JOIN v_trajectory_ranked t ON gd.I = t.I AND gd.signal_id = t.signal_id
LEFT JOIN v_stability_ranked s ON gd.I = s.I AND gd.signal_id = s.signal_id
LEFT JOIN v_geometry_ranked c ON gd.I = c.I AND gd.signal_id = c.signal_id;


-- ------------------------------------------------------------
-- SUMMARY REPORT VIEW
-- Aggregates rankings across all signals/time
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
