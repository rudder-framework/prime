-- ============================================================
-- ORTHON Classification SQL
-- Interprets PRISM computed values
--
-- PRISM computes numbers. ORTHON classifies.
-- ============================================================

-- ------------------------------------------------------------
-- TRAJECTORY CLASSIFICATION
-- Uses Lyapunov exponent (gold standard for chaos detection)
--
-- Lyapunov thresholds:
--   λ > 0.1   : chaotic (sensitive dependence on initial conditions)
--   λ > 0.01  : quasi-periodic (edge of chaos)
--   λ > -0.01 : oscillating (limit cycle)
--   λ > -0.1  : converging (damped oscillation)
--   λ < -0.1  : stable (fixed point attractor)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_trajectory_type AS
SELECT
    gd.I,
    gd.signal_id,
    gd.unit_id,

    -- Raw computed values from PRISM
    d.lyapunov_max,
    gd.effective_dim_velocity,
    gd.effective_dim_acceleration,

    -- Classification (ORTHON decides based on Lyapunov)
    CASE
        WHEN d.lyapunov_max > 0.1 THEN 'chaotic'
        WHEN d.lyapunov_max > 0.01 THEN 'quasi_periodic'
        -- Fallback to derivative-based when Lyapunov unavailable
        WHEN d.lyapunov_max IS NULL AND gd.effective_dim_velocity < -0.1 THEN 'collapsing'
        WHEN d.lyapunov_max IS NULL AND gd.effective_dim_velocity > 0.1 THEN 'expanding'
        WHEN d.lyapunov_max > -0.01 THEN 'oscillating'
        WHEN d.lyapunov_max > -0.1 THEN 'converging'
        ELSE 'stable'
    END AS trajectory_type,

    -- Confidence based on data quality
    CASE
        WHEN d.lyapunov_max IS NOT NULL THEN 'high'
        WHEN gd.effective_dim IS NOT NULL THEN 'medium'
        ELSE 'low'
    END AS classification_confidence

FROM geometry_dynamics gd
LEFT JOIN dynamics d ON gd.I = d.I AND gd.signal_id = d.signal_id;


-- ------------------------------------------------------------
-- STABILITY CLASSIFICATION
-- Based on Lyapunov exponent sign and magnitude
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_stability_class AS
SELECT
    I,
    signal_id,
    unit_id,
    lyapunov_max,

    CASE
        WHEN lyapunov_max > 0.05 THEN 'unstable'
        WHEN lyapunov_max < -0.05 THEN 'stable'
        WHEN lyapunov_max IS NULL THEN 'unknown'
        ELSE 'marginally_stable'
    END AS stability_class,

    -- Numeric stability score (-1 to +1, negative is stable)
    CASE
        WHEN lyapunov_max IS NULL THEN 0
        ELSE LEAST(1.0, GREATEST(-1.0, lyapunov_max * 10))
    END AS stability_score

FROM dynamics;


-- ------------------------------------------------------------
-- COLLAPSE DETECTION
-- Based on effective_dim velocity (loss of degrees of freedom)
--
-- Collapse = sustained negative velocity in effective_dim
-- This indicates the system is losing complexity
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_collapse_status AS
SELECT
    I,
    signal_id,
    unit_id,
    effective_dim,
    effective_dim_velocity,
    collapse_onset_idx,
    collapse_onset_fraction,

    -- Current collapse status
    CASE
        WHEN effective_dim_velocity < -0.1 THEN 'collapsing'
        WHEN effective_dim_velocity > 0.1 THEN 'expanding'
        WHEN ABS(effective_dim_velocity) < 0.01 THEN 'stable'
        ELSE 'drifting'
    END AS collapse_status,

    -- Collapse lifecycle stage (if collapse detected)
    CASE
        WHEN collapse_onset_fraction IS NULL THEN 'none_detected'
        WHEN collapse_onset_fraction < 0.2 THEN 'early_warning'
        WHEN collapse_onset_fraction < 0.5 THEN 'mid_life'
        WHEN collapse_onset_fraction < 0.8 THEN 'late_stage'
        ELSE 'imminent'
    END AS collapse_stage,

    -- Time remaining estimate (as fraction of lifecycle)
    CASE
        WHEN collapse_onset_fraction IS NULL THEN NULL
        ELSE 1.0 - collapse_onset_fraction
    END AS remaining_fraction

FROM geometry_dynamics;


-- ------------------------------------------------------------
-- SIGNAL TYPE CLASSIFICATION
-- Based on typology metrics from PRISM
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_signal_classification AS
SELECT
    signal_id,
    unit_id,
    smoothness,
    periodicity_ratio,
    kurtosis,
    skewness,
    memory_proxy,

    -- Signal morphology
    CASE
        WHEN smoothness > 0.9 THEN 'smooth'
        WHEN smoothness < 0.3 THEN 'noisy'
        WHEN kurtosis > 6 THEN 'impulsive'
        ELSE 'mixed'
    END AS signal_type,

    -- Periodicity
    CASE
        WHEN ABS(periodicity_ratio) > 0.7 THEN 'periodic'
        WHEN ABS(periodicity_ratio) > 0.3 THEN 'quasi_periodic'
        ELSE 'aperiodic'
    END AS periodicity_type,

    -- Tail behavior (outlier tendency)
    CASE
        WHEN kurtosis > 5 THEN 'heavy_tails'
        WHEN kurtosis > 3 THEN 'moderate_tails'
        WHEN kurtosis < 2 THEN 'light_tails'
        ELSE 'normal_tails'
    END AS tail_type,

    -- Memory/persistence (Hurst-like)
    CASE
        WHEN memory_proxy > 0.6 THEN 'trending'
        WHEN memory_proxy < 0.4 THEN 'mean_reverting'
        ELSE 'random_walk'
    END AS memory_type

FROM typology;


-- ------------------------------------------------------------
-- ANOMALY SEVERITY
-- Based on z-score magnitude
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_anomaly_severity AS
SELECT
    I,
    signal_id,
    unit_id,
    value,
    z_score,
    is_anomaly,

    CASE
        WHEN ABS(z_score) > 5 THEN 'critical'
        WHEN ABS(z_score) > 4 THEN 'severe'
        WHEN ABS(z_score) > 3 THEN 'moderate'
        WHEN ABS(z_score) > 2 THEN 'mild'
        ELSE 'normal'
    END AS severity,

    CASE
        WHEN z_score > 3 THEN 'high_spike'
        WHEN z_score < -3 THEN 'low_spike'
        WHEN z_score > 2 THEN 'elevated'
        WHEN z_score < -2 THEN 'depressed'
        ELSE 'normal'
    END AS anomaly_direction

FROM zscore;


-- ------------------------------------------------------------
-- COUPLING STRENGTH
-- Based on pairwise correlation/distance
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_coupling_strength AS
SELECT
    I,
    signal_a,
    signal_b,
    unit_id,
    correlation,
    distance,
    cosine_similarity,

    CASE
        WHEN ABS(correlation) > 0.9 THEN 'strongly_coupled'
        WHEN ABS(correlation) > 0.7 THEN 'moderately_coupled'
        WHEN ABS(correlation) > 0.4 THEN 'weakly_coupled'
        ELSE 'uncoupled'
    END AS coupling_strength,

    CASE
        WHEN correlation > 0.7 THEN 'positive'
        WHEN correlation < -0.7 THEN 'negative'
        ELSE 'neutral'
    END AS coupling_direction

FROM signal_pairwise;


-- ------------------------------------------------------------
-- UNIFIED HEALTH VIEW
-- Combines all classifications into single health assessment
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_system_health AS
SELECT
    gd.I,
    gd.signal_id,
    gd.unit_id,

    -- Computed values from PRISM
    gd.effective_dim,
    gd.effective_dim_velocity,
    d.lyapunov_max,

    -- Classifications from ORTHON
    t.trajectory_type,
    t.classification_confidence,
    s.stability_class,
    c.collapse_status,
    c.collapse_stage,

    -- Overall health score (0-1, higher is healthier)
    CASE
        WHEN c.collapse_status = 'collapsing' THEN 0.2
        WHEN s.stability_class = 'unstable' THEN 0.3
        WHEN t.trajectory_type = 'chaotic' THEN 0.4
        WHEN c.collapse_status = 'drifting' THEN 0.6
        WHEN s.stability_class = 'marginally_stable' THEN 0.7
        WHEN t.trajectory_type = 'oscillating' THEN 0.8
        WHEN t.trajectory_type = 'quasi_periodic' THEN 0.85
        WHEN s.stability_class = 'stable' THEN 1.0
        ELSE 0.5  -- unknown
    END AS health_score,

    -- Risk level
    CASE
        WHEN c.collapse_stage IN ('late_stage', 'imminent') THEN 'critical'
        WHEN c.collapse_status = 'collapsing' THEN 'high'
        WHEN s.stability_class = 'unstable' THEN 'high'
        WHEN t.trajectory_type = 'chaotic' THEN 'elevated'
        WHEN c.collapse_status = 'drifting' THEN 'moderate'
        ELSE 'low'
    END AS risk_level

FROM geometry_dynamics gd
LEFT JOIN dynamics d ON gd.I = d.I AND gd.signal_id = d.signal_id
LEFT JOIN v_trajectory_type t ON gd.I = t.I AND gd.signal_id = t.signal_id
LEFT JOIN v_stability_class s ON gd.I = s.I AND gd.signal_id = s.signal_id
LEFT JOIN v_collapse_status c ON gd.I = c.I AND gd.signal_id = c.signal_id;


-- ------------------------------------------------------------
-- SUMMARY REPORT VIEW
-- Aggregates health across all signals/time
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_health_summary AS
SELECT
    unit_id,

    -- Counts by trajectory type
    COUNT(*) AS total_observations,
    SUM(CASE WHEN trajectory_type = 'stable' THEN 1 ELSE 0 END) AS n_stable,
    SUM(CASE WHEN trajectory_type = 'oscillating' THEN 1 ELSE 0 END) AS n_oscillating,
    SUM(CASE WHEN trajectory_type = 'quasi_periodic' THEN 1 ELSE 0 END) AS n_quasi_periodic,
    SUM(CASE WHEN trajectory_type = 'chaotic' THEN 1 ELSE 0 END) AS n_chaotic,
    SUM(CASE WHEN trajectory_type = 'collapsing' THEN 1 ELSE 0 END) AS n_collapsing,

    -- Health metrics
    AVG(health_score) AS mean_health_score,
    MIN(health_score) AS min_health_score,

    -- Risk counts
    SUM(CASE WHEN risk_level = 'critical' THEN 1 ELSE 0 END) AS n_critical,
    SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) AS n_high_risk,

    -- Overall status
    CASE
        WHEN SUM(CASE WHEN risk_level = 'critical' THEN 1 ELSE 0 END) > 0 THEN 'CRITICAL'
        WHEN SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) > 0 THEN 'WARNING'
        WHEN AVG(health_score) < 0.5 THEN 'DEGRADED'
        WHEN AVG(health_score) < 0.8 THEN 'NOMINAL'
        ELSE 'HEALTHY'
    END AS overall_status

FROM v_system_health
GROUP BY unit_id;
