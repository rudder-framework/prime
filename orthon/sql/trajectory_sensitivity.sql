-- =============================================================================
-- TRAJECTORY SENSITIVITY INTERPRETATION (FTLE & Saddle Analysis)
-- =============================================================================
--
-- Interprets PRISM trajectory sensitivity outputs:
--   - ftle.parquet: Finite-Time Lyapunov Exponents
--   - saddle_detection.parquet: Saddle point proximity
--   - trajectory_sensitivity.parquet: Variable importance
--
-- Mathematical basis:
--   FTLE(x₀, T) = (1/T) * ln(σ_max(Φ_T))
--   Saddle: Jacobian eigenvalues with mixed signs
--   Sensitivity: ||Φ[:, i]|| for variable i
--
-- =============================================================================

-- -----------------------------------------------------------------------------
-- View: v_ftle_regime
-- Classifies current FTLE regime
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_ftle_regime AS
SELECT
    cohort,
    I,
    ftle_current,
    ftle_mean,
    ftle_std,

    CASE
        WHEN ftle_current > 0.5 THEN 'highly_sensitive'
        WHEN ftle_current > 0.1 THEN 'sensitive'
        WHEN ftle_current > 0.01 THEN 'weakly_sensitive'
        WHEN ftle_current > -0.01 THEN 'neutral'
        WHEN ftle_current > -0.1 THEN 'weakly_stable'
        ELSE 'stable'
    END AS ftle_regime,

    -- Trend detection
    CASE
        WHEN ftle_current > ftle_mean + ftle_std THEN 'rising'
        WHEN ftle_current < ftle_mean - ftle_std THEN 'falling'
        ELSE 'stable'
    END AS ftle_trend,

    -- Alert level
    CASE
        WHEN ftle_current > 0.5 THEN 'critical'
        WHEN ftle_current > 0.3 THEN 'warning'
        WHEN ftle_current > 0.1 THEN 'attention'
        ELSE 'normal'
    END AS ftle_alert

FROM ftle;

-- -----------------------------------------------------------------------------
-- View: v_saddle_proximity
-- Assesses proximity to saddle points (unstable equilibria)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_saddle_proximity AS
SELECT
    cohort,
    I,
    saddle_score_current,
    basin_stability_current,
    current_stability_type,
    n_saddle_points,

    -- Proximity classification
    CASE
        WHEN current_stability_type = 'saddle' AND saddle_score_current > 0.8
            THEN 'at_saddle'
        WHEN saddle_score_current > 0.5 THEN 'near_saddle'
        WHEN saddle_score_current > 0.3 THEN 'approaching'
        ELSE 'distant'
    END AS saddle_proximity,

    -- Basin stability classification
    CASE
        WHEN basin_stability_current < 0.2 THEN 'critical'
        WHEN basin_stability_current < 0.4 THEN 'low'
        WHEN basin_stability_current < 0.6 THEN 'moderate'
        ELSE 'high'
    END AS basin_stability_class,

    -- Transition risk
    CASE
        WHEN saddle_score_current > 0.7 AND basin_stability_current < 0.3
            THEN 'high_transition_risk'
        WHEN saddle_score_current > 0.5 OR basin_stability_current < 0.4
            THEN 'moderate_transition_risk'
        ELSE 'low_transition_risk'
    END AS transition_risk

FROM saddle_detection;

-- -----------------------------------------------------------------------------
-- View: v_variable_sensitivity
-- Variable sensitivity rankings
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_variable_sensitivity AS
SELECT
    cohort,
    I,
    signal_id,
    current_sensitivity,
    current_rank,
    mean_sensitivity,

    -- Sensitivity level
    CASE
        WHEN current_rank = 1 THEN 'dominant'
        WHEN current_rank <= 3 THEN 'important'
        ELSE 'secondary'
    END AS sensitivity_level,

    -- Is this variable the current focus?
    (current_rank = 1) AS is_dominant

FROM trajectory_sensitivity;

-- -----------------------------------------------------------------------------
-- View: v_sensitivity_transitions
-- Detects when dominant variable changes
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_sensitivity_transitions AS
WITH ranked AS (
    SELECT
        cohort,
        I,
        signal_id,
        current_rank,
        LAG(signal_id) OVER (PARTITION BY cohort ORDER BY I) AS prev_dominant
    FROM trajectory_sensitivity
    WHERE current_rank = 1
)
SELECT
    cohort,
    I,
    signal_id AS new_dominant,
    prev_dominant,
    (signal_id != prev_dominant AND prev_dominant IS NOT NULL) AS is_transition

FROM ranked;

-- -----------------------------------------------------------------------------
-- View: v_trajectory_health
-- Unified trajectory health assessment
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_trajectory_health AS
SELECT
    f.cohort,
    f.I,

    -- FTLE metrics
    f.ftle_current,
    f.ftle_regime,
    f.ftle_alert,

    -- Saddle metrics
    s.saddle_score_current,
    s.saddle_proximity,
    s.basin_stability_current,
    s.transition_risk,

    -- Combined health score (0-1, higher = healthier)
    GREATEST(0, LEAST(1,
        0.5 * (1 - COALESCE(f.ftle_current, 0))
        + 0.3 * COALESCE(s.basin_stability_current, 1)
        + 0.2 * (1 - COALESCE(s.saddle_score_current, 0))
    )) AS trajectory_health_score,

    -- Overall alert level
    CASE
        WHEN f.ftle_alert = 'critical' OR s.transition_risk = 'high_transition_risk'
            THEN 'critical'
        WHEN f.ftle_alert = 'warning' OR s.transition_risk = 'moderate_transition_risk'
            THEN 'warning'
        WHEN f.ftle_alert = 'attention' THEN 'attention'
        ELSE 'normal'
    END AS overall_alert

FROM v_ftle_regime f
LEFT JOIN v_saddle_proximity s ON f.cohort = s.cohort AND f.I = s.I;

-- -----------------------------------------------------------------------------
-- View: v_trajectory_summary
-- Summary statistics per cohort
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_trajectory_summary AS
SELECT
    cohort,

    -- FTLE summary
    AVG(ftle_current) AS mean_ftle,
    MAX(ftle_current) AS max_ftle,
    MODE() WITHIN GROUP (ORDER BY ftle_regime) AS typical_regime,

    -- Saddle summary
    AVG(saddle_score_current) AS mean_saddle_score,
    MAX(saddle_score_current) AS max_saddle_score,
    AVG(basin_stability_current) AS mean_basin_stability,

    -- Health summary
    AVG(trajectory_health_score) AS mean_health_score,
    MIN(trajectory_health_score) AS min_health_score,

    -- Alert counts
    SUM(CASE WHEN overall_alert = 'critical' THEN 1 ELSE 0 END) AS n_critical,
    SUM(CASE WHEN overall_alert = 'warning' THEN 1 ELSE 0 END) AS n_warning,
    SUM(CASE WHEN overall_alert = 'attention' THEN 1 ELSE 0 END) AS n_attention,

    -- Overall assessment
    CASE
        WHEN AVG(trajectory_health_score) < 0.3 THEN 'unhealthy'
        WHEN AVG(trajectory_health_score) < 0.6 THEN 'stressed'
        ELSE 'healthy'
    END AS trajectory_assessment,

    COUNT(DISTINCT I) AS n_windows

FROM v_trajectory_health
GROUP BY cohort;

-- -----------------------------------------------------------------------------
-- View: v_lcs_detection
-- Lagrangian Coherent Structure detection
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_lcs_detection AS
SELECT
    cohort,
    I,
    n_lcs_points,
    lcs_fraction,
    stretching_anisotropy_mean,

    -- LCS presence
    CASE
        WHEN lcs_fraction > 0.1 THEN 'strong_lcs'
        WHEN lcs_fraction > 0.05 THEN 'moderate_lcs'
        WHEN lcs_fraction > 0.01 THEN 'weak_lcs'
        ELSE 'no_lcs'
    END AS lcs_class,

    -- Anisotropy classification
    CASE
        WHEN stretching_anisotropy_mean > 10 THEN 'highly_anisotropic'
        WHEN stretching_anisotropy_mean > 3 THEN 'anisotropic'
        ELSE 'isotropic'
    END AS anisotropy_class

FROM ftle
WHERE n_lcs_points IS NOT NULL;

-- -----------------------------------------------------------------------------
-- View: v_influence_network
-- Variable influence relationships
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_influence_network AS
SELECT
    cohort,
    I,
    from_variable,
    to_variable,
    influence_strength,

    -- Influence classification
    CASE
        WHEN influence_strength > 0.5 THEN 'strong'
        WHEN influence_strength > 0.2 THEN 'moderate'
        ELSE 'weak'
    END AS influence_class

FROM trajectory_sensitivity_influence
WHERE influence_strength > 0.1;  -- Filter weak influences

-- -----------------------------------------------------------------------------
-- View: v_early_warning_trajectory
-- Early warning signals from trajectory analysis
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_early_warning_trajectory AS
SELECT
    h.cohort,
    h.I,
    h.trajectory_health_score,
    h.overall_alert,

    -- Rate of change in health score
    h.trajectory_health_score - LAG(h.trajectory_health_score, 10)
        OVER (PARTITION BY h.cohort ORDER BY h.I) AS health_change_rate,

    -- Trend in FTLE
    h.ftle_current - LAG(h.ftle_current, 10)
        OVER (PARTITION BY h.cohort ORDER BY h.I) AS ftle_trend,

    -- Early warning flag
    CASE
        WHEN h.trajectory_health_score < 0.5
            AND h.trajectory_health_score < LAG(h.trajectory_health_score, 10)
                OVER (PARTITION BY h.cohort ORDER BY h.I)
            THEN TRUE
        ELSE FALSE
    END AS declining_health,

    CASE
        WHEN h.ftle_current > 0.2
            AND h.ftle_current > LAG(h.ftle_current, 10)
                OVER (PARTITION BY h.cohort ORDER BY h.I) + 0.1
            THEN TRUE
        ELSE FALSE
    END AS rising_sensitivity

FROM v_trajectory_health h;
