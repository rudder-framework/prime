-- =============================================================================
-- TRANSFER FUNCTION ANALYSIS (Laplace Domain Interpretation)
-- =============================================================================
--
-- Estimates system dynamics from CONTROL → RESPONSE signal pairs.
-- Operates on PRISM outputs: signal_pairwise.parquet, signal_vector.parquet
--
-- Prerequisites:
--   - typology.parquet with signal_role (CONTROL/RESPONSE)
--   - signal_pairwise.parquet with coherence, correlation, lag
--   - signal_vector.parquet with spectral features
--
-- Mathematical basis:
--   H(s) = Y(s)/X(s) where s = σ + jω
--   Estimated via frequency response: H(jω) = Pxy(ω)/Pxx(ω)
--
-- =============================================================================

-- -----------------------------------------------------------------------------
-- View: v_control_response_pairs
-- Identifies valid CONTROL → RESPONSE pairs from typology
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_control_response_pairs AS
SELECT
    c.signal_id AS control_signal,
    r.signal_id AS response_signal,
    c.n_levels AS control_levels,
    c.step_count AS control_steps,
    r.spectral_entropy AS response_complexity
FROM typology c
CROSS JOIN typology r
WHERE c.signal_role = 'CONTROL'
  AND r.signal_role = 'RESPONSE'
  AND c.signal_id != r.signal_id
  AND c.cohort = r.cohort;  -- Same unit/cohort

-- -----------------------------------------------------------------------------
-- View: v_transfer_function_estimate
-- Estimates transfer function parameters from pairwise metrics
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_transfer_function_estimate AS
SELECT
    p.signal_a AS control_signal,
    p.signal_b AS response_signal,
    p.cohort,

    -- DC Gain estimate (correlation at lag 0 × amplitude ratio)
    p.correlation * (sv_r.signal_std / NULLIF(sv_c.signal_std, 0)) AS dc_gain_estimate,

    -- Bandwidth estimate from coherence rolloff
    -- High coherence at low freq, drops at bandwidth
    p.mean_coherence AS coherence_quality,

    -- Time constant from lag at max correlation
    p.lag_at_max_xcorr / NULLIF(p.sample_rate, 0) AS delay_seconds,
    p.lag_at_max_xcorr AS delay_samples,

    -- Coupling strength
    ABS(p.correlation) AS coupling_strength,

    -- Causality direction (from Granger if available)
    p.granger_f_ab AS granger_control_to_response,
    p.granger_p_ab AS granger_pvalue,

    -- Classification
    CASE
        WHEN ABS(p.correlation) > 0.8 AND p.mean_coherence > 0.7
            THEN 'strong_coupling'
        WHEN ABS(p.correlation) > 0.5 AND p.mean_coherence > 0.5
            THEN 'moderate_coupling'
        WHEN ABS(p.correlation) > 0.3
            THEN 'weak_coupling'
        ELSE 'uncoupled'
    END AS coupling_class,

    -- System order estimate (from spectral rolloff)
    CASE
        WHEN sv_r.spectral_slope > -1.5 THEN 'first_order'
        WHEN sv_r.spectral_slope > -2.5 THEN 'second_order'
        ELSE 'higher_order'
    END AS estimated_order

FROM signal_pairwise p
JOIN v_control_response_pairs pairs
    ON p.signal_a = pairs.control_signal
    AND p.signal_b = pairs.response_signal
LEFT JOIN signal_vector sv_c
    ON sv_c.signal_id = p.signal_a AND sv_c.I = p.I
LEFT JOIN signal_vector sv_r
    ON sv_r.signal_id = p.signal_b AND sv_r.I = p.I
WHERE p.mean_coherence > 0.3;  -- Filter low-quality estimates

-- -----------------------------------------------------------------------------
-- View: v_system_dynamics_summary
-- Aggregates transfer function estimates per control-response pair
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_system_dynamics_summary AS
SELECT
    control_signal,
    response_signal,
    cohort,

    -- Averaged estimates
    AVG(dc_gain_estimate) AS mean_dc_gain,
    STDDEV(dc_gain_estimate) AS dc_gain_stability,

    AVG(delay_seconds) AS mean_delay,
    AVG(coupling_strength) AS mean_coupling,
    AVG(coherence_quality) AS mean_coherence,

    -- Dominant characteristics
    MODE() WITHIN GROUP (ORDER BY coupling_class) AS dominant_coupling,
    MODE() WITHIN GROUP (ORDER BY estimated_order) AS dominant_order,

    -- Confidence
    COUNT(*) AS n_windows,
    AVG(CASE WHEN granger_pvalue < 0.05 THEN 1.0 ELSE 0.0 END) AS granger_significant_ratio

FROM v_transfer_function_estimate
GROUP BY control_signal, response_signal, cohort;

-- -----------------------------------------------------------------------------
-- View: v_step_response_characteristics
-- Estimates step response from CONTROL step events
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_step_response_characteristics AS
SELECT
    b.signal_id AS control_signal,
    r.signal_id AS response_signal,
    b.I AS step_time,
    b.magnitude AS step_magnitude,
    b.direction AS step_direction,

    -- Response characteristics (from signal_vector after step)
    sv_post.signal_mean - sv_pre.signal_mean AS response_magnitude,
    (sv_post.signal_mean - sv_pre.signal_mean) / NULLIF(b.magnitude, 0) AS gain_at_step,

    -- Settling indicator (variance reduction post-step)
    sv_post.signal_std / NULLIF(sv_pre.signal_std, 0) AS variance_ratio_post_step

FROM breaks b
JOIN v_control_response_pairs pairs ON b.signal_id = pairs.control_signal
JOIN typology r ON r.signal_id = pairs.response_signal
LEFT JOIN signal_vector sv_pre
    ON sv_pre.signal_id = r.signal_id
    AND sv_pre.I = b.I - 1
LEFT JOIN signal_vector sv_post
    ON sv_post.signal_id = r.signal_id
    AND sv_post.I BETWEEN b.I + 1 AND b.I + 10
WHERE b.magnitude > 0.1;  -- Significant steps only

-- -----------------------------------------------------------------------------
-- View: v_laplace_domain_interpretation
-- Final interpretation with standard control theory parameters
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_laplace_domain_interpretation AS
SELECT
    control_signal,
    response_signal,
    cohort,

    mean_dc_gain AS K,  -- Static gain
    mean_delay AS T_d,  -- Transport delay

    -- Time constant estimate: τ ≈ delay for first-order systems
    CASE
        WHEN dominant_order = 'first_order' THEN mean_delay
        ELSE mean_delay * 0.5  -- Rough approximation for higher order
    END AS tau_estimate,

    -- Bandwidth estimate: ω_bw ≈ 1/τ
    1.0 / NULLIF(mean_delay, 0) AS bandwidth_estimate,

    -- Transfer function model suggestion
    CASE dominant_order
        WHEN 'first_order' THEN
            'G(s) = ' || ROUND(mean_dc_gain, 2) || ' / (' || ROUND(mean_delay, 2) || 's + 1)'
        WHEN 'second_order' THEN
            'G(s) = K·ωn² / (s² + 2ζωn·s + ωn²)'
        ELSE 'Higher-order: use system identification'
    END AS suggested_model,

    mean_coupling,
    mean_coherence,
    n_windows,
    granger_significant_ratio AS causality_confidence

FROM v_system_dynamics_summary
WHERE n_windows >= 5;  -- Require sufficient data
