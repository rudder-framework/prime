-- =============================================================================
-- NAVIER-STOKES INTERPRETATION (Fluid Dynamics Analysis)
-- =============================================================================
--
-- Interprets signal features through the lens of fluid dynamics.
-- Operates on PRISM outputs to detect turbulence, flow regimes, vorticity.
--
-- Prerequisites:
--   - signal_vector.parquet with spectral features
--   - typology.parquet with signal classification
--   - Domain knowledge: which signals are velocity, pressure, temperature
--
-- Mathematical basis:
--   ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
--
--   Reynolds number: Re = ρuL/μ = uL/ν
--   Kolmogorov cascade: E(k) ∝ k^(-5/3) in inertial subrange
--
-- =============================================================================

-- -----------------------------------------------------------------------------
-- View: v_flow_signal_classification
-- Classifies signals by their role in fluid systems
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_flow_signal_classification AS
SELECT
    signal_id,
    cohort,

    -- Infer signal type from name patterns (customize per domain)
    CASE
        WHEN LOWER(signal_id) LIKE '%velocity%' OR LOWER(signal_id) LIKE '%speed%'
            OR LOWER(signal_id) LIKE '%flow%' THEN 'velocity'
        WHEN LOWER(signal_id) LIKE '%pressure%' OR LOWER(signal_id) LIKE '%psi%'
            THEN 'pressure'
        WHEN LOWER(signal_id) LIKE '%temp%' THEN 'temperature'
        WHEN LOWER(signal_id) LIKE '%visc%' THEN 'viscosity'
        WHEN LOWER(signal_id) LIKE '%dens%' THEN 'density'
        ELSE 'unknown'
    END AS fluid_role,

    temporal_pattern,
    spectral,
    spectral_slope,
    spectral_entropy

FROM typology;

-- -----------------------------------------------------------------------------
-- View: v_turbulence_detection
-- Detects turbulent vs laminar flow from spectral characteristics
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_turbulence_detection AS
SELECT
    sv.signal_id,
    sv.cohort,
    sv.I,
    fc.fluid_role,

    -- Spectral slope analysis (Kolmogorov: -5/3 ≈ -1.67)
    sv.spectral_slope,

    CASE
        WHEN sv.spectral_slope BETWEEN -2.0 AND -1.3
            THEN 'turbulent_inertial'  -- Kolmogorov cascade
        WHEN sv.spectral_slope BETWEEN -3.0 AND -2.0
            THEN 'turbulent_dissipation'  -- Dissipation range
        WHEN sv.spectral_slope > -1.0
            THEN 'laminar_or_periodic'
        WHEN sv.spectral_slope < -3.0
            THEN 'strongly_damped'
        ELSE 'transitional'
    END AS flow_regime,

    -- Turbulence intensity proxy (spectral entropy)
    sv.spectral_entropy AS turbulence_intensity,

    -- Energy cascade indicator
    CASE
        WHEN sv.spectral_slope BETWEEN -1.8 AND -1.5
            AND sv.spectral_entropy > 0.7
            THEN TRUE
        ELSE FALSE
    END AS kolmogorov_cascade_detected,

    -- Intermittency (from kurtosis - turbulence is intermittent)
    sv.kurtosis,
    CASE
        WHEN sv.kurtosis > 6 THEN 'highly_intermittent'
        WHEN sv.kurtosis > 3 THEN 'intermittent'
        ELSE 'steady'
    END AS intermittency_class

FROM signal_vector sv
JOIN v_flow_signal_classification fc ON sv.signal_id = fc.signal_id
WHERE fc.fluid_role IN ('velocity', 'pressure');

-- -----------------------------------------------------------------------------
-- View: v_reynolds_number_proxy
-- Estimates Reynolds number regime from signal characteristics
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_reynolds_number_proxy AS
SELECT
    cohort,
    I,

    -- Velocity signal characteristics
    MAX(CASE WHEN fluid_role = 'velocity' THEN signal_mean END) AS mean_velocity,
    MAX(CASE WHEN fluid_role = 'velocity' THEN signal_std END) AS velocity_fluctuation,
    MAX(CASE WHEN fluid_role = 'velocity' THEN spectral_slope END) AS velocity_spectral_slope,

    -- Reynolds regime estimate (from turbulence indicators)
    CASE
        WHEN MAX(CASE WHEN fluid_role = 'velocity' THEN spectral_entropy END) > 0.8
            AND MAX(CASE WHEN fluid_role = 'velocity' THEN kurtosis END) > 4
            THEN 'high_re_turbulent'  -- Re >> 4000
        WHEN MAX(CASE WHEN fluid_role = 'velocity' THEN spectral_entropy END) > 0.5
            THEN 'transitional'  -- Re ~ 2300-4000
        ELSE 'low_re_laminar'  -- Re < 2300
    END AS reynolds_regime,

    -- Turbulent kinetic energy proxy: TKE ∝ <u'²>
    POWER(MAX(CASE WHEN fluid_role = 'velocity' THEN signal_std END), 2) AS tke_proxy

FROM signal_vector sv
JOIN v_flow_signal_classification fc ON sv.signal_id = fc.signal_id
GROUP BY cohort, I;

-- -----------------------------------------------------------------------------
-- View: v_vorticity_indicators
-- Detects vortex shedding and rotational flow patterns
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_vorticity_indicators AS
SELECT
    sv.signal_id,
    sv.cohort,
    sv.I,

    -- Periodic vortex shedding (Strouhal behavior)
    sv.dominant_frequency AS shedding_frequency,
    sv.spectral_peak_snr AS shedding_strength,

    CASE
        WHEN t.temporal_pattern = 'PERIODIC'
            AND sv.spectral_peak_snr > 10
            THEN 'vortex_shedding_detected'
        WHEN t.temporal_pattern = 'QUASI_PERIODIC'
            THEN 'irregular_vortices'
        ELSE 'no_clear_vorticity'
    END AS vorticity_pattern,

    -- Strouhal number proxy (if characteristic length known)
    -- St = fL/U, typically 0.2 for cylinder wake
    sv.dominant_frequency AS strouhal_proxy

FROM signal_vector sv
JOIN typology t ON sv.signal_id = t.signal_id
JOIN v_flow_signal_classification fc ON sv.signal_id = fc.signal_id
WHERE fc.fluid_role = 'velocity';

-- -----------------------------------------------------------------------------
-- View: v_pressure_velocity_coupling
-- Analyzes pressure-velocity relationships (Bernoulli, water hammer)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_pressure_velocity_coupling AS
SELECT
    p.signal_a AS velocity_signal,
    p.signal_b AS pressure_signal,
    p.cohort,
    p.I,

    p.correlation AS pv_correlation,
    p.lag_at_max_xcorr AS phase_lag_samples,

    -- Bernoulli: p + ½ρv² = const → anticorrelation expected
    CASE
        WHEN p.correlation < -0.5 THEN 'bernoulli_dominated'
        WHEN p.correlation > 0.5 THEN 'source_pressure'  -- Pump/compressor
        ELSE 'complex_coupling'
    END AS coupling_type,

    -- Water hammer detection (sharp pressure spikes with velocity changes)
    CASE
        WHEN ABS(p.correlation) > 0.7
            AND sv_p.kurtosis > 6
            THEN 'water_hammer_risk'
        ELSE 'normal'
    END AS transient_risk

FROM signal_pairwise p
JOIN v_flow_signal_classification fc_a ON p.signal_a = fc_a.signal_id
JOIN v_flow_signal_classification fc_b ON p.signal_b = fc_b.signal_id
LEFT JOIN signal_vector sv_p ON sv_p.signal_id = p.signal_b AND sv_p.I = p.I
WHERE fc_a.fluid_role = 'velocity'
  AND fc_b.fluid_role = 'pressure';

-- -----------------------------------------------------------------------------
-- View: v_navier_stokes_summary
-- Overall fluid dynamics health summary per cohort
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_navier_stokes_summary AS
SELECT
    cohort,

    -- Flow regime summary
    MODE() WITHIN GROUP (ORDER BY flow_regime) AS dominant_flow_regime,
    AVG(turbulence_intensity) AS mean_turbulence_intensity,
    SUM(CASE WHEN kolmogorov_cascade_detected THEN 1 ELSE 0 END)::FLOAT
        / COUNT(*) AS kolmogorov_fraction,

    -- Intermittency
    MODE() WITHIN GROUP (ORDER BY intermittency_class) AS dominant_intermittency,

    -- Stability assessment
    CASE
        WHEN AVG(turbulence_intensity) > 0.8
            AND MODE() WITHIN GROUP (ORDER BY intermittency_class) = 'highly_intermittent'
            THEN 'unstable_turbulent'
        WHEN AVG(turbulence_intensity) < 0.3
            THEN 'stable_laminar'
        ELSE 'transitional'
    END AS stability_assessment,

    COUNT(DISTINCT I) AS n_windows

FROM v_turbulence_detection
GROUP BY cohort;

-- -----------------------------------------------------------------------------
-- View: v_energy_cascade_analysis
-- Detailed energy cascade analysis for turbulent flows
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_energy_cascade_analysis AS
SELECT
    signal_id,
    cohort,
    I,

    spectral_slope,

    -- Energy in different spectral ranges
    band_low_rel AS large_scale_energy,  -- Energy-containing range
    band_mid_rel AS inertial_energy,     -- Inertial subrange
    band_high_rel AS dissipation_energy, -- Dissipation range

    -- Cascade quality (how close to -5/3 law)
    ABS(spectral_slope - (-5.0/3.0)) AS kolmogorov_deviation,

    CASE
        WHEN ABS(spectral_slope - (-5.0/3.0)) < 0.2 THEN 'ideal_cascade'
        WHEN ABS(spectral_slope - (-5.0/3.0)) < 0.5 THEN 'good_cascade'
        ELSE 'non_ideal_cascade'
    END AS cascade_quality,

    -- Energy transfer direction
    CASE
        WHEN band_low_rel > band_high_rel * 2 THEN 'forward_cascade'  -- Large → small
        WHEN band_high_rel > band_low_rel * 2 THEN 'inverse_cascade'  -- Small → large (2D turbulence)
        ELSE 'balanced'
    END AS cascade_direction

FROM signal_vector sv
JOIN v_flow_signal_classification fc ON sv.signal_id = fc.signal_id
WHERE fc.fluid_role = 'velocity';
