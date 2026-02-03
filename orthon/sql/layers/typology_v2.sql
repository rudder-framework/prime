-- ==========================================================================
-- ORTHON Typology v2: 10-Dimension Signal Classification
-- ==========================================================================
--
-- Reads from typology_raw.parquet (computed by PRISM Level 1+2 Python)
-- and classifies each signal across 10 dimensions.
--
-- PRISM computes the raw statistical measures.
-- ORTHON classifies from those measures. This file is the classifier.
--
-- Input:  typology_raw.parquet (one row per signal, all raw test values)
-- Output: typology.parquet (one row per signal, 10 dimensions + derived)
--
-- Replaces the proxy-based typology.sql once PRISM Level 2 is implemented.
--
-- ==========================================================================

SELECT
    signal_id,
    unit_id,

    -- ======================================================================
    -- RAW VALUES (pass-through from Python for debugging/auditing)
    -- ======================================================================
    n_samples,
    adf_pvalue,
    kpss_pvalue,
    variance_ratio,
    acf_half_life,
    hurst,
    perm_entropy,
    sample_entropy,
    spectral_flatness,
    spectral_slope,
    harmonic_noise_ratio,
    spectral_peak_snr,
    dominant_frequency,
    is_first_bin_peak,  -- True when dominant_freq is artifact from 1/f slope
    turning_point_ratio,
    lyapunov_proxy,
    determinism_score,
    arch_pvalue,
    rolling_var_std,
    kurtosis,
    skewness,
    crest_factor,
    unique_ratio,
    is_integer,
    sparsity,
    signal_std,

    -- ======================================================================
    -- DIMENSION 1: CONTINUITY
    -- ======================================================================
    CASE
        WHEN signal_std < 0.001                          THEN 'CONSTANT'
        WHEN sparsity > 0.9                              THEN 'EVENT'
        WHEN is_integer AND unique_ratio < 0.05          THEN 'DISCRETE'
        ELSE 'CONTINUOUS'
    END AS continuity,

    -- ======================================================================
    -- DIMENSION 2: STATIONARITY (from ADF + KPSS joint test)
    -- ======================================================================
    -- ADF: H0 = unit root (non-stationary). Low p → reject → stationary
    -- KPSS: H0 = stationary. Low p → reject → non-stationary
    --
    -- CRITICAL FIX: Monotonic trends (like exponential decay) can pass ADF
    -- (bounded signals have no unit root) but fail KPSS (clearly not stationary).
    -- When turning_point_ratio < 0.5 (monotonic), trust KPSS over ADF.
    -- ======================================================================
    CASE
        WHEN signal_std < 0.001                          THEN 'CONSTANT'
        -- MONOTONIC OVERRIDE: Very few turning points = trending, trust KPSS
        WHEN turning_point_ratio < 0.5 AND kpss_pvalue < 0.05 THEN 'NON_STATIONARY'
        -- Standard joint test interpretation
        WHEN adf_pvalue < 0.05 AND kpss_pvalue >= 0.05  THEN 'STATIONARY'
        WHEN adf_pvalue >= 0.05 AND kpss_pvalue < 0.05  THEN 'NON_STATIONARY'
        -- Both reject: oscillating or borderline - check for true oscillation
        WHEN adf_pvalue < 0.05 AND kpss_pvalue < 0.05 AND turning_point_ratio > 0.8
            THEN 'STATIONARY'  -- Oscillating around mean, genuinely stationary
        WHEN adf_pvalue < 0.05 AND kpss_pvalue < 0.05   THEN 'DIFFERENCE_STATIONARY'
        WHEN adf_pvalue >= 0.05 AND kpss_pvalue >= 0.05 THEN 'TREND_STATIONARY'
        ELSE 'NON_STATIONARY'
    END AS stationarity,

    -- ======================================================================
    -- DIMENSION 3: TEMPORAL PATTERN
    -- ======================================================================
    -- Decision tree: TRENDING must come before PERIODIC to catch monotonic signals
    --
    -- CRITICAL FIX: is_first_bin_peak gates out fake periodicity from 1/f slopes.
    -- Monotonic signals (exponential decay, linear ramps) have very low turning_point_ratio
    -- and should NEVER be classified as PERIODIC regardless of spectral SNR.
    --
    -- Order: CONSTANT → TRENDING → CHAOTIC → PERIODIC → QUASI_PERIODIC → MEAN_REVERTING → RANDOM
    -- ======================================================================
    CASE
        WHEN signal_std < 0.001                                              THEN 'CONSTANT'

        -- TRENDING (monotonic): very few turning points = one-way trajectory
        -- This catches exponential decay, linear drift, asymptotic approach
        WHEN turning_point_ratio < 0.5                                       THEN 'TRENDING'

        -- TRENDING (persistent): moderate turning points but strong persistence + 1/f spectrum
        WHEN turning_point_ratio < 0.8 AND hurst > 0.65 AND spectral_slope < -1.0 THEN 'TRENDING'

        -- CHAOTIC: positive Lyapunov proxy, some structure (not pure noise)
        WHEN lyapunov_proxy > 0.05 AND perm_entropy > 0.3 AND perm_entropy < 0.95 THEN 'CHAOTIC'

        -- PERIODIC: ONLY when dominant_frequency is real (not first-bin artifact)
        -- Must have: real spectral peak + narrow bandwidth + enough oscillation
        WHEN NOT COALESCE(is_first_bin_peak, FALSE)
             AND dominant_frequency > 0
             AND spectral_peak_snr > 20
             AND spectral_flatness < 0.1
             AND turning_point_ratio > 0.5                                   THEN 'PERIODIC'
        WHEN NOT COALESCE(is_first_bin_peak, FALSE)
             AND dominant_frequency > 0
             AND spectral_peak_snr > 10
             AND spectral_flatness < 0.3
             AND turning_point_ratio > 0.5                                   THEN 'PERIODIC'

        -- QUASI_PERIODIC: moderate spectral concentration, real frequency
        WHEN NOT COALESCE(is_first_bin_peak, FALSE)
             AND spectral_peak_snr > 5
             AND spectral_flatness < 0.5
             AND turning_point_ratio > 0.5                                   THEN 'QUASI_PERIODIC'

        -- MEAN_REVERTING: anti-persistent behavior (bounces off limits)
        WHEN hurst < 0.45                                                    THEN 'MEAN_REVERTING'

        -- Default: RANDOM
        ELSE 'RANDOM'
    END AS temporal_pattern,

    -- ======================================================================
    -- DIMENSION 4: MEMORY
    -- ======================================================================
    CASE
        WHEN signal_std < 0.001   THEN NULL
        WHEN hurst > 0.65         THEN 'LONG_MEMORY'
        WHEN hurst < 0.45         THEN 'ANTI_PERSISTENT'
        ELSE 'SHORT_MEMORY'
    END AS memory,

    -- ======================================================================
    -- DIMENSION 5: COMPLEXITY
    -- ======================================================================
    CASE
        WHEN signal_std < 0.001   THEN NULL
        WHEN perm_entropy < 0.3   THEN 'LOW'
        WHEN perm_entropy > 0.7   THEN 'HIGH'
        ELSE 'MEDIUM'
    END AS complexity,

    -- ======================================================================
    -- DIMENSION 6: DISTRIBUTION SHAPE
    -- ======================================================================
    -- Priority: heavy/light tails first, then skew direction
    CASE
        WHEN signal_std < 0.001                          THEN 'CONSTANT'
        WHEN kurtosis > 4.0                              THEN 'HEAVY_TAILED'
        WHEN kurtosis < 2.5                              THEN 'LIGHT_TAILED'
        WHEN skewness > 0.5                              THEN 'SKEWED_RIGHT'
        WHEN skewness < -0.5                             THEN 'SKEWED_LEFT'
        ELSE 'GAUSSIAN'
    END AS distribution,

    -- ======================================================================
    -- DIMENSION 7: AMPLITUDE CHARACTER
    -- ======================================================================
    CASE
        WHEN signal_std < 0.001                          THEN 'CONSTANT'
        WHEN crest_factor > 6 AND kurtosis > 6           THEN 'IMPULSIVE'
        WHEN crest_factor < 4 AND spectral_flatness < 0.3 THEN 'SMOOTH'
        WHEN spectral_flatness > 0.7                     THEN 'NOISY'
        ELSE 'MIXED'
    END AS amplitude,

    -- ======================================================================
    -- DIMENSION 8: SPECTRAL CHARACTER
    -- ======================================================================
    -- CRITICAL FIX: is_first_bin_peak means spectral peak is artifact from 1/f slope.
    -- When first-bin artifact detected, classify as ONE_OVER_F regardless of HNR.
    -- ======================================================================
    CASE
        WHEN signal_std < 0.001                          THEN NULL
        -- First-bin artifact = 1/f spectrum (trending/monotonic signals)
        WHEN COALESCE(is_first_bin_peak, FALSE)          THEN 'ONE_OVER_F'
        -- True harmonics: multiple peaks at integer multiples
        WHEN harmonic_noise_ratio > 5                    THEN 'HARMONIC'
        -- Pure tones: essentially zero flatness = single frequency
        WHEN spectral_flatness < 0.01                    THEN 'NARROWBAND'
        -- ONE_OVER_F: steep spectral slope (power law)
        WHEN spectral_slope < -1.5                       THEN 'ONE_OVER_F'
        -- NARROWBAND: concentrated power but not 1/f
        WHEN spectral_flatness < 0.2 AND spectral_slope > -1.5 THEN 'NARROWBAND'
        WHEN spectral_flatness > 0.8                     THEN 'BROADBAND'
        ELSE 'BROADBAND'
    END AS spectral,

    -- ======================================================================
    -- DIMENSION 9: VOLATILITY
    -- ======================================================================
    CASE
        WHEN signal_std < 0.001                          THEN NULL
        WHEN arch_pvalue < 0.05 AND rolling_var_std > 0.5 THEN 'VOLATILITY_CLUSTERING'
        WHEN variance_ratio > 2.0 OR variance_ratio < 0.5 THEN 'HETEROSCEDASTIC'
        ELSE 'HOMOSCEDASTIC'
    END AS volatility,

    -- ======================================================================
    -- DIMENSION 10: DETERMINISM
    -- ======================================================================
    -- Note: Pure periodic signals may have low recurrence-based determinism
    -- but are truly deterministic - use spectral characteristics as fallback
    -- Only apply fallback when turning_point_ratio indicates actual periodicity
    CASE
        WHEN signal_std < 0.001                          THEN NULL
        WHEN determinism_score > 0.8                     THEN 'DETERMINISTIC'
        -- Periodic signals with very narrow spectrum AND regular oscillation are deterministic
        WHEN spectral_peak_snr > 50 AND spectral_flatness < 0.01 AND turning_point_ratio < 0.5 THEN 'DETERMINISTIC'
        WHEN spectral_peak_snr > 20 AND spectral_flatness < 0.05 AND turning_point_ratio < 0.3 THEN 'DETERMINISTIC'
        WHEN determinism_score < 0.3                     THEN 'STOCHASTIC'
        ELSE 'MIXED'
    END AS determinism,

    -- ======================================================================
    -- DERIVED: RECOMMENDED WINDOW SIZE
    -- ======================================================================
    CASE
        WHEN signal_std < 0.001                          THEN NULL
        WHEN acf_half_life IS NOT NULL AND acf_half_life > 16
            THEN CAST(GREATEST(64, 4 * acf_half_life) AS INTEGER)
        WHEN hurst > 0.65                                THEN 256
        WHEN hurst < 0.45                                THEN 64
        ELSE 128
    END AS recommended_window,

    -- ======================================================================
    -- DERIVED: DERIVATIVE DEPTH
    -- ======================================================================
    CASE
        WHEN signal_std < 0.001                          THEN 0
        -- Trending signals need velocity + acceleration
        WHEN turning_point_ratio < 0.5 AND hurst > 0.65 THEN 2
        -- Non-stationary needs at least velocity
        WHEN adf_pvalue >= 0.05                          THEN 2
        -- Stationary: velocity only
        ELSE 1
    END AS derivative_depth,

    -- ======================================================================
    -- DERIVED: EIGENVALUE BUDGET
    -- ======================================================================
    CASE
        WHEN signal_std < 0.001                          THEN 0
        WHEN perm_entropy < 0.3                          THEN 3
        WHEN perm_entropy > 0.7                          THEN 8
        ELSE 5
    END AS eigenvalue_budget,

    -- ======================================================================
    -- WINDOW FACTOR (from typology_raw, for PRISM adaptive windowing)
    -- ======================================================================
    -- Multiplier for engine base windows based on signal characteristics.
    -- Range: 0.5 to 3.0. Default 1.0 for standard signals.
    -- Higher values for: narrowband, low-freq, periodic, noisy, anti-persistent
    -- ======================================================================
    window_factor

FROM typology_raw
ORDER BY unit_id, signal_id
