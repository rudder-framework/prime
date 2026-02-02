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
    -- Note: When both reject, check ADF strength - very low ADF means strongly stationary
    CASE
        WHEN signal_std < 0.001                          THEN 'CONSTANT'
        WHEN adf_pvalue < 0.05 AND kpss_pvalue >= 0.05  THEN 'STATIONARY'
        WHEN adf_pvalue >= 0.05 AND kpss_pvalue < 0.05  THEN 'NON_STATIONARY'
        -- Both reject: if ADF very strongly rejects (p < 0.001), treat as stationary
        WHEN adf_pvalue < 0.001 AND kpss_pvalue < 0.05  THEN 'STATIONARY'
        WHEN adf_pvalue < 0.05 AND kpss_pvalue < 0.05   THEN 'DIFFERENCE_STATIONARY'
        WHEN adf_pvalue >= 0.05 AND kpss_pvalue >= 0.05 THEN 'TREND_STATIONARY'
        ELSE 'NON_STATIONARY'
    END AS stationarity,

    -- ======================================================================
    -- DIMENSION 3: TEMPORAL PATTERN
    -- ======================================================================
    -- Decision tree reordered: trending/chaotic before periodic (random walks have spectral peaks)
    -- 1. TRENDING: low turning points + long memory + non-stationary spectral slope
    -- 2. CHAOTIC: positive Lyapunov with some determinism
    -- 3. PERIODIC: high spectral peak SNR with truly flat spectrum elsewhere
    -- 4. QUASI_PERIODIC: moderate spectral concentration
    -- 5. MEAN_REVERTING: anti-persistent (Hurst < 0.45)
    -- 6. RANDOM: everything else
    CASE
        WHEN signal_std < 0.001                                              THEN 'CONSTANT'
        -- TRENDING: few turning points, long memory, steep spectral slope (1/f or steeper)
        WHEN turning_point_ratio < 0.8 AND hurst > 0.65 AND spectral_slope < -1.0 THEN 'TRENDING'
        -- CHAOTIC: positive Lyapunov proxy, some structure (not pure noise)
        WHEN lyapunov_proxy > 0.05 AND perm_entropy > 0.3 AND perm_entropy < 0.95 THEN 'CHAOTIC'
        -- PERIODIC: high spectral peak AND very narrow (low flatness) AND normal turning points
        WHEN spectral_peak_snr > 20 AND spectral_flatness < 0.1 AND turning_point_ratio > 0.1 THEN 'PERIODIC'
        WHEN spectral_peak_snr > 10 AND spectral_flatness < 0.3 AND turning_point_ratio > 0.1 THEN 'PERIODIC'
        -- QUASI_PERIODIC: moderate spectral concentration
        WHEN spectral_peak_snr > 5 AND spectral_flatness < 0.5 AND turning_point_ratio > 0.3 THEN 'QUASI_PERIODIC'
        -- MEAN_REVERTING: anti-persistent behavior
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
    -- Check ONE_OVER_F before NARROWBAND (random walks have both low flatness and steep slope)
    -- Exception: if spectral_flatness is essentially zero, it's truly narrowband (single frequency)
    CASE
        WHEN signal_std < 0.001                          THEN NULL
        WHEN harmonic_noise_ratio > 5                    THEN 'HARMONIC'
        -- Pure tones: essentially zero flatness means single frequency (NARROWBAND)
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
    END AS eigenvalue_budget

FROM typology_raw
ORDER BY unit_id, signal_id
