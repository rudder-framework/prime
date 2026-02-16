# Prime Typology Classification Configuration
# =============================================
# All thresholds and settings for signal classification.
# Tune these values based on domain and dataset characteristics.
#
# Usage:
#   from prime.config import TYPOLOGY_CONFIG
#   threshold = TYPOLOGY_CONFIG['temporal']['trending']['hurst_strong']
#
# PR4 Updates:
#   - spectral_override for noisy periodic (guitar A3 fix)
#   - segment_trend for oscillating trends (battery fix)
#   - bounded_deterministic for smooth chaos (pendulum/Rössler fix)

TYPOLOGY_CONFIG = {
    
    # =========================================================
    # FFT / Spectral Artifact Detection
    # =========================================================
    'artifacts': {
        # First-bin artifact: dominant_freq ≈ 1/fft_size with red slope
        'first_bin_tolerance': 0.01,      # Within 1% of 1/fft_size
        'first_bin_slope_threshold': -0.3, # Slope must be negative (1/f)
        'default_fft_size': 256,
    },
    
    # =========================================================
    # Genuine Periodicity Gates (all must pass)
    # =========================================================
    'periodic': {
        'spectral_flatness_max': 0.15,    # Periodic = concentrated energy (< 15% spread)
        'snr_min': 15.0,                  # Strong peak required (~30x above noise floor)
        'hurst_max': 0.95,                # Not a trend
        'turning_point_ratio_max': 0.95,  # Not monotonic
        # ACF must exist (not None) - no threshold needed
        
        # NEW: Spectral override - if spectral evidence is overwhelming,
        # skip TPR check (noisy periodic signals have inflated TPR)
        'spectral_override': {
            'enabled': True,
            'snr_threshold': 30.0,        # 30 dB = 1000:1 signal-to-noise
            'flatness_threshold': 0.1,    # < 10% energy outside peak
        },
    },
    
    # =========================================================
    # Temporal Pattern Classification
    # =========================================================
    'temporal': {
        'trending': {
            # Gate 1: Strong hurst alone is sufficient
            'hurst_strong': 0.99,         # Definitionally trending
            
            # Gate 2: High hurst + non-decaying ACF
            'hurst_moderate': 0.85,       # Combined with acf=NaN
            
            # Gate 3: Relative ACF half-life (for finite but long ACF)
            'acf_ratio_min': 0.10,        # acf_half_life / n_samples
            
            # Gate 4: Low entropy + high hurst
            'sample_entropy_max': 0.15,   # For noisy trends
            'hurst_entropy_gate': 0.90,   # Combined with low entropy
            
            # NEW: Global segment trend detection
            # Catches oscillating trends like battery degradation
            'segment_trend': {
                'enabled': True,
                'n_segments': 4,
                'monotonic_required': True,
                'min_change_ratio': 0.20,  # 20% change in segment means
                'hurst_floor': 0.60,       # Only apply if hurst > this
            },
            
            # NEW: Variance growth check for distinguishing trends from bounded
            'variance_growth': {
                'enabled': True,
                'half_ratio_min': 1.5,     # var(2nd half) / var(1st half)
            },
        },
        
        'random': {
            'spectral_flatness_min': 0.5,  # Flat spectrum (was 0.9 — too strict for turbofan)
            'perm_entropy_min': 0.95,      # High complexity (was 0.99 — real random ~0.96-0.98)
        },
        
        'chaotic': {
            'lyapunov_proxy_min': 0.01,    # Per-sample units: chaotic 0.017-0.027, non-chaotic 0.002
            'perm_entropy_min': 0.95,      # High complexity
            'min_samples': 500,            # Lyapunov unreliable below this
            'determinism_score_min': 0.3,  # Optional: filter noise
            # Clean deterministic chaos (ODE-integrated systems like Rossler)
            # These have LOW entropy (structured) + positive Lyapunov
            'clean_chaos': {
                'enabled': True,
                'lyapunov_proxy_min': 0.01,   # Per-sample units: chaotic 0.017-0.027, non-chaotic 0.002
                'perm_entropy_max': 0.6,      # Structured, not random
                'sample_entropy_max': 0.3,    # Predictable local structure
            },
        },
        
        'quasi_periodic': {
            # ALL three conditions required (multi-feature AND gate)
            'spectral_flatness_max': 0.3,     # Some spectral concentration
            'spectral_peak_snr_min': 10.0,    # Detectable peak above noise
            'perm_entropy_max': 0.9,          # Not noise-dominated
        },
        
        'constant': {
            # Detect near-constant signals (hurst estimation fails)
            'variance_ratio_max': 0.001,   # var / mean² threshold
            'range_ratio_max': 0.01,       # (max-min) / mean threshold
            'hurst_default': 0.5,          # Hurst returns this on failure
        },

        'discrete': {
            # Few unique integer values (categorical, state machine, digital)
            'unique_ratio_max': 0.05,      # < 5% unique values
            # Also requires is_integer = True
        },

        'impulsive': {
            # Extreme spikes (impacts, faults, transients)
            'kurtosis_min': 20,            # Heavy tails
            'crest_factor_min': 10,        # Peak / RMS ratio
        },

        'event': {
            # Sparse signals with rare occurrences
            'sparsity_min': 0.8,           # > 80% zeros
            'kurtosis_min': 10,            # Some tail weight
        },

        # DRIFTING: Non-stationary persistent signals with directional drift
        # High Hurst (0.85-0.95) but don't meet the strict hurst >= 0.99
        # threshold for TRENDING. Noisy trend buried in stochastic variation.
        # Originally discovered from C-MAPSS turbofan RUL analysis.
        # Key discriminators:
        #   - acf_half_life correlates with lifecycle (r = 0.74-0.80)
        #   - variance_ratio correlates with lifecycle (r = -0.47)
        #   - d1 (rate of change) predicts RUL via formula: RUL = (threshold - current) / d1
        'drifting': {
            'enabled': True,
            'hurst_min': 0.85,             # High persistence (but not 0.99)
            'hurst_max': 0.99,             # Below strict TRENDING threshold
            'perm_entropy_min': 0.90,      # Noisy (not clean monotonic)
            'stationarity': 'NON_STATIONARY',  # Mean/variance changing
            # Optional: variance_ratio check for non-bounded behavior
            'variance_ratio_min': 0.2,     # Some variance growth (not bounded chaos)
        },

        # NEW: Bounded deterministic detection
        # For smooth chaos that looks like trending but is bounded
        'bounded_deterministic': {
            'enabled': True,
            'hurst_min': 0.95,             # High local persistence
            'perm_entropy_max': 0.5,       # Low complexity (smooth)
            'variance_ratio_max': 3.0,     # var(max_window) / var(min_window)
            # If hurst is high but variance is bounded → not a true trend
        },

        # STATIONARY: explicit stationarity evidence required
        # Signals that don't pass these gates fall through to RANDOM
        'stationary': {
            'adf_pvalue_max': 0.05,           # ADF rejects unit root
            'variance_ratio_min': 0.85,       # Variance stable
            'variance_ratio_max': 1.15,       # Variance stable
        },

        # NEW: Integrated process detection (stationarity test override)
        # For signals that have clear unit root behavior but bounded variance
        # Example: wrapping angles (theta2 in double pendulum)
        'integrated_process': {
            'enabled': True,
            'adf_pvalue_min': 0.10,        # Fail to reject unit root (non-stationary)
            'kpss_pvalue_max': 0.05,       # Reject stationarity (non-stationary)
            # Also requires acf_half_life = None (never decays)
            # All three conditions must be true to classify as DRIFTING
        },
    },
    
    # =========================================================
    # Spectral Classification
    # =========================================================
    'spectral': {
        'harmonic': {
            'hnr_min': 3.0,                # Harmonic-to-noise ratio
            # Also requires temporal_pattern == PERIODIC
        },
        
        'broadband': {
            'spectral_flatness_min': 0.3,  # Flat spectrum (was 0.8 — turbofan signals ~0.3-0.5)
        },
        
        'red_noise': {
            'spectral_slope_max': -0.5,    # 1/f spectrum
        },
        
        'blue_noise': {
            'spectral_slope_min': 0.2,     # Rising spectrum
        },

        'narrowband': {
            'spectral_flatness_max': 0.1,  # Energy concentrated at specific frequencies
            'spectral_peak_snr_min': 20.0, # Dominant peak far above noise floor
        },

        # Default: BROADBAND (none of the above)
    },
    
    # =========================================================
    # Stationarity Correction
    # =========================================================
    'stationarity': {
        'deterministic_trend': {
            'n_segments': 4,               # Split signal into N segments
            'strength_threshold': 2.0,     # Segment divergence / std
        },
        
        'override_conditions': {
            'mean_shift_ratio_min': 1.0,
            'variance_ratio_extreme_low': 0.01,
            'variance_ratio_extreme_high': 100.0,
        },
    },
    
    # =========================================================
    # Window / Stride Recommendations
    # =========================================================
    'windowing': {
        'min_samples': 64,                 # Minimum viable window
        'default_window': 128,             # Fallback window size
        'stride_overlap_target': 0.5,      # 50% overlap default
        'short_series_threshold': 500,     # Use stride=1 below this
    },
    
    # =========================================================
    # Engine Selection by Temporal Pattern
    # =========================================================
    'engines': {
        'trending': {
            'add': ['hurst', 'rate_of_change_ratio', 'trend_r2', 
                    'detrend_std', 'cusum'],
            'remove': ['harmonics_ratio', 'band_ratios', 'thd', 
                       'frequency_bands'],
        },
        
        'periodic': {
            'add': ['harmonics_ratio', 'thd', 'band_ratios', 
                    'fundamental_freq', 'phase_coherence'],
            'remove': ['hurst', 'trend_r2'],
        },
        
        'chaotic': {
            'add': ['lyapunov', 'correlation_dimension', 
                    'recurrence_rate', 'determinism'],
            'remove': ['trend_r2', 'harmonics_ratio'],
        },
        
        'random': {
            'add': ['spectral_entropy', 'band_power'],
            'remove': ['trend_r2', 'harmonics_ratio', 'lyapunov'],
        },
        
        'constant': {
            'add': [],
            'remove': ['*'],  # Skip all engines for constant signals
        },

        'drifting': {
            # Engines for drifting signal analysis (persistent noisy trends)
            # Based on C-MAPSS turbofan analysis: d1 (rate of change) is key
            'add': ['hurst', 'rate_of_change', 'trend_r2', 'variance_ratio',
                    'acf_decay', 'sample_entropy', 'cusum', 'detrend_std'],
            'remove': ['harmonics_ratio', 'thd', 'frequency_bands'],
        },
    },
    
    # =========================================================
    # Visualization Selection by Temporal Pattern
    # =========================================================
    'visualizations': {
        'trending': {
            'add': ['trend_overlay', 'segment_comparison', 'cusum_plot'],
            'remove': ['waterfall', 'recurrence'],
        },
        
        'periodic': {
            'add': ['waterfall', 'phase_portrait', 'spectrum'],
            'remove': ['trend_overlay'],
        },
        
        'chaotic': {
            'add': ['phase_portrait', 'recurrence', 'lyapunov_spectrum'],
            'remove': ['trend_overlay'],
        },

        'drifting': {
            'add': ['trend_overlay', 'variance_evolution', 'rul_projection'],
            'remove': ['waterfall', 'recurrence'],
        },
    },
}


# =========================================================
# Helper functions to access config
# =========================================================

def get_threshold(path: str, default=None):
    """
    Get a threshold value by dot-notation path.
    
    Example:
        get_threshold('temporal.trending.hurst_strong')  # Returns 0.99
        get_threshold('periodic.snr_min')                # Returns 6.0
    """
    keys = path.split('.')
    value = TYPOLOGY_CONFIG
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def get_engine_adjustments(temporal_pattern) -> dict:
    """Get engine add/remove lists for a temporal pattern."""
    pattern = temporal_pattern[0] if not isinstance(temporal_pattern, str) and hasattr(temporal_pattern, '__iter__') else temporal_pattern
    return TYPOLOGY_CONFIG['engines'].get(
        str(pattern).lower(),
        {'add': [], 'remove': []}
    )


def get_viz_adjustments(temporal_pattern) -> dict:
    """Get visualization add/remove lists for a temporal pattern."""
    pattern = temporal_pattern[0] if not isinstance(temporal_pattern, str) and hasattr(temporal_pattern, '__iter__') else temporal_pattern
    return TYPOLOGY_CONFIG['visualizations'].get(
        pattern.lower(),
        {'add': [], 'remove': []}
    )


# =========================================================
# Config validation
# =========================================================

def validate_config():
    """Check config for internal consistency."""
    errors = []
    
    # Periodic hurst_max should be <= trending hurst_strong
    if TYPOLOGY_CONFIG['periodic']['hurst_max'] > TYPOLOGY_CONFIG['temporal']['trending']['hurst_strong']:
        errors.append("periodic.hurst_max should be <= temporal.trending.hurst_strong")
    
    # Random spectral_flatness should be > broadband threshold
    if TYPOLOGY_CONFIG['temporal']['random']['spectral_flatness_min'] <= TYPOLOGY_CONFIG['spectral']['broadband']['spectral_flatness_min']:
        errors.append("random.spectral_flatness_min should be > broadband.spectral_flatness_min")
    
    return errors


if __name__ == '__main__':
    # Print config summary
    import json
    print(json.dumps(TYPOLOGY_CONFIG, indent=2))
    
    # Validate
    errors = validate_config()
    if errors:
        print("\nValidation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\nConfig valid ✓")
