"""
Typology Configuration
======================
All thresholds for signal classification and window sizing.
Single source of truth. Both Prime and Manifold import this.

Usage:
    from typology.config import CONFIG
    threshold = CONFIG['temporal']['trending']['hurst_strong']
"""

CONFIG = {

    # =================================================================
    # FFT / Spectral Artifact Detection
    # =================================================================
    'artifacts': {
        'first_bin_tolerance': 0.01,
        'first_bin_slope_threshold': -0.3,
        'default_fft_size': 256,
    },

    # =================================================================
    # Genuine Periodicity Gates (all must pass)
    # =================================================================
    'periodic': {
        'spectral_flatness_max': 0.15,
        'snr_min': 15.0,
        'hurst_max': 0.95,
        'turning_point_ratio_max': 0.95,
        'spectral_override': {
            'enabled': True,
            'snr_threshold': 30.0,
            'flatness_threshold': 0.1,
        },
    },

    # =================================================================
    # Temporal Pattern Classification
    # =================================================================
    'temporal': {
        'trending': {
            'hurst_strong': 0.99,
            'hurst_moderate': 0.85,
            'acf_ratio_min': 0.10,
            'sample_entropy_max': 0.15,
            'hurst_entropy_gate': 0.90,
            'segment_trend': {
                'enabled': True,
                'n_segments': 4,
                'monotonic_required': True,
                'min_change_ratio': 0.20,
                'hurst_floor': 0.60,
            },
            'variance_growth': {
                'enabled': True,
                'half_ratio_min': 1.5,
            },
        },
        'random': {
            'spectral_flatness_min': 0.5,
            'perm_entropy_min': 0.95,
        },
        'chaotic': {
            'lyapunov_proxy_min': 0.01,
            'perm_entropy_min': 0.35,
            'min_samples': 500,
            'determinism_score_min': 0.3,
            'clean_chaos': {
                'enabled': True,
                'lyapunov_proxy_min': 0.15,
                'perm_entropy_max': 0.6,
                'sample_entropy_max': 0.3,
            },
        },
        'quasi_periodic': {
            'spectral_flatness_max': 0.3,
            'spectral_peak_snr_min': 10.0,
            'perm_entropy_max': 0.9,
        },
        'constant': {
            'variance_ratio_max': 0.001,
            'range_ratio_max': 0.01,
            'hurst_default': 0.5,
        },
        'discrete': {
            'unique_ratio_max': 0.05,
        },
        'impulsive': {
            'kurtosis_min': 20,
            'crest_factor_min': 10,
        },
        'event': {
            'sparsity_min': 0.8,
            'kurtosis_min': 10,
        },
        'stationary': {
            'adf_pvalue_max': 0.05,
            'variance_ratio_min': 0.85,
            'variance_ratio_max': 1.15,
        },
        'bounded_deterministic': {
            'enabled': True,
            'hurst_min': 0.95,
            'perm_entropy_max': 0.5,
            'variance_ratio_max': 3.0,
        },
    },

    # =================================================================
    # Spectral Classification
    # =================================================================
    'spectral': {
        'harmonic': {
            'hnr_min': 3.0,
        },
        'broadband': {
            'spectral_flatness_min': 0.8,
        },
        'red_noise': {
            'spectral_slope_max': -0.5,
        },
        'blue_noise': {
            'spectral_slope_min': 0.2,
        },
    },

    # =================================================================
    # Stationarity Overrides
    # =================================================================
    'stationarity': {
        'deterministic_trend': {
            'n_segments': 4,
            'strength_threshold': 2.0,
        },
        'override_conditions': {
            'mean_shift_ratio_min': 1.0,
            'variance_ratio_extreme_low': 0.01,
            'variance_ratio_extreme_high': 100.0,
        },
    },

    # =================================================================
    # Memory Classification
    # =================================================================
    'memory': {
        'long_memory_min': 0.65,
        'anti_persistent_max': 0.45,
    },

    # =================================================================
    # Complexity Classification
    # =================================================================
    'complexity': {
        'low_max': 0.3,
        'high_min': 0.7,
    },

    # =================================================================
    # Window Sizing (characteristic time → window/stride)
    # =================================================================
    'window': {
        'multiplier': 2.5,
        'min_window': 64,
        'max_window': 2048,
        'min_stride': 8,
        'stride_fraction': {
            'fast': 0.25,
            'medium': 0.50,
            'slow': 0.75,
        },
        'default_stride_fraction': 0.50,
        'dynamics_speed': {
            'fast_threshold': 0.01,
            'slow_threshold': 0.10,
        },
    },

    # =================================================================
    # Observation-level defaults (before any measures computed)
    # =================================================================
    'defaults': {
        'window_by_length': [
            # (max_n_samples, window, stride)
            (128,  None, None),     # too short to window → use full signal
            (512,  64,   16),
            (2048, 128,  32),
            (None, 256,  64),       # None = unlimited
        ],
    },

    # =================================================================
    # Engine min/max caps (from literature review)
    # =================================================================
    'caps': {
        'ftle':      {'min': 500,  'default': 5000, 'max': 10000},
        'sampen':    {'min': 100,  'default': 1000, 'max': 2000},
        'rqa':       {'min': 200,  'default': 1000, 'max': 2000},
    },

    # =================================================================
    # Engine gating by typology (what Manifold runs per signal type)
    # =================================================================
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
            'remove': ['*'],
        },
    },
}


def get(path: str, default=None):
    """
    Get a config value by dot-separated path.

    Usage:
        get('temporal.trending.hurst_strong')  → 0.99
        get('caps.ftle.min')                   → 500
    """
    keys = path.split('.')
    val = CONFIG
    for key in keys:
        if isinstance(val, dict) and key in val:
            val = val[key]
        else:
            return default
    return val
