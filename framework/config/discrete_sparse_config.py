# RUDDER Typology Configuration - Discrete & Sparse Extension
# ===========================================================
# PR #5: Adds detection for non-continuous signal types
#
# These gates run BEFORE the continuous classification (PR4).
# Order: CONSTANT → DISCRETE → IMPULSIVE → EVENT → BINARY → STEP → (continuous)
#
# Usage:
#   from framework.config import TYPOLOGY_CONFIG
#   # Merge with PR4 config, or use standalone

DISCRETE_SPARSE_CONFIG = {

    # =========================================================
    # CONSTANT: Zero or negligible variance
    # =========================================================
    # Examples: stuck sensor, zero channel, disabled input
    'constant': {
        'enabled': True,
        'signal_std_max': 1e-10,       # Effectively zero
        'unique_ratio_max': 0.001,     # < 0.1% unique values
        'hurst_default_match': 0.5,    # Hurst returns 0.5 on failure
        # Any ONE of these conditions triggers CONSTANT
    },

    # =========================================================
    # DISCRETE: Integer/categorical with few levels
    # =========================================================
    # Examples: gear position, error codes, state machine output
    'discrete': {
        'enabled': True,
        'is_integer_required': True,   # All values must be integers
        'unique_ratio_max': 0.05,      # < 5% unique values
        'unique_count_max': 50,        # Or fewer than 50 distinct values
        # Must satisfy: is_integer AND (unique_ratio < max OR unique_count < max)
    },

    # =========================================================
    # BINARY: Exactly two distinct values
    # =========================================================
    # Examples: on/off switch, fault flag, presence detection
    'binary': {
        'enabled': True,
        'unique_count_exact': 2,       # Exactly 2 distinct values
        # Note: Supersedes DISCRETE if exactly 2 values
    },

    # =========================================================
    # IMPULSIVE: Rare extreme spikes dominate signal
    # =========================================================
    # Examples: bearing impact, flash crash, lightning strike
    'impulsive': {
        'enabled': True,
        'kurtosis_min': 20.0,          # Extreme tails
        'crest_factor_min': 10.0,      # Peak / RMS > 10
        # Must satisfy BOTH conditions
    },

    # =========================================================
    # EVENT: Sparse activity, mostly baseline
    # =========================================================
    # Examples: earthquake catalog, network intrusion, rare alarms
    'event': {
        'enabled': True,
        'sparsity_min': 0.80,          # > 80% at baseline (zero or mode)
        'kurtosis_min': 10.0,          # Heavy tails from rare events
        # Must satisfy BOTH conditions
    },

    # =========================================================
    # STEP: Piecewise constant, regime changes
    # =========================================================
    # Examples: thermostat setpoint, rate decisions, control mode
    'step': {
        'enabled': True,
        'derivative_sparsity_min': 0.90,  # diff() is 90% zeros
        'unique_ratio_max': 0.10,         # Few distinct levels
        'is_integer_required': False,     # Can be continuous levels
    },

    # =========================================================
    # INTERMITTENT: Alternating active/quiet periods
    # =========================================================
    # Examples: voice activity, rainfall, bursty traffic
    'intermittent': {
        'enabled': True,
        'zero_run_ratio_min': 0.30,    # At least 30% of signal is zero-runs
        'activity_clustering': True,    # Activity is clustered, not uniform
        'sparsity_range': [0.30, 0.80], # Between EVENT and continuous
    },
}


# =========================================================
# Spectral classification for discrete/sparse types
# =========================================================
DISCRETE_SPARSE_SPECTRAL = {
    'constant': 'NONE',           # No meaningful spectrum
    'binary': 'SWITCHING',        # On/off transitions
    'discrete': 'QUANTIZED',      # Discrete levels
    'impulsive': 'BROADBAND',     # Impulses have flat spectrum
    'event': 'SPARSE',            # Mostly silence
    'step': 'DC_DOMINANT',        # Low frequency dominated
    'intermittent': 'BURSTY',     # Time-varying spectrum
}


# =========================================================
# Engine recommendations for discrete/sparse types
# =========================================================
DISCRETE_SPARSE_ENGINES = {
    'constant': {
        'add': [],
        'remove': ['*'],  # Skip all engines
    },
    'binary': {
        'add': ['transition_count', 'duty_cycle', 'mean_time_between'],
        'remove': ['harmonics_ratio', 'hurst', 'lyapunov'],
    },
    'discrete': {
        'add': ['level_histogram', 'transition_matrix', 'dwell_times'],
        'remove': ['harmonics_ratio', 'spectral_entropy'],
    },
    'impulsive': {
        'add': ['peak_detection', 'inter_arrival', 'peak_amplitude_dist'],
        'remove': ['trend_r2', 'harmonics_ratio'],
    },
    'event': {
        'add': ['event_rate', 'inter_event_time', 'event_amplitude'],
        'remove': ['hurst', 'trend_r2', 'harmonics_ratio'],
    },
    'step': {
        'add': ['changepoint_detection', 'level_means', 'regime_duration'],
        'remove': ['harmonics_ratio', 'lyapunov'],
    },
    'intermittent': {
        'add': ['burst_detection', 'activity_ratio', 'silence_distribution'],
        'remove': ['trend_r2'],
    },
}


def get_discrete_threshold(type_name: str, param: str, default=None):
    """Get threshold for discrete/sparse classification."""
    cfg = DISCRETE_SPARSE_CONFIG.get(type_name, {})
    return cfg.get(param, default)
