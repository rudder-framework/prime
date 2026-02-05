"""
Manifest Generator v2.5 - Per-Engine Window Specification
==========================================================

Generates manifest.yaml from typology results with:
- Corrected classifications (PR4 continuous, PR5 discrete/sparse)
- **Inclusive engine selection**: "If it's a maybe, run it"
- Enhanced with entropy/ACF diagnostics for better type discrimination
- CONSTANT signal handling (skip_signals - only type that removes engines)
- System window for state_vector/geometry alignment
- Multi-scale representation (spectral vs trajectory)
- Window method tracking (how window was determined)
- Validation against config
- **Per-engine minimum window requirements** (v2.5)

Key features in v2.5:
- engine_windows: Minimum window sizes for FFT-based and long-range engines
- engine_window_overrides: Per-signal overrides when signal window < engine min
- system.window: Common window for state_vector/geometry alignment
- window_method: Tracks how window was determined (period, acf_half_life, etc.)
- window_confidence: high/medium/low confidence in window selection
- representation: spectral (fast signals) vs trajectory (slow signals)

ORTHON classifies → Manifest specifies → PRISM executes
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import yaml

# Engine/viz recommendations by type
# PHILOSOPHY: Inclusive - add all potentially useful engines, only remove
# when an engine is DEFINITELY inappropriate (would produce misleading results).
# "If it's a maybe, run it."
ENGINE_ADJUSTMENTS = {
    # === Continuous types (PR4) ===
    # Enhanced with entropy and ACF measures for better type discrimination
    'trending': {
        'add': ['hurst', 'rate_of_change', 'trend_r2', 'detrend_std', 'cusum',
                'spectral', 'kurtosis', 'skewness', 'crest_factor',
                'sample_entropy', 'acf_decay', 'variance_growth'],  # Entropy/ACF diagnostics
        'remove': [],  # Harmonics might catch oscillating trends - keep it
    },
    'periodic': {
        'add': ['harmonics', 'thd', 'frequency_bands', 'fundamental_freq', 'phase_coherence',
                'spectral', 'kurtosis', 'skewness', 'crest_factor', 'hurst',
                'snr'],  # Signal-to-noise ratio
        'remove': [],  # Hurst can show persistence in periodic - keep it
    },
    'chaotic': {
        'add': ['lyapunov', 'correlation_dimension', 'recurrence_rate', 'determinism',
                'spectral', 'kurtosis', 'skewness', 'crest_factor', 'hurst',
                'harmonics', 'frequency_bands',
                'sample_entropy', 'perm_entropy', 'embedding_dim'],  # Key chaos measures
        'remove': [],
    },
    'random': {
        'add': ['spectral_entropy', 'band_power', 'spectral', 'kurtosis', 'skewness',
                'crest_factor', 'hurst', 'frequency_bands',
                'sample_entropy', 'perm_entropy', 'acf_decay'],  # Entropy cross-checks
        'remove': [],  # Even random signals benefit from full characterization
    },
    'quasi_periodic': {
        'add': ['frequency_bands', 'spectral', 'harmonics', 'hurst', 'kurtosis',
                'skewness', 'crest_factor', 'rate_of_change',
                'sample_entropy'],  # Distinguishes from chaos
        'remove': [],
    },
    'stationary': {
        'add': ['spectral', 'kurtosis', 'skewness', 'crest_factor', 'hurst',
                'frequency_bands', 'spectral_entropy',
                'acf_decay', 'variance_ratio', 'adf_stat'],  # Stationarity tests
        'remove': [],
    },

    # === Discrete/sparse types (PR5) ===
    # These are more selective because continuous engines genuinely don't apply
    'constant': {
        'add': [],
        'remove': ['*'],  # Skip all - no information to extract
    },
    'binary': {
        'add': ['transition_count', 'duty_cycle', 'mean_time_between',
                'kurtosis', 'skewness',
                'switching_frequency'],  # For periodic switching patterns
        'remove': [],  # Let PRISM decide what makes sense
    },
    'discrete': {
        'add': ['level_histogram', 'transition_matrix', 'dwell_times',
                'kurtosis', 'skewness', 'spectral',
                'level_count', 'entropy'],  # Quantify discretization
        'remove': [],
    },
    'impulsive': {
        'add': ['peak_detection', 'inter_arrival', 'peak_amplitude_dist',
                'kurtosis', 'skewness', 'crest_factor', 'spectral', 'hurst',
                'envelope', 'rise_time'],  # Shape analysis
        'remove': [],  # Impulsive signals have interesting spectral content
    },
    'event': {
        'add': ['event_rate', 'inter_event_time', 'event_amplitude',
                'kurtosis', 'skewness', 'crest_factor', 'spectral'],
        'remove': [],
    },
    'step': {
        'add': ['changepoint_detection', 'level_means', 'regime_duration',
                'kurtosis', 'skewness', 'spectral', 'hurst'],
        'remove': [],
    },
    'intermittent': {
        'add': ['burst_detection', 'activity_ratio', 'silence_distribution',
                'kurtosis', 'skewness', 'crest_factor', 'spectral', 'hurst'],
        'remove': [],
    },
}

VIZ_ADJUSTMENTS = {
    # PHILOSOPHY: Inclusive - add useful visualizations, only remove for CONSTANT
    'trending': {
        'add': ['trend_overlay', 'segment_comparison', 'cusum_plot', 'spectral_density'],
        'remove': [],  # Keep waterfall - might show spectral drift
    },
    'periodic': {
        'add': ['waterfall', 'phase_portrait', 'spectrum', 'spectral_density'],
        'remove': [],
    },
    'chaotic': {
        'add': ['phase_portrait', 'recurrence', 'lyapunov_spectrum', 'spectral_density'],
        'remove': [],
    },
    'random': {
        'add': ['histogram', 'spectral_density', 'waterfall'],
        'remove': [],
    },
    'quasi_periodic': {
        'add': ['waterfall', 'spectral_density', 'phase_portrait'],
        'remove': [],
    },
    'stationary': {
        'add': ['histogram', 'spectral_density'],
        'remove': [],
    },
    'constant': {
        'add': [],
        'remove': ['*'],  # Only exception - nothing to visualize
    },
    'binary': {
        'add': ['state_timeline', 'transition_diagram', 'histogram'],
        'remove': [],
    },
    'discrete': {
        'add': ['state_timeline', 'level_histogram', 'spectral_density'],
        'remove': [],
    },
    'impulsive': {
        'add': ['spike_plot', 'amplitude_histogram', 'spectral_density'],
        'remove': [],
    },
    'event': {
        'add': ['event_timeline', 'inter_event_histogram'],
        'remove': [],
    },
    'step': {
        'add': ['state_timeline', 'level_histogram'],
        'remove': [],
    },
    'intermittent': {
        'add': ['activity_timeline', 'burst_histogram', 'spectral_density'],
        'remove': [],
    },
}

# Default base engines (before type-specific adjustments)
# These run on ALL signals (except CONSTANT) - key discriminators
BASE_ENGINES = [
    # Distribution
    'crest_factor', 'kurtosis', 'skewness',
    # Spectral
    'spectral',
    # Complexity/Information (discriminators)
    'sample_entropy', 'perm_entropy',
    # Memory
    'hurst', 'acf_decay',
]
BASE_VISUALIZATIONS = ['spectral_density']

# Engine minimum sample requirements
# FFT-based engines require minimum 64 samples to produce meaningful results.
# Long-range dependency engines (hurst) need longer series for statistical significance.
# Engines not listed here work fine with any window size >= 32.
ENGINE_MIN_WINDOWS = {
    # FFT-based engines (need 64 samples minimum)
    'spectral': 64,
    'harmonics': 64,
    'fundamental_freq': 64,
    'thd': 64,
    'frequency_bands': 64,
    'spectral_entropy': 64,
    'band_power': 64,
    # Entropy engines (need sufficient samples for embedding)
    'sample_entropy': 64,
    'perm_entropy': 32,  # Works with smaller windows
    # Long-range dependency engines
    'hurst': 128,  # Needs longer series for R/S analysis
    # Engines that work fine at 32: crest_factor, kurtosis, skewness, acf_decay, snr, phase_coherence
}


def compute_engine_window_overrides(
    engines: List[str],
    signal_window: int,
    engine_min_windows: Dict[str, int] = None,
) -> Dict[str, int]:
    """
    Compute engine-specific window overrides when signal window is insufficient.

    Args:
        engines: List of engines requested for this signal
        signal_window: The signal's base window size
        engine_min_windows: Engine minimum requirements (default: ENGINE_MIN_WINDOWS)

    Returns:
        Dict mapping engine names to their required window sizes (only for engines
        that need a larger window than signal_window)
    """
    if engine_min_windows is None:
        engine_min_windows = ENGINE_MIN_WINDOWS

    overrides = {}
    for engine in engines:
        min_required = engine_min_windows.get(engine, 0)
        if signal_window < min_required:
            overrides[engine] = min_required

    return overrides


def get_window_params(temporal_pattern: str, n_samples: int, typology_row: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get window/stride parameters based on signal type and typology measures.

    Returns:
        dict with window_size, stride, derivative_depth, window_method, window_confidence
    """
    pattern = temporal_pattern.upper()

    # Extract typology measures if available
    seasonal_period = None
    acf_half_life = None
    acf_decay_lag = None
    dominant_freq = None
    if typology_row:
        seasonal_period = typology_row.get('seasonal_period')
        acf_half_life = typology_row.get('acf_half_life')
        acf_decay_lag = typology_row.get('acf_decay_lag')
        dominant_freq = typology_row.get('dominant_freq')

    # Discrete/sparse types: minimal windowing
    if pattern in ('CONSTANT', 'BINARY', 'DISCRETE', 'EVENT'):
        return {
            'window_size': n_samples,  # Full signal
            'stride': n_samples,       # No overlap
            'derivative_depth': 0,
            'window_method': 'full_signal',
            'window_confidence': 'high',
        }

    # Try data-driven window selection first
    window_size = None
    window_method = 'default'
    window_confidence = 'low'

    # Priority 1: Use seasonal period (4 cycles)
    if seasonal_period and seasonal_period > 0:
        window_size = int(4 * seasonal_period)
        window_method = 'period'
        window_confidence = 'high'

    # Priority 2: Use ACF half-life (4x for decorrelation)
    elif acf_half_life and acf_half_life > 0:
        window_size = int(4 * acf_half_life)
        window_method = 'acf_half_life'
        window_confidence = 'high'

    # Priority 3: Use ACF decay lag (for long-memory)
    elif acf_decay_lag and acf_decay_lag > 0:
        window_size = int(8 * acf_decay_lag)
        window_method = 'long_memory'
        window_confidence = 'medium'

    # Priority 4: Use dominant frequency
    elif dominant_freq and dominant_freq > 0:
        period = int(1 / dominant_freq) if dominant_freq < 0.5 else 2
        window_size = int(4 * period)
        window_method = 'frequency'
        window_confidence = 'medium'

    # Clamp window to valid range
    if window_size:
        window_size = max(32, min(2048, window_size))
        window_size = min(window_size, n_samples // 2)

    # Type-specific defaults if no data-driven window
    if window_size is None:
        if pattern == 'TRENDING':
            window_size = 128
            window_method = 'default'
            window_confidence = 'low'
        elif pattern == 'IMPULSIVE':
            window_size = 64
            window_method = 'default'
            window_confidence = 'low'
        else:
            window_size = 128
            window_method = 'default'
            window_confidence = 'low'

    # Stride based on type
    if pattern == 'TRENDING':
        stride = window_size // 4  # 75% overlap
        derivative_depth = 2
    elif pattern == 'IMPULSIVE':
        stride = window_size // 4  # 75% overlap
        derivative_depth = 1
    else:
        stride = window_size // 2  # 50% overlap
        derivative_depth = 1

    return {
        'window_size': window_size,
        'stride': stride,
        'derivative_depth': derivative_depth,
        'window_method': window_method,
        'window_confidence': window_confidence,
    }


def apply_engine_adjustments(
    base_engines: List[str],
    temporal_pattern: str,
) -> List[str]:
    """Apply type-specific engine add/remove."""
    engines = list(base_engines)
    pattern = temporal_pattern.lower()

    adjustments = ENGINE_ADJUSTMENTS.get(pattern, {})

    # Remove first
    for eng in adjustments.get('remove', []):
        if eng == '*':
            return []
        if eng in engines:
            engines.remove(eng)

    # Then add
    for eng in adjustments.get('add', []):
        if eng not in engines:
            engines.append(eng)

    return engines


def apply_viz_adjustments(
    base_viz: List[str],
    temporal_pattern: str,
) -> List[str]:
    """Apply type-specific visualization add/remove."""
    viz = list(base_viz)
    pattern = temporal_pattern.lower()

    adjustments = VIZ_ADJUSTMENTS.get(pattern, {})

    for v in adjustments.get('remove', []):
        if v == '*':
            return []
        if v in viz:
            viz.remove(v)

    for v in adjustments.get('add', []):
        if v not in viz:
            viz.append(v)

    return viz


def get_output_hints(temporal_pattern: str, spectral: str) -> Dict[str, Any]:
    """Get output configuration hints for PRISM."""
    pattern = temporal_pattern.upper()

    hints = {}

    # Spectral output mode
    if pattern in ('PERIODIC', 'QUASI_PERIODIC'):
        hints['spectral'] = {
            'output_mode': 'per_bin',
            'n_bins': 'auto',
            'include_phase': False,
            'note': 'waterfall-ready output',
        }
    elif pattern == 'TRENDING':
        hints['spectral'] = {
            'output_mode': 'summary',
        }

    # Harmonics for HARMONIC spectral
    if spectral == 'HARMONIC':
        hints['harmonics'] = {
            'n_harmonics': 5,
            'include_thd': True,
        }

    return hints


def build_signal_config(
    signal_id: str,
    cohort: str,
    typology_row: Dict[str, Any],
    base_engines: List[str] = None,
    base_viz: List[str] = None,
) -> Dict[str, Any]:
    """
    Build manifest config for a single signal.

    Args:
        signal_id: Signal identifier
        cohort: Cohort identifier
        typology_row: Row from typology with temporal_pattern, spectral, etc.
        base_engines: Starting engine list (default: BASE_ENGINES)
        base_viz: Starting visualization list (default: BASE_VISUALIZATIONS)

    Returns:
        Signal configuration dict
    """
    if base_engines is None:
        base_engines = list(BASE_ENGINES)
    if base_viz is None:
        base_viz = list(BASE_VISUALIZATIONS)

    temporal = typology_row.get('temporal_pattern', 'STATIONARY')
    spectral = typology_row.get('spectral', 'NARROWBAND')
    n_samples = typology_row.get('n_samples', 1000)

    # Get type-specific adjustments
    engines = apply_engine_adjustments(base_engines, temporal)
    visualizations = apply_viz_adjustments(base_viz, temporal)
    window_params = get_window_params(temporal, n_samples, typology_row)
    output_hints = get_output_hints(temporal, spectral)

    config = {
        'engines': engines,
        'rolling_engines': [],
        'window_size': window_params['window_size'],
        'window_method': window_params['window_method'],
        'window_confidence': window_params['window_confidence'],
        'stride': window_params['stride'],
        'derivative_depth': window_params['derivative_depth'],
        'eigenvalue_budget': 5,
        'typology': {
            'temporal_pattern': temporal,
            'spectral': spectral,
        },
        'visualizations': visualizations,
    }

    if output_hints:
        config['output_hints'] = output_hints

    # Compute engine window overrides if signal window is smaller than engine requirements
    engine_overrides = compute_engine_window_overrides(engines, window_params['window_size'])
    if engine_overrides:
        config['engine_window_overrides'] = engine_overrides

    # Mark discrete/sparse types
    if temporal in ('CONSTANT', 'BINARY', 'DISCRETE', 'IMPULSIVE', 'EVENT', 'STEP', 'INTERMITTENT'):
        config['is_discrete_sparse'] = True

    return config


def build_manifest(
    typology_df,
    observations_path: str = 'observations.parquet',
    typology_path: str = 'typology.parquet',
    output_dir: str = 'output/',
    job_id: str = None,
    base_engines: List[str] = None,
    pair_engines: List[str] = None,
    symmetric_pair_engines: List[str] = None,
) -> Dict[str, Any]:
    """
    Build complete manifest from typology DataFrame.

    Args:
        typology_df: DataFrame with signal_id, cohort, temporal_pattern, spectral, n_samples
        observations_path: Path to observations parquet
        typology_path: Path to typology parquet
        output_dir: Output directory for PRISM
        job_id: Optional job ID (auto-generated if None)
        base_engines: Base engine list
        pair_engines: Pairwise engine list
        symmetric_pair_engines: Symmetric pairwise engine list

    Returns:
        Complete manifest dict
    """
    if job_id is None:
        job_id = f"orthon-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if base_engines is None:
        base_engines = list(BASE_ENGINES)
    if pair_engines is None:
        pair_engines = ['granger', 'transfer_entropy']
    if symmetric_pair_engines is None:
        symmetric_pair_engines = ['cointegration', 'correlation', 'mutual_info']

    # Build cohort structure
    cohorts = {}
    skip_signals = []
    all_engines = set()

    for _, row in typology_df.iterrows():
        signal_id = row['signal_id']
        cohort = row['cohort']
        temporal = row.get('temporal_pattern', 'STATIONARY')

        # Initialize cohort if needed
        if cohort not in cohorts:
            cohorts[cohort] = {}

        # Skip CONSTANT signals
        if temporal == 'CONSTANT':
            skip_signals.append(f"{cohort}/{signal_id}")
            continue

        # Build signal config
        config = build_signal_config(
            signal_id=signal_id,
            cohort=cohort,
            typology_row=row.to_dict(),
            base_engines=base_engines,
        )

        cohorts[cohort][signal_id] = config
        all_engines.update(config['engines'])

    # Count signals
    total_signals = len(typology_df)
    constant_signals = len(skip_signals)
    active_signals = total_signals - constant_signals

    # Calculate system window (median of all signal windows for alignment)
    all_windows = []
    all_strides = []
    for cohort_signals in cohorts.values():
        for sig_config in cohort_signals.values():
            all_windows.append(sig_config['window_size'])
            all_strides.append(sig_config['stride'])

    if all_windows:
        system_window = int(sorted(all_windows)[len(all_windows) // 2])  # median
        system_stride = int(sorted(all_strides)[len(all_strides) // 2])
    else:
        system_window = 128
        system_stride = 64

    # Build engine_windows section (only engines with requirements > default min_samples)
    engine_windows = {k: v for k, v in ENGINE_MIN_WINDOWS.items() if v > 32}
    engine_windows['note'] = 'Minimum window sizes for FFT-based and long-range engines'

    # Build manifest
    manifest = {
        'version': '2.5',
        'job_id': job_id,
        'created_at': datetime.now().isoformat(),
        'generator': 'orthon.manifest_generator v2.5 (per-engine window spec)',

        'paths': {
            'observations': observations_path,
            'typology': typology_path,
            'output_dir': output_dir,
        },

        'system': {
            'window': system_window,
            'stride': system_stride,
            'note': 'Common window for state_vector/geometry alignment',
        },

        'engine_windows': engine_windows,

        'summary': {
            'total_signals': total_signals,
            'total_cohorts': len(cohorts),
            'active_signals': active_signals,
            'constant_signals': constant_signals,
            'signal_engines': sorted(all_engines),
            'rolling_engines': [],
            'pair_engines': pair_engines,
            'symmetric_pair_engines': symmetric_pair_engines,
            'n_signal_engines': len(all_engines),
            'n_rolling_engines': 0,
        },

        'params': {
            'default_window': 128,
            'default_stride': 64,
            'min_samples': 64,
            'note': 'per-signal windows override system window when needed',
        },

        'cohorts': cohorts,

        'pair_engines': pair_engines,
        'symmetric_pair_engines': symmetric_pair_engines,
        'skip_signals': skip_signals,
    }

    return manifest


def validate_manifest(manifest: Dict[str, Any]) -> List[str]:
    """
    Validate manifest structure and consistency.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Required top-level keys
    required = ['version', 'job_id', 'paths', 'summary', 'cohorts']
    for key in required:
        if key not in manifest:
            errors.append(f"Missing required key: {key}")

    # Check cohort structure
    cohorts = manifest.get('cohorts', {})
    for cohort_id, signals in cohorts.items():
        for signal_id, config in signals.items():
            # Required signal keys
            if 'engines' not in config:
                errors.append(f"{cohort_id}/{signal_id}: missing 'engines'")
            if 'typology' not in config:
                errors.append(f"{cohort_id}/{signal_id}: missing 'typology'")

            # Typology must have temporal_pattern
            typology = config.get('typology', {})
            if 'temporal_pattern' not in typology:
                errors.append(f"{cohort_id}/{signal_id}: missing temporal_pattern")

            # CONSTANT signals should be in skip_signals, not cohorts
            if typology.get('temporal_pattern') == 'CONSTANT':
                errors.append(f"{cohort_id}/{signal_id}: CONSTANT signal should be in skip_signals")

    # Check skip_signals format
    skip_signals = manifest.get('skip_signals', [])
    for sig in skip_signals:
        if '/' not in sig:
            errors.append(f"skip_signals entry '{sig}' should be 'cohort/signal_id' format")

    return errors


def manifest_to_yaml(manifest: Dict[str, Any]) -> str:
    """Convert manifest dict to YAML string."""
    return yaml.dump(manifest, default_flow_style=False, sort_keys=False, allow_unicode=True)


def save_manifest(manifest: Dict[str, Any], path: str) -> None:
    """Save manifest to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
