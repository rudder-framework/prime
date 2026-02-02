"""
Manifest Generator v2.2 - PR4/PR5 Integration
==============================================

Generates manifest.yaml from typology results with:
- Corrected classifications (PR4 continuous, PR5 discrete/sparse)
- Type-specific engine selection
- CONSTANT signal handling (skip_signals)
- Validation against config

ORTHON classifies → Manifest specifies → PRISM executes
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import yaml

# Engine/viz recommendations by type
ENGINE_ADJUSTMENTS = {
    # === Continuous types (PR4) ===
    'trending': {
        'add': ['hurst', 'rate_of_change', 'trend_r2', 'detrend_std', 'cusum'],
        'remove': ['harmonics', 'frequency_bands', 'thd'],
    },
    'periodic': {
        'add': ['harmonics', 'thd', 'frequency_bands', 'fundamental_freq', 'phase_coherence'],
        'remove': ['hurst', 'trend_r2'],
    },
    'chaotic': {
        'add': ['lyapunov', 'correlation_dimension', 'recurrence_rate', 'determinism'],
        'remove': ['trend_r2', 'harmonics'],
    },
    'random': {
        'add': ['spectral_entropy', 'band_power'],
        'remove': ['trend_r2', 'harmonics', 'lyapunov'],
    },
    'quasi_periodic': {
        'add': ['frequency_bands', 'spectral'],
        'remove': ['trend_r2'],
    },
    'stationary': {
        'add': ['spectral', 'kurtosis', 'skewness'],
        'remove': ['trend_r2', 'hurst'],
    },

    # === Discrete/sparse types (PR5) ===
    'constant': {
        'add': [],
        'remove': ['*'],  # Skip all
    },
    'binary': {
        'add': ['transition_count', 'duty_cycle', 'mean_time_between'],
        'remove': ['harmonics', 'hurst', 'lyapunov', 'spectral'],
    },
    'discrete': {
        'add': ['level_histogram', 'transition_matrix', 'dwell_times'],
        'remove': ['harmonics', 'spectral_entropy'],
    },
    'impulsive': {
        'add': ['peak_detection', 'inter_arrival', 'peak_amplitude_dist'],
        'remove': ['trend_r2', 'harmonics'],
    },
    'event': {
        'add': ['event_rate', 'inter_event_time', 'event_amplitude'],
        'remove': ['hurst', 'trend_r2', 'harmonics'],
    },
    'step': {
        'add': ['changepoint_detection', 'level_means', 'regime_duration'],
        'remove': ['harmonics', 'lyapunov'],
    },
    'intermittent': {
        'add': ['burst_detection', 'activity_ratio', 'silence_distribution'],
        'remove': ['trend_r2'],
    },
}

VIZ_ADJUSTMENTS = {
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
    'random': {
        'add': ['histogram', 'spectral_density'],
        'remove': ['trend_overlay', 'waterfall'],
    },
    'quasi_periodic': {
        'add': ['waterfall', 'spectral_density'],
        'remove': ['trend_overlay'],
    },
    'constant': {
        'add': [],
        'remove': ['*'],
    },
    'binary': {
        'add': ['state_timeline', 'transition_diagram'],
        'remove': ['waterfall', 'spectral_density'],
    },
    'discrete': {
        'add': ['state_timeline', 'level_histogram'],
        'remove': ['waterfall'],
    },
    'impulsive': {
        'add': ['spike_plot', 'amplitude_histogram'],
        'remove': ['waterfall'],
    },
    'event': {
        'add': ['event_timeline', 'inter_event_histogram'],
        'remove': ['waterfall', 'spectral_density'],
    },
}

# Default base engines (before type-specific adjustments)
BASE_ENGINES = ['crest_factor', 'kurtosis', 'skewness', 'spectral']
BASE_VISUALIZATIONS = ['spectral_density']


def get_window_params(temporal_pattern: str, n_samples: int) -> Dict[str, int]:
    """
    Get window/stride parameters based on signal type.

    Returns:
        dict with window_size, stride, derivative_depth
    """
    pattern = temporal_pattern.upper()

    # Discrete/sparse types: minimal windowing
    if pattern in ('CONSTANT', 'BINARY', 'DISCRETE', 'EVENT'):
        return {
            'window_size': n_samples,  # Full signal
            'stride': n_samples,       # No overlap
            'derivative_depth': 0,
        }

    # Trending: smaller stride for change detection
    if pattern == 'TRENDING':
        return {
            'window_size': 128,
            'stride': 32,  # 75% overlap
            'derivative_depth': 2,
        }

    # Impulsive: small windows to catch spikes
    if pattern == 'IMPULSIVE':
        return {
            'window_size': 64,
            'stride': 16,
            'derivative_depth': 1,
        }

    # Default: standard windowing
    return {
        'window_size': 128,
        'stride': 64,  # 50% overlap
        'derivative_depth': 1,
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
    window_params = get_window_params(temporal, n_samples)
    output_hints = get_output_hints(temporal, spectral)

    config = {
        'engines': engines,
        'rolling_engines': [],
        'window_size': window_params['window_size'],
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

    # Build manifest
    manifest = {
        'version': '2.2',
        'job_id': job_id,
        'created_at': datetime.now().isoformat(),
        'generator': 'orthon.manifest_generator v2.2 (PR4/PR5 integrated)',

        'paths': {
            'observations': observations_path,
            'typology': typology_path,
            'output_dir': output_dir,
        },

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
            'note': 'stride computed per-signal from temporal pattern',
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
