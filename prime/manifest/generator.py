"""
Manifest Generator v2.6 - Intervention Mode Support
====================================================

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
- **Intervention mode** (v2.6) - for fault injection / event response datasets

Key features in v2.5:
- engine_windows: Minimum window sizes for FFT-based and long-range engines
- engine_window_overrides: Per-signal overrides when signal window < engine min
- system.window: Common window for state_vector/geometry alignment
- window_method: Tracks how window was determined (period, acf_half_life, etc.)
- window_confidence: high/medium/low confidence in window selection
- representation: spectral (fast signals) vs trajectory (slow signals)

Key features in v2.6 (Intervention Mode):
- intervention.enabled: True to enable intervention mode
- intervention.event_index: Sample index where intervention occurs (e.g., 20)
- intervention.pre_samples: Samples before intervention (default: event_index)
- intervention.post_samples: Samples after intervention (default: n_samples - event_index)
- When enabled:
  - No windowing: compute over full cohort span
  - FTLE computed per cohort (not pooled)
  - Granger computed pre vs post intervention
  - Breaks reported relative to event_index
  - Eigenvalue trajectories tracked across full span

Prime classifies → Manifest specifies → Manifold executes
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import os
import yaml


def _derive_output_dir(observations_path: str) -> str:
    """Derive output directory from observations filename.

    {prefix}_observations.parquet → {prefix}_output/
    observations.parquet          → output/
    """
    obs_filename = os.path.basename(observations_path)
    parent = os.path.dirname(observations_path)

    if obs_filename == "observations.parquet":
        prefix = ""
    else:
        prefix = obs_filename.replace("_observations.parquet", "")

    if prefix:
        return os.path.join(parent, f"{prefix}_output") if parent else f"{prefix}_output"
    else:
        return os.path.join(parent, "output") if parent else "output"


def _normalize_temporal(val) -> list:
    """Normalize temporal_pattern to list[str], handling str, list, numpy array."""
    if isinstance(val, str):
        return [val]
    # Handle numpy arrays, polars Series, or any iterable
    try:
        return [str(x) for x in val]
    except TypeError:
        return [str(val)]

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
        'remove': [],  # Let Manifold decide what makes sense
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
        'add': ['trend_overlay', 'segment_comparison', 'cusum_plot', 'spectral_density', 'waterfall'],
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
        'add': ['histogram', 'spectral_density', 'waterfall'],
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


def get_window_params(temporal_pattern, n_samples: int, typology_row: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get window/stride parameters based on signal type and typology measures.

    Uses primary classification only for window selection.

    Returns:
        dict with window_size, stride, derivative_depth, window_method, window_confidence
    """
    patterns = _normalize_temporal(temporal_pattern)
    pattern = patterns[0].upper()

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
    temporal_pattern,
) -> List[str]:
    """
    Apply type-specific engine add/remove.

    Accepts str or list[str] for dual classification.
    Removes from primary only, adds from ALL patterns (union).
    """
    patterns = _normalize_temporal(temporal_pattern)
    engines = list(base_engines)

    for i, pattern in enumerate(patterns):
        p = pattern.lower()
        adjustments = ENGINE_ADJUSTMENTS.get(p, {})

        # Removes from primary only (first element)
        if i == 0:
            for eng in adjustments.get('remove', []):
                if eng == '*':
                    return []
                if eng in engines:
                    engines.remove(eng)

        # Adds from ALL patterns (union)
        for eng in adjustments.get('add', []):
            if eng not in engines:
                engines.append(eng)

    return engines


def apply_viz_adjustments(
    base_viz: List[str],
    temporal_pattern,
) -> List[str]:
    """
    Apply type-specific visualization add/remove.

    Accepts str or list[str] for dual classification.
    Removes from primary only, adds from ALL patterns (union).
    """
    patterns = _normalize_temporal(temporal_pattern)
    viz = list(base_viz)

    for i, pattern in enumerate(patterns):
        p = pattern.lower()
        adjustments = VIZ_ADJUSTMENTS.get(p, {})

        # Removes from primary only
        if i == 0:
            for v in adjustments.get('remove', []):
                if v == '*':
                    return []
                if v in viz:
                    viz.remove(v)

        # Adds from ALL patterns (union)
        for v in adjustments.get('add', []):
            if v not in viz:
                viz.append(v)

    return viz


def get_output_hints(temporal_pattern, spectral: str) -> Dict[str, Any]:
    """Get output configuration hints for Manifold."""
    patterns = _normalize_temporal(temporal_pattern)
    pattern = patterns[0].upper()

    hints = {}

    # Spectral output mode — all continuous types get per_bin (waterfall-ready)
    continuous_types = ('PERIODIC', 'QUASI_PERIODIC', 'TRENDING', 'CHAOTIC',
                        'RANDOM', 'STATIONARY', 'DRIFTING')
    if pattern in continuous_types:
        hints['spectral'] = {
            'output_mode': 'per_bin',
            'n_bins': 'auto',
            'include_phase': False,
            'note': 'waterfall-ready output',
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

    # Handle dual classification: temporal_pattern may be list[str], str, or numpy array
    temporal_raw = typology_row.get('temporal_pattern', 'STATIONARY')
    temporal = _normalize_temporal(temporal_raw)
    temporal_primary = typology_row.get('temporal_primary', temporal[0])
    if not isinstance(temporal_primary, str):
        temporal_primary = str(temporal_primary)

    spectral = typology_row.get('spectral', 'NARROWBAND')
    n_samples = typology_row.get('n_samples', 1000)

    # Use engines from typology if present (single source of truth from classify step)
    typology_engines = typology_row.get('engines')
    if typology_engines is not None and len(typology_engines) > 0:
        engines = list(typology_engines)
    else:
        engines = apply_engine_adjustments(base_engines, temporal)
    visualizations = apply_viz_adjustments(base_viz, temporal)
    window_params = get_window_params(temporal_primary, n_samples, typology_row)
    output_hints = get_output_hints(temporal_primary, spectral)

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
            'temporal_primary': temporal_primary,
            'temporal_secondary': temporal[1] if len(temporal) > 1 else None,
            'classification_confidence': typology_row.get('classification_confidence', 'clear'),
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
    if temporal_primary in ('CONSTANT', 'BINARY', 'DISCRETE', 'IMPULSIVE', 'EVENT', 'STEP', 'INTERMITTENT'):
        config['is_discrete_sparse'] = True

    return config


def build_manifest(
    typology_df,
    observations_path: str = 'observations.parquet',
    typology_path: str = 'typology.parquet',
    output_dir: str = None,
    job_id: str = None,
    base_engines: List[str] = None,
    pair_engines: List[str] = None,
    symmetric_pair_engines: List[str] = None,
    intervention: Dict[str, Any] = None,
    axis: str = 'time',
    run_id: int = 1,
) -> Dict[str, Any]:
    """
    Build complete manifest from typology DataFrame.

    Args:
        typology_df: DataFrame with signal_id, cohort, temporal_pattern, spectral, n_samples
        observations_path: Path to observations parquet
        typology_path: Path to typology parquet
        output_dir: Output directory for Manifold. If None, derived from
                    observations_path: {prefix}_observations.parquet → {prefix}_output/
        job_id: Optional job ID (auto-generated if None)
        base_engines: Base engine list
        pair_engines: Pairwise engine list
        symmetric_pair_engines: Symmetric pairwise engine list
        intervention: Intervention mode config dict with:
            - enabled: True to enable intervention mode
            - event_index: Sample index where intervention occurs (e.g., 20 for TEP)
            - pre_samples: Optional, samples before intervention (default: event_index)
            - post_samples: Optional, samples after intervention (default: computed)
            When enabled, Manifold computes:
            - Full-span eigenvalue trajectories per cohort (no windowing)
            - FTLE per cohort (not pooled across cohorts)
            - Granger pre vs post intervention
            - Breaks relative to event_index
        axis: Signal used as ordering axis (default: "time")
        run_id: Sequential run number for this domain (default: 1)

    Returns:
        Complete manifest dict
    """
    if output_dir is None:
        output_dir = _derive_output_dir(observations_path)

    if job_id is None:
        job_id = f"prime-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if base_engines is None:
        base_engines = list(BASE_ENGINES)
    if pair_engines is None:
        pair_engines = ['granger', 'transfer_entropy']
    if symmetric_pair_engines is None:
        symmetric_pair_engines = ['cointegration', 'correlation', 'mutual_info']

    # Process intervention mode
    intervention_enabled = False
    intervention_config = None
    if intervention and intervention.get('enabled', False):
        intervention_enabled = True
        event_index = intervention.get('event_index', 0)
        intervention_config = {
            'enabled': True,
            'event_index': event_index,
            'pre_samples': intervention.get('pre_samples', event_index),
            'post_samples': intervention.get('post_samples'),  # None = compute from data
            'note': 'Intervention mode: full-span per-cohort analysis with pre/post split',
        }

    # Build cohort structure
    cohorts = {}
    skip_signals = []
    all_engines = set()

    for _, row in typology_df.iterrows():
        signal_id = row['signal_id']
        cohort = row['cohort']
        # Handle dual classification: temporal_pattern may be list, str, or numpy array
        temporal_primary = row.get('temporal_primary', None)
        if temporal_primary is None:
            tp = row.get('temporal_pattern', 'STATIONARY')
            temporal_primary = _normalize_temporal(tp)[0]
        if not isinstance(temporal_primary, str):
            temporal_primary = str(temporal_primary)

        # Initialize cohort if needed
        if cohort not in cohorts:
            cohorts[cohort] = {}

        # Skip CONSTANT signals
        if temporal_primary == 'CONSTANT':
            skip_signals.append(f"{cohort}/{signal_id}")
            continue

        # Build signal config
        config = build_signal_config(
            signal_id=signal_id,
            cohort=cohort,
            typology_row=row.to_dict(),
            base_engines=base_engines,
        )

        # Override window params for intervention mode (full span, no windowing)
        if intervention_enabled:
            n_samples = row.get('n_samples', 1000)
            config['window_size'] = n_samples
            config['stride'] = n_samples  # Single window = full span
            config['window_method'] = 'intervention_full_span'
            config['window_confidence'] = 'high'
            # Remove engine window overrides - we're using full span
            config.pop('engine_window_overrides', None)

        cohorts[cohort][signal_id] = config
        all_engines.update(config['engines'])

    # Count signals
    total_signals = len(typology_df)
    constant_signals = len(skip_signals)
    active_signals = total_signals - constant_signals

    # Calculate system window
    all_windows = []
    all_strides = []
    for cohort_signals in cohorts.values():
        for sig_config in cohort_signals.values():
            all_windows.append(sig_config['window_size'])
            all_strides.append(sig_config['stride'])

    if intervention_enabled:
        # Intervention mode: system window = max (full span per cohort)
        system_window = max(all_windows) if all_windows else 1000
        system_stride = system_window  # No overlap in intervention mode
    elif all_windows:
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
        'version': '2.6',
        'job_id': job_id,
        'created_at': datetime.now().isoformat(),
        'generator': 'prime.manifest_generator v2.6 (intervention mode)',

        'paths': {
            'observations': observations_path,
            'typology': typology_path,
            'output_dir': output_dir,
        },

        'ordering_signal': axis,

        'signal_0': {
            'name': axis,
            'unit': 'arbitrary',
        },

        'parameterization': {
            'axis_signal': axis,
            'run_id': run_id,
            'source': 'observations.parquet',
        },

        'system': {
            'window': system_window,
            'stride': system_stride,
            'mode': 'auto',
            'note': 'Full-span per-cohort (intervention mode)' if intervention_enabled else 'Common window for state_vector/geometry alignment',
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

    # Add intervention section if enabled
    if intervention_config:
        manifest['intervention'] = intervention_config

    # Add atlas section (system-level engines: stages 16-23)
    manifest['atlas'] = build_atlas_config(
        typology_df,
        n_signals=active_signals,
        intervention=intervention,
    )

    return manifest


def build_atlas_config(
    typology_df,
    n_signals: int = 0,
    intervention: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Build atlas section for system-level engines (stages 16-23).

    Atlas engines operate on the system/cohort level, not per-signal.
    They are included in the manifest as a top-level `atlas` section.

    Args:
        typology_df: DataFrame with typology classifications
        n_signals: Number of active (non-constant) signals
        intervention: Intervention config dict (if any)

    Returns:
        Atlas configuration dict for manifest
    """
    atlas = {}

    # FTLE: forward is computed by core pipeline (stage_08),
    # atlas adds backward and rolling
    atlas['ftle'] = {
        'directions': ['forward', 'backward'],
        'rolling': True,
        'rolling_window': 200,
        'rolling_stride': 50,
    }

    # velocity_field: always enabled
    atlas['velocity_field'] = {
        'enabled': True,
        'smooth': 'savgol',
        'include_components': False,
    }

    # ridge_proximity: requires ftle_rolling + velocity_field
    atlas['ridge_proximity'] = {
        'enabled': True,
        'ridge_threshold': 0.05,
    }

    # break_sequence: enabled if breaks exist
    break_config = {
        'enabled': True,
    }
    if intervention and intervention.get('enabled'):
        break_config['reference_index'] = intervention.get('event_index', 0)
    atlas['break_sequence'] = break_config

    # Segments: auto-detect from intervention or leave for user
    segments = None
    if intervention and intervention.get('enabled'):
        event_idx = intervention.get('event_index', 0)
        segments = [
            {'name': 'pre', 'range': [0, event_idx - 1]},
            {'name': 'post', 'range': [event_idx, None]},
        ]
    if segments:
        atlas['segments'] = segments

    # segment_comparison + info_flow_delta: enabled when segments defined
    atlas['segment_comparison'] = {
        'enabled': segments is not None,
    }
    atlas['info_flow_delta'] = {
        'enabled': segments is not None,
    }

    return atlas


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
            tp = typology.get('temporal_primary', typology.get('temporal_pattern'))
            tp_check = tp[0] if isinstance(tp, list) else tp
            if tp_check == 'CONSTANT':
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
