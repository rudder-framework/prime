"""
PR10: Manifest Generator v2.4
==============================

Adds:
- system_window: common window for state_vector/geometry
- representation_type per signal: SPECTRAL vs TRAJECTORY
- features list per signal based on representation

The EEG paradigm: all signals at common time scale,
fast signals → band powers, slow signals → trajectory.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import yaml

from .system_window import (
    compute_system_window,
    compute_signal_representation,
    compute_system_representation,
    summarize_representations,
    REPRESENTATION_CONFIG,
)


# ============================================================
# Engine Selection (from PR9)
# ============================================================

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

ENGINE_ADDITIONS = {
    'TRENDING': ['hurst', 'rate_of_change', 'trend_r2', 'detrend_std', 'cusum',
                 'sample_entropy', 'acf_decay'],
    'PERIODIC': ['harmonics', 'thd', 'frequency_bands', 'fundamental_freq',
                 'phase_coherence', 'hurst', 'sample_entropy'],
    'QUASI_PERIODIC': ['frequency_bands', 'harmonics', 'hurst', 'rate_of_change',
                       'sample_entropy', 'acf_decay'],
    'CHAOTIC': ['lyapunov', 'correlation_dimension', 'recurrence_rate',
                'determinism', 'harmonics', 'sample_entropy', 'perm_entropy'],
    'RANDOM': ['spectral_entropy', 'band_power', 'hurst', 'frequency_bands',
               'sample_entropy', 'perm_entropy', 'acf_decay'],
    'STATIONARY': ['hurst', 'frequency_bands', 'spectral_entropy',
                   'sample_entropy', 'acf_decay'],
    'CONSTANT': [],
    'BINARY': ['transition_count', 'duty_cycle', 'mean_time_between'],
    'DISCRETE': ['level_histogram', 'transition_matrix', 'dwell_times'],
    'IMPULSIVE': ['peak_detection', 'inter_arrival', 'peak_amplitude_dist', 'hurst'],
    'EVENT': ['event_rate', 'inter_event_time', 'event_amplitude'],
    'STEP': ['changepoint_detection', 'level_means', 'regime_duration', 'hurst'],
    'INTERMITTENT': ['burst_detection', 'activity_ratio', 'silence_distribution', 'hurst'],
}


def get_engines_for_type(temporal_pattern: str) -> List[str]:
    """Get engine list for a temporal pattern type."""
    if temporal_pattern == 'CONSTANT':
        return []
    engines = set(BASE_ENGINES)
    engines.update(ENGINE_ADDITIONS.get(temporal_pattern, []))
    return sorted(engines)


# ============================================================
# Manifest Generator v2.4
# ============================================================

def generate_manifest(
    typology_df,
    observations_path: str,
    typology_path: str,
    output_dir: str,
    job_id: str = None,
    system_window_method: str = 'max',
) -> Dict[str, Any]:
    """
    Generate manifest v2.4 with system window and representations.
    
    Args:
        typology_df: DataFrame with typology_raw measures + temporal_pattern
        observations_path: Path to observations.parquet
        typology_path: Path to typology.parquet
        output_dir: Output directory
        job_id: Optional job ID
        system_window_method: 'max' | 'median' | 'p90'
        
    Returns:
        Complete manifest dict
    """
    if job_id is None:
        job_id = f"orthon-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Import here to avoid circular dependency
    from .characteristic_time import compute_window_config
    
    cohort_col = 'cohort' if 'cohort' in typology_df.columns else None
    
    # First pass: compute per-signal characteristic times
    signal_char_times = []
    signal_configs_raw = {}
    
    for _, row in typology_df.iterrows():
        signal_id = row['signal_id']
        cohort = row[cohort_col] if cohort_col else 'default'
        temporal = row.get('temporal_pattern', 'STATIONARY')
        
        # Skip CONSTANT
        if temporal == 'CONSTANT':
            continue
        
        # Compute characteristic time (from PR9)
        window_cfg = compute_window_config(row.to_dict())
        char_time = window_cfg['characteristic_time']
        
        signal_char_times.append({
            'signal_id': signal_id,
            'characteristic_time': char_time,
        })
        
        signal_configs_raw[(cohort, signal_id)] = {
            'row': row.to_dict(),
            'window_cfg': window_cfg,
            'temporal': temporal,
        }
    
    # Compute system window and representations
    system_rep = compute_system_representation(
        signal_char_times,
        system_window_method=system_window_method,
    )
    
    system_window = system_rep['system_window']
    system_stride = system_rep['system_stride']
    
    # Second pass: build cohort structure with representations
    cohorts = {}
    skip_signals = []
    all_engines = set()
    
    for _, row in typology_df.iterrows():
        signal_id = row['signal_id']
        cohort = row[cohort_col] if cohort_col else 'default'
        temporal = row.get('temporal_pattern', 'STATIONARY')
        
        if cohort not in cohorts:
            cohorts[cohort] = {}
        
        if temporal == 'CONSTANT':
            skip_signals.append(f"{cohort}/{signal_id}")
            continue
        
        # Get precomputed configs
        raw_cfg = signal_configs_raw.get((cohort, signal_id), {})
        window_cfg = raw_cfg.get('window_cfg', {})
        char_time = window_cfg.get('characteristic_time', 64)
        
        # Get representation config
        rep_cfg = system_rep['signals'].get(signal_id, {})
        representation = rep_cfg.get('representation', 'spectral')
        features = rep_cfg.get('features', [])
        
        # Get engines
        engines = get_engines_for_type(temporal)
        all_engines.update(engines)
        
        # Derivative depth
        derivative_depth = 2 if temporal == 'TRENDING' else 1
        if temporal in ['CONSTANT', 'BINARY', 'DISCRETE']:
            derivative_depth = 0
        
        config = {
            'engines': engines,
            'rolling_engines': [],
            
            # Per-signal window (for signal_vector characterization)
            'signal_window': window_cfg.get('window_size', 128),
            'signal_stride': window_cfg.get('stride', 64),
            
            # Characteristic time metadata
            'characteristic_time': char_time,
            'characteristic_source': window_cfg.get('characteristic_source', 'fallback'),
            'dynamics_speed': window_cfg.get('dynamics_speed', 'medium'),
            
            # Representation for state_vector
            'representation': representation,
            'state_features': features,
            'tau_ratio': rep_cfg.get('tau_ratio', 0),
            
            'derivative_depth': derivative_depth,
            'eigenvalue_budget': 5,
            
            'typology': {
                'temporal_pattern': temporal,
                'spectral': row.get('spectral', 'NARROWBAND'),
            },
        }
        
        # Add bands for spectral signals
        if representation == 'spectral':
            config['bands'] = rep_cfg.get('bands', REPRESENTATION_CONFIG['default_bands'])
        
        cohorts[cohort][signal_id] = config
    
    # Summary
    rep_summary = summarize_representations(system_rep)
    total_signals = len(typology_df)
    constant_count = len(skip_signals)
    
    manifest = {
        'version': '2.4',
        'job_id': job_id,
        'created_at': datetime.now().isoformat(),
        'generator': 'orthon.manifest_generator v2.4 (PR10: system_window + representation)',
        
        'paths': {
            'observations': observations_path,
            'typology': typology_path,
            'output_dir': output_dir,
        },
        
        # System-level window (for state_vector/geometry alignment)
        'system': {
            'window': system_window,
            'stride': system_stride,
            'window_method': system_window_method,
        },
        
        # Representation summary
        'representation_summary': {
            'n_spectral': rep_summary['n_spectral'],
            'n_trajectory': rep_summary['n_trajectory'],
            'spectral_signals': rep_summary['spectral_signals'],
            'trajectory_signals': rep_summary['trajectory_signals'],
            'bands': REPRESENTATION_CONFIG['default_bands'],
        },
        
        'summary': {
            'total_signals': total_signals,
            'total_cohorts': len(cohorts),
            'active_signals': total_signals - constant_count,
            'constant_signals': constant_count,
            'signal_engines': sorted(all_engines),
            'n_signal_engines': len(all_engines),
        },
        
        'params': {
            'spectral_threshold': REPRESENTATION_CONFIG['spectral_threshold'],
            'note': 'signals with τ/system_window < threshold get spectral representation',
        },
        
        'cohorts': cohorts,
        
        'pair_engines': ['granger', 'transfer_entropy'],
        'symmetric_pair_engines': ['cointegration', 'correlation', 'mutual_info'],
        'skip_signals': skip_signals,
    }
    
    return manifest


def manifest_to_yaml(manifest: Dict[str, Any]) -> str:
    """Convert manifest to YAML string."""
    return yaml.dump(manifest, default_flow_style=False, sort_keys=False)


def save_manifest(manifest: Dict[str, Any], path: str) -> None:
    """Save manifest to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
