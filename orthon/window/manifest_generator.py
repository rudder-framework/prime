"""
PR9: Updated Manifest Generator
================================

Generates manifest with window/stride derived from characteristic time.

Key change: Window/stride now data-driven from typology_raw measures,
not hardcoded by type.

Manifest output per signal:
- window_size: derived from characteristic_time
- stride: derived from dynamics_speed  
- characteristic_time: the computed time scale
- characteristic_source: which measure determined it
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import yaml

from .characteristic_time import compute_window_config


# ============================================================
# Engine Selection (unchanged from PR6)
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
    'CONSTANT': [],  # Skip all
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
    additions = ENGINE_ADDITIONS.get(temporal_pattern, [])
    engines.update(additions)
    return sorted(engines)


# ============================================================
# Signal Config Builder
# ============================================================

def build_signal_config(
    signal_id: str,
    cohort: str,
    typology_row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build signal config with data-driven window/stride.
    
    Args:
        signal_id: Signal identifier
        cohort: Cohort/unit identifier
        typology_row: Row from typology_raw with all measures
        
    Returns:
        Complete signal config for manifest
    """
    temporal_pattern = typology_row.get('temporal_pattern', 'STATIONARY')
    spectral = typology_row.get('spectral', 'NARROWBAND')
    n_samples = typology_row.get('n_samples', 1000)
    
    # Get engines
    engines = get_engines_for_type(temporal_pattern)
    
    # Compute window/stride from characteristic time
    window_cfg = compute_window_config(typology_row)
    
    # Derivative depth based on temporal pattern
    derivative_depth = 2 if temporal_pattern == 'TRENDING' else 1
    if temporal_pattern in ['CONSTANT', 'BINARY', 'DISCRETE']:
        derivative_depth = 0
    
    return {
        'engines': engines,
        'rolling_engines': [],
        
        # Data-driven window/stride
        'window_size': window_cfg['window_size'],
        'stride': window_cfg['stride'],
        'characteristic_time': window_cfg['characteristic_time'],
        'characteristic_source': window_cfg['characteristic_source'],
        'dynamics_speed': window_cfg['dynamics_speed'],
        
        'derivative_depth': derivative_depth,
        'eigenvalue_budget': 5,
        
        'typology': {
            'temporal_pattern': temporal_pattern,
            'spectral': spectral,
        },
        
        'is_discrete_sparse': temporal_pattern in [
            'BINARY', 'DISCRETE', 'IMPULSIVE', 'EVENT', 'STEP', 'INTERMITTENT'
        ],
    }


# ============================================================
# Manifest Generator
# ============================================================

def generate_manifest(
    typology_df,
    observations_path: str,
    typology_path: str,
    output_dir: str,
    job_id: str = None,
) -> Dict[str, Any]:
    """
    Generate manifest with data-driven window/stride.
    
    Args:
        typology_df: DataFrame with typology_raw + temporal_pattern
        observations_path: Path to observations.parquet
        typology_path: Path to typology.parquet
        output_dir: Output directory for PRISM
        job_id: Optional job ID
        
    Returns:
        Complete manifest dict
    """
    if job_id is None:
        job_id = f"orthon-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Detect cohort column
    cohort_col = 'cohort' if 'cohort' in typology_df.columns else None
    
    cohorts = {}
    skip_signals = []
    all_engines = set()
    
    for _, row in typology_df.iterrows():
        signal_id = row['signal_id']
        cohort = row[cohort_col] if cohort_col else 'default'
        temporal = row.get('temporal_pattern', 'STATIONARY')
        
        # Initialize cohort
        if cohort not in cohorts:
            cohorts[cohort] = {}
        
        # Skip CONSTANT
        if temporal == 'CONSTANT':
            skip_signals.append(f"{cohort}/{signal_id}")
            continue
        
        # Build signal config
        config = build_signal_config(signal_id, cohort, row.to_dict())
        cohorts[cohort][signal_id] = config
        all_engines.update(config['engines'])
    
    # Summary stats
    total_signals = len(typology_df)
    constant_count = len(skip_signals)
    active_signals = total_signals - constant_count
    
    manifest = {
        'version': '2.3',
        'job_id': job_id,
        'created_at': datetime.now().isoformat(),
        'generator': 'orthon.manifest_generator v2.3 (PR9: characteristic_time)',
        
        'paths': {
            'observations': observations_path,
            'typology': typology_path,
            'output_dir': output_dir,
        },
        
        'summary': {
            'total_signals': total_signals,
            'total_cohorts': len(cohorts),
            'active_signals': active_signals,
            'constant_signals': constant_count,
            'signal_engines': sorted(all_engines),
            'rolling_engines': [],
            'pair_engines': ['granger', 'transfer_entropy'],
            'symmetric_pair_engines': ['cointegration', 'correlation', 'mutual_info'],
            'n_signal_engines': len(all_engines),
            'n_rolling_engines': 0,
        },
        
        'params': {
            'note': 'window/stride computed from characteristic_time per signal',
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
