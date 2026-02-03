"""
PR10: System Window + Representation Type
==========================================

Computes:
1. system_window - max(characteristic_time) across all signals
2. representation_type per signal - SPECTRAL vs TRAJECTORY

The EEG insight: all signals represented at common window, but
fast signals → spectral features (band powers)
slow signals → trajectory features (value, slope, curvature)

No data loss. Physically meaningful. Common framework.
"""

from typing import Dict, Any, List, Tuple
import math


# ============================================================
# Configuration
# ============================================================

REPRESENTATION_CONFIG = {
    # Threshold: if τ / system_window < threshold → SPECTRAL
    # Above threshold → TRAJECTORY
    'spectral_threshold': 0.3,  # τ < 30% of system_window → spectral
    
    # Default frequency bands (Hz-agnostic, normalized to Nyquist)
    # These are fractions of the sampling rate
    'default_bands': [0.001, 0.01, 0.05, 0.1, 0.25, 0.5],
    
    # Spectral features to extract per band
    'spectral_features': [
        'band_power',
        'band_power_ratio',  # relative to total
        'band_dominant_freq',
        'band_phase_stability',
    ],
    
    # Trajectory features to extract
    'trajectory_features': [
        'value_start',
        'value_end', 
        'value_mean',
        'slope',
        'curvature',
        'residual_std',
        'monotonicity',
    ],
    
    # Minimum system window
    'min_system_window': 64,
    
    # Maximum system window  
    'max_system_window': 4096,
}


# ============================================================
# System Window Computation
# ============================================================

def compute_system_window(
    characteristic_times: List[float],
    method: str = 'max',
) -> int:
    """
    Compute system-level window from per-signal characteristic times.
    
    Args:
        characteristic_times: List of τ values for each signal
        method: 'max' | 'median' | 'p90'
        
    Returns:
        System window size (integer)
    """
    cfg = REPRESENTATION_CONFIG
    
    if not characteristic_times:
        return cfg['min_system_window']
    
    # Filter out invalid values
    valid = [t for t in characteristic_times if t is not None and not math.isnan(t) and t > 0]
    
    if not valid:
        return cfg['min_system_window']
    
    if method == 'max':
        system_tau = max(valid)
    elif method == 'median':
        sorted_valid = sorted(valid)
        n = len(sorted_valid)
        system_tau = sorted_valid[n // 2]
    elif method == 'p90':
        sorted_valid = sorted(valid)
        idx = int(len(sorted_valid) * 0.9)
        system_tau = sorted_valid[min(idx, len(sorted_valid) - 1)]
    else:
        system_tau = max(valid)
    
    # Apply multiplier for full capture (same as PR9)
    system_window = int(system_tau * 2.5)
    
    # Apply bounds
    system_window = max(system_window, cfg['min_system_window'])
    system_window = min(system_window, cfg['max_system_window'])
    
    return system_window


# ============================================================
# Representation Type Classification
# ============================================================

def classify_representation(
    characteristic_time: float,
    system_window: int,
) -> str:
    """
    Classify signal representation type.
    
    τ << system_window → SPECTRAL (characterized by frequency content)
    τ ≈ system_window  → TRAJECTORY (characterized by path/trend)
    
    Args:
        characteristic_time: Signal's τ
        system_window: System-level window
        
    Returns:
        'spectral' or 'trajectory'
    """
    cfg = REPRESENTATION_CONFIG
    
    if characteristic_time is None or math.isnan(characteristic_time):
        return 'spectral'  # Default to spectral if unknown
    
    ratio = characteristic_time / system_window
    
    if ratio < cfg['spectral_threshold']:
        return 'spectral'
    else:
        return 'trajectory'


def get_features_for_representation(representation: str) -> List[str]:
    """Get feature list for a representation type."""
    cfg = REPRESENTATION_CONFIG
    
    if representation == 'spectral':
        return cfg['spectral_features']
    else:
        return cfg['trajectory_features']


def get_bands() -> List[float]:
    """Get default frequency bands."""
    return REPRESENTATION_CONFIG['default_bands']


# ============================================================
# Per-Signal Representation Config
# ============================================================

def compute_signal_representation(
    characteristic_time: float,
    system_window: int,
    sample_rate: float = None,
) -> Dict[str, Any]:
    """
    Compute representation config for a signal.
    
    Args:
        characteristic_time: Signal's τ
        system_window: System-level window
        sample_rate: Optional sample rate for band Hz conversion
        
    Returns:
        {
            representation: 'spectral' | 'trajectory',
            features: [...],
            bands: [...] (if spectral),
            tau_ratio: float,
        }
    """
    representation = classify_representation(characteristic_time, system_window)
    features = get_features_for_representation(representation)
    
    result = {
        'representation': representation,
        'features': features,
        'tau_ratio': characteristic_time / system_window if system_window > 0 else 0,
    }
    
    if representation == 'spectral':
        result['bands'] = get_bands()
        if sample_rate:
            # Convert normalized bands to Hz
            result['bands_hz'] = [b * sample_rate for b in result['bands']]
    
    return result


# ============================================================
# Batch: Process all signals
# ============================================================

def compute_system_representation(
    signals: List[Dict[str, Any]],
    system_window_method: str = 'max',
) -> Dict[str, Any]:
    """
    Compute system window and per-signal representations.
    
    Args:
        signals: List of dicts with 'signal_id' and 'characteristic_time'
        system_window_method: 'max' | 'median' | 'p90'
        
    Returns:
        {
            system_window: int,
            system_stride: int,
            signals: {
                signal_id: {representation config}
            }
        }
    """
    # Extract characteristic times
    char_times = [s.get('characteristic_time', 0) for s in signals]
    
    # Compute system window
    system_window = compute_system_window(char_times, method=system_window_method)
    
    # System stride (50% overlap at system level)
    system_stride = system_window // 2
    
    # Per-signal representations
    signal_configs = {}
    for sig in signals:
        signal_id = sig['signal_id']
        tau = sig.get('characteristic_time', 0)
        
        rep_config = compute_signal_representation(tau, system_window)
        rep_config['characteristic_time'] = tau
        
        signal_configs[signal_id] = rep_config
    
    return {
        'system_window': system_window,
        'system_stride': system_stride,
        'signals': signal_configs,
    }


# ============================================================
# Summary statistics
# ============================================================

def summarize_representations(system_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize representation distribution.
    
    Returns:
        {
            n_spectral: int,
            n_trajectory: int,
            spectral_signals: [...],
            trajectory_signals: [...],
        }
    """
    spectral = []
    trajectory = []
    
    for signal_id, config in system_config['signals'].items():
        if config['representation'] == 'spectral':
            spectral.append(signal_id)
        else:
            trajectory.append(signal_id)
    
    return {
        'n_spectral': len(spectral),
        'n_trajectory': len(trajectory),
        'spectral_signals': spectral,
        'trajectory_signals': trajectory,
    }
