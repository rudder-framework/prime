"""
PR9: Window/Stride from Characteristic Time
=============================================

Computes the characteristic time scale of each signal from typology_raw
measures, then derives optimal window and stride.

characteristic_time = "how fast does this signal change"
window = 2-3× characteristic_time (capture the behavior)  
stride = f(overlap needed based on dynamics speed)

State_vector and geometry inherit these - they just need the window.
"""

from typing import Dict, Any, Optional, Tuple
import math


# ============================================================
# Configuration
# ============================================================

WINDOW_CONFIG = {
    # Window = characteristic_time × multiplier
    'window_multiplier': 2.5,
    
    # Bounds
    'min_window': 64,
    'max_window': 2048,
    'min_stride': 8,
    
    # Stride fraction by dynamics speed
    'stride_fraction': {
        'fast': 0.25,      # 75% overlap
        'medium': 0.50,    # 50% overlap  
        'slow': 0.75,      # 25% overlap
    },
    'default_stride_fraction': 0.50,
    
    # Dynamics speed thresholds (char_time / n_samples)
    'dynamics_speed': {
        'fast_threshold': 0.01,   # < 1% = fast
        'slow_threshold': 0.10,   # > 10% = slow
    },
}


# ============================================================
# Characteristic Time Computation
# ============================================================

def compute_characteristic_time(
    acf_half_life: Optional[float],
    dominant_frequency: Optional[float],
    turning_point_ratio: Optional[float],
    hurst: Optional[float],
    n_samples: int,
    inter_event_time: Optional[float] = None,
    derivative_sparsity: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Compute characteristic time scale from typology_raw measures.
    
    Priority:
    1. acf_half_life - memory length
    2. 1/dominant_frequency - period  
    3. inter_event_time - event spacing
    4. level_duration (from derivative_sparsity) - step duration
    5. hurst-based estimate - persistence
    6. turning_point_ratio - oscillation rate
    7. fallback - 1% of signal length
    
    Returns:
        (characteristic_time, source)
    """
    candidates = []
    
    # 1. ACF half-life (memory)
    if _valid(acf_half_life) and acf_half_life > 0:
        candidates.append((acf_half_life, 'acf_half_life'))
    
    # 2. Dominant frequency → period
    if _valid(dominant_frequency) and dominant_frequency > 0:
        period = 1.0 / dominant_frequency
        if period < n_samples:
            candidates.append((period, 'dominant_frequency'))
    
    # 3. Inter-event time (sparse/impulsive)
    if _valid(inter_event_time) and inter_event_time > 0:
        candidates.append((inter_event_time, 'inter_event_time'))
    
    # 4. Level duration from derivative_sparsity (step signals)
    # High derivative_sparsity = signal mostly flat = long levels
    if _valid(derivative_sparsity) and derivative_sparsity > 0.5:
        # Estimate: if 90% of derivatives are zero, avg level = n_samples * 0.1
        level_duration = n_samples * (1 - derivative_sparsity)
        level_duration = max(level_duration, 16)
        candidates.append((level_duration, 'derivative_sparsity'))
    
    # 5. Hurst-based (persistent/trending signals)
    if _valid(hurst) and hurst > 0.7:
        hurst_time = n_samples * (hurst - 0.5) * 0.2
        hurst_time = max(hurst_time, 32)
        candidates.append((hurst_time, 'hurst'))
    
    # 6. Turning point ratio (oscillation rate)
    if _valid(turning_point_ratio) and 0.1 < turning_point_ratio < 0.9:
        tpr_time = 2.0 / turning_point_ratio
        tpr_time = min(tpr_time, n_samples * 0.1)
        candidates.append((tpr_time, 'turning_point_ratio'))
    
    # Select by priority
    priority = ['acf_half_life', 'dominant_frequency', 'inter_event_time',
                'derivative_sparsity', 'hurst', 'turning_point_ratio']
    
    if candidates:
        candidates.sort(key=lambda x: priority.index(x[1]) if x[1] in priority else 99)
        return candidates[0]
    
    # Fallback
    return (max(n_samples * 0.01, 64), 'fallback')


def _valid(value) -> bool:
    """Check if value is valid (not None, not NaN)."""
    if value is None:
        return False
    try:
        return not math.isnan(value)
    except (TypeError, ValueError):
        return False


# ============================================================
# Dynamics Speed Classification  
# ============================================================

def classify_dynamics_speed(characteristic_time: float, n_samples: int) -> str:
    """Classify dynamics as fast/medium/slow."""
    cfg = WINDOW_CONFIG['dynamics_speed']
    ratio = characteristic_time / n_samples
    
    if ratio < cfg['fast_threshold']:
        return 'fast'
    elif ratio > cfg['slow_threshold']:
        return 'slow'
    return 'medium'


# ============================================================
# Window/Stride Computation
# ============================================================

def compute_window_stride(
    characteristic_time: float,
    n_samples: int,
    dynamics_speed: str = None,
) -> Dict[str, Any]:
    """
    Compute window and stride from characteristic time.
    
    Returns: {window, stride, overlap}
    """
    cfg = WINDOW_CONFIG
    
    # Window = multiplier × characteristic_time
    window = int(characteristic_time * cfg['window_multiplier'])
    window = max(window, cfg['min_window'])
    window = min(window, cfg['max_window'])
    window = min(window, n_samples)
    
    # Dynamics speed
    if dynamics_speed is None:
        dynamics_speed = classify_dynamics_speed(characteristic_time, n_samples)
    
    # Stride
    stride_frac = cfg['stride_fraction'].get(dynamics_speed, cfg['default_stride_fraction'])
    stride = int(window * stride_frac)
    stride = max(stride, cfg['min_stride'])
    stride = min(stride, window)
    
    return {
        'window': window,
        'stride': stride,
        'overlap': round(1.0 - (stride / window), 3),
    }


# ============================================================
# Main API: From typology_raw row
# ============================================================

def compute_window_config(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute window/stride from a typology_raw row.
    
    Returns:
        {
            window_size: int,
            stride: int,
            overlap: float,
            characteristic_time: float,
            characteristic_source: str,
            dynamics_speed: str,
        }
    """
    n_samples = row.get('n_samples', 1000)
    
    char_time, source = compute_characteristic_time(
        acf_half_life=row.get('acf_half_life'),
        dominant_frequency=row.get('dominant_frequency'),
        turning_point_ratio=row.get('turning_point_ratio'),
        hurst=row.get('hurst'),
        n_samples=n_samples,
        inter_event_time=row.get('inter_event_time'),
        derivative_sparsity=row.get('derivative_sparsity'),
    )
    
    dynamics_speed = classify_dynamics_speed(char_time, n_samples)
    ws = compute_window_stride(char_time, n_samples, dynamics_speed)
    
    return {
        'window_size': ws['window'],
        'stride': ws['stride'],
        'overlap': ws['overlap'],
        'characteristic_time': round(char_time, 2),
        'characteristic_source': source,
        'dynamics_speed': dynamics_speed,
    }


# ============================================================
# Batch: Add to DataFrame
# ============================================================

def add_window_columns(df):
    """
    Add window config columns to typology_raw DataFrame.
    
    Adds: window_size, stride, characteristic_time, characteristic_source, dynamics_speed
    """
    import pandas as pd
    
    results = df.apply(lambda row: compute_window_config(row.to_dict()), axis=1)
    results_df = pd.DataFrame(results.tolist())
    
    for col in ['window_size', 'stride', 'characteristic_time', 'characteristic_source', 'dynamics_speed']:
        df[col] = results_df[col]
    
    return df
