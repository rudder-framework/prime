"""
Window Sizing
=============
Computes window_size and stride from signal characteristics.

Two modes:
    1. from_length(n_samples) — no measures needed, pure defaults
    2. from_measures(measures) — uses ACF, frequency, hurst, etc.

Manifold calls from_length() before any computation.
Manifold calls from_measures() after signal_vector window 1 exists.
Prime calls from_measures() for manifest optimization on re-runs.
"""

import math
from typing import Dict, Any, Tuple

from typology.config import CONFIG


# =================================================================
# Public API
# =================================================================

def from_length(n_samples: int) -> Dict[str, Any]:
    """
    Default window/stride from observation count alone.
    No typology measures needed. Called before any computation.

    Args:
        n_samples: Number of observations for this signal.

    Returns:
        {window_size, stride, source: 'length_default'}
    """
    for max_n, window, stride in CONFIG['defaults']['window_by_length']:
        if max_n is None or n_samples <= max_n:
            if window is None:
                # Too short to window
                return {
                    'window_size': n_samples,
                    'stride': n_samples,
                    'source': 'too_short',
                }
            return {
                'window_size': window,
                'stride': stride,
                'source': 'length_default',
            }

    # Shouldn't reach here, but fallback
    return {'window_size': 256, 'stride': 64, 'source': 'fallback'}


def from_measures(measures: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimal window/stride from signal measures (ACF, frequency, hurst, etc.).
    Called after signal_vector or typology_raw exists.

    Args:
        measures: Dict with optional keys:
            acf_half_life, dominant_frequency, turning_point_ratio,
            hurst, n_samples, derivative_sparsity, inter_event_time

    Returns:
        {window_size, stride, overlap, characteristic_time,
         characteristic_source, dynamics_speed}
    """
    n_samples = measures.get('n_samples', 1000)

    char_time, source = _characteristic_time(
        acf_half_life=measures.get('acf_half_life'),
        dominant_frequency=measures.get('dominant_frequency'),
        turning_point_ratio=measures.get('turning_point_ratio'),
        hurst=measures.get('hurst'),
        n_samples=n_samples,
        inter_event_time=measures.get('inter_event_time'),
        derivative_sparsity=measures.get('derivative_sparsity'),
    )

    dynamics_speed = _classify_speed(char_time, n_samples)
    ws = _window_stride(char_time, n_samples, dynamics_speed)

    return {
        'window_size': ws['window'],
        'stride': ws['stride'],
        'overlap': ws['overlap'],
        'characteristic_time': round(char_time, 2),
        'characteristic_source': source,
        'dynamics_speed': dynamics_speed,
    }


def system_window(characteristic_times: list, method: str = 'max') -> int:
    """
    Compute system-level window from per-signal characteristic times.
    Uses the slowest signal to set the system window.

    Args:
        characteristic_times: List of τ values per signal.
        method: 'max' | 'median' | 'p90'

    Returns:
        System window size (int).
    """
    valid = [t for t in characteristic_times
             if t is not None and not math.isnan(t) and t > 0]

    if not valid:
        return CONFIG['window']['min_window']

    if method == 'max':
        tau = max(valid)
    elif method == 'median':
        s = sorted(valid)
        tau = s[len(s) // 2]
    elif method == 'p90':
        s = sorted(valid)
        tau = s[min(int(len(s) * 0.9), len(s) - 1)]
    else:
        tau = max(valid)

    window = int(tau * CONFIG['window']['multiplier'])
    window = max(window, CONFIG['window']['min_window'])
    window = min(window, CONFIG['window']['max_window'])
    return window


# =================================================================
# Internals
# =================================================================

def _valid(value):
    if value is None:
        return False
    try:
        return not math.isnan(value)
    except (TypeError, ValueError):
        return False


def _characteristic_time(
    acf_half_life, dominant_frequency, turning_point_ratio,
    hurst, n_samples, inter_event_time=None, derivative_sparsity=None,
) -> Tuple[float, str]:
    """
    Compute characteristic time from available measures.

    Priority:
        1. ACF half-life (memory length)
        2. 1/dominant_frequency (period)
        3. inter_event_time (event spacing)
        4. derivative_sparsity (step duration)
        5. hurst-based estimate (persistence)
        6. turning_point_ratio (oscillation rate)
        7. fallback (1% of signal length)
    """
    candidates = []

    # 1. ACF half-life
    if _valid(acf_half_life) and acf_half_life > 0:
        candidates.append((acf_half_life, 'acf_half_life'))

    # 2. Dominant frequency → period
    if _valid(dominant_frequency) and dominant_frequency > 0:
        period = 1.0 / dominant_frequency
        if period < n_samples:
            candidates.append((period, 'dominant_frequency'))

    # 3. Inter-event time
    if _valid(inter_event_time) and inter_event_time > 0:
        candidates.append((inter_event_time, 'inter_event_time'))

    # 4. Derivative sparsity → step duration
    if _valid(derivative_sparsity) and derivative_sparsity > 0.5:
        # High sparsity = signal holds levels for long periods
        level_duration = n_samples * derivative_sparsity * 0.1
        candidates.append((level_duration, 'derivative_sparsity'))

    # 5. Hurst-based estimate
    if _valid(hurst):
        if hurst > 0.8:
            # Very persistent → slow dynamics → long characteristic time
            tau_hurst = n_samples * 0.05
            candidates.append((tau_hurst, 'hurst_persistent'))
        elif hurst < 0.3:
            # Anti-persistent → fast dynamics → short characteristic time
            tau_hurst = n_samples * 0.005
            candidates.append((tau_hurst, 'hurst_anti_persistent'))

    # 6. Turning point ratio
    if _valid(turning_point_ratio) and turning_point_ratio > 0:
        if turning_point_ratio < 0.4:
            # Few turning points → slow dynamics
            tau_tpr = n_samples * 0.03
            candidates.append((tau_tpr, 'turning_point_ratio'))
        elif turning_point_ratio > 0.8:
            # Many turning points → fast dynamics
            tau_tpr = n_samples * 0.005
            candidates.append((tau_tpr, 'turning_point_ratio'))

    # Pick best candidate (prefer direct measurements)
    if candidates:
        return candidates[0]

    # 7. Fallback
    return (n_samples * 0.01, 'fallback')


def _classify_speed(characteristic_time: float, n_samples: int) -> str:
    cfg = CONFIG['window']['dynamics_speed']
    ratio = characteristic_time / n_samples if n_samples > 0 else 0

    if ratio < cfg['fast_threshold']:
        return 'fast'
    if ratio > cfg['slow_threshold']:
        return 'slow'
    return 'medium'


def _window_stride(
    characteristic_time: float,
    n_samples: int,
    dynamics_speed: str,
) -> Dict[str, Any]:
    cfg = CONFIG['window']

    window = int(characteristic_time * cfg['multiplier'])
    window = max(window, cfg['min_window'])
    window = min(window, cfg['max_window'])
    window = min(window, n_samples)

    stride_frac = cfg['stride_fraction'].get(
        dynamics_speed, cfg['default_stride_fraction'])
    stride = int(window * stride_frac)
    stride = max(stride, cfg['min_stride'])
    stride = min(stride, window)

    overlap = round(1.0 - (stride / window), 3) if window > 0 else 0

    return {'window': window, 'stride': stride, 'overlap': overlap}
