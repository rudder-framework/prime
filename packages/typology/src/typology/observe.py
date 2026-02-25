"""
Observation-Level Measures
==========================
Cheap O(n) computations on raw signal values. Pure numpy.
No pmtvs dependency. No windowing. Called BEFORE signal_vector exists.

These provide:
    1. Enough info for initial window sizing (ACF, dominant freq, TPR)
    2. Quick classification (is_constant, is_integer, basic stats)
    3. Initial typology for engine gating

Manifold calls: typology.observe(values) → measures + window + initial class
"""

import numpy as np
from typing import Dict, Any


def observe(values: np.ndarray) -> Dict[str, Any]:
    """
    Compute cheap measures from raw signal values.

    Args:
        values: 1D array of signal observations, sorted by index.

    Returns:
        Dict with all cheap measures + window recommendation + initial class.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[~np.isnan(values)]
    n = len(values)

    if n == 0:
        return _empty(n)

    # Basic stats (O(n))
    mean = float(np.mean(values))
    std = float(np.std(values))
    n_unique = len(np.unique(values))
    is_constant = n_unique <= 2 or std < 1e-10

    if is_constant:
        return _constant_result(n, mean)

    # Distribution (O(n))
    kurt = float(_kurtosis(values))
    skew = float(_skewness(values))
    crest = float(np.max(np.abs(values)) / np.sqrt(np.mean(values ** 2)))

    # Continuity (O(n))
    unique_ratio = float(len(np.unique(values)) / n) if n > 0 else 1.0
    is_integer = bool(np.all(values == np.floor(values)))
    sparsity = float(np.sum(values == 0) / n) if n > 0 else 0.0

    # Derivative (O(n))
    diff = np.diff(values)
    derivative_sparsity = float(np.sum(np.abs(diff) < 1e-10) / len(diff)) if len(diff) > 0 else 0.0

    # Turning point ratio (O(n))
    tpr = _turning_point_ratio(values)

    # ACF half-life (O(n log n) via FFT)
    acf_half_life = _acf_half_life(values)

    # Dominant frequency (O(n log n) via FFT)
    dom_freq, spectral_flatness, spectral_slope, snr, is_first_bin = _spectral_quick(values)

    # Hurst estimate (O(n) — simple R/S on 2 scales, not full Mandelbrot)
    hurst = _hurst_quick(values)

    # Permutation entropy (O(n) — fixed order 3)
    perm_entropy = _perm_entropy_quick(values)

    measures = {
        'n_samples': n,
        'signal_mean': mean,
        'signal_std': std,
        'is_constant': is_constant,
        'kurtosis': kurt,
        'skewness': skew,
        'crest_factor': crest,
        'unique_ratio': unique_ratio,
        'is_integer': is_integer,
        'sparsity': sparsity,
        'derivative_sparsity': derivative_sparsity,
        'turning_point_ratio': tpr,
        'acf_half_life': acf_half_life,
        'dominant_frequency': dom_freq,
        'spectral_flatness': spectral_flatness,
        'spectral_slope': spectral_slope,
        'spectral_peak_snr': snr,
        'is_first_bin_peak': is_first_bin,
        'hurst': hurst,
        'perm_entropy': perm_entropy,
    }

    return measures


# =================================================================
# Statistical helpers — pure numpy, no dependencies
# =================================================================

def _kurtosis(x):
    """Excess kurtosis (Fisher)."""
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-15:
        return 0.0
    z = (x - m) / s
    return float(np.mean(z ** 4) - 3.0)


def _skewness(x):
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-15:
        return 0.0
    z = (x - m) / s
    return float(np.mean(z ** 3))


def _turning_point_ratio(x):
    """Fraction of points that are local extrema."""
    if len(x) < 3:
        return 0.667
    d = np.diff(x)
    sign_changes = np.sum(d[:-1] * d[1:] < 0)
    return float(sign_changes / (len(x) - 2))


def _acf_half_life(x, max_lag=None):
    """ACF half-life via FFT. Returns None if ACF never drops below 0.5."""
    n = len(x)
    if n < 20:
        return None
    if max_lag is None:
        max_lag = min(n // 4, 500)

    # FFT-based autocorrelation
    xc = x - np.mean(x)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    f = np.fft.rfft(xc, fft_size)
    acf_full = np.fft.irfft(f * np.conj(f))[:n]
    if acf_full[0] < 1e-15:
        return None
    acf = acf_full[:max_lag] / acf_full[0]

    # Find first crossing below 0.5
    below = np.where(acf < 0.5)[0]
    if len(below) == 0:
        return None  # Never decays — infinite memory
    return float(below[0])


def _spectral_quick(x):
    """Quick spectral analysis via FFT. Returns (dom_freq, flatness, slope, snr, is_first_bin)."""
    n = len(x)
    if n < 16:
        return None, 0.5, 0.0, 0.0, False

    xc = x - np.mean(x)
    fft = np.fft.rfft(xc)
    psd = np.abs(fft[1:]) ** 2  # Skip DC
    if len(psd) == 0 or np.sum(psd) < 1e-15:
        return None, 0.5, 0.0, 0.0, False

    # Dominant frequency (normalized to [0, 0.5])
    peak_idx = int(np.argmax(psd))
    dom_freq = float((peak_idx + 1) / n)

    # Is it a first-bin artifact?
    is_first_bin = peak_idx == 0

    # Spectral flatness (geometric mean / arithmetic mean)
    log_psd = np.log(psd + 1e-30)
    geo_mean = np.exp(np.mean(log_psd))
    arith_mean = np.mean(psd)
    flatness = float(geo_mean / arith_mean) if arith_mean > 0 else 0.0

    # Spectral slope (log-log linear regression)
    log_freq = np.log(np.arange(1, len(psd) + 1))
    slope = float(np.polyfit(log_freq, log_psd, 1)[0]) if len(psd) > 2 else 0.0

    # SNR of dominant peak
    if len(psd) > 3:
        sorted_psd = np.sort(psd)
        noise_floor = np.mean(sorted_psd[:len(sorted_psd) // 2])
        snr = float(10 * np.log10(psd[peak_idx] / noise_floor)) if noise_floor > 0 else 0.0
    else:
        snr = 0.0

    return dom_freq, flatness, slope, snr, is_first_bin


def _hurst_quick(x):
    """
    Quick Hurst estimate using simplified R/S on 2 scales.
    Not as accurate as full R/S, but O(n) and good enough for window sizing.
    """
    n = len(x)
    if n < 64:
        return 0.5

    def _rs(series):
        """R/S statistic for a single series."""
        m = np.mean(series)
        y = np.cumsum(series - m)
        r = float(np.max(y) - np.min(y))
        s = float(np.std(series, ddof=1))
        if s < 1e-15:
            return 0.0
        return r / s

    # Two scales: full and halves
    rs_full = _rs(x)
    half = n // 2
    rs_half = (_rs(x[:half]) + _rs(x[half:2 * half])) / 2

    if rs_half < 1e-15 or rs_full < 1e-15:
        return 0.5

    # H = log(RS_full / RS_half) / log(2)
    h = np.log(rs_full / rs_half) / np.log(2)
    return float(np.clip(h, 0.0, 1.0))


def _perm_entropy_quick(x, order=3):
    """
    Permutation entropy of order 3. O(n).
    Normalized to [0, 1] where 1 = maximum complexity.
    """
    n = len(x)
    if n < order + 1:
        return 0.5

    # Count ordinal patterns
    from math import factorial
    n_perms = factorial(order)
    counts = {}

    for i in range(n - order + 1):
        pattern = tuple(np.argsort(x[i:i + order]))
        counts[pattern] = counts.get(pattern, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 0.5

    probs = np.array([c / total for c in counts.values()])
    entropy = -np.sum(probs * np.log(probs + 1e-30))
    max_entropy = np.log(n_perms)

    return float(entropy / max_entropy) if max_entropy > 0 else 0.5


# =================================================================
# Edge cases
# =================================================================

def _empty(n):
    return {
        'n_samples': n, 'signal_mean': 0.0, 'signal_std': 0.0,
        'is_constant': True, 'kurtosis': 0.0, 'skewness': 0.0,
        'crest_factor': 0.0, 'unique_ratio': 0.0, 'is_integer': False,
        'sparsity': 1.0, 'derivative_sparsity': 1.0,
        'turning_point_ratio': 0.0, 'acf_half_life': None,
        'dominant_frequency': None, 'spectral_flatness': 0.0,
        'spectral_slope': 0.0, 'spectral_peak_snr': 0.0,
        'is_first_bin_peak': False, 'hurst': 0.5, 'perm_entropy': 0.0,
    }


def _constant_result(n, mean):
    return {
        'n_samples': n, 'signal_mean': mean, 'signal_std': 0.0,
        'is_constant': True, 'kurtosis': 0.0, 'skewness': 0.0,
        'crest_factor': 0.0, 'unique_ratio': 0.0, 'is_integer': False,
        'sparsity': 0.0, 'derivative_sparsity': 1.0,
        'turning_point_ratio': 0.0, 'acf_half_life': None,
        'dominant_frequency': None, 'spectral_flatness': 0.0,
        'spectral_slope': 0.0, 'spectral_peak_snr': 0.0,
        'is_first_bin_peak': False, 'hurst': 0.5, 'perm_entropy': 0.0,
    }
