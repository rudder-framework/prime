"""
Rolling signal stability metrics.

Hilbert transform gives analytic signal:
    z(t) = x(t) + i*H[x(t)]
    amplitude(t) = |z(t)|       instantaneous envelope
    phase(t) = arg(z(t))        instantaneous phase
    frequency(t) = dφ/dt / 2π   instantaneous frequency

Stability = low variance in amplitude envelope over a window.
"""

import numpy as np
from typing import Dict, Any, List, Optional


def compute_signal_stability(
    values: np.ndarray,
    window_size: int = 100,
    stride: int = 50,
) -> List[Dict[str, Any]]:
    """
    Compute rolling stability metrics for a single signal.

    Parameters
    ----------
    values : np.ndarray
        1D time series.
    window_size : int
        Rolling window size.
    stride : int
        Step between windows.

    Returns
    -------
    list of dict — one per window.
    """
    values = np.asarray(values, dtype=np.float64).flatten()
    n = len(values)
    results = []

    for start in range(0, n - window_size + 1, stride):
        window = values[start:start + window_size]
        valid = window[np.isfinite(window)]

        if len(valid) < window_size // 2:
            continue

        row = {'I': start + window_size // 2}

        # Hilbert envelope
        try:
            from scipy.signal import hilbert
            analytic = hilbert(valid)
            envelope = np.abs(analytic)
            phase = np.unwrap(np.angle(analytic))

            row['mean_amplitude'] = float(np.mean(envelope))
            row['amplitude_std'] = float(np.std(envelope))
            row['amplitude_cv'] = float(np.std(envelope) / (np.mean(envelope) + 1e-30))

            # Instantaneous frequency
            dt = 1.0
            inst_freq = np.diff(phase) / (2 * np.pi * dt)
            row['mean_frequency'] = float(np.mean(inst_freq))
            row['frequency_std'] = float(np.std(inst_freq))
        except ImportError:
            # Fallback: simple envelope approximation
            envelope = np.abs(valid - np.mean(valid))
            row['mean_amplitude'] = float(np.mean(envelope))
            row['amplitude_std'] = float(np.std(envelope))
            row['amplitude_cv'] = float(np.std(envelope) / (np.mean(envelope) + 1e-30))
            row['mean_frequency'] = np.nan
            row['frequency_std'] = np.nan

        # Wavelet energy proxy (variance in frequency bands)
        row['signal_energy'] = float(np.sum(valid ** 2) / len(valid))
        row['signal_rms'] = float(np.sqrt(np.mean(valid ** 2)))

        # Stability ratio: low amplitude_cv = stable
        row['stability_ratio'] = float(1.0 / (1.0 + row['amplitude_cv']))

        results.append(row)

    return results
