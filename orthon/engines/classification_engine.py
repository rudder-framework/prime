"""
Level 2: Signal Classification Engine

Classify each signal's behavior pattern using decision tree:
- CONSTANT: No variation
- STATIONARY: Stable statistics
- TRENDING: Monotonic drift
- PERIODIC: Dominant frequency
- QUASI_PERIODIC: Mixed periodicities
- CHAOTIC: Sensitive dependence
- RANDOM: No structure
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict
from scipy import stats
from scipy.fft import fft, fftfreq


class SignalClass(Enum):
    """Signal behavior classification."""
    CONSTANT = "constant"
    STATIONARY = "stationary"
    TRENDING = "trending"
    PERIODIC = "periodic"
    QUASI_PERIODIC = "quasi_periodic"
    CHAOTIC = "chaotic"
    RANDOM = "random"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of signal classification."""
    signal_class: SignalClass
    metrics: Dict[str, float]
    confidence: float
    description: str


def classify_signal(
    values: np.ndarray,
    is_stationary: Optional[bool] = None,
) -> ClassificationResult:
    """
    Classify signal behavior pattern.

    Args:
        values: Time series values
        is_stationary: Optional pre-computed stationarity flag

    Returns:
        ClassificationResult with classification and metrics
    """
    n = len(values)
    if n < 10:
        return ClassificationResult(
            signal_class=SignalClass.UNKNOWN,
            metrics={},
            confidence=0.0,
            description="Insufficient data for classification",
        )

    # Compute metrics
    metrics = _compute_classification_metrics(values)

    # Decision tree
    # Level 1: Is constant?
    if metrics['cv'] < 0.001:
        return ClassificationResult(
            signal_class=SignalClass.CONSTANT,
            metrics=metrics,
            confidence=1.0,
            description="No variation detected",
        )

    # Level 2: Is non-stationary?
    if is_stationary is not None:
        non_stationary = not is_stationary
    else:
        # Infer from metrics
        non_stationary = metrics['trend_r2'] > 0.5 or metrics['hurst'] > 0.7

    if non_stationary:
        # Has dominant frequency?
        if metrics['spectral_peak_snr'] > 10:  # 10 dB SNR
            # Peak at bottom of spectrum = trend
            if metrics['peak_at_bottom']:
                return ClassificationResult(
                    signal_class=SignalClass.TRENDING,
                    metrics=metrics,
                    confidence=min(metrics['trend_r2'], 0.95),
                    description="Monotonic drift with low-frequency dominance",
                )
            else:
                return ClassificationResult(
                    signal_class=SignalClass.QUASI_PERIODIC,
                    metrics=metrics,
                    confidence=metrics['spectral_flatness'],
                    description="Non-stationary with periodic component",
                )
        else:
            # Few turning points = trend
            if metrics['turning_point_ratio'] < 0.7:
                return ClassificationResult(
                    signal_class=SignalClass.TRENDING,
                    metrics=metrics,
                    confidence=min(1 - metrics['turning_point_ratio'], 0.95),
                    description="Monotonic drift with few oscillations",
                )
            else:
                # Check for chaos
                if metrics['lyapunov_proxy'] > 0.5 and metrics['perm_entropy'] > 0.8:
                    return ClassificationResult(
                        signal_class=SignalClass.CHAOTIC,
                        metrics=metrics,
                        confidence=min(metrics['lyapunov_proxy'], 0.95),
                        description="Sensitive dependence on initial conditions",
                    )
                else:
                    return ClassificationResult(
                        signal_class=SignalClass.RANDOM,
                        metrics=metrics,
                        confidence=metrics['perm_entropy'],
                        description="Non-stationary random process",
                    )
    else:
        # Stationary signal
        # Has dominant frequency?
        if metrics['spectral_peak_snr'] > 15:  # Higher threshold for stationary
            return ClassificationResult(
                signal_class=SignalClass.PERIODIC,
                metrics=metrics,
                confidence=1 - metrics['spectral_flatness'],
                description="Periodic signal with dominant frequency",
            )
        else:
            return ClassificationResult(
                signal_class=SignalClass.STATIONARY,
                metrics=metrics,
                confidence=0.9 - metrics['trend_r2'],
                description="Stationary signal with stable statistics",
            )


def _compute_classification_metrics(values: np.ndarray) -> Dict[str, float]:
    """Compute metrics used in classification."""
    n = len(values)

    # Basic statistics
    mean = np.mean(values)
    std = np.std(values)
    cv = std / abs(mean) if abs(mean) > 1e-10 else 0.0

    # Trend
    t = np.arange(n)
    try:
        slope, intercept, r_value, _, _ = stats.linregress(t, values)
        trend_r2 = r_value ** 2
        trend_slope = slope
    except Exception:
        trend_r2 = 0.0
        trend_slope = 0.0

    # Spectral analysis
    spectral_flatness, spectral_peak_snr, dominant_freq, peak_at_bottom = _compute_spectral_metrics(values)

    # Turning point ratio
    turning_ratio = _compute_turning_point_ratio(values)

    # Permutation entropy
    perm_entropy = _compute_permutation_entropy(values)

    # Hurst exponent proxy (simplified)
    hurst = _compute_hurst_proxy(values)

    # Lyapunov proxy (local divergence rate)
    lyapunov_proxy = _compute_lyapunov_proxy(values)

    return {
        'cv': float(cv),
        'trend_slope': float(trend_slope),
        'trend_r2': float(trend_r2),
        'spectral_flatness': float(spectral_flatness),
        'spectral_peak_snr': float(spectral_peak_snr),
        'dominant_freq': float(dominant_freq),
        'peak_at_bottom': bool(peak_at_bottom),
        'turning_point_ratio': float(turning_ratio),
        'perm_entropy': float(perm_entropy),
        'hurst': float(hurst),
        'lyapunov_proxy': float(lyapunov_proxy),
    }


def _compute_spectral_metrics(values: np.ndarray) -> tuple:
    """Compute spectral analysis metrics."""
    n = len(values)
    if n < 8:
        return 0.0, 0.0, 0.0, False

    # Detrend
    values_detrend = values - np.mean(values)

    # FFT
    fft_vals = fft(values_detrend)
    power = np.abs(fft_vals[:n // 2]) ** 2
    freqs = fftfreq(n, d=1.0)[:n // 2]

    # Avoid DC
    power = power[1:]
    freqs = freqs[1:]

    if len(power) == 0 or np.sum(power) < 1e-20:
        return 0.0, 0.0, 0.0, False

    # Spectral flatness
    power_norm = power / np.sum(power)
    log_power = np.log(power_norm + 1e-20)
    spectral_flatness = np.exp(np.mean(log_power)) / (np.mean(power_norm) + 1e-20)
    spectral_flatness = np.clip(spectral_flatness, 0, 1)

    # Peak SNR
    peak_idx = np.argmax(power)
    peak_power = power[peak_idx]
    noise_power = np.median(power)
    spectral_peak_snr = 10 * np.log10(peak_power / (noise_power + 1e-20))

    # Dominant frequency
    dominant_freq = freqs[peak_idx] if len(freqs) > peak_idx else 0.0

    # Peak at bottom (trend indicator)
    peak_at_bottom = peak_idx < len(power) * 0.1

    return spectral_flatness, spectral_peak_snr, dominant_freq, peak_at_bottom


def _compute_turning_point_ratio(values: np.ndarray) -> float:
    """Compute turning point ratio."""
    n = len(values)
    if n < 3:
        return 1.0

    turning_points = 0
    for i in range(1, n - 1):
        if (values[i] > values[i - 1] and values[i] > values[i + 1]) or \
           (values[i] < values[i - 1] and values[i] < values[i + 1]):
            turning_points += 1

    expected = 2 * (n - 2) / 3
    return turning_points / expected if expected > 0 else 1.0


def _compute_permutation_entropy(values: np.ndarray, m: int = 3) -> float:
    """Compute permutation entropy."""
    n = len(values)
    if n < m + 1:
        return 0.0

    from collections import Counter
    import math

    patterns = []
    for i in range(n - m + 1):
        subseq = values[i:i + m]
        pattern = tuple(np.argsort(subseq))
        patterns.append(pattern)

    counts = Counter(patterns)
    total = len(patterns)

    probs = np.array(list(counts.values())) / total
    entropy = -np.sum(probs * np.log2(probs + 1e-20))

    max_entropy = np.log2(math.factorial(m))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _compute_hurst_proxy(values: np.ndarray) -> float:
    """Compute simplified Hurst exponent proxy."""
    n = len(values)
    if n < 16:
        return 0.5

    # Simple R/S analysis at single scale
    half = n // 2
    first_half = values[:half]
    second_half = values[half:]

    def rs_stat(x):
        mean = np.mean(x)
        cumdev = np.cumsum(x - mean)
        R = np.max(cumdev) - np.min(cumdev)
        S = np.std(x, ddof=1)
        return R / S if S > 1e-10 else 0

    rs1 = rs_stat(first_half)
    rs2 = rs_stat(second_half)
    rs_full = rs_stat(values)

    # H approximation from R/S scaling
    if rs1 > 0 and rs_full > 0:
        H = np.log(rs_full / rs1) / np.log(2)
        return np.clip(H, 0, 1)
    return 0.5


def _compute_lyapunov_proxy(values: np.ndarray) -> float:
    """
    Compute Lyapunov exponent proxy (local divergence rate).

    Simplified estimation using nearest neighbor divergence.
    """
    n = len(values)
    if n < 20:
        return 0.0

    # Embedding
    m = 3  # embedding dimension
    tau = 1  # time delay

    if n < m * tau + 10:
        return 0.0

    # Create embedding vectors
    L = n - (m - 1) * tau
    embedded = np.zeros((L, m))
    for i in range(m):
        embedded[:, i] = values[i * tau:i * tau + L]

    # Find nearest neighbors and track divergence
    divergences = []
    for i in range(L - 1):
        # Find nearest neighbor (excluding immediate neighbors)
        min_dist = np.inf
        min_idx = -1
        for j in range(L - 1):
            if abs(i - j) > tau:
                dist = np.linalg.norm(embedded[i] - embedded[j])
                if dist < min_dist and dist > 1e-10:
                    min_dist = dist
                    min_idx = j

        if min_idx >= 0:
            # Track divergence
            final_dist = np.linalg.norm(embedded[i + 1] - embedded[min_idx + 1])
            if min_dist > 1e-10:
                divergence = np.log(final_dist / min_dist)
                if np.isfinite(divergence):
                    divergences.append(divergence)

    if len(divergences) > 5:
        # Mean divergence rate (proxy for Lyapunov exponent)
        lyap = np.mean(divergences)
        # Normalize to [0, 1] range for classification
        return np.clip((lyap + 0.5) / 1.0, 0, 1)

    return 0.0
