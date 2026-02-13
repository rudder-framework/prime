"""
Stability Engine

Assess dynamical stability via Lyapunov analysis and Critical Slowing Down (CSD).

Core metrics:
- Lyapunov exponent: λ (trajectory divergence rate)
- CSD indicators: AR1→1, Variance↑, Recovery rate→0
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from scipy import stats


@dataclass
class StabilityResult:
    """Result of stability analysis."""
    lyapunov_exponent: float        # λ (>0 = unstable)
    lyapunov_confidence: float      # Confidence in estimate
    ar1: float                      # Lag-1 autocorrelation
    ar1_trend: float                # Trend in AR1 over time
    variance_trend: float           # Trend in variance over time
    csd_detected: bool              # Critical slowing down detected
    csd_confidence: float           # Confidence in CSD detection
    recovery_rate: float            # τ = -1/λ
    is_stable: bool
    description: str


def compute_stability(
    values: np.ndarray,
    window_ar1s: Optional[List[float]] = None,
    window_variances: Optional[List[float]] = None,
) -> StabilityResult:
    """
    Compute stability metrics for a time series.

    Args:
        values: Time series values
        window_ar1s: Optional list of AR1 values from rolling windows
        window_variances: Optional list of variances from rolling windows

    Returns:
        StabilityResult with stability assessment
    """
    n = len(values)

    if n < 20:
        return _empty_stability_result("Insufficient data for stability analysis")

    # === LYAPUNOV EXPONENT ===
    lyapunov, lyap_confidence = _compute_lyapunov(values)

    # === AR1 (autocorrelation at lag 1) ===
    if np.std(values) > 1e-10:
        ar1 = np.corrcoef(values[:-1], values[1:])[0, 1]
        if not np.isfinite(ar1):
            ar1 = 0.0
    else:
        ar1 = 1.0

    # === CSD DETECTION ===
    # AR1 trend (should increase toward 1)
    if window_ar1s and len(window_ar1s) > 3:
        ar1_trend = _compute_kendall_trend(window_ar1s)
    else:
        # Compute from data
        ar1_trend = _compute_ar1_trend(values)

    # Variance trend (should increase)
    if window_variances and len(window_variances) > 3:
        variance_trend = _compute_kendall_trend(window_variances)
    else:
        variance_trend = _compute_variance_trend(values)

    # CSD detection: both trends positive and significant
    csd_detected = ar1_trend > 0.1 and variance_trend > 0.1
    csd_confidence = min(abs(ar1_trend), abs(variance_trend)) if csd_detected else 0.0

    # === RECOVERY RATE ===
    # τ = -1/λ (time to return to equilibrium)
    if lyapunov < -0.01:
        recovery_rate = -1.0 / lyapunov
    elif lyapunov > 0.01:
        recovery_rate = np.inf  # Unstable, never recovers
    else:
        recovery_rate = 1000.0  # Slow recovery (near critical)

    # === STABILITY CLASSIFICATION ===
    is_stable = lyapunov < 0 and ar1 < 0.95

    if lyapunov > 0.1:
        description = "UNSTABLE: Trajectories diverging exponentially"
    elif lyapunov > 0:
        description = "MARGINALLY UNSTABLE: Weak exponential divergence"
    elif ar1 > 0.95:
        description = "CRITICAL: Near bifurcation (AR1→1)"
    elif csd_detected:
        description = "CSD DETECTED: Approaching critical transition"
    elif lyapunov < -0.1:
        description = "STABLE: Perturbations decay rapidly"
    else:
        description = "STABLE: Normal dynamical behavior"

    return StabilityResult(
        lyapunov_exponent=float(lyapunov),
        lyapunov_confidence=float(lyap_confidence),
        ar1=float(ar1),
        ar1_trend=float(ar1_trend),
        variance_trend=float(variance_trend),
        csd_detected=csd_detected,
        csd_confidence=float(csd_confidence),
        recovery_rate=float(min(recovery_rate, 10000)),
        is_stable=is_stable,
        description=description,
    )


def _compute_lyapunov(values: np.ndarray) -> Tuple[float, float]:
    """
    Compute Lyapunov exponent using Rosenstein method.

    λ = lim(t→∞) (1/t) × log(|δx(t)| / |δx(0)|)

    Returns:
        Tuple of (lyapunov_exponent, confidence)
    """
    n = len(values)
    if n < 50:
        return 0.0, 0.0

    # Embedding parameters
    m = 3   # embedding dimension
    tau = 1  # time delay
    min_separation = 10  # Theiler window

    L = n - (m - 1) * tau
    if L < 20:
        return 0.0, 0.0

    # Create embedding
    embedded = np.zeros((L, m))
    for i in range(m):
        embedded[:, i] = values[i * tau:i * tau + L]

    # Find nearest neighbors
    divergences = []
    max_iter = min(L - min_separation, 100)

    for i in range(max_iter):
        # Find nearest neighbor (excluding Theiler window)
        min_dist = np.inf
        min_idx = -1

        for j in range(L):
            if abs(i - j) > min_separation:
                dist = np.linalg.norm(embedded[i] - embedded[j])
                if dist < min_dist and dist > 1e-10:
                    min_dist = dist
                    min_idx = j

        if min_idx >= 0 and min_idx + 1 < L and i + 1 < L:
            # Track divergence over time
            for dt in range(1, min(10, L - max(i, min_idx))):
                d0 = np.linalg.norm(embedded[i] - embedded[min_idx])
                dt_dist = np.linalg.norm(embedded[i + dt] - embedded[min_idx + dt])
                if d0 > 1e-10:
                    divergences.append((dt, np.log(dt_dist / d0)))

    if len(divergences) < 10:
        return 0.0, 0.0

    # Fit log(divergence) vs time
    divergences = np.array(divergences)
    try:
        slope, _, r_value, _, _ = stats.linregress(divergences[:, 0], divergences[:, 1])
        confidence = r_value ** 2  # R² as confidence
        return float(slope), float(confidence)
    except Exception:
        return 0.0, 0.0


def _compute_ar1_trend(values: np.ndarray, n_windows: int = 10) -> float:
    """Compute trend in AR1 over rolling windows."""
    n = len(values)
    if n < 50:
        return 0.0

    window_size = n // n_windows
    ar1s = []

    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, n)
        window = values[start:end]

        if len(window) > 5 and np.std(window) > 1e-10:
            ar1 = np.corrcoef(window[:-1], window[1:])[0, 1]
            if np.isfinite(ar1):
                ar1s.append(ar1)

    return _compute_kendall_trend(ar1s) if len(ar1s) > 3 else 0.0


def _compute_variance_trend(values: np.ndarray, n_windows: int = 10) -> float:
    """Compute trend in variance over rolling windows."""
    n = len(values)
    if n < 50:
        return 0.0

    window_size = n // n_windows
    variances = []

    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, n)
        window = values[start:end]

        if len(window) > 5:
            variances.append(np.var(window))

    return _compute_kendall_trend(variances) if len(variances) > 3 else 0.0


def _compute_kendall_trend(values: List[float]) -> float:
    """Compute normalized Kendall tau trend."""
    if len(values) < 3:
        return 0.0

    t = np.arange(len(values))
    try:
        tau, p_value = stats.kendalltau(t, values)
        # Return tau if significant, else 0
        return float(tau) if p_value < 0.1 else float(tau) * 0.5
    except Exception:
        return 0.0


def _empty_stability_result(description: str) -> StabilityResult:
    """Return empty stability result."""
    return StabilityResult(
        lyapunov_exponent=0.0,
        lyapunov_confidence=0.0,
        ar1=0.0,
        ar1_trend=0.0,
        variance_trend=0.0,
        csd_detected=False,
        csd_confidence=0.0,
        recovery_rate=0.0,
        is_stable=True,
        description=description,
    )


def compute_csd_indicators(
    ar1_series: List[float],
    variance_series: List[float],
) -> Dict:
    """
    Compute Critical Slowing Down indicators from time series.

    Args:
        ar1_series: AR1 values over time (rolling windows)
        variance_series: Variance values over time

    Returns:
        Dict with CSD indicators and detection
    """
    if len(ar1_series) < 5 or len(variance_series) < 5:
        return {
            'ar1_trend_tau': 0.0,
            'var_trend_tau': 0.0,
            'csd_detected': False,
            'csd_strength': 0.0,
        }

    ar1_tau = _compute_kendall_trend(ar1_series)
    var_tau = _compute_kendall_trend(variance_series)

    # CSD detected if both increasing
    csd_detected = ar1_tau > 0.1 and var_tau > 0.1

    # CSD strength (product of trends)
    csd_strength = ar1_tau * var_tau if csd_detected else 0.0

    return {
        'ar1_trend_tau': float(ar1_tau),
        'var_trend_tau': float(var_tau),
        'csd_detected': csd_detected,
        'csd_strength': float(csd_strength),
        'latest_ar1': float(ar1_series[-1]),
        'latest_variance': float(variance_series[-1]),
    }
