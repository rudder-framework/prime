"""
ORTHON Typology Raw Computation

Computes the raw statistical measures that feed into typology_v2.sql.
This is the ONLY computation ORTHON performs - everything else is classification.

PRISM computes engine outputs. ORTHON computes typology and classifies.

Output: typology_raw.parquet with one row per (cohort, signal_id)

Usage:
    python -m orthon.ingest.typology_raw data/observations.parquet data/typology_raw.parquet
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.signal import welch
from statsmodels.tsa.stattools import adfuller, kpss, acf


@dataclass
class SignalProfile:
    """Raw statistical profile for a single signal."""
    signal_id: str
    cohort: Optional[str]
    n_samples: int

    # Stationarity (Dimension 2)
    adf_pvalue: float
    kpss_pvalue: float
    variance_ratio: float
    acf_half_life: Optional[float]

    # Memory (Dimension 4)
    hurst: float

    # Complexity (Dimension 5)
    perm_entropy: float
    sample_entropy: float

    # Spectral (Dimension 8)
    spectral_flatness: float
    spectral_slope: float
    harmonic_noise_ratio: float
    spectral_peak_snr: float
    dominant_frequency: float
    is_first_bin_peak: bool  # True = dominant_freq is artifact from 1/f slope

    # Temporal Pattern (Dimension 3)
    turning_point_ratio: float
    lyapunov_proxy: float

    # Determinism (Dimension 10)
    determinism_score: float

    # Volatility (Dimension 9)
    arch_pvalue: float
    rolling_var_std: float

    # Distribution (Dimension 6) - can be SQL but include for completeness
    kurtosis: float
    skewness: float
    crest_factor: float

    # Continuity (Dimension 1)
    unique_ratio: float
    is_integer: bool
    is_constant: bool  # CV-based constant detection
    sparsity: float
    signal_std: float
    signal_mean: float
    derivative_sparsity: float  # STEP detection: fraction of zero derivatives
    zero_run_ratio: float  # INTERMITTENT detection: avg zero run / total length

    # Window Factor (for adaptive windowing in PRISM)
    window_factor: float = 1.0  # Multiplier for engine base windows


# ============================================================
# DIMENSION 2: STATIONARITY
# ============================================================

def compute_adf_pvalue(values: np.ndarray, max_lag: int = None) -> float:
    """
    Augmented Dickey-Fuller test.
    H0: unit root (non-stationary). Low p-value -> reject -> stationary.
    """
    try:
        if len(values) < 20:
            return 1.0  # Not enough data, assume non-stationary
        result = adfuller(values, maxlag=max_lag, autolag='AIC')
        return float(result[1])  # p-value
    except Exception:
        return 1.0


def compute_kpss_pvalue(values: np.ndarray) -> float:
    """
    KPSS test.
    H0: stationary. Low p-value -> reject -> non-stationary.
    """
    try:
        if len(values) < 20:
            return 0.0  # Not enough data, assume non-stationary
        # Use 'c' for level stationarity (constant mean)
        result = kpss(values, regression='c', nlags='auto')
        return float(result[1])  # p-value
    except Exception:
        return 0.0


def compute_variance_ratio(values: np.ndarray, window: int = 50) -> float:
    """
    Ratio of rolling variance to global variance.
    High ratio (>2) or low ratio (<0.5) indicates heteroscedasticity.
    """
    try:
        if len(values) < window * 2:
            return 1.0

        global_var = np.var(values)
        if global_var < 1e-10:
            return 1.0

        # Rolling variance using numpy stride tricks
        n = len(values) - window + 1
        rolling_vars = np.array([np.var(values[i:i+window]) for i in range(n)])
        mean_rolling_var = np.mean(rolling_vars)

        return float(mean_rolling_var / global_var)
    except Exception:
        return 1.0


def compute_acf_half_life(values: np.ndarray, max_lag: int = 100) -> Optional[float]:
    """
    Find lag at which ACF drops to 0.5 (half-life of autocorrelation).
    Returns None if ACF doesn't decay to 0.5 within max_lag.
    """
    try:
        if len(values) < max_lag:
            max_lag = len(values) // 2
        if max_lag < 2:
            return None

        acf_values = acf(values, nlags=max_lag, fft=True)

        # Find first lag where ACF < 0.5
        for lag, ac in enumerate(acf_values):
            if ac < 0.5:
                return float(lag)
        return None  # Doesn't decay fast enough
    except Exception:
        return None


# ============================================================
# DIMENSION 4: MEMORY (Hurst Exponent)
# ============================================================

def compute_hurst(values: np.ndarray) -> float:
    """
    Hurst exponent via R/S analysis.
    H > 0.5: persistent (trending)
    H = 0.5: random walk
    H < 0.5: anti-persistent (mean-reverting)
    """
    try:
        n = len(values)
        if n < 20:
            return 0.5

        # Use multiple window sizes
        max_k = min(n // 2, 100)
        min_k = 10

        if max_k <= min_k:
            return 0.5

        rs_values = []
        ns = []

        for k in range(min_k, max_k + 1, max(1, (max_k - min_k) // 10)):
            # Number of windows
            n_windows = n // k
            if n_windows < 1:
                continue

            rs_list = []
            for i in range(n_windows):
                window = values[i * k:(i + 1) * k]
                mean_adj = window - np.mean(window)
                cumsum = np.cumsum(mean_adj)
                r = np.max(cumsum) - np.min(cumsum)
                s = np.std(window, ddof=1)
                if s > 1e-10:
                    rs_list.append(r / s)

            if rs_list:
                rs_values.append(np.mean(rs_list))
                ns.append(k)

        if len(ns) < 3:
            return 0.5

        # Linear regression on log-log plot
        log_n = np.log(ns)
        log_rs = np.log(rs_values)
        slope, _, _, _, _ = stats.linregress(log_n, log_rs)

        # Clamp to reasonable range
        return float(np.clip(slope, 0.0, 1.0))
    except Exception:
        return 0.5


# ============================================================
# DIMENSION 5: COMPLEXITY
# ============================================================

def compute_permutation_entropy(values: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """
    Permutation entropy - measures complexity of ordinal patterns.
    Range: 0 (deterministic) to 1 (random).
    """
    try:
        n = len(values)
        if n < (order - 1) * delay + order:
            return 0.5

        # Generate ordinal patterns
        from math import factorial

        patterns = {}
        n_patterns = n - (order - 1) * delay

        for i in range(n_patterns):
            # Extract embedding
            indices = [i + j * delay for j in range(order)]
            embedding = [values[idx] for idx in indices]

            # Get ordinal pattern (rank order)
            pattern = tuple(np.argsort(embedding))
            patterns[pattern] = patterns.get(pattern, 0) + 1

        # Compute entropy
        total = sum(patterns.values())
        probs = np.array([count / total for count in patterns.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Normalize by maximum entropy
        max_entropy = np.log2(factorial(order))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.5
    except Exception:
        return 0.5


def compute_sample_entropy(values: np.ndarray, m: int = 2, r: float = None) -> float:
    """
    Sample entropy - measures unpredictability.
    Lower values = more regular/predictable.
    """
    try:
        n = len(values)
        if n < m + 2:
            return 0.5

        if r is None:
            r = 0.2 * np.std(values)
        if r < 1e-10:
            return 0.0

        def count_matches(template_len):
            count = 0
            for i in range(n - template_len):
                for j in range(i + 1, n - template_len):
                    if np.max(np.abs(values[i:i+template_len] - values[j:j+template_len])) < r:
                        count += 1
            return count

        A = count_matches(m + 1)
        B = count_matches(m)

        if B == 0:
            return 0.0

        return float(-np.log(A / B)) if A > 0 else 0.0
    except Exception:
        return 0.5


# ============================================================
# DIMENSION 8: SPECTRAL CHARACTER
# ============================================================

def compute_spectral_profile(values: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
    """
    Compute spectral characteristics.
    Returns dict with flatness, slope, harmonic_noise_ratio, peak_snr, dominant_freq.

    IMPORTANT: Detects first-bin artifact where slow/trending signals concentrate
    energy at the lowest FFT frequency. This is NOT a real spectral peak - it's
    just where 1/f or monotonic signals put their energy. When detected:
    - is_first_bin_peak = True
    - dominant_frequency = 0.0 (artifact, not real period)
    - spectral_peak_snr still computed (for debugging) but shouldn't drive classification
    """
    try:
        n = len(values)
        if n < 64:
            return {
                'spectral_flatness': 0.5,
                'spectral_slope': 0.0,
                'harmonic_noise_ratio': 0.0,
                'spectral_peak_snr': 0.0,
                'dominant_frequency': 0.0,
                'is_first_bin_peak': False,
            }

        # Compute PSD using Welch's method
        nperseg = min(256, n // 4)
        freqs, psd = welch(values, fs=fs, nperseg=nperseg)

        # Remove DC component
        if len(freqs) > 1:
            freqs = freqs[1:]
            psd = psd[1:]

        if len(psd) == 0 or np.sum(psd) < 1e-10:
            return {
                'spectral_flatness': 0.5,
                'spectral_slope': 0.0,
                'harmonic_noise_ratio': 0.0,
                'spectral_peak_snr': 0.0,
                'dominant_frequency': 0.0,
                'is_first_bin_peak': False,
            }

        # Spectral flatness (Wiener entropy)
        # Geometric mean / arithmetic mean
        log_psd = np.log(psd + 1e-10)
        geo_mean = np.exp(np.mean(log_psd))
        arith_mean = np.mean(psd)
        flatness = geo_mean / arith_mean if arith_mean > 1e-10 else 0.5

        # Spectral slope (log-log fit)
        log_freqs = np.log(freqs + 1e-10)
        slope, _, _, _, _ = stats.linregress(log_freqs, log_psd)

        # Peak detection
        peak_idx = np.argmax(psd)
        peak_power = psd[peak_idx]
        peak_freq = freqs[peak_idx]

        # SNR of peak vs noise floor (median)
        noise_floor = np.median(psd)
        peak_snr = 10 * np.log10(peak_power / noise_floor) if noise_floor > 1e-10 else 0.0

        # Harmonic-to-noise ratio (simplified)
        # Compare power at peak and harmonics vs broadband
        total_power = np.sum(psd)
        hnr = peak_power / (total_power - peak_power) if total_power > peak_power else 0.0

        # ================================================================
        # FIRST-BIN ARTIFACT DETECTION
        # ================================================================
        # If peak is in the first 3 bins AND spectral slope is negative (1/f-like),
        # this is NOT a real periodic signal - it's just where slow/trending
        # signals concentrate their energy.
        #
        # True periodic signals have peaks AWAY from the first bin (at their
        # actual oscillation frequency).
        #
        # Threshold: slope < -0.3 indicates falling spectrum (energy at low freqs)
        # ================================================================
        is_first_bin = peak_idx < 3
        is_falling_spectrum = slope < -0.3

        if is_first_bin and is_falling_spectrum:
            # This is an artifact - null out the dominant frequency
            return {
                'spectral_flatness': float(np.clip(flatness, 0, 1)),
                'spectral_slope': float(slope),
                'harmonic_noise_ratio': float(hnr),
                'spectral_peak_snr': float(peak_snr),
                'dominant_frequency': 0.0,  # Artifact - no real period
                'is_first_bin_peak': True,
            }

        return {
            'spectral_flatness': float(np.clip(flatness, 0, 1)),
            'spectral_slope': float(slope),
            'harmonic_noise_ratio': float(hnr),
            'spectral_peak_snr': float(peak_snr),
            'dominant_frequency': float(peak_freq),
            'is_first_bin_peak': False,
        }
    except Exception:
        return {
            'spectral_flatness': 0.5,
            'spectral_slope': 0.0,
            'harmonic_noise_ratio': 0.0,
            'spectral_peak_snr': 0.0,
            'dominant_frequency': 0.0,
            'is_first_bin_peak': False,
        }


# ============================================================
# DIMENSION 3: TEMPORAL PATTERN
# ============================================================

def compute_turning_point_ratio(values: np.ndarray) -> float:
    """
    Ratio of turning points to expected for random series.
    Low ratio (<0.5) suggests trending, high ratio suggests oscillation.
    Expected for random: 2/3 * (n-2)
    """
    try:
        n = len(values)
        if n < 3:
            return 0.67

        # Count turning points (local maxima and minima)
        turning_points = 0
        for i in range(1, n - 1):
            if (values[i] > values[i-1] and values[i] > values[i+1]) or \
               (values[i] < values[i-1] and values[i] < values[i+1]):
                turning_points += 1

        expected = (2/3) * (n - 2)
        return float(turning_points / expected) if expected > 0 else 0.67
    except Exception:
        return 0.67


def compute_lyapunov_proxy(values: np.ndarray, lag: int = 1) -> float:
    """
    Simplified Lyapunov exponent proxy using average divergence.
    Positive = chaotic, negative = stable.

    Note: This is a proxy, not the full Rosenstein algorithm.
    For true Lyapunov, use PRISM's dynamics layer.
    """
    try:
        n = len(values)
        if n < 100:
            return 0.0

        # Simple proxy: rate of divergence of nearby trajectories
        # Using embedding dimension 3
        m = 3
        embedded = np.array([values[i:i+m] for i in range(n - m)])

        # Find nearest neighbors and measure divergence
        divergences = []
        for i in range(min(100, len(embedded) - lag)):
            dists = np.linalg.norm(embedded - embedded[i], axis=1)
            dists[max(0, i-5):i+6] = np.inf  # Exclude temporal neighbors

            nn_idx = np.argmin(dists)
            if nn_idx + lag < len(embedded) and i + lag < len(embedded):
                d0 = dists[nn_idx]
                d1 = np.linalg.norm(embedded[i + lag] - embedded[nn_idx + lag])
                if d0 > 1e-10:
                    divergences.append(np.log(d1 / d0) / lag)

        if not divergences:
            return 0.0

        return float(np.mean(divergences))
    except Exception:
        return 0.0


# ============================================================
# DIMENSION 10: DETERMINISM
# ============================================================

def compute_determinism_score(values: np.ndarray, threshold: float = None) -> float:
    """
    Determinism from recurrence plot analysis.
    High score (>0.8) = deterministic, low score (<0.3) = stochastic.

    Simplified version using diagonal line percentage.
    """
    try:
        n = len(values)
        if n < 50:
            return 0.5

        # Subsample for efficiency
        if n > 500:
            step = n // 500
            values = values[::step]
            n = len(values)

        if threshold is None:
            threshold = 0.1 * np.std(values)

        # Build recurrence matrix
        dists = np.abs(values.reshape(-1, 1) - values.reshape(1, -1))
        recurrence = (dists < threshold).astype(int)

        # Count diagonal lines (length >= 2)
        total_recurrence = np.sum(recurrence) - n  # Exclude main diagonal
        if total_recurrence < 1:
            return 0.5

        # Count points on diagonal lines of length >= 2
        diagonal_count = 0
        for k in range(1, n):
            diag = np.diag(recurrence, k)
            # Count consecutive 1s
            in_line = False
            line_len = 0
            for val in diag:
                if val == 1:
                    line_len += 1
                    in_line = True
                else:
                    if in_line and line_len >= 2:
                        diagonal_count += line_len
                    line_len = 0
                    in_line = False
            if in_line and line_len >= 2:
                diagonal_count += line_len

        # Determinism = diagonal points / total recurrence
        det = diagonal_count / total_recurrence if total_recurrence > 0 else 0.5
        return float(np.clip(det, 0, 1))
    except Exception:
        return 0.5


# ============================================================
# DIMENSION 9: VOLATILITY
# ============================================================

def compute_arch_test(values: np.ndarray) -> Tuple[float, float]:
    """
    ARCH test for heteroscedasticity (volatility clustering).
    Returns (p-value, rolling_var_std).
    Low p-value = significant ARCH effects = volatility clustering.
    """
    try:
        n = len(values)
        if n < 50:
            return 0.5, 0.0

        # Compute returns/residuals
        residuals = np.diff(values)
        sq_residuals = residuals ** 2

        # Rolling variance std
        window = min(50, n // 4)
        if window < 10:
            return 0.5, 0.0

        rolling_vars = np.array([np.var(residuals[i:i+window])
                                 for i in range(len(residuals) - window + 1)])
        rolling_var_std = np.std(rolling_vars) / (np.mean(rolling_vars) + 1e-10)

        # Simple ARCH(1) test via autocorrelation of squared residuals
        if len(sq_residuals) > 10:
            acf_sq = acf(sq_residuals, nlags=5, fft=True)
            # Ljung-Box style test statistic
            lb_stat = n * np.sum(acf_sq[1:] ** 2)
            # Approximate p-value from chi-square
            p_value = 1 - stats.chi2.cdf(lb_stat, df=5)
        else:
            p_value = 0.5

        return float(p_value), float(rolling_var_std)
    except Exception:
        return 0.5, 0.0


# ============================================================
# DIMENSION 1: CONTINUITY (can be SQL, but include for completeness)
# ============================================================

def _is_constant(signal_std: float, signal_mean: float) -> bool:
    """
    Detect constant signals using relative threshold.

    A signal is constant if:
    1. Absolute std < 1e-10 (numerical zero), OR
    2. Coefficient of variation < 1e-6 (relative to mean)
    """
    if signal_std < 1e-10:
        return True

    if signal_mean != 0 and abs(signal_std / signal_mean) < 1e-6:
        return True

    return False


def compute_continuity_features(values: np.ndarray) -> Dict[str, Any]:
    """
    Features for continuity dimension.

    Includes:
    - derivative_sparsity: fraction of zero derivatives (detects STEP signals)
    - zero_run_ratio: avg consecutive zero run length / total (detects INTERMITTENT)
    """
    try:
        n = len(values)
        unique_vals = np.unique(values)
        n_unique = len(unique_vals)

        # Unique ratio
        unique_ratio = n_unique / n if n > 0 else 0

        # Is integer?
        is_integer = np.allclose(values, np.round(values))

        # Sparsity (fraction of zeros)
        sparsity = np.sum(values == 0) / n if n > 0 else 0

        # Standard deviation and mean
        signal_std = np.std(values)
        signal_mean = np.mean(values)

        # Is constant? (using coefficient of variation)
        is_constant = _is_constant(signal_std, signal_mean)

        # ================================================================
        # DERIVATIVE SPARSITY - for STEP signal detection
        # High value (>0.8) indicates step/plateau signal
        # ================================================================
        if n > 1:
            derivatives = np.diff(values)
            # Use threshold relative to signal std to handle noise
            threshold = 0.01 * signal_std if signal_std > 1e-10 else 1e-10
            zero_derivs = np.sum(np.abs(derivatives) < threshold)
            derivative_sparsity = zero_derivs / len(derivatives)
        else:
            derivative_sparsity = 0.0

        # ================================================================
        # ZERO RUN RATIO - for INTERMITTENT signal detection
        # Measures average consecutive zero run length relative to total
        # High value indicates intermittent/bursty signal with long gaps
        # ================================================================
        if n > 1:
            # Find runs of zeros
            is_zero = np.abs(values) < 1e-10
            runs = []
            current_run = 0
            for z in is_zero:
                if z:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)

            if runs:
                avg_run_length = np.mean(runs)
                zero_run_ratio = avg_run_length / n
            else:
                zero_run_ratio = 0.0
        else:
            zero_run_ratio = 0.0

        return {
            'unique_ratio': float(unique_ratio),
            'is_integer': bool(is_integer),
            'is_constant': bool(is_constant),
            'sparsity': float(sparsity),
            'signal_std': float(signal_std),
            'signal_mean': float(signal_mean),
            'derivative_sparsity': float(derivative_sparsity),
            'zero_run_ratio': float(zero_run_ratio),
        }
    except Exception:
        return {
            'unique_ratio': 1.0,
            'is_integer': False,
            'is_constant': False,
            'sparsity': 0.0,
            'signal_std': 1.0,
            'signal_mean': 0.0,
            'derivative_sparsity': 0.0,
            'zero_run_ratio': 0.0,
        }


# ============================================================
# WINDOW FACTOR - for adaptive windowing in PRISM
# ============================================================

def compute_window_factor(
    spectral_flatness: float,
    spectral_slope: float,
    spectral_peak_snr: float,
    dominant_frequency: float,
    hurst: float,
    perm_entropy: float,
    turning_point_ratio: float,
    adf_pvalue: float,
) -> float:
    """
    Compute window_factor based on signal characteristics.

    Higher factor = signal needs larger windows for reliable analysis.
    Range: 0.5 to 3.0

    Factors that increase window requirement:
    - Narrowband spectrum (need more samples to resolve peaks)
    - Low-frequency content (need longer observation for slow dynamics)
    - Periodic/quasi-periodic patterns (need to capture full cycles)
    - High noise / high entropy (need more averaging)
    - Anti-persistent behavior (noisy, need more samples)
    - Non-stationarity (need context to detect drift)

    Returns:
        Window multiplier (1.0 = base, 2.0 = double window, etc.)
    """
    factor = 1.0

    # ================================================================
    # SPECTRAL CHARACTERISTICS
    # ================================================================

    # Narrowband signals (low flatness, high peak SNR) need more resolution
    if spectral_flatness < 0.3 and spectral_peak_snr > 10:
        factor *= 1.5  # Need more samples to resolve spectral peaks

    # Red noise / 1/f signals (steep negative slope) have energy at low freqs
    if spectral_slope < -1.0:
        factor *= 1.25  # Low-frequency content needs longer observation

    # Periodic signals with low dominant frequency need to capture cycles
    if dominant_frequency > 0 and dominant_frequency < 0.1:
        # Very slow oscillation - need longer window
        factor *= 1.4

    # ================================================================
    # TEMPORAL PATTERN
    # ================================================================

    # Low turning point ratio = trending/persistent = non-stationary
    if turning_point_ratio < 0.5:
        factor *= 1.25  # Trending signals need context

    # Non-stationary by ADF (high p-value = unit root)
    if adf_pvalue > 0.1:
        factor *= 1.2  # Non-stationary needs larger context

    # ================================================================
    # MEMORY / PERSISTENCE
    # ================================================================

    # Anti-persistent (Hurst < 0.4) = rough/noisy, needs more averaging
    if hurst < 0.4:
        factor *= 1.3

    # Highly persistent (Hurst > 0.8) = slow dynamics
    if hurst > 0.8:
        factor *= 1.2

    # ================================================================
    # COMPLEXITY / NOISE
    # ================================================================

    # High entropy = noisy/complex, needs more averaging
    if perm_entropy > 0.9:
        factor *= 1.2

    # ================================================================
    # CLAMP TO REASONABLE RANGE
    # ================================================================
    factor = max(0.5, min(3.0, factor))

    return round(factor, 2)


# ============================================================
# DISTRIBUTION (Dimension 6) - can be SQL but include
# ============================================================

def compute_distribution_features(values: np.ndarray) -> Dict[str, float]:
    """
    Distribution shape features.
    """
    try:
        kurt = float(stats.kurtosis(values, fisher=True) + 3)  # Excess + 3 = regular kurtosis
        skew = float(stats.skew(values))

        # Crest factor
        rms = np.sqrt(np.mean(values ** 2))
        peak = np.max(np.abs(values))
        crest = peak / rms if rms > 1e-10 else 1.0

        return {
            'kurtosis': kurt,
            'skewness': skew,
            'crest_factor': float(crest),
        }
    except Exception:
        return {
            'kurtosis': 3.0,
            'skewness': 0.0,
            'crest_factor': 1.0,
        }


# ============================================================
# MAIN: COMPUTE FULL SIGNAL PROFILE
# ============================================================

def compute_signal_profile(
    values: np.ndarray,
    signal_id: str,
    cohort: Optional[str] = None
) -> SignalProfile:
    """
    Compute complete raw typology profile for a signal.

    Args:
        values: Signal values (sorted by I)
        signal_id: Signal identifier
        cohort: Optional unit identifier

    Returns:
        SignalProfile with all raw measures
    """
    n = len(values)

    # Skip if constant
    std = np.std(values)
    if std < 1e-10:
        # Return minimal profile for constant signals
        return SignalProfile(
            signal_id=signal_id,
            cohort=cohort,
            n_samples=n,
            adf_pvalue=1.0,
            kpss_pvalue=1.0,
            variance_ratio=1.0,
            acf_half_life=None,
            hurst=0.5,
            perm_entropy=0.0,
            sample_entropy=0.0,
            spectral_flatness=0.0,
            spectral_slope=0.0,
            harmonic_noise_ratio=0.0,
            spectral_peak_snr=0.0,
            dominant_frequency=0.0,
            is_first_bin_peak=False,
            turning_point_ratio=0.0,
            lyapunov_proxy=0.0,
            determinism_score=0.0,
            arch_pvalue=1.0,
            rolling_var_std=0.0,
            kurtosis=3.0,
            skewness=0.0,
            crest_factor=1.0,
            unique_ratio=0.0,
            is_integer=False,
            is_constant=True,
            sparsity=0.0,
            signal_std=0.0,
            signal_mean=0.0,
            derivative_sparsity=1.0,  # Constant = all zero derivatives
            zero_run_ratio=0.0,
            window_factor=0.5,  # Constant signals need minimal windows
        )

    # Compute all features
    spectral = compute_spectral_profile(values)
    arch_p, roll_var_std = compute_arch_test(values)
    continuity = compute_continuity_features(values)
    distribution = compute_distribution_features(values)

    # Compute intermediate values needed for window_factor
    adf_pvalue = compute_adf_pvalue(values)
    hurst = compute_hurst(values)
    perm_entropy = compute_permutation_entropy(values)
    turning_point_ratio = compute_turning_point_ratio(values)

    # Compute window_factor based on signal characteristics
    window_factor = compute_window_factor(
        spectral_flatness=spectral['spectral_flatness'],
        spectral_slope=spectral['spectral_slope'],
        spectral_peak_snr=spectral['spectral_peak_snr'],
        dominant_frequency=spectral['dominant_frequency'],
        hurst=hurst,
        perm_entropy=perm_entropy,
        turning_point_ratio=turning_point_ratio,
        adf_pvalue=adf_pvalue,
    )

    return SignalProfile(
        signal_id=signal_id,
        cohort=cohort,
        n_samples=n,

        # Stationarity
        adf_pvalue=adf_pvalue,
        kpss_pvalue=compute_kpss_pvalue(values),
        variance_ratio=compute_variance_ratio(values),
        acf_half_life=compute_acf_half_life(values),

        # Memory
        hurst=hurst,

        # Complexity
        perm_entropy=perm_entropy,
        sample_entropy=compute_sample_entropy(values),

        # Spectral
        spectral_flatness=spectral['spectral_flatness'],
        spectral_slope=spectral['spectral_slope'],
        harmonic_noise_ratio=spectral['harmonic_noise_ratio'],
        spectral_peak_snr=spectral['spectral_peak_snr'],
        dominant_frequency=spectral['dominant_frequency'],
        is_first_bin_peak=spectral.get('is_first_bin_peak', False),

        # Temporal
        turning_point_ratio=turning_point_ratio,
        lyapunov_proxy=compute_lyapunov_proxy(values),

        # Determinism
        determinism_score=compute_determinism_score(values),

        # Volatility
        arch_pvalue=arch_p,
        rolling_var_std=roll_var_std,

        # Distribution
        kurtosis=distribution['kurtosis'],
        skewness=distribution['skewness'],
        crest_factor=distribution['crest_factor'],

        # Continuity
        unique_ratio=continuity['unique_ratio'],
        is_integer=continuity['is_integer'],
        is_constant=continuity['is_constant'],
        sparsity=continuity['sparsity'],
        signal_std=continuity['signal_std'],
        signal_mean=continuity['signal_mean'],
        derivative_sparsity=continuity['derivative_sparsity'],
        zero_run_ratio=continuity['zero_run_ratio'],

        # Window factor for PRISM
        window_factor=window_factor,
    )


def profile_to_dict(profile: SignalProfile) -> Dict[str, Any]:
    """Convert SignalProfile to dict for DataFrame creation."""
    return {
        'signal_id': profile.signal_id,
        'cohort': profile.cohort,
        'n_samples': profile.n_samples,
        'adf_pvalue': profile.adf_pvalue,
        'kpss_pvalue': profile.kpss_pvalue,
        'variance_ratio': profile.variance_ratio,
        'acf_half_life': profile.acf_half_life,
        'hurst': profile.hurst,
        'perm_entropy': profile.perm_entropy,
        'sample_entropy': profile.sample_entropy,
        'spectral_flatness': profile.spectral_flatness,
        'spectral_slope': profile.spectral_slope,
        'harmonic_noise_ratio': profile.harmonic_noise_ratio,
        'spectral_peak_snr': profile.spectral_peak_snr,
        'dominant_frequency': profile.dominant_frequency,
        'is_first_bin_peak': profile.is_first_bin_peak,
        'turning_point_ratio': profile.turning_point_ratio,
        'lyapunov_proxy': profile.lyapunov_proxy,
        'determinism_score': profile.determinism_score,
        'arch_pvalue': profile.arch_pvalue,
        'rolling_var_std': profile.rolling_var_std,
        'kurtosis': profile.kurtosis,
        'skewness': profile.skewness,
        'crest_factor': profile.crest_factor,
        'unique_ratio': profile.unique_ratio,
        'is_integer': profile.is_integer,
        'is_constant': profile.is_constant,
        'sparsity': profile.sparsity,
        'signal_std': profile.signal_std,
        'signal_mean': profile.signal_mean,
        'derivative_sparsity': profile.derivative_sparsity,
        'zero_run_ratio': profile.zero_run_ratio,
        'window_factor': profile.window_factor,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def compute_typology_raw(
    observations_path: str,
    output_path: str = "typology_raw.parquet",
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute raw typology for all signals in observations.parquet.

    Args:
        observations_path: Path to observations.parquet
        output_path: Where to write typology_raw.parquet
        verbose: Print progress

    Returns:
        DataFrame with raw typology measures
    """
    if verbose:
        print(f"ORTHON Typology Raw Computation")
        print(f"  Input: {observations_path}")

    # Read observations
    df = pl.read_parquet(observations_path)

    # Get unique (cohort, signal_id) combinations
    if 'cohort' in df.columns:
        groups = df.select(['cohort', 'signal_id']).unique().sort(['cohort', 'signal_id'])
    else:
        groups = df.select(['signal_id']).unique().sort('signal_id')
        groups = groups.with_columns(pl.lit(None).alias('cohort'))

    if verbose:
        print(f"  Signals: {len(groups)}")

    # Compute profile for each signal
    profiles = []
    for row in groups.iter_rows(named=True):
        signal_id = row['signal_id']
        cohort = row.get('cohort')

        # Filter to this signal
        if cohort is not None:
            signal_df = df.filter(
                (pl.col('signal_id') == signal_id) &
                (pl.col('cohort') == cohort)
            )
        else:
            signal_df = df.filter(pl.col('signal_id') == signal_id)

        # Sort by I and get values
        signal_df = signal_df.sort('I')
        values = signal_df['value'].to_numpy()

        if verbose:
            print(f"    {signal_id}: {len(values)} samples", end='')

        # Compute profile
        profile = compute_signal_profile(values, signal_id, cohort)
        profiles.append(profile_to_dict(profile))

        if verbose:
            print(f" -> H={profile.hurst:.2f}, PE={profile.perm_entropy:.2f}")

    # Create DataFrame
    result_df = pl.DataFrame(profiles)

    # Write
    result_df.write_parquet(output_path)

    if verbose:
        print(f"\n  Output: {output_path}")
        print(f"  Signals processed: {len(profiles)}")

    return result_df


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("ORTHON Typology Raw Computation")
        print("=" * 40)
        print("\nComputes raw statistical measures for 10-dimension typology.")
        print("\nUsage:")
        print("  python -m orthon.ingest.typology_raw <observations.parquet> [typology_raw.parquet]")
        print("\nOutput feeds into typology_v2.sql for classification.")
        sys.exit(1)

    obs_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "typology_raw.parquet"

    compute_typology_raw(obs_path, out_path)
