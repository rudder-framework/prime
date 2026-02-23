"""
Typology Raw Computation

Computes the raw statistical measures that feed into typology_v2.sql.
This is the ONLY computation Prime performs - everything else is classification.

Manifold computes engine outputs. Prime computes typology and classifies.

Output: typology_raw.parquet with one row per (cohort, signal_id)

Usage:
    python -m prime.ingest.typology_raw data/observations.parquet data/typology_raw.parquet
"""

import os
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from pmtvs import (
    hurst_exponent,
    permutation_entropy,
    sample_entropy,
    lyapunov_rosenstein,
    BACKEND as PRIMITIVES_BACKEND,
    skewness as _skewness,
    kurtosis as _kurtosis,
    crest_factor as _crest_factor,
    spectral_flatness as _spectral_flatness,
    dominant_frequency as _dominant_frequency,
    harmonic_ratio as _harmonic_ratio,
    psd as _psd,
    acf_decay_time as _acf_decay_time,
    adf_test as _adf_test,
    kpss_test as _kpss_test,
    arch_test as _arch_test,
    determinism_from_signal as _determinism_from_signal,
)


def _spectral_slope(values: np.ndarray) -> float:
    """
    Compute spectral slope via linear regression in log-log space.
    Negative slope indicates 1/f-like (red) noise, near-zero is white noise.
    """
    try:
        n = len(values)
        if n < 16:
            return 0.0
        # Compute PSD
        psd_vals = np.abs(np.fft.rfft(values)) ** 2
        freqs = np.fft.rfftfreq(n)
        # Skip DC component
        freqs = freqs[1:]
        psd_vals = psd_vals[1:]
        # Filter out zeros for log
        mask = (freqs > 0) & (psd_vals > 0)
        if np.sum(mask) < 2:
            return 0.0
        log_freq = np.log10(freqs[mask])
        log_psd = np.log10(psd_vals[mask])
        # Linear regression
        slope, _ = np.polyfit(log_freq, log_psd, 1)
        return float(slope)
    except Exception:
        return 0.0


def _signal_to_noise(values: np.ndarray) -> dict:
    """
    Estimate signal-to-noise ratio in dB.
    Uses the ratio of signal variance to noise variance estimate.
    """
    try:
        n = len(values)
        if n < 4:
            return {'db': 0.0}
        # Estimate noise as high-frequency residual
        # Simple approach: noise = diff of diff (second derivative captures noise)
        signal_var = np.var(values)
        noise_estimate = np.var(np.diff(values)) / 2  # Variance of diff / 2 approximates noise var
        if noise_estimate < 1e-10:
            return {'db': 60.0}  # Very clean signal
        if signal_var < 1e-10:
            return {'db': 0.0}
        snr = signal_var / noise_estimate
        db = 10 * np.log10(snr) if snr > 0 else 0.0
        return {'db': float(np.clip(db, -20, 60))}
    except Exception:
        return {'db': 0.0}

# Parallel workers — set PRIME_WORKERS=N to override, default = 4
PRIME_WORKERS = int(os.environ.get("PRIME_WORKERS", "0")) or 4


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

    # Window Factor (for adaptive windowing in Manifold)
    window_factor: float = 1.0  # Multiplier for engine base windows


# ============================================================
# DIMENSION 2: STATIONARITY
# ============================================================

def _adf_stat_to_pvalue(stat: float) -> float:
    """
    Convert ADF test statistic to approximate p-value.

    Uses interpolation on Dickey-Fuller distribution critical values
    for the constant-only ('c') model at large T (asymptotic).
    Values from MacKinnon (1994, 2010) and Fuller (1976).
    """
    # (critical_value, p-value) pairs — critical values ascending
    # Standard DF critical values from MacKinnon (1994, 2010), constant-only model.
    # Extended into the deep left tail for resolution at very low p-values.
    _cvs = np.array([
        -15.0, -12.0, -10.0, -8.0, -6.5, -5.5, -5.00, -4.50,
        -3.96, -3.63, -3.43, -3.12, -2.86, -2.57,
        -2.22, -1.95, -1.60, -1.20, -0.93, -0.49, -0.07, 0.67,
    ])
    _pvals = np.array([
        1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 2e-4, 5e-4, 8e-4,
        0.001, 0.005, 0.01, 0.025, 0.05, 0.10,
        0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99,
    ])
    return float(np.clip(np.interp(stat, _cvs, _pvals), 0.0, 1.0))


def compute_adf_pvalue(values: np.ndarray, max_lag: int = None) -> float:
    """
    Augmented Dickey-Fuller test via pmtvs.
    H0: unit root (non-stationary). Low p-value -> reject -> stationary.

    pmtvs.adf_test returns (test_statistic, critical_value_5pct).
    We take the test statistic and convert to a p-value via the
    Dickey-Fuller distribution critical value table.
    """
    try:
        if len(values) < 20:
            return 1.0  # Not enough data, assume non-stationary
        result = _adf_test(values, max_lag=max_lag)
        stat = float(result[0])  # test statistic (index 0)
        if not np.isfinite(stat):
            return 1.0
        return _adf_stat_to_pvalue(stat)
    except Exception:
        return 1.0


def compute_kpss_pvalue(values: np.ndarray) -> float:
    """
    KPSS test via pmtvs.
    H0: stationary. Low p-value -> reject -> non-stationary.
    """
    try:
        if len(values) < 20:
            return 0.0  # Not enough data, assume non-stationary
        result = _kpss_test(values, regression='c')
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
    """ACF decay time via pmtvs (formerly acf_half_life)."""
    try:
        if len(values) < 4:
            return None
        result = _acf_decay_time(values, threshold=0.5)
        return float(result) if result is not None and np.isfinite(result) else None
    except Exception:
        return None


# ============================================================
# DIMENSION 8: SPECTRAL CHARACTER
# ============================================================

def compute_spectral_profile(values: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
    """Spectral characteristics via individual pmtvs functions."""
    _defaults = {
        'spectral_flatness': 0.5, 'spectral_slope': 0.0,
        'harmonic_noise_ratio': 0.0, 'spectral_peak_snr': 0.0,
        'dominant_frequency': 0.0, 'is_first_bin_peak': False,
    }
    try:
        if len(values) < 64:
            return _defaults

        flatness = float(_spectral_flatness(values))
        slope = float(_spectral_slope(values))
        dom_freq = float(_dominant_frequency(values))
        hnr = float(_harmonic_ratio(values))
        snr_result = _signal_to_noise(values)
        snr = float(snr_result['db']) if isinstance(snr_result, dict) else float(snr_result)

        # Detect if dominant frequency is in the first FFT bin (1/f artifact)
        n = len(values)
        freq_resolution = fs / n
        is_first_bin = dom_freq <= freq_resolution * 1.5

        return {
            'spectral_flatness': flatness,
            'spectral_slope': slope,
            'harmonic_noise_ratio': hnr,
            'spectral_peak_snr': snr,
            'dominant_frequency': dom_freq,
            'is_first_bin_peak': is_first_bin,
        }
    except Exception:
        return _defaults


# ============================================================
# DIMENSION 3: TEMPORAL PATTERN
# ============================================================

def compute_turning_point_ratio(values: np.ndarray) -> float:
    """Turning point ratio: fraction of interior points that are local extrema."""
    try:
        n = len(values)
        if n < 3:
            return 0.67
        diff = np.diff(values)
        sign_changes = np.diff(np.sign(diff))
        turning_points = np.sum(sign_changes != 0)
        return float(turning_points / (n - 2))
    except Exception:
        return 0.67


# ============================================================
# DIMENSION 10: DETERMINISM
# ============================================================

def compute_determinism_score(values: np.ndarray, threshold: float = None) -> float:
    """Determinism via pmtvs RQA."""
    try:
        if len(values) < 50:
            return 0.5
        return float(_determinism_from_signal(values))
    except Exception:
        return 0.5


# ============================================================
# DIMENSION 9: VOLATILITY
# ============================================================

def compute_arch_test(values: np.ndarray) -> Tuple[float, float]:
    """
    ARCH test via pmtvs + rolling variance std.
    Returns (p-value, rolling_var_std).

    pmtvs.arch_test returns a tuple (statistic, pvalue).
    Returns NaN for arch_pvalue when signal is too short.
    """
    n = len(values)
    if n < 50:
        return float('nan'), 0.0

    # ARCH p-value from pmtvs — returns tuple (statistic, pvalue)
    try:
        result = _arch_test(values)
        p_value = float(result[1])  # pvalue is index 1
        if not np.isfinite(p_value):
            p_value = float('nan')
    except Exception:
        p_value = float('nan')

    # Rolling variance std (volatility clustering measure)
    try:
        residuals = np.diff(values)
        window = max(20, n // 10)
        if window >= len(residuals):
            window = max(10, len(residuals) // 4)
        if window < 10:
            return p_value, 0.0

        rolling_vars = np.array([np.var(residuals[i:i+window])
                                 for i in range(len(residuals) - window + 1)])
        rolling_var_std = np.std(rolling_vars) / (np.mean(rolling_vars) + 1e-10)

        return p_value, float(rolling_var_std)
    except Exception:
        return p_value, 0.0


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
    Continuity features, partially via pmtvs.

    Includes:
    - derivative_sparsity: fraction of zero derivatives (detects STEP signals)
    - zero_run_ratio: avg consecutive zero run length / total (detects INTERMITTENT)
    """
    try:
        n = len(values)

        # Basic continuity features (formerly from pmtvs.continuity_features)
        unique_count = len(np.unique(values))
        unique_ratio = unique_count / n if n > 0 else 1.0
        is_integer = bool(np.allclose(values, np.round(values), atol=1e-6))
        sparsity = float(np.sum(np.abs(values) < 1e-10) / n) if n > 0 else 0.0

        # Signal stats
        signal_std = float(np.std(values))
        signal_mean = float(np.mean(values))
        is_constant = _is_constant(signal_std, signal_mean)

        # Derivative sparsity — STEP signal detection
        if n > 1:
            derivatives = np.diff(values)
            threshold = 0.01 * signal_std if signal_std > 1e-10 else 1e-10
            zero_derivs = np.sum(np.abs(derivatives) < threshold)
            derivative_sparsity = zero_derivs / len(derivatives)
        else:
            derivative_sparsity = 0.0

        # Zero run ratio — INTERMITTENT signal detection
        if n > 1:
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
                zero_run_ratio = np.mean(runs) / n
            else:
                zero_run_ratio = 0.0
        else:
            zero_run_ratio = 0.0

        return {
            'unique_ratio': float(unique_ratio),
            'is_integer': bool(is_integer),
            'is_constant': bool(is_constant),
            'sparsity': float(sparsity),
            'signal_std': signal_std,
            'signal_mean': signal_mean,
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
# WINDOW FACTOR - for adaptive windowing in Manifold
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
    Distribution shape features via pmtvs.
    """
    try:
        kurt = float(_kurtosis(values, fisher=True)) + 3  # Excess + 3 = regular kurtosis
        skew = float(_skewness(values))
        crest = float(_crest_factor(values))

        return {
            'kurtosis': kurt,
            'skewness': skew,
            'crest_factor': crest,
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
        values: Signal values (sorted by signal_0)
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
            signal_mean=float(np.mean(values)),
            derivative_sparsity=1.0,  # Constant = all zero derivatives
            zero_run_ratio=0.0,
            window_factor=1.0,  # Default window — Manifold handles constants
        )

    # Compute all features
    spectral = compute_spectral_profile(values)
    arch_p, roll_var_std = compute_arch_test(values)
    continuity = compute_continuity_features(values)
    distribution = compute_distribution_features(values)

    # Compute intermediate values needed for window_factor
    adf_pvalue = compute_adf_pvalue(values)
    hurst = hurst_exponent(values)
    perm_entropy = permutation_entropy(values)
    turning_point_ratio = compute_turning_point_ratio(values)

    # Lyapunov via pmtvs (full Rosenstein, not proxy)
    try:
        lyap = lyapunov_rosenstein(values)[0]
        if np.isnan(lyap):
            lyap = 0.0
    except Exception:
        lyap = 0.0

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
        sample_entropy=sample_entropy(values),

        # Spectral
        spectral_flatness=spectral['spectral_flatness'],
        spectral_slope=spectral['spectral_slope'],
        harmonic_noise_ratio=spectral['harmonic_noise_ratio'],
        spectral_peak_snr=spectral['spectral_peak_snr'],
        dominant_frequency=spectral['dominant_frequency'],
        is_first_bin_peak=spectral.get('is_first_bin_peak', False),

        # Temporal
        turning_point_ratio=turning_point_ratio,
        lyapunov_proxy=lyap,

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

        # Window factor for Manifold
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
# PARALLEL WORKER (must be top-level for pickling)
# ============================================================

def _compute_one_signal(
    observations_path: str,
    signal_id: str,
    cohort: Optional[str],
) -> Dict[str, Any]:
    """Worker function for parallel typology computation."""
    lazy = pl.scan_parquet(observations_path)

    if cohort is not None:
        signal_df = (
            lazy.filter(
                (pl.col('signal_id') == signal_id) &
                (pl.col('cohort') == cohort)
            )
            .sort('signal_0')
            .select(['signal_0', 'value'])
            .collect()
        )
    else:
        signal_df = (
            lazy.filter(pl.col('signal_id') == signal_id)
            .sort('signal_0')
            .select(['signal_0', 'value'])
            .collect()
        )

    values = signal_df['value'].to_numpy()
    del signal_df

    profile = compute_signal_profile(values, signal_id, cohort)
    return profile_to_dict(profile)


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

    Memory: O(largest_signal), NOT O(total_dataset).
    Scans lazily for signal list, then pulls one signal at a time.
    Set PRIME_WORKERS=N for N-way parallel computation.

    Args:
        observations_path: Path to observations.parquet
        output_path: Where to write typology_raw.parquet
        verbose: Print progress

    Returns:
        DataFrame with raw typology measures
    """
    workers = PRIME_WORKERS

    if verbose:
        print(f"Typology Raw Computation")
        print(f"  Backend: pmtvs ({PRIMITIVES_BACKEND})")
        print(f"  Workers: {workers}")
        print(f"  Input: {observations_path}")

    # Lazy scan — only reads metadata, not the full dataset
    lazy = pl.scan_parquet(observations_path)
    schema_cols = lazy.collect_schema().names()
    has_cohort = 'cohort' in schema_cols

    # Get unique (cohort, signal_id) combinations — small result, safe to collect
    if has_cohort:
        groups = lazy.select(['cohort', 'signal_id']).unique().sort(['cohort', 'signal_id']).collect()
    else:
        groups = lazy.select(['signal_id']).unique().sort('signal_id').collect()
        groups = groups.with_columns(pl.lit(None).alias('cohort'))

    if verbose:
        print(f"  Signals: {len(groups)}")

    profiles = []

    if workers > 1:
        # ── Parallel processing ──
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row in groups.iter_rows(named=True):
                signal_id = row['signal_id']
                cohort = row.get('cohort')
                fut = pool.submit(
                    _compute_one_signal,
                    observations_path, signal_id, cohort,
                )
                futures[fut] = signal_id

            done = 0
            total = len(futures)
            for fut in as_completed(futures):
                signal_id = futures[fut]
                try:
                    profile_dict = fut.result()
                    profiles.append(profile_dict)
                except Exception as e:
                    if verbose:
                        print(f"    {signal_id}: FAILED ({e})")
                done += 1
                if verbose and (done % 500 == 0 or done == total):
                    print(f"    {done}/{total} signals complete")
    else:
        # ── Sequential processing ──
        for row in groups.iter_rows(named=True):
            signal_id = row['signal_id']
            cohort = row.get('cohort')

            if cohort is not None:
                signal_df = (
                    lazy.filter(
                        (pl.col('signal_id') == signal_id) &
                        (pl.col('cohort') == cohort)
                    )
                    .sort('signal_0')
                    .select(['signal_0', 'value'])
                    .collect()
                )
            else:
                signal_df = (
                    lazy.filter(pl.col('signal_id') == signal_id)
                    .sort('signal_0')
                    .select(['signal_0', 'value'])
                    .collect()
                )

            values = signal_df['value'].to_numpy()
            del signal_df

            if verbose:
                print(f"    {signal_id}: {len(values)} samples", end='')

            profile = compute_signal_profile(values, signal_id, cohort)
            profiles.append(profile_to_dict(profile))
            del values

            if verbose:
                print(f" -> H={profile.hurst:.2f}, PE={profile.perm_entropy:.2f}")

    # Create DataFrame (small — one row per signal)
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
        print("Typology Raw Computation")
        print("=" * 40)
        print("\nComputes raw statistical measures for 10-dimension typology.")
        print("\nUsage:")
        print("  python -m prime.ingest.typology_raw <observations.parquet> [typology_raw.parquet]")
        print("\nOutput feeds into typology_v2.sql for classification.")
        sys.exit(1)

    obs_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "typology_raw.parquet"

    compute_typology_raw(obs_path, out_path)
