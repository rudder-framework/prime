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
)
from pmtvs.individual.statistics import (
    skewness as _skewness,
    kurtosis as _kurtosis,
    crest_factor as _crest_factor,
)
from pmtvs.individual.spectral import spectral_profile as _spectral_profile
from pmtvs.individual.acf import acf_half_life as _acf_half_life
from pmtvs.individual.temporal import turning_point_ratio as _turning_point_ratio
from pmtvs.individual.continuity import continuity_features as _continuity_features
from pmtvs.stat_tests.stationarity_tests import (
    adf_test as _adf_test,
    kpss_test as _kpss_test,
)
from pmtvs.stat_tests.volatility import arch_test as _arch_test
from pmtvs.dynamical.rqa import determinism_from_signal as _determinism_from_signal

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

def compute_adf_pvalue(values: np.ndarray, max_lag: int = None) -> float:
    """
    Augmented Dickey-Fuller test via pmtvs.
    H0: unit root (non-stationary). Low p-value -> reject -> stationary.
    """
    try:
        if len(values) < 20:
            return 1.0  # Not enough data, assume non-stationary
        result = _adf_test(values, max_lag=max_lag)
        return float(result[1])  # p-value
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
    """ACF half-life via pmtvs."""
    try:
        if len(values) < 4:
            return None
        result = _acf_half_life(values, threshold=0.5, max_lag=max_lag)
        return float(result) if result is not None else None
    except Exception:
        return None


# ============================================================
# DIMENSION 8: SPECTRAL CHARACTER
# ============================================================

def compute_spectral_profile(values: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
    """Spectral characteristics via pmtvs."""
    _defaults = {
        'spectral_flatness': 0.5, 'spectral_slope': 0.0,
        'harmonic_noise_ratio': 0.0, 'spectral_peak_snr': 0.0,
        'dominant_frequency': 0.0, 'is_first_bin_peak': False,
    }
    try:
        if len(values) < 64:
            return _defaults
        return _spectral_profile(values, fs=fs)
    except Exception:
        return _defaults


# ============================================================
# DIMENSION 3: TEMPORAL PATTERN
# ============================================================

def compute_turning_point_ratio(values: np.ndarray) -> float:
    """Turning point ratio via pmtvs."""
    try:
        if len(values) < 3:
            return 0.67
        return float(_turning_point_ratio(values))
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
    """
    try:
        n = len(values)
        if n < 50:
            return 0.5, 0.0

        # ARCH p-value from pmtvs
        result = _arch_test(values)
        p_value = float(result['pvalue']) if not np.isnan(result['pvalue']) else 0.5

        # Rolling variance std (volatility clustering measure)
        residuals = np.diff(values)
        window = min(50, n // 4)
        if window < 10:
            return p_value, 0.0

        rolling_vars = np.array([np.var(residuals[i:i+window])
                                 for i in range(len(residuals) - window + 1)])
        rolling_var_std = np.std(rolling_vars) / (np.mean(rolling_vars) + 1e-10)

        return p_value, float(rolling_var_std)
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
    Continuity features, partially via pmtvs.

    Includes:
    - derivative_sparsity: fraction of zero derivatives (detects STEP signals)
    - zero_run_ratio: avg consecutive zero run length / total (detects INTERMITTENT)
    """
    try:
        n = len(values)

        # Basic features from pmtvs
        basic = _continuity_features(values)
        unique_ratio = basic['unique_ratio']
        is_integer = basic['is_integer']
        sparsity = basic['sparsity']

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
            .sort('I')
            .select(['I', 'value'])
            .collect()
        )
    else:
        signal_df = (
            lazy.filter(pl.col('signal_id') == signal_id)
            .sort('I')
            .select(['I', 'value'])
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
                    .sort('I')
                    .select(['I', 'value'])
                    .collect()
                )
            else:
                signal_df = (
                    lazy.filter(pl.col('signal_id') == signal_id)
                    .sort('I')
                    .select(['I', 'value'])
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
