"""
Signal Classification — 10 Dimensions
======================================
Classifies a signal from pre-computed features (from signal_vector or
typology_raw measures). Pure threshold logic. No heavy computation.

Both Manifold (per-window) and Prime (per-signal summary) call this.

Dimensions:
    1. continuity    — CONSTANT | EVENT | DISCRETE | CONTINUOUS
    2. stationarity  — STATIONARY | TREND_STATIONARY | DIFFERENCE_STATIONARY | NON_STATIONARY
    3. temporal      — CONSTANT | TRENDING | CHAOTIC | PERIODIC | QUASI_PERIODIC |
                       DRIFTING | MEAN_REVERTING | IMPULSIVE | EVENT | DISCRETE |
                       STATIONARY | RANDOM
    4. memory        — LONG_MEMORY | SHORT_MEMORY | ANTI_PERSISTENT
    5. complexity    — LOW | MEDIUM | HIGH
    6. spectral      — NARROWBAND | HARMONIC | BROADBAND | ONE_OVER_F |
                       RED_NOISE | BLUE_NOISE
    7. determinism   — DETERMINISTIC | STOCHASTIC | MIXED
    8. distribution  — GAUSSIAN | HEAVY_TAILED | SKEWED | UNIFORM
    9. amplitude     — STABLE | MODULATED | HETEROSCEDASTIC
   10. volatility    — LOW | MODERATE | HIGH | CLUSTERED
"""

import math
from typing import Dict, Any, Optional
from typology.config import CONFIG


# =================================================================
# Public API
# =================================================================

def classify(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify a signal across all 10 dimensions.

    Args:
        features: Dict of pre-computed measures. Keys may include:
            signal_std, signal_mean, sparsity, is_integer, unique_ratio,
            kurtosis, crest_factor, skewness, hurst, perm_entropy,
            sample_entropy, spectral_flatness, spectral_slope,
            spectral_peak_snr, dominant_frequency, is_first_bin_peak,
            turning_point_ratio, lyapunov_proxy, determinism_score,
            adf_pvalue, kpss_pvalue, variance_ratio, acf_half_life,
            arch_pvalue, rolling_var_std, n_samples

    Returns:
        Dict with keys: continuity, stationarity, temporal, memory,
        complexity, spectral, determinism, distribution, amplitude,
        volatility, confidence
    """
    return {
        'continuity':  _continuity(features),
        'stationarity': _stationarity(features),
        'temporal':    _temporal(features),
        'memory':      _memory(features),
        'complexity':  _complexity(features),
        'spectral':    _spectral(features),
        'determinism': _determinism(features),
        'distribution': _distribution(features),
        'amplitude':   _amplitude(features),
        'volatility':  _volatility(features),
    }


# =================================================================
# Helpers
# =================================================================

def _get(features, key, default=None):
    """Safely get a feature value, handling NaN."""
    val = features.get(key, default)
    if val is None:
        return default
    try:
        if math.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return val


def _is_constant(features):
    """Check if signal is constant."""
    std = _get(features, 'signal_std')
    mean = _get(features, 'signal_mean')
    if std is not None and std < 1e-10:
        return True
    if std is not None and mean is not None and abs(mean) > 1e-10:
        if std / abs(mean) < 1e-6:
            return True
    return False


# =================================================================
# Dimension 1: Continuity
# =================================================================

def _continuity(f):
    if _is_constant(f):
        return 'CONSTANT'
    if _get(f, 'sparsity', 0) > 0.9:
        return 'EVENT'
    if _get(f, 'is_integer', False) and _get(f, 'unique_ratio', 1.0) < 0.05:
        return 'DISCRETE'
    return 'CONTINUOUS'


# =================================================================
# Dimension 2: Stationarity
# =================================================================

def _stationarity(f):
    if _is_constant(f):
        return 'STATIONARY'

    adf = _get(f, 'adf_pvalue')
    kpss = _get(f, 'kpss_pvalue')
    tpr = _get(f, 'turning_point_ratio', 0.667)

    # Monotonic signals: trust KPSS over ADF
    if tpr < 0.5:
        if kpss is not None and kpss < 0.05:
            return 'NON_STATIONARY'
        return 'TREND_STATIONARY'

    if adf is None and kpss is None:
        return 'NON_STATIONARY'  # can't determine, assume worst

    adf_stationary = adf is not None and adf < 0.05
    kpss_stationary = kpss is not None and kpss >= 0.05

    if adf_stationary and kpss_stationary:
        return 'STATIONARY'
    if adf_stationary and not kpss_stationary:
        return 'TREND_STATIONARY'
    if not adf_stationary and kpss_stationary:
        return 'DIFFERENCE_STATIONARY'
    return 'NON_STATIONARY'


# =================================================================
# Dimension 3: Temporal Pattern (the big one)
# =================================================================

def _temporal(f):
    """
    Decision tree (order matters):
        0. CONSTANT
        1. DISCRETE
        2. IMPULSIVE
        3. EVENT
        4. CHAOTIC (bounded deterministic)
        5. CHAOTIC (general)
        6. TRENDING
        7. DRIFTING
        8. PERIODIC
        9. QUASI_PERIODIC
       10. STATIONARY
       11. RANDOM (default)
    """
    cfg = CONFIG['temporal']

    # --- CONSTANT ---
    if _is_constant(f):
        return 'CONSTANT'

    # --- DISCRETE ---
    if _get(f, 'is_integer', False):
        if _get(f, 'unique_ratio', 1.0) < cfg['discrete']['unique_ratio_max']:
            return 'DISCRETE'

    # --- IMPULSIVE ---
    kurt = _get(f, 'kurtosis', 3.0)
    crest = _get(f, 'crest_factor', 1.0)
    if kurt > cfg['impulsive']['kurtosis_min'] and crest > cfg['impulsive']['crest_factor_min']:
        return 'IMPULSIVE'

    # --- EVENT ---
    if _get(f, 'sparsity', 0) > cfg['event']['sparsity_min']:
        if kurt > cfg['event']['kurtosis_min']:
            return 'EVENT'

    # --- CHAOTIC ---
    hurst = _get(f, 'hurst', 0.5)
    pe = _get(f, 'perm_entropy', 0.5)
    se = _get(f, 'sample_entropy', 1.0)
    lyap = _get(f, 'lyapunov_proxy', 0.0)
    vr = _get(f, 'variance_ratio')
    n = _get(f, 'n_samples', 5000)
    det = _get(f, 'determinism_score', 1.0)

    # Bounded deterministic chaos (Rössler, pendulum)
    bd_cfg = cfg.get('bounded_deterministic', {})
    if bd_cfg.get('enabled', False) and vr is not None:
        if (hurst >= bd_cfg['hurst_min'] and
                pe <= bd_cfg['perm_entropy_max'] and
                vr <= bd_cfg['variance_ratio_max']):
            cc = cfg['chaotic'].get('clean_chaos', {})
            if cc.get('enabled', False):
                if (lyap > cc['lyapunov_proxy_min'] and
                        pe < cc['perm_entropy_max'] and
                        se < cc['sample_entropy_max']):
                    return 'CHAOTIC'
            # Bounded but not chaotic → skip TRENDING, fall through

    # General chaotic
    ch = cfg['chaotic']
    if n >= ch['min_samples'] and lyap > ch['lyapunov_proxy_min'] and pe > ch['perm_entropy_min']:
        if det > ch.get('determinism_score_min', 0):
            return 'CHAOTIC'

    # --- TRENDING (only if not chaotic/bounded) ---
    if not (bd_cfg.get('enabled', False) and vr is not None and
            hurst >= bd_cfg['hurst_min'] and pe <= bd_cfg['perm_entropy_max']):
        tr = cfg['trending']
        acf = _get(f, 'acf_half_life')

        if hurst >= tr['hurst_strong']:
            return 'TRENDING'

        if hurst > tr['hurst_moderate']:
            acf_absent = acf is None
            acf_long = (not acf_absent and n > 0 and acf / n > tr['acf_ratio_min'])
            entropy_low = se < tr['sample_entropy_max']
            if (acf_absent or acf_long) and entropy_low:
                return 'TRENDING'

    # --- DRIFTING ---
    stationarity = _get(f, 'stationarity')
    if _is_drifting(hurst, pe, vr, stationarity):
        return 'DRIFTING'

    # --- PERIODIC ---
    sf = _get(f, 'spectral_flatness', 0.5)
    snr = _get(f, 'spectral_peak_snr', 0.0)
    dom_freq = _get(f, 'dominant_frequency')
    tpr = _get(f, 'turning_point_ratio', 0.667)
    first_bin = _get(f, 'is_first_bin_peak', False)
    slope = _get(f, 'spectral_slope', 0.0)
    acf = _get(f, 'acf_half_life')

    if _is_periodic(dom_freq, acf, tpr, snr, sf, slope, hurst, n):
        return 'PERIODIC'

    # --- QUASI_PERIODIC ---
    qp = cfg['quasi_periodic']
    if (sf < qp['spectral_flatness_max'] and
            snr > qp['spectral_peak_snr_min'] and
            pe < qp['perm_entropy_max']):
        return 'QUASI_PERIODIC'

    # --- STATIONARY ---
    st = cfg['stationary']
    adf = _get(f, 'adf_pvalue')
    if adf is not None and adf < st['adf_pvalue_max']:
        if vr is not None and st['variance_ratio_min'] <= vr <= st['variance_ratio_max']:
            return 'STATIONARY'

    # --- RANDOM (null hypothesis) ---
    return 'RANDOM'


def _is_drifting(hurst, pe, vr, stationarity):
    """Non-stationary persistent with high entropy."""
    if hurst is None or hurst < 0.85:
        return False
    if pe is not None and pe < 0.5:
        return False
    if stationarity == 'STATIONARY':
        return False
    if vr is not None and 0.85 <= vr <= 1.15:
        return False
    return True


def _is_periodic(dom_freq, acf, tpr, snr, sf, slope, hurst, n):
    """Multi-gate periodicity check."""
    pcfg = CONFIG['periodic']

    if dom_freq is None or dom_freq <= 0:
        return False

    # Spectral override: overwhelming evidence
    ov = pcfg.get('spectral_override', {})
    if ov.get('enabled', False):
        if snr > ov['snr_threshold'] and sf < ov['flatness_threshold']:
            return True

    # Standard gates
    if sf > pcfg['spectral_flatness_max']:
        return False
    if snr < pcfg['snr_min']:
        return False
    if hurst > pcfg['hurst_max']:
        return False
    if tpr > pcfg['turning_point_ratio_max']:
        return False
    return True


# =================================================================
# Dimension 4: Memory
# =================================================================

def _memory(f):
    if _is_constant(f):
        return None
    hurst = _get(f, 'hurst', 0.5)
    if hurst > CONFIG['memory']['long_memory_min']:
        return 'LONG_MEMORY'
    if hurst < CONFIG['memory']['anti_persistent_max']:
        return 'ANTI_PERSISTENT'
    return 'SHORT_MEMORY'


# =================================================================
# Dimension 5: Complexity
# =================================================================

def _complexity(f):
    if _is_constant(f):
        return None
    pe = _get(f, 'perm_entropy', 0.5)
    if pe < CONFIG['complexity']['low_max']:
        return 'LOW'
    if pe > CONFIG['complexity']['high_min']:
        return 'HIGH'
    return 'MEDIUM'


# =================================================================
# Dimension 6: Spectral
# =================================================================

def _spectral(f):
    if _is_constant(f):
        return None
    sf = _get(f, 'spectral_flatness', 0.5)
    slope = _get(f, 'spectral_slope', 0.0)
    hnr = _get(f, 'harmonic_noise_ratio', 0.0)
    snr = _get(f, 'spectral_peak_snr', 0.0)

    scfg = CONFIG['spectral']

    if hnr > scfg['harmonic']['hnr_min'] and snr > 10:
        return 'HARMONIC'
    if sf > scfg['broadband']['spectral_flatness_min']:
        return 'BROADBAND'
    if slope < scfg['red_noise']['spectral_slope_max']:
        return 'RED_NOISE'
    if slope > scfg['blue_noise']['spectral_slope_min']:
        return 'BLUE_NOISE'
    if snr > 10 and sf < 0.3:
        return 'NARROWBAND'
    return 'ONE_OVER_F'


# =================================================================
# Dimension 7: Determinism
# =================================================================

def _determinism(f):
    if _is_constant(f):
        return None
    det = _get(f, 'determinism_score')
    if det is None:
        return None
    if det > 0.8:
        return 'DETERMINISTIC'
    if det < 0.3:
        return 'STOCHASTIC'
    return 'MIXED'


# =================================================================
# Dimension 8: Distribution
# =================================================================

def _distribution(f):
    if _is_constant(f):
        return None
    kurt = _get(f, 'kurtosis', 3.0)
    skew = _get(f, 'skewness', 0.0)

    if kurt > 5:
        return 'HEAVY_TAILED'
    if abs(skew) > 1.0:
        return 'SKEWED'
    if kurt < 2.5:
        return 'UNIFORM'
    return 'GAUSSIAN'


# =================================================================
# Dimension 9: Amplitude
# =================================================================

def _amplitude(f):
    if _is_constant(f):
        return None
    rvs = _get(f, 'rolling_var_std', 0.0)
    if rvs > 0.5:
        return 'HETEROSCEDASTIC'
    if rvs > 0.2:
        return 'MODULATED'
    return 'STABLE'


# =================================================================
# Dimension 10: Volatility
# =================================================================

def _volatility(f):
    if _is_constant(f):
        return None
    arch_p = _get(f, 'arch_pvalue', 1.0)
    rvs = _get(f, 'rolling_var_std', 0.0)

    if arch_p < 0.01:
        return 'CLUSTERED'
    if rvs > 0.5:
        return 'HIGH'
    if rvs > 0.2:
        return 'MODERATE'
    return 'LOW'
