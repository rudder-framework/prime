"""
Level 2 Typology Corrections - Config-Driven Classification

This module contains corrected classification functions that use
centralized configuration from orthon.config.typology_config.

All thresholds are configurable - no magic numbers in the logic.

PR4 Updates:
  - spectral_override for noisy periodic (guitar A3 fix)
  - segment_trend for oscillating trends (battery fix)  
  - bounded_deterministic for smooth chaos (pendulum/Rössler fix)
"""

from typing import Any, Dict, List, Optional
import math

# Import config (with fallback for standalone testing)
try:
    from orthon.config import TYPOLOGY_CONFIG, get_threshold
except ImportError:
    from .config.typology_config import TYPOLOGY_CONFIG, get_threshold


# ============================================================
# FIX 1: First-Bin Artifact Detection
# ============================================================

def is_first_bin_artifact(
    dominant_frequency: Optional[float],
    n_samples: int,
    fft_size: int = None,
    spectral_slope: float = 0.0,
) -> bool:
    """
    Detect if dominant_frequency is a first-bin FFT artifact.
    
    Uses config: artifacts.first_bin_tolerance, artifacts.first_bin_slope_threshold
    """
    if dominant_frequency is None or math.isnan(dominant_frequency):
        return False
    
    if fft_size is None:
        fft_size = get_threshold('artifacts.default_fft_size', 256)
    
    first_bin = 1.0 / fft_size
    tolerance = get_threshold('artifacts.first_bin_tolerance', 0.01)
    slope_thresh = get_threshold('artifacts.first_bin_slope_threshold', -0.3)
    
    freq_match = abs(dominant_frequency - first_bin) < (first_bin * tolerance)
    has_red_slope = spectral_slope < slope_thresh
    
    return freq_match and has_red_slope


# ============================================================
# FIX 2: Genuine Periodicity Validation (with spectral override)
# ============================================================

def is_genuine_periodic(
    dominant_frequency: Optional[float],
    acf_half_life: Optional[float],
    turning_point_ratio: float,
    spectral_peak_snr: float,
    spectral_flatness: float,
    spectral_slope: float,
    hurst: Optional[float],
    n_samples: int,
    fft_size: int = None,
) -> bool:
    """
    Validate if a signal is genuinely periodic (not a trend or noise).
    
    NEW: Spectral override - if SNR > 30 dB and flatness < 0.1,
    skip TPR check (noisy periodic signals have inflated TPR).
    
    Uses config: periodic.*, artifacts.*
    """
    if fft_size is None:
        fft_size = get_threshold('artifacts.default_fft_size', 256)
    
    cfg = TYPOLOGY_CONFIG['periodic']
    
    # Gate 1: Real spectral peak (not first-bin artifact)
    if dominant_frequency is None or math.isnan(dominant_frequency):
        return False
    if is_first_bin_artifact(dominant_frequency, n_samples, fft_size, spectral_slope):
        return False
    
    # Gate 2: Not broadband
    if spectral_flatness > cfg['spectral_flatness_max']:
        return False
    
    # Gate 3: Prominent spectral peak
    if spectral_peak_snr < cfg['snr_min']:
        return False
    
    # Gate 4: ACF shows oscillation (must exist)
    if acf_half_life is None:
        return False
    
    # Gate 5: Not a trend (hurst < threshold)
    if hurst is not None and hurst > cfg['hurst_max']:
        return False
    
    # NEW: Spectral override - overwhelming spectral evidence
    override_cfg = cfg.get('spectral_override', {})
    if override_cfg.get('enabled', False):
        snr_thresh = override_cfg.get('snr_threshold', 30.0)
        flat_thresh = override_cfg.get('flatness_threshold', 0.1)
        
        if spectral_peak_snr > snr_thresh and spectral_flatness < flat_thresh:
            # Spectral evidence is overwhelming - skip TPR check
            return True
    
    # Gate 6: Not monotonic (standard check)
    if turning_point_ratio > cfg['turning_point_ratio_max']:
        return False
    
    return True


# ============================================================
# FIX 3: Segment Trend Detection (for oscillating trends)
# ============================================================

def has_segment_trend(
    segment_means: List[float],
    total_mean: float,
    hurst: float,
) -> bool:
    """
    Detect global trend via segment analysis.
    
    Catches oscillating trends like battery degradation where
    local oscillations mask the global monotonic decline.
    
    Uses config: temporal.trending.segment_trend.*
    """
    cfg = TYPOLOGY_CONFIG['temporal']['trending'].get('segment_trend', {})
    
    if not cfg.get('enabled', False):
        return False
    
    # Check hurst floor
    hurst_floor = cfg.get('hurst_floor', 0.60)
    if hurst < hurst_floor:
        return False
    
    if not segment_means or len(segment_means) < 2:
        return False
    
    # Check monotonicity
    if cfg.get('monotonic_required', True):
        diffs = [segment_means[i+1] - segment_means[i] for i in range(len(segment_means)-1)]
        all_increasing = all(d >= 0 for d in diffs)
        all_decreasing = all(d <= 0 for d in diffs)
        if not (all_increasing or all_decreasing):
            return False
    
    # Check magnitude of change
    min_change = cfg.get('min_change_ratio', 0.20)
    if total_mean == 0:
        return False
    
    total_change = abs(segment_means[-1] - segment_means[0]) / abs(total_mean)
    return total_change >= min_change


# ============================================================
# FIX 4: Bounded Deterministic Detection (for smooth chaos)
# ============================================================

def is_bounded_deterministic(
    hurst: float,
    perm_entropy: float,
    variance_ratio: Optional[float],
) -> bool:
    """
    Detect smooth deterministic chaos (e.g., double pendulum, Rössler).
    
    These systems have high hurst (locally smooth) but bounded variance
    (not expanding like true trends).
    
    Uses config: temporal.bounded_deterministic.*
    
    NOTE: Requires explicit variance_ratio to be computed and passed.
    Returns False if variance_ratio is None (can't determine boundedness).
    """
    cfg = TYPOLOGY_CONFIG['temporal'].get('bounded_deterministic', {})
    
    if not cfg.get('enabled', False):
        return False
    
    # Must have explicit variance_ratio to determine boundedness
    if variance_ratio is None:
        return False
    
    hurst_min = cfg.get('hurst_min', 0.95)
    pe_max = cfg.get('perm_entropy_max', 0.5)
    var_ratio_max = cfg.get('variance_ratio_max', 3.0)
    
    # High hurst (locally smooth)
    if hurst < hurst_min:
        return False
    
    # Low perm entropy (not noisy)
    if perm_entropy > pe_max:
        return False
    
    # Bounded variance (not expanding)
    if variance_ratio > var_ratio_max:
        return False
    
    return True


# ============================================================
# FIX 5: Temporal Pattern Classification
# ============================================================

def classify_temporal_pattern(
    row: Dict[str, Any],
    fft_size: int = None,
) -> str:
    """
    Classify temporal pattern using config-driven thresholds.

    Decision tree:
        0. signal_std == 0 OR variance_ratio < threshold → CONSTANT
        1. is_integer AND unique_ratio < 0.05 → DISCRETE
        2. kurtosis > 20 AND crest_factor > 10 → IMPULSIVE
        3. sparsity > 0.8 AND kurtosis > 10 → EVENT
        4. Check bounded_deterministic (smooth chaos override)
        5. Check segment_trend (oscillating trend override)
        6. hurst >= hurst_strong → TRENDING
        7. hurst > hurst_moderate AND (acf=NaN OR acf_ratio > threshold) AND se < threshold → TRENDING
        8. is_genuine_periodic() → PERIODIC
        9. spectral_flatness > threshold AND perm_entropy > threshold → RANDOM
        10. n >= min_samples AND lyapunov > threshold AND pe > threshold → CHAOTIC
        11. turning_point_ratio < threshold → QUASI_PERIODIC
        12. default → STATIONARY
    """
    if fft_size is None:
        fft_size = get_threshold('artifacts.default_fft_size', 256)
    
    # Extract measures
    dominant_freq = row.get('dominant_frequency')
    acf_half_life = row.get('acf_half_life')
    turning_point_ratio = row.get('turning_point_ratio', 0.667)
    spectral_peak_snr = row.get('spectral_peak_snr', 0.0)
    spectral_flatness = row.get('spectral_flatness', 0.5)
    spectral_slope = row.get('spectral_slope', 0.0)
    hurst = row.get('hurst', 0.5)
    perm_entropy = row.get('perm_entropy', 0.5)
    sample_entropy = row.get('sample_entropy', 1.0)
    lyapunov_proxy = row.get('lyapunov_proxy', 0.0)
    n_samples = row.get('n_samples', 5000)

    # For CONSTANT detection
    signal_std = row.get('signal_std', None)
    signal_mean = row.get('signal_mean', row.get('mean', None))

    # Optional: segment means and variance ratio (if available)
    segment_means = row.get('segment_means', [])
    total_mean = signal_mean if signal_mean is not None else 1.0
    variance_ratio = row.get('variance_ratio')  # None if not computed

    # Get config sections
    trend_cfg = TYPOLOGY_CONFIG['temporal']['trending']
    random_cfg = TYPOLOGY_CONFIG['temporal']['random']
    chaotic_cfg = TYPOLOGY_CONFIG['temporal']['chaotic']
    qp_cfg = TYPOLOGY_CONFIG['temporal']['quasi_periodic']
    const_cfg = TYPOLOGY_CONFIG['temporal'].get('constant', {})

    # ========================================
    # CONSTANT detection (must come FIRST)
    # PR8 FIX: Use robust multi-criteria detection
    # Philosophy: When in doubt, return False. Let PRISM compute.
    # ========================================
    from orthon.typology.constant_detection import classify_constant_from_row
    if classify_constant_from_row(row):
        return 'CONSTANT'

    unique_ratio = row.get('unique_ratio', 1.0)

    # ========================================
    # DISCRETE detection (few unique integer values)
    # Categorical/ordinal data, state machines, digital signals
    # ========================================
    is_integer = row.get('is_integer', False)
    discrete_cfg = TYPOLOGY_CONFIG['temporal'].get('discrete', {})
    if is_integer and unique_ratio < discrete_cfg.get('unique_ratio_max', 0.05):
        return 'DISCRETE'

    # ========================================
    # IMPULSIVE detection (extreme spikes)
    # Impact events, mechanical faults, electrical transients
    # ========================================
    kurtosis = row.get('kurtosis', 3.0)
    crest_factor = row.get('crest_factor', 1.0)
    impulsive_cfg = TYPOLOGY_CONFIG['temporal'].get('impulsive', {})
    if kurtosis > impulsive_cfg.get('kurtosis_min', 20) and \
       crest_factor > impulsive_cfg.get('crest_factor_min', 10):
        return 'IMPULSIVE'

    # ========================================
    # EVENT detection (sparse with high kurtosis)
    # Rare occurrences, trigger signals, fault indicators
    # ========================================
    sparsity = row.get('sparsity', 0.0)
    event_cfg = TYPOLOGY_CONFIG['temporal'].get('event', {})
    if sparsity > event_cfg.get('sparsity_min', 0.8) and \
       kurtosis > event_cfg.get('kurtosis_min', 10):
        return 'EVENT'

    # ========================================
    # Check for bounded deterministic (smooth chaos)
    # This must come BEFORE trending checks
    # ========================================
    if is_bounded_deterministic(hurst or 0.5, perm_entropy, variance_ratio):
        # High hurst + bounded variance = not a trend, likely smooth chaos
        # Fall through to CHAOTIC or QUASI_PERIODIC depending on other measures
        pass  # Don't return TRENDING even if hurst is high
    else:
        # ========================================
        # TRENDING checks (only if not bounded deterministic)
        # ========================================
        
        # NEW: Segment trend detection (oscillating trends)
        if segment_means and has_segment_trend(segment_means, total_mean, hurst or 0.5):
            return 'TRENDING'
        
        # Strong hurst alone → TRENDING
        if hurst is not None and hurst >= trend_cfg['hurst_strong']:
            return 'TRENDING'
        
        # Moderate hurst + long ACF + low entropy → TRENDING
        if hurst is not None and hurst > trend_cfg['hurst_moderate']:
            acf_absent = acf_half_life is None or (isinstance(acf_half_life, float) and math.isnan(acf_half_life))
            acf_long = (not acf_absent and n_samples > 0 and 
                        acf_half_life / n_samples > trend_cfg['acf_ratio_min'])
            entropy_low = sample_entropy < trend_cfg['sample_entropy_max']
            
            if (acf_absent or acf_long) and entropy_low:
                return 'TRENDING'
    
    # ========================================
    # PERIODIC check (with spectral override)
    # ========================================
    if is_genuine_periodic(
        dominant_freq, acf_half_life, turning_point_ratio,
        spectral_peak_snr, spectral_flatness, spectral_slope,
        hurst, n_samples, fft_size,
    ):
        return 'PERIODIC'
    
    # ========================================
    # RANDOM check
    # ========================================
    if spectral_flatness > random_cfg['spectral_flatness_min'] and \
       perm_entropy > random_cfg['perm_entropy_min']:
        return 'RANDOM'
    
    # ========================================
    # CHAOTIC check
    # ========================================
    if n_samples >= chaotic_cfg['min_samples'] and \
       lyapunov_proxy > chaotic_cfg['lyapunov_proxy_min'] and \
       perm_entropy > chaotic_cfg['perm_entropy_min']:
        det_min = chaotic_cfg.get('determinism_score_min')
        det_score = row.get('determinism_score', 1.0)
        if det_min is None or det_score > det_min:
            return 'CHAOTIC'
    
    # ========================================
    # QUASI_PERIODIC check
    # ========================================
    if turning_point_ratio < qp_cfg['turning_point_ratio_max']:
        return 'QUASI_PERIODIC'
    
    # Default
    return 'STATIONARY'


# ============================================================
# FIX 6: Spectral Classification
# ============================================================

def classify_spectral(
    row: Dict[str, Any],
    temporal_pattern: str,
) -> str:
    """
    Classify spectral characteristics using config-driven thresholds.
    
    Uses config: spectral.*
    """
    spectral_flatness = row.get('spectral_flatness', 0.5)
    spectral_slope = row.get('spectral_slope', 0.0)
    harmonic_noise_ratio = row.get('harmonic_noise_ratio', 0.0)
    
    cfg = TYPOLOGY_CONFIG['spectral']
    
    # PERIODIC signals: bypass slope-based classification
    if temporal_pattern == 'PERIODIC':
        if harmonic_noise_ratio > cfg['harmonic']['hnr_min']:
            return 'HARMONIC'
        return 'NARROWBAND'
    
    # Broadband: flat spectrum
    if spectral_flatness > cfg['broadband']['spectral_flatness_min']:
        return 'BROADBAND'
    
    # Red noise: 1/f spectrum
    if spectral_slope < cfg['red_noise']['spectral_slope_max']:
        return 'RED_NOISE'
    
    # Blue noise: rising spectrum
    if spectral_slope > cfg['blue_noise']['spectral_slope_min']:
        return 'BLUE_NOISE'
    
    # Default
    return 'NARROWBAND'


# ============================================================
# FIX 7: Engine Corrections
# ============================================================

def correct_engines(
    engines: List[str],
    temporal_pattern: str,
    spectral: str,
) -> List[str]:
    """
    Adjust engine list based on temporal/spectral classification.
    
    Uses config: engines.*
    """
    engines = list(engines)  # Copy
    
    # Get adjustments for temporal pattern
    adjustments = TYPOLOGY_CONFIG['engines'].get(temporal_pattern.lower(), {})
    
    # Remove inappropriate engines
    for eng in adjustments.get('remove', []):
        if eng == '*':
            return []  # Remove all (e.g., for CONSTANT signals)
        if eng in engines:
            engines.remove(eng)
    
    # Add appropriate engines
    for eng in adjustments.get('add', []):
        if eng not in engines:
            engines.append(eng)
    
    # Spectral-specific additions
    if spectral == 'RED_NOISE' and 'psd_slope' not in engines:
        engines.append('psd_slope')
    
    return engines


# ============================================================
# FIX 8: Visualization Corrections
# ============================================================

def correct_visualizations(
    visualizations: List[str],
    temporal_pattern: str,
) -> List[str]:
    """
    Adjust visualization list based on temporal pattern.
    
    Uses config: visualizations.*
    """
    visualizations = list(visualizations)  # Copy
    
    adjustments = TYPOLOGY_CONFIG['visualizations'].get(temporal_pattern.lower(), {})
    
    for viz in adjustments.get('remove', []):
        if viz in visualizations:
            visualizations.remove(viz)
    
    for viz in adjustments.get('add', []):
        if viz not in visualizations:
            visualizations.append(viz)
    
    return visualizations


# ============================================================
# Main correction function
# ============================================================

def apply_corrections(row: Dict[str, Any], fft_size: int = None) -> Dict[str, Any]:
    """
    Apply all corrections to a typology row.
    
    Args:
        row: Dict with raw measures and current classifications
        fft_size: FFT size used in spectral analysis
        
    Returns:
        Corrected row dict
    """
    if fft_size is None:
        fft_size = get_threshold('artifacts.default_fft_size', 256)
    
    row = dict(row)  # Copy
    
    # Check for first-bin artifact
    dom_freq = row.get('dominant_frequency')
    n_samples = row.get('n_samples', 5000)
    spectral_slope = row.get('spectral_slope', 0.0)
    
    is_artifact = is_first_bin_artifact(dom_freq, n_samples, fft_size, spectral_slope)
    row['dominant_frequency_is_artifact'] = is_artifact
    row['dominant_frequency_corrected'] = None if is_artifact else dom_freq
    
    # Reclassify temporal pattern
    new_temporal = classify_temporal_pattern(row, fft_size)
    row['temporal_pattern'] = new_temporal
    
    # Reclassify spectral
    new_spectral = classify_spectral(row, new_temporal)
    row['spectral'] = new_spectral
    
    # Correct engines if present
    if 'engines' in row:
        row['engines'] = correct_engines(row['engines'], new_temporal, new_spectral)
    
    # Correct visualizations if present
    if 'visualizations' in row:
        row['visualizations'] = correct_visualizations(
            row['visualizations'], new_temporal
        )
    
    return row
