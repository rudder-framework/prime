"""
Level 2 Classification Corrections

Fixes identified during CSTR chemical reactor dataset review (2025-02-02).

Problem Summary:
    6/7 signals misclassified as PERIODIC when they are monotonic trends.
    Root cause: spectral peak detector picks up lowest FFT bin as
    "dominant frequency" on slow-evolving signals with 1/f spectral slope.

Corrections:
    1. is_first_bin_artifact() - detects when dominant_freq = 1/N_fft
    2. is_genuine_periodic() - validates periodicity claim against ACF/turning points
    3. correct_temporal_pattern() - applies corrected classification logic
    4. correct_spectral_class() - fixes HARMONIC label that cascades from fake PERIODIC

These functions operate on the typology_raw row dict (27 raw measures)
and return corrected classification values. They are designed to be
called AFTER compute_raw_measures() and BEFORE writing typology.parquet.

Usage:
    row = compute_raw_measures(signal)  # existing
    row = correct_temporal_pattern(row)  # NEW: apply fix
    row = correct_spectral_class(row)    # NEW: apply fix
"""

from typing import Dict, Any, Optional
import numpy as np


# ============================================================
# FIX 1: Spectral Peak Artifact Detection
# ============================================================

def is_first_bin_artifact(
    dominant_frequency: float,
    n_samples: int,
    fft_size: int = 256,
    spectral_slope: float = 0.0,
    slope_threshold: float = -0.3,
) -> bool:
    """
    Detect if dominant_frequency is just the first FFT bin.

    The first FFT bin has frequency 1/N_fft. When a slow-evolving signal
    concentrates energy at low frequencies (negative spectral slope),
    the FFT peak detector picks this bin as "dominant" — but it's just
    the lowest resolvable frequency, not a real spectral peak.

    Chemical reactor CSTR example:
        All 7 signals reported dominant_freq = 0.00390625 = 1/256.
        spectral_slope ranged from -0.57 to -0.74.
        These are trending signals, not periodic.

    Args:
        dominant_frequency: Reported dominant frequency from raw measures
        n_samples: Total number of samples in signal
        fft_size: FFT window size used (default 256)
        spectral_slope: Power spectrum slope (negative = energy at low freq)
        slope_threshold: Slopes below this indicate 1/f-like spectrum

    Returns:
        True if the dominant frequency is likely an artifact
    """
    if dominant_frequency is None or np.isnan(dominant_frequency):
        return False

    first_bin_freq = 1.0 / fft_size

    # Check if dominant_freq matches first bin (with tolerance)
    freq_is_first_bin = abs(dominant_frequency - first_bin_freq) < first_bin_freq * 0.1

    # Check if spectrum has negative slope (energy concentrated at low freq)
    has_red_spectrum = spectral_slope < slope_threshold

    return freq_is_first_bin and has_red_spectrum


def compute_corrected_dominant_freq(
    dominant_frequency: float,
    n_samples: int,
    spectral_slope: float = 0.0,
    spectral_flatness: float = 1.0,
    fft_size: int = 256,
) -> Optional[float]:
    """
    Return corrected dominant frequency, or None if artifact.

    When the spectral peak is an FFT-bin artifact:
    - Return None (no real dominant frequency exists)
    - Downstream: window recommender should use ACF-based sizing, not period-based

    When spectral_flatness is high (> 0.8), the spectrum is broadband
    and no single frequency dominates — also return None.

    Args:
        dominant_frequency: Raw dominant frequency
        n_samples: Signal length
        spectral_slope: Spectral slope
        spectral_flatness: Spectral flatness (0=peaked, 1=flat)
        fft_size: FFT window size used

    Returns:
        Corrected frequency, or None if no genuine peak
    """
    if dominant_frequency is None or np.isnan(dominant_frequency):
        return None

    # Artifact check
    if is_first_bin_artifact(dominant_frequency, n_samples, fft_size, spectral_slope):
        return None

    # Broadband check
    if spectral_flatness > 0.8:
        return None

    return dominant_frequency


# ============================================================
# FIX 2: Genuine Periodicity Validation
# ============================================================

def is_genuine_periodic(
    dominant_frequency: Optional[float],
    acf_half_life: Optional[float],
    turning_point_ratio: float,
    spectral_peak_snr: float,
    spectral_flatness: float,
    spectral_slope: float,
    hurst: float,
    n_samples: int,
    fft_size: int = 256,
) -> bool:
    """
    Validate whether a signal is genuinely periodic.

    A truly periodic signal must show ALL of these:
    1. A real spectral peak (not first-bin artifact)
    2. ACF that decays and then recovers (oscillation in ACF)
    3. Turning point ratio consistent with oscillation (< 0.95)
    4. Spectral peak SNR above noise floor

    Monotonic trends fail multiple checks:
    - Their spectral peak is at the first bin (artifact)
    - Their ACF never decays (acf_half_life = None)
    - Their turning point ratio is ~0.96 (near maximum for random)
    - Their Hurst exponent is ~1.0 (extreme persistence)

    Chemical CSTR example:
        conc_A: turning_point_ratio=0.959, hurst=1.0, ACF never decays
        → NOT periodic (exponential decay)
        temperature: spectral_flatness=0.985, SNR=2.27
        → NOT periodic (broadband random walk)

    Args:
        dominant_frequency: From raw measures
        acf_half_life: From Level 1 (None if ACF never crossed 0.5)
        turning_point_ratio: Fraction of points that are local extrema
        spectral_peak_snr: Signal-to-noise of dominant peak (dB)
        spectral_flatness: Spectral flatness (0=peaked, 1=flat)
        spectral_slope: Power spectrum slope
        hurst: Hurst exponent
        n_samples: Signal length
        fft_size: FFT window size

    Returns:
        True only if signal shows genuine periodic behavior
    """
    # Gate 1: Must have a real spectral peak (not first-bin artifact)
    if dominant_frequency is None or np.isnan(dominant_frequency):
        return False
    if is_first_bin_artifact(dominant_frequency, n_samples, fft_size, spectral_slope):
        return False

    # Gate 2: Broadband signals are not periodic
    if spectral_flatness > 0.7:
        return False

    # Gate 3: Spectral peak must be genuinely prominent
    # Low SNR means no real peak above noise floor
    if spectral_peak_snr < 6.0:  # ~4x above noise
        return False

    # Gate 4: ACF must show oscillation pattern
    # If ACF never drops below 0.5, signal is monotonic, not oscillating
    if acf_half_life is None:
        return False

    # Gate 5: Hurst ≈ 1.0 signals are deterministic trends, not oscillations
    if hurst is not None and hurst > 0.95:
        return False

    # Gate 6: Turning point ratio near 2/3 is random; near 1.0 is monotonic
    # Periodic signals have turning_point_ratio < 0.9
    # (each cycle has exactly 2 turning points per period)
    if turning_point_ratio > 0.95:
        return False

    return True


# ============================================================
# FIX 3: Corrected Temporal Pattern Classification
# ============================================================

def classify_temporal_pattern(
    row: Dict[str, Any],
    fft_size: int = 256,
) -> str:
    """
    Corrected temporal pattern classification.

    Replaces the decision tree branch that was producing false PERIODIC
    labels on monotonic trending signals.

    Decision tree (corrected):
        1. If is_genuine_periodic() → PERIODIC
        2. If hurst >= 0.99 → TRENDING
           (Hurst ≈ 1.0 is sufficient evidence for trending regardless
            of spectral features, sample entropy, or ACF structure.
            Catches noisy degradation trends like C-MAPSS turbofan sensors.)
        3. If hurst > 0.85 and ACF never decays → TRENDING
        4. If sample_entropy < 0.02 and Hurst > 0.9 → TRENDING
        5. If spectral_flatness > 0.9 and perm_entropy > 0.99 → RANDOM
        6. If n_samples >= 500 and lyapunov_proxy > 0.5 and perm_entropy > 0.95 → CHAOTIC
           (Lyapunov proxy unreliable below ~500 samples; in C-MAPSS with
            n=154-182, 94% of RANDOM signals have lyap > 0.5.)
        7. If turning_point_ratio < 0.7 → QUASI_PERIODIC
        8. Default → STATIONARY

    Args:
        row: Dict with all raw measures
        fft_size: FFT window size used in spectral analysis

    Returns:
        One of: PERIODIC, TRENDING, RANDOM, CHAOTIC, QUASI_PERIODIC, STATIONARY
    """
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

    # 1. Genuine periodic signal
    if is_genuine_periodic(
        dominant_freq, acf_half_life, turning_point_ratio,
        spectral_peak_snr, spectral_flatness, spectral_slope,
        hurst, n_samples, fft_size,
    ):
        return 'PERIODIC'

    # 2. Hurst ≈ 1.0: definitionally trending.
    #    This is the strongest single indicator of trending behavior.
    #    Catches noisy degradation trends (C-MAPSS sensor_07/08/11/12/13/15/20/21)
    #    that have enough noise to fool spectral/entropy gates but are
    #    unmistakably trending by Hurst analysis.
    if hurst is not None and bool(hurst >= 0.99):
        return 'TRENDING'

    # 3. Trending: high persistence + non-decaying ACF
    #    This catches exponential decay (conc_A), accumulation (conc_C),
    #    and rise-then-fall dynamics (conc_B) where acf never decays
    if hurst is not None and bool(hurst > 0.85) and acf_half_life is None:
        return 'TRENDING'

    # 4. Also trending: very low sample entropy with extreme Hurst
    #    (near-deterministic evolution)
    if bool(sample_entropy < 0.02) and hurst is not None and bool(hurst > 0.9):
        return 'TRENDING'

    # 5. Random: flat spectrum + high permutation entropy
    if bool(spectral_flatness > 0.9) and bool(perm_entropy > 0.99):
        return 'RANDOM'

    # 6. Chaotic: positive Lyapunov proxy + high complexity
    #    Minimum sample length guard: Lyapunov proxy unreliable below ~500 samples.
    #    Without this, 94% of RANDOM signals in short series (n<200) get false CHAOTIC.
    if n_samples >= 500 and bool(lyapunov_proxy > 0.5) and bool(perm_entropy > 0.95):
        return 'CHAOTIC'

    # 7. Quasi-periodic: low turning point ratio but not fully periodic
    if bool(turning_point_ratio < 0.7):
        return 'QUASI_PERIODIC'

    # 8. Default
    return 'STATIONARY'


# ============================================================
# FIX 4: Corrected Spectral Classification
# ============================================================

def classify_spectral(
    row: Dict[str, Any],
    temporal_pattern: str,
    fft_size: int = 256,
) -> str:
    """
    Corrected spectral classification.

    The HARMONIC label was cascading from the false PERIODIC classification.
    Now uses the corrected temporal_pattern to gate HARMONIC.

    Decision tree:
        1. If temporal_pattern == PERIODIC and HNR > 3 → HARMONIC
        2. If temporal_pattern == PERIODIC (lower HNR) → NARROWBAND
           (harmonic series with decreasing amplitude creates negative
            spectral slope that looks like 1/f but is discrete peaks)
        3. If spectral_flatness > 0.8 → BROADBAND
        4. If spectral_slope < -0.5 → RED_NOISE (1/f spectrum)
        5. If spectral_slope > 0.2 → BLUE_NOISE
        6. Default → NARROWBAND

    Bearing vibration example:
        f0 + 2f0 + 3f0 harmonics with SNR > 30 dB create spectral_slope ≈ -0.6
        but this is NOT 1/f noise — it's a harmonic series with rolloff.
        Slope check must be skipped for PERIODIC signals.

    Chemical CSTR example:
        Monotonic trends with spectral_slope ≈ -0.6 have continuous 1/f spectrum.
        These are correctly classified as RED_NOISE (temporal = TRENDING).

    Args:
        row: Dict with raw measures
        temporal_pattern: Corrected temporal pattern from classify_temporal_pattern()
        fft_size: FFT window size

    Returns:
        One of: HARMONIC, BROADBAND, RED_NOISE, BLUE_NOISE, NARROWBAND
    """
    spectral_flatness = row.get('spectral_flatness', 0.5)
    spectral_slope = row.get('spectral_slope', 0.0)
    harmonic_noise_ratio = row.get('harmonic_noise_ratio', 0.0)

    # PERIODIC signals: spectral class is determined by harmonic content,
    # NOT by spectral slope. Harmonic series with decreasing amplitude
    # creates negative slope that mimics 1/f noise but is actually
    # discrete narrowband peaks.
    if temporal_pattern == 'PERIODIC':
        if harmonic_noise_ratio > 3.0:
            return 'HARMONIC'
        return 'NARROWBAND'

    # Broadband (flat spectrum)
    if spectral_flatness > 0.8:
        return 'BROADBAND'

    # Red noise / 1/f spectrum (energy at low frequencies)
    # Applies only to non-periodic signals (trending, random, etc.)
    if spectral_slope < -0.5:
        return 'RED_NOISE'

    # Blue noise (energy at high frequencies)
    if spectral_slope > 0.2:
        return 'BLUE_NOISE'

    return 'NARROWBAND'


# ============================================================
# FIX 5: Corrected Visualization Selection
# ============================================================

def correct_visualizations(
    visualizations: list,
    temporal_pattern: str,
    spectral_class: str,
) -> list:
    """
    Remove inappropriate visualizations from corrected classification.

    When PERIODIC was wrong, waterfall and recurrence were selected.
    For TRENDING signals, replace with appropriate choices.

    Args:
        visualizations: Original visualization list
        temporal_pattern: Corrected temporal pattern
        spectral_class: Corrected spectral classification

    Returns:
        Corrected visualization list
    """
    viz = set(visualizations)

    # Remove spectral viz for non-periodic signals
    if temporal_pattern != 'PERIODIC':
        viz.discard('waterfall')
        viz.discard('recurrence')

    # Add appropriate viz for trending
    if temporal_pattern == 'TRENDING':
        viz.add('trend_overlay')
        viz.add('segment_comparison')

    # Add appropriate viz for red noise
    if spectral_class == 'RED_NOISE':
        viz.add('psd_slope')  # Show the 1/f characteristic

    return sorted(viz)


# ============================================================
# FIX 6: Corrected Engine Selection
# ============================================================

def correct_engines(
    engines: list,
    temporal_pattern: str,
    spectral_class: str,
) -> list:
    """
    Remove inappropriate engines from corrected classification.

    PERIODIC → TRENDING means:
    - Remove: harmonics_ratio, band_ratios, thd, frequency_bands
    - Add: hurst, rate_of_change_ratio, trend_r2

    Args:
        engines: Original engine list
        temporal_pattern: Corrected temporal pattern
        spectral_class: Corrected spectral classification

    Returns:
        Corrected engine list
    """
    eng = set(engines)

    # Remove harmonic engines for non-periodic signals
    if temporal_pattern != 'PERIODIC':
        eng.discard('harmonics_ratio')
        eng.discard('band_ratios')
        eng.discard('thd')
        eng.discard('frequency_bands')

    # Add trending engines
    if temporal_pattern == 'TRENDING':
        eng.add('hurst')
        eng.add('rate_of_change_ratio')
        eng.add('trend_r2')
        eng.add('detrend_std')

    return sorted(eng)


# ============================================================
# MASTER CORRECTION FUNCTION
# ============================================================

def apply_corrections(row: Dict[str, Any], fft_size: int = 256) -> Dict[str, Any]:
    """
    Apply all classification corrections to a typology row.

    Call this after compute_raw_measures() and the original classify_signal()
    to fix known issues. Returns a new dict with corrected fields.

    Args:
        row: Complete typology row (raw measures + original classifications)
        fft_size: FFT window size used in spectral analysis

    Returns:
        Row with corrected classification fields
    """
    corrected = dict(row)

    # Fix dominant_frequency artifact
    corrected_freq = compute_corrected_dominant_freq(
        row.get('dominant_frequency'),
        row.get('n_samples', 5000),
        row.get('spectral_slope', 0.0),
        row.get('spectral_flatness', 0.5),
        fft_size,
    )
    corrected['dominant_frequency_corrected'] = corrected_freq
    corrected['dominant_frequency_is_artifact'] = (corrected_freq is None and
                                                    row.get('dominant_frequency') is not None)

    # Fix temporal pattern
    corrected_temporal = classify_temporal_pattern(row, fft_size)
    corrected['temporal_pattern'] = corrected_temporal

    # Fix spectral classification (depends on corrected temporal)
    corrected_spectral = classify_spectral(row, corrected_temporal, fft_size)
    corrected['spectral'] = corrected_spectral

    # Fix visualizations
    if 'visualizations' in row and isinstance(row['visualizations'], list):
        corrected['visualizations'] = correct_visualizations(
            row['visualizations'], corrected_temporal, corrected_spectral,
        )

    # Fix engine selection
    if 'engines' in row and isinstance(row['engines'], list):
        corrected['engines'] = correct_engines(
            row['engines'], corrected_temporal, corrected_spectral,
        )

    return corrected
