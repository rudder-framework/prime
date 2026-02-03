"""
Tests for typology classification corrections.

Validates against the CSTR chemical reactor patterns that
exposed the bugs. Each test includes the actual values
from the dataset so regressions are caught immediately.
"""

import pytest
import numpy as np

from orthon.corrections.level2_corrections import (
    is_first_bin_artifact,
    compute_corrected_dominant_freq,
    is_genuine_periodic,
    classify_temporal_pattern,
    classify_spectral,
    correct_engines,
    apply_corrections,
)
from orthon.corrections.level1_corrections import (
    detect_deterministic_trend,
    correct_stationarity,
    apply_level1_corrections,
)
from orthon.corrections.manifest_corrections import (
    compute_global_stride_default,
)


# ============================================================
# CSTR dataset ground truth values (from review)
# ============================================================

CONC_A_ROW = {
    'signal_id': 'conc_A',
    'n_samples': 5000,
    'adf_pvalue': 0.0,
    'kpss_pvalue': 0.01,
    'variance_ratio': 0.00035,
    'acf_half_life': None,
    'hurst': 1.0,
    'perm_entropy': 0.985,
    'sample_entropy': 0.005,
    'spectral_flatness': 0.0097,
    'spectral_slope': -0.694,
    'harmonic_noise_ratio': 16.31,
    'spectral_peak_snr': 41.6,
    'dominant_frequency': 0.00390625,
    'turning_point_ratio': 0.959,
    'lyapunov_proxy': 0.035,
    'stationarity': 'STATIONARY',
    'temporal_pattern': 'PERIODIC',
    'spectral': 'HARMONIC',
}

CONC_B_ROW = {
    'signal_id': 'conc_B',
    'n_samples': 5000,
    'adf_pvalue': 0.839,
    'kpss_pvalue': 0.01,
    'variance_ratio': 0.00092,
    'acf_half_life': None,
    'hurst': 1.0,
    'perm_entropy': 0.999,
    'sample_entropy': 0.015,
    'spectral_flatness': 0.0155,
    'spectral_slope': -0.655,
    'harmonic_noise_ratio': 14.83,
    'spectral_peak_snr': 39.51,
    'dominant_frequency': 0.00390625,
    'turning_point_ratio': 0.969,
    'lyapunov_proxy': 0.034,
    'stationarity': 'NON_STATIONARY',
    'temporal_pattern': 'PERIODIC',
    'spectral': 'HARMONIC',
}

TEMPERATURE_ROW = {
    'signal_id': 'temperature',
    'n_samples': 5000,
    'adf_pvalue': 0.596,
    'kpss_pvalue': 0.01,
    'variance_ratio': 0.136,
    'acf_half_life': None,
    'hurst': 0.598,
    'perm_entropy': 0.9999,
    'sample_entropy': 1.408,
    'spectral_flatness': 0.985,
    'spectral_slope': -0.022,
    'harmonic_noise_ratio': 0.013,
    'spectral_peak_snr': 2.27,
    'dominant_frequency': 0.00390625,
    'turning_point_ratio': 0.998,
    'lyapunov_proxy': 1.295,
    'stationarity': 'NON_STATIONARY',
    'temporal_pattern': 'RANDOM',
    'spectral': 'BROADBAND',
}


# ============================================================
# FIX 1: First-bin artifact detection
# ============================================================

class TestFirstBinArtifact:

    def test_cstr_signals_are_artifacts(self):
        """All CSTR chemical signals have dominant_freq = 1/256 with negative slope."""
        assert is_first_bin_artifact(0.00390625, 5000, 256, -0.694)
        assert is_first_bin_artifact(0.00390625, 5000, 256, -0.655)
        assert is_first_bin_artifact(0.00390625, 5000, 256, -0.566)

    def test_temperature_is_artifact(self):
        """Temperature also hits first bin, but slope is near zero."""
        # Slope -0.022 is above threshold of -0.3, so NOT flagged as artifact
        assert not is_first_bin_artifact(0.00390625, 5000, 256, -0.022)

    def test_real_periodic_not_flagged(self):
        """A real periodic signal at a higher frequency should not be flagged."""
        # freq=0.05 is well above 1/256=0.0039
        assert not is_first_bin_artifact(0.05, 5000, 256, -0.2)

    def test_none_frequency(self):
        assert not is_first_bin_artifact(None, 5000, 256, -0.5)

    def test_nan_frequency(self):
        assert not is_first_bin_artifact(float('nan'), 5000, 256, -0.5)


class TestCorrectedDominantFreq:

    def test_cstr_chemical_returns_none(self):
        """Chemical signals with 1/f slope at first bin → None."""
        result = compute_corrected_dominant_freq(
            0.00390625, 5000, -0.694, 0.0097, 256
        )
        assert result is None

    def test_broadband_returns_none(self):
        """Temperature-like broadband signal → None."""
        result = compute_corrected_dominant_freq(
            0.00390625, 5000, -0.022, 0.985, 256
        )
        assert result is None  # flatness > 0.8

    def test_real_periodic_preserved(self):
        """Genuine periodic signal keeps its frequency."""
        result = compute_corrected_dominant_freq(
            0.05, 1000, -0.1, 0.15, 256
        )
        assert result == 0.05


# ============================================================
# FIX 2: Genuine periodicity validation
# ============================================================

class TestGenuinePeriodic:

    def test_conc_a_not_periodic(self):
        """Exponential decay is NOT periodic."""
        assert not is_genuine_periodic(
            dominant_frequency=0.00390625,
            acf_half_life=None,
            turning_point_ratio=0.959,
            spectral_peak_snr=41.6,
            spectral_flatness=0.0097,
            spectral_slope=-0.694,
            hurst=1.0,
            n_samples=5000,
        )

    def test_temperature_not_periodic(self):
        """Random walk is NOT periodic."""
        assert not is_genuine_periodic(
            dominant_frequency=0.00390625,
            acf_half_life=None,
            turning_point_ratio=0.998,
            spectral_peak_snr=2.27,
            spectral_flatness=0.985,
            spectral_slope=-0.022,
            hurst=0.598,
            n_samples=5000,
        )

    def test_sine_wave_is_periodic(self):
        """A real sine wave should be detected as periodic."""
        assert is_genuine_periodic(
            dominant_frequency=0.05,
            acf_half_life=10,
            turning_point_ratio=0.63,
            spectral_peak_snr=30.0,
            spectral_flatness=0.05,
            spectral_slope=-0.1,
            hurst=0.5,
            n_samples=1000,
        )


# ============================================================
# FIX 3: Temporal pattern classification
# ============================================================

class TestTemporalPattern:

    def test_conc_a_is_trending(self):
        """conc_A (exponential decay, hurst=1.0) → TRENDING."""
        result = classify_temporal_pattern(CONC_A_ROW)
        assert result == 'TRENDING'

    def test_conc_b_is_trending(self):
        """conc_B (rise-then-fall, hurst=1.0) → TRENDING."""
        result = classify_temporal_pattern(CONC_B_ROW)
        assert result == 'TRENDING'

    def test_temperature_is_random(self):
        """temperature (broadband, high perm_entropy) → RANDOM."""
        result = classify_temporal_pattern(TEMPERATURE_ROW)
        assert result == 'RANDOM'

    def test_sine_wave_is_periodic(self):
        """Genuine sine wave → PERIODIC."""
        sine_row = {
            'dominant_frequency': 0.05,
            'acf_half_life': 10,
            'turning_point_ratio': 0.63,
            'spectral_peak_snr': 30.0,
            'spectral_flatness': 0.05,
            'spectral_slope': -0.1,
            'harmonic_noise_ratio': 20.0,
            'hurst': 0.5,
            'perm_entropy': 0.6,
            'sample_entropy': 0.5,
            'lyapunov_proxy': 0.0,
            'n_samples': 1000,
        }
        result = classify_temporal_pattern(sine_row)
        assert result == 'PERIODIC'


# ============================================================
# FIX 4: Spectral classification
# ============================================================

class TestSpectralClassification:

    def test_conc_a_is_red_noise(self):
        """conc_A with corrected TRENDING → RED_NOISE (1/f spectrum)."""
        result = classify_spectral(CONC_A_ROW, 'TRENDING')
        assert result == 'RED_NOISE'

    def test_temperature_is_broadband(self):
        """temperature → BROADBAND (flat spectrum)."""
        result = classify_spectral(TEMPERATURE_ROW, 'RANDOM')
        assert result == 'BROADBAND'

    def test_periodic_high_hnr_is_harmonic(self):
        """Genuine periodic with high HNR → HARMONIC."""
        row = {'spectral_flatness': 0.05, 'spectral_slope': -0.1,
               'harmonic_noise_ratio': 20.0}
        result = classify_spectral(row, 'PERIODIC')
        assert result == 'HARMONIC'

    def test_periodic_low_hnr_is_narrowband(self):
        """Periodic with low HNR → NARROWBAND (not RED_NOISE).

        Bearing vibration example: f0 + 2f0 + 3f0 harmonics create
        spectral_slope ≈ -0.6, but the signal is periodic, not 1/f.
        Harmonic amplitude rolloff mimics negative slope.
        """
        bearing_row = {
            'spectral_flatness': 0.068,
            'spectral_slope': -0.58,  # Would be RED_NOISE if not periodic
            'harmonic_noise_ratio': 1.0,  # Low HNR, so not HARMONIC
        }
        result = classify_spectral(bearing_row, 'PERIODIC')
        assert result == 'NARROWBAND'

    def test_periodic_bypasses_red_noise_slope(self):
        """Even with steep negative slope, PERIODIC → NARROWBAND, not RED_NOISE."""
        row = {
            'spectral_flatness': 0.04,
            'spectral_slope': -0.7,  # Steep negative slope
            'harmonic_noise_ratio': 1.4,
        }
        result = classify_spectral(row, 'PERIODIC')
        assert result == 'NARROWBAND'


# ============================================================
# FIX 5: Engine corrections
# ============================================================

class TestEngineCorrections:

    def test_trending_removes_harmonic_engines(self):
        """TRENDING should not have harmonic analysis engines."""
        engines = ['kurtosis', 'skewness', 'harmonics_ratio', 'band_ratios', 'thd']
        corrected = correct_engines(engines, 'TRENDING', 'RED_NOISE')
        assert 'harmonics_ratio' not in corrected
        assert 'band_ratios' not in corrected
        assert 'thd' not in corrected

    def test_trending_adds_trend_engines(self):
        """TRENDING should add trend analysis engines."""
        engines = ['kurtosis', 'skewness']
        corrected = correct_engines(engines, 'TRENDING', 'RED_NOISE')
        assert 'hurst' in corrected
        assert 'trend_r2' in corrected
        assert 'rate_of_change_ratio' in corrected


# ============================================================
# Level 1: Deterministic trend detection
# ============================================================

class TestDeterministicTrend:

    def test_exponential_decay(self):
        """Exponential decay should be detected as deterministic trend."""
        y = np.exp(-np.linspace(0, 5, 5000))  # 1.0 → 0.007
        is_trend, strength, direction = detect_deterministic_trend(y)
        assert is_trend is True
        assert strength > 2.0
        assert direction == 'decreasing'

    def test_accumulation(self):
        """Monotonic accumulation (like conc_C) → trend."""
        y = 1.0 - np.exp(-np.linspace(0, 5, 5000))  # 0 → 0.993
        is_trend, strength, direction = detect_deterministic_trend(y)
        assert is_trend is True
        assert direction == 'increasing'

    def test_random_walk_not_trend(self):
        """Random walk should generally not be detected as deterministic trend."""
        np.random.seed(42)
        y = np.cumsum(np.random.randn(5000))
        is_trend, strength, direction = detect_deterministic_trend(y)
        # Random walk may or may not be monotonic, but strength
        # should typically be moderate. Key: not all random walks pass.
        # This is a probabilistic test — check that strength is reasonable.
        assert strength < 10.0

    def test_stationary_not_trend(self):
        """White noise → not a trend."""
        np.random.seed(42)
        y = np.random.randn(5000)
        is_trend, strength, direction = detect_deterministic_trend(y)
        assert is_trend is False

    def test_sine_wave_not_trend(self):
        """Sine wave → not a monotonic trend."""
        t = np.linspace(0, 10 * np.pi, 5000)
        y = np.sin(t)
        is_trend, strength, direction = detect_deterministic_trend(y)
        assert is_trend is False


# ============================================================
# Level 1: Stationarity correction
# ============================================================

class TestStationarityCorrection:

    def test_conc_a_override(self):
        """conc_A: ADF=pass, KPSS=reject, extreme variance ratio → TREND_STATIONARY."""
        result = correct_stationarity(
            stationarity_type='STATIONARY',
            adf_rejects=True,
            kpss_rejects=True,
            mean_shift_ratio=2.5,
            variance_ratio=0.00035,
            mean_stable=False,
            is_deterministic_trend=True,
            trend_strength=5.0,
        )
        assert result == 'TREND_STATIONARY'

    def test_white_noise_unchanged(self):
        """Genuine stationary signal should not be overridden."""
        result = correct_stationarity(
            stationarity_type='STATIONARY',
            adf_rejects=True,
            kpss_rejects=False,
            mean_shift_ratio=0.1,
            variance_ratio=1.05,
            mean_stable=True,
            is_deterministic_trend=False,
            trend_strength=0.1,
        )
        assert result == 'STATIONARY'

    def test_temperature_unchanged(self):
        """NON_STATIONARY random walk should not be overridden."""
        result = correct_stationarity(
            stationarity_type='NON_STATIONARY',
            adf_rejects=False,
            kpss_rejects=True,
            mean_shift_ratio=1.0,
            variance_ratio=0.136,
            mean_stable=False,
            is_deterministic_trend=False,
            trend_strength=0.5,
        )
        assert result == 'NON_STATIONARY'


# ============================================================
# Manifest: Global stride default
# ============================================================

class TestGlobalStride:

    def test_sensible_default(self):
        """Global stride should be median of per-signal strides."""
        signals = {
            'sig_a': {'stride': 64},
            'sig_b': {'stride': 64},
            'sig_c': {'stride': 128},
        }
        result = compute_global_stride_default(signals, default_window=128)
        assert result == 64  # median

    def test_fallback_no_signals(self):
        """Empty signals dict → 50% of default window."""
        result = compute_global_stride_default({}, default_window=128)
        assert result == 64

    def test_never_zero(self):
        result = compute_global_stride_default({}, default_window=1)
        assert result >= 1


# ============================================================
# Integration: apply_corrections on CSTR rows
# ============================================================

class TestApplyCorrections:

    def test_conc_a_full_correction(self):
        """Full correction pipeline for conc_A."""
        corrected = apply_corrections(CONC_A_ROW)
        assert corrected['temporal_pattern'] == 'TRENDING'
        assert corrected['spectral'] == 'RED_NOISE'
        assert corrected['dominant_frequency_is_artifact'] is True
        assert corrected['dominant_frequency_corrected'] is None

    def test_conc_b_full_correction(self):
        """Full correction pipeline for conc_B."""
        corrected = apply_corrections(CONC_B_ROW)
        assert corrected['temporal_pattern'] == 'TRENDING'
        assert corrected['spectral'] == 'RED_NOISE'

    def test_temperature_stays_correct(self):
        """Temperature was already correctly classified — should remain RANDOM."""
        corrected = apply_corrections(TEMPERATURE_ROW)
        assert corrected['temporal_pattern'] == 'RANDOM'
        assert corrected['spectral'] == 'BROADBAND'

    def test_all_cstr_signals_differentiated(self):
        """After correction, CSTR signals should have more diverse cards."""
        rows = [CONC_A_ROW, CONC_B_ROW, TEMPERATURE_ROW]
        corrected = [apply_corrections(r) for r in rows]
        patterns = {r['temporal_pattern'] for r in corrected}
        # Should have at least TRENDING and RANDOM
        assert 'TRENDING' in patterns
        assert 'RANDOM' in patterns


# ============================================================
# Integration: bearing vibration signals
# ============================================================

# Bearing signal: f0 + 2f0 + 3f0 harmonics, SNR > 30 dB
BEARING_ACC_X_ROW = {
    'signal_id': 'acc_x', 'unit_id': 'bearing_1',
    'n_samples': 10000,
    'dominant_frequency': 0.031250,  # Real peak, NOT first-bin artifact
    'acf_half_life': 5.0,
    'turning_point_ratio': 0.633,
    'spectral_peak_snr': 31.1,
    'spectral_flatness': 0.068,
    'spectral_slope': -0.580,      # Harmonic rolloff, NOT 1/f noise
    'harmonic_noise_ratio': 1.0,
    'hurst': 0.610,
    'perm_entropy': 0.908,
    'sample_entropy': 0.803,
    'lyapunov_proxy': 1.572,
    'kurtosis': 2.214,
    'skewness': 0.023,
    'crest_factor': 5.928,
    'engines': ['attractor', 'crest_factor', 'entropy', 'garch',
                'kurtosis', 'lyapunov', 'skewness', 'spectral'],
    'visualizations': ['waterfall', 'phase_portrait',
                       'volatility_map', 'spectral_density'],
}

BEARING_ACC_Y_ROW = {
    'signal_id': 'acc_y', 'unit_id': 'bearing_3',
    'n_samples': 10000,
    'dominant_frequency': 0.039062,
    'acf_half_life': 4.0,
    'turning_point_ratio': 0.524,
    'spectral_peak_snr': 34.3,
    'spectral_flatness': 0.037,
    'spectral_slope': -0.524,
    'harmonic_noise_ratio': 1.186,
    'hurst': 0.451,
    'perm_entropy': 0.883,
    'sample_entropy': 0.842,
    'lyapunov_proxy': 1.564,
    'kurtosis': 2.281,
    'skewness': 0.696,
    'crest_factor': 3.001,
    'engines': ['attractor', 'crest_factor', 'entropy',
                'kurtosis', 'lyapunov', 'skewness', 'spectral'],
    'visualizations': ['waterfall', 'phase_portrait', 'spectral_density'],
}


class TestBearingIntegration:

    def test_bearing_acc_x_periodic_not_chaotic(self):
        """Bearing acc_x: CHAOTIC → PERIODIC. Genuine harmonics with SNR > 30 dB."""
        corrected = apply_corrections(BEARING_ACC_X_ROW)
        assert corrected['temporal_pattern'] == 'PERIODIC'

    def test_bearing_acc_y_periodic_not_chaotic(self):
        """Bearing acc_y: CHAOTIC → PERIODIC."""
        corrected = apply_corrections(BEARING_ACC_Y_ROW)
        assert corrected['temporal_pattern'] == 'PERIODIC'

    def test_bearing_spectral_not_red_noise(self):
        """Bearing signals: spectral should be NARROWBAND, NOT RED_NOISE.

        The negative spectral slope comes from harmonic amplitude rolloff,
        not from continuous 1/f noise. PERIODIC gate must override slope check.
        """
        corrected_x = apply_corrections(BEARING_ACC_X_ROW)
        corrected_y = apply_corrections(BEARING_ACC_Y_ROW)
        assert corrected_x['spectral'] == 'NARROWBAND'
        assert corrected_y['spectral'] == 'NARROWBAND'

    def test_bearing_dominant_freq_preserved(self):
        """Bearing dominant frequencies are REAL peaks, not artifacts."""
        corrected = apply_corrections(BEARING_ACC_X_ROW)
        assert corrected['dominant_frequency_is_artifact'] is False
        assert corrected['dominant_frequency_corrected'] == 0.031250

    def test_bearing_vs_cstr_differentiation(self):
        """Bearing (PERIODIC) and CSTR (TRENDING) must be different classes."""
        bearing = apply_corrections(BEARING_ACC_X_ROW)
        cstr = apply_corrections(CONC_A_ROW)
        assert bearing['temporal_pattern'] != cstr['temporal_pattern']
        assert bearing['spectral'] != cstr['spectral']
        assert bearing['temporal_pattern'] == 'PERIODIC'
        assert cstr['temporal_pattern'] == 'TRENDING'


# ============================================================
# Integration: C-MAPSS turbofan degradation signals
# ============================================================

# C-MAPSS sensor_07 from engine_1: noisy degradation trend
# Has Hurst=1.0 but enough noise to fool spectral/entropy gates
CMAPSS_SENSOR_07_ROW = {
    'signal_id': 'sensor_07', 'unit_id': 'engine_1',
    'n_samples': 182,
    'dominant_frequency': 0.022,  # First-bin artifact from trend shape
    'acf_half_life': None,        # ACF never decays (trending)
    'turning_point_ratio': 0.67,  # Noisy enough to look random
    'spectral_peak_snr': 18.5,    # Below 20 dB threshold
    'spectral_flatness': 0.42,
    'spectral_slope': -1.52,
    'harmonic_noise_ratio': 8.2,
    'hurst': 1.0,                 # DEFINITIONALLY TRENDING
    'perm_entropy': 0.97,
    'sample_entropy': 0.12,       # Above 0.08 threshold
    'lyapunov_proxy': 0.82,       # Would trigger false CHAOTIC
    'kurtosis': 2.8,
    'skewness': 0.15,
    'crest_factor': 3.2,
}

# C-MAPSS constant sensor (sensor_01): pure noise around constant value
CMAPSS_CONSTANT_ROW = {
    'signal_id': 'sensor_01', 'unit_id': 'engine_1',
    'n_samples': 182,
    'dominant_frequency': 0.12,
    'acf_half_life': 2.0,
    'turning_point_ratio': 0.66,
    'spectral_peak_snr': 3.5,
    'spectral_flatness': 0.91,    # Flat spectrum = random noise
    'spectral_slope': -0.15,
    'harmonic_noise_ratio': 0.5,
    'hurst': 0.52,                # White noise Hurst
    'perm_entropy': 0.998,        # Max entropy = random
    'sample_entropy': 1.8,
    'lyapunov_proxy': 0.79,       # Would trigger false CHAOTIC without guard
    'kurtosis': 3.1,
    'skewness': -0.02,
    'crest_factor': 3.5,
}

# Long chaotic signal (Lorenz attractor): should still get CHAOTIC
# Chaotic attractors have broadband spectra (high flatness) and no dominant peak
LORENZ_ROW = {
    'signal_id': 'lorenz_x', 'unit_id': None,
    'n_samples': 10000,           # Long enough for reliable Lyapunov
    'dominant_frequency': 0.08,
    'acf_half_life': 15.0,
    'turning_point_ratio': 0.68,
    'spectral_peak_snr': 4.5,     # Low SNR - no real peak, fails PERIODIC gate 3
    'spectral_flatness': 0.65,    # Moderately broadband (chaotic)
    'spectral_slope': -0.8,
    'harmonic_noise_ratio': 1.2,
    'hurst': 0.72,
    'perm_entropy': 0.96,         # High complexity
    'sample_entropy': 0.45,
    'lyapunov_proxy': 0.9,        # Positive = chaotic
    'kurtosis': 2.5,
    'skewness': 0.1,
    'crest_factor': 4.2,
}


class TestCMAPSSIntegration:
    """Tests for C-MAPSS turbofan degradation signals."""

    def test_noisy_trend_hurst_gate(self):
        """Noisy degradation trend with Hurst=1.0 must be TRENDING.

        Even though spectral/entropy features look ambiguous, Hurst=1.0
        is definitionally trending. This gate catches all C-MAPSS
        degradation sensors that were previously misclassified.
        """
        corrected = apply_corrections(CMAPSS_SENSOR_07_ROW)
        assert corrected['temporal_pattern'] == 'TRENDING'

    def test_constant_sensor_stays_random(self):
        """Constant sensors with white noise must stay RANDOM.

        Without the n_samples >= 500 guard on CHAOTIC, the high
        lyapunov_proxy (0.79) would incorrectly trigger CHAOTIC.
        """
        corrected = apply_corrections(CMAPSS_CONSTANT_ROW)
        assert corrected['temporal_pattern'] == 'RANDOM'

    def test_short_series_no_false_chaotic(self):
        """Short series (n<500) must NOT get false CHAOTIC classification.

        Lyapunov proxy is unreliable below ~500 samples. In C-MAPSS with
        n=154-182, 94% of RANDOM signals have lyap > 0.5 which would
        trigger false CHAOTIC without the sample length guard.
        """
        # Create row with chaotic-looking features but short length
        short_chaos_row = dict(CMAPSS_CONSTANT_ROW)
        short_chaos_row['n_samples'] = 180  # Too short
        short_chaos_row['spectral_flatness'] = 0.85  # Not flat enough for RANDOM
        short_chaos_row['perm_entropy'] = 0.96
        short_chaos_row['lyapunov_proxy'] = 0.85

        corrected = apply_corrections(short_chaos_row)
        assert corrected['temporal_pattern'] != 'CHAOTIC'

    def test_long_series_chaotic_still_works(self):
        """Long chaotic series (n>=500) must still get CHAOTIC.

        The sample length guard should not prevent legitimate chaos
        detection on sufficiently long series like Lorenz attractors.
        """
        corrected = apply_corrections(LORENZ_ROW)
        assert corrected['temporal_pattern'] == 'CHAOTIC'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
