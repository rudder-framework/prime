"""
Test: Window Recommender

Validates window selection logic against benchmark signals with known
timescales. Each test checks that the recommender picks the right rule
and produces a sensible window size.

Run:
    python -m pytest tests/test_window_recommender.py -v
    python tests/test_window_recommender.py  # standalone
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orthon.manifest.window_recommender import (
    recommend_window,
    compute_stride,
    recommend_stride,
    eigenvalue_window_check,
    MIN_WINDOW,
    MAX_WINDOW,
    DEFAULT_WINDOW,
    CYCLES_TO_CAPTURE,
    ACF_MULTIPLIER,
)


# ============================================================
# BENCHMARK TYPOLOGY ROWS
# ============================================================
# Each row simulates what Level 1 + Level 2 typology would produce.
# The raw measures (acf_half_life, seasonal_period, etc.) are the
# key inputs; classifications are secondary context.
# ============================================================

SIGNALS = {

    # --- Pure sine wave: period = 50 samples ---
    'sine_50': {
        'signal_id': 'sine_50',
        'continuity': 'CONTINUOUS',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'PERIODIC',
        'memory': 'SHORT_MEMORY',
        'seasonal_period': 50.0,    # 50 samples per cycle
        'dominant_freq': 0.02,      # 1/50
        'acf_half_life': 12.0,
        'acf_decayed': True,
        'n_samples': 10000,
    },

    # --- Sine with short period: 10 samples ---
    'sine_10': {
        'signal_id': 'sine_10',
        'continuity': 'CONTINUOUS',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'PERIODIC',
        'memory': 'SHORT_MEMORY',
        'seasonal_period': 10.0,
        'dominant_freq': 0.1,
        'acf_half_life': 3.0,
        'acf_decayed': True,
        'n_samples': 5000,
    },

    # --- Sine with long period: 500 samples ---
    'sine_500': {
        'signal_id': 'sine_500',
        'continuity': 'CONTINUOUS',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'PERIODIC',
        'memory': 'SHORT_MEMORY',
        'seasonal_period': 500.0,
        'dominant_freq': 0.002,
        'acf_half_life': 125.0,
        'acf_decayed': True,
        'n_samples': 100000,
    },

    # --- White noise: no period, fast ACF decay ---
    'white_noise': {
        'signal_id': 'white_noise',
        'continuity': 'CONTINUOUS',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'RANDOM',
        'memory': 'SHORT_MEMORY',
        'seasonal_period': None,
        'dominant_freq': None,
        'acf_half_life': 1.0,
        'acf_decay_lag': 1,
        'acf_decayed': True,
        'n_samples': 10000,
    },

    # --- Random walk: non-stationary, long memory ---
    'random_walk': {
        'signal_id': 'random_walk',
        'continuity': 'CONTINUOUS',
        'stationarity': 'NON_STATIONARY',
        'temporal_pattern': 'TRENDING',
        'memory': 'LONG_MEMORY',
        'seasonal_period': None,
        'dominant_freq': None,
        'acf_half_life': None,      # ACF never decayed
        'acf_decay_lag': 200,
        'acf_decayed': False,
        'n_samples': 5000,
    },

    # --- AR(1) with phi=0.95: slow ACF decay ---
    'ar1_slow': {
        'signal_id': 'ar1_slow',
        'continuity': 'CONTINUOUS',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'MEAN_REVERTING',
        'memory': 'LONG_MEMORY',
        'seasonal_period': None,
        'dominant_freq': None,
        'acf_half_life': 14.0,     # ln(2)/ln(0.95) ≈ 13.5
        'acf_decay_lag': 20,
        'acf_decayed': True,
        'n_samples': 5000,
    },

    # --- Lorenz attractor: chaotic, broadband ---
    'lorenz': {
        'signal_id': 'lorenz',
        'continuity': 'CONTINUOUS',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'CHAOTIC',
        'memory': 'SHORT_MEMORY',
        'seasonal_period': None,
        'dominant_freq': None,
        'acf_half_life': 8.0,
        'acf_decay_lag': 5,
        'acf_decayed': True,
        'n_samples': 20000,
    },

    # --- Degrading turbofan: non-stationary trend ---
    'turbofan': {
        'signal_id': 'turbofan',
        'continuity': 'CONTINUOUS',
        'stationarity': 'NON_STATIONARY',
        'temporal_pattern': 'TRENDING',
        'memory': 'LONG_MEMORY',
        'seasonal_period': None,
        'dominant_freq': None,
        'acf_half_life': None,
        'acf_decay_lag': 150,
        'acf_decayed': False,
        'n_samples': 300,           # short lifecycle
    },

    # --- Bearing vibration: periodic + impulsive ---
    'bearing': {
        'signal_id': 'bearing',
        'continuity': 'CONTINUOUS',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'PERIODIC',
        'memory': 'SHORT_MEMORY',
        'seasonal_period': 120.0,   # ball pass frequency period
        'dominant_freq': 0.00833,   # 1/120
        'acf_half_life': 30.0,
        'acf_decayed': True,
        'n_samples': 100000,
    },

    # --- Constant sensor ---
    'constant': {
        'signal_id': 'constant',
        'continuity': 'CONSTANT',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'CONSTANT',
        'memory': None,
        'seasonal_period': None,
        'dominant_freq': None,
        'acf_half_life': None,
        'acf_decayed': None,
        'n_samples': 1000,
    },

    # --- Very short signal: only 100 samples ---
    'short_signal': {
        'signal_id': 'short_signal',
        'continuity': 'CONTINUOUS',
        'stationarity': 'NON_STATIONARY',
        'temporal_pattern': 'TRENDING',
        'memory': 'SHORT_MEMORY',
        'seasonal_period': None,
        'dominant_freq': None,
        'acf_half_life': 5.0,
        'acf_decay_lag': 8,
        'acf_decayed': True,
        'n_samples': 100,
    },

    # --- Quasi-periodic with weak period ---
    'quasi_periodic': {
        'signal_id': 'quasi_periodic',
        'continuity': 'CONTINUOUS',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'QUASI_PERIODIC',
        'memory': 'SHORT_MEMORY',
        'seasonal_period': 75.0,
        'dominant_freq': 0.0133,
        'acf_half_life': 20.0,
        'acf_decayed': True,
        'n_samples': 50000,
    },
}


# ============================================================
# RULE TESTS
# ============================================================

def test_constant_gets_zero():
    """CONSTANT → window = 0."""
    rec = recommend_window(SIGNALS['constant'])
    assert rec.window_size == 0, f"Constant should be 0, got {rec.window_size}"
    assert rec.method == 'constant'


def test_periodic_uses_period():
    """PERIODIC signals should use 4 × period."""
    rec = recommend_window(SIGNALS['sine_50'])
    expected = CYCLES_TO_CAPTURE * 50  # 200
    assert rec.window_size == expected, f"Sine 50 should be {expected}, got {rec.window_size}"
    assert rec.method == 'period'


def test_short_period_hits_min():
    """Very short period: 4 × 10 = 40 → clamped to MIN_WINDOW."""
    rec = recommend_window(SIGNALS['sine_10'])
    expected = max(MIN_WINDOW, CYCLES_TO_CAPTURE * 10)
    assert rec.window_size == expected, f"Sine 10 should be {expected}, got {rec.window_size}"


def test_long_period():
    """Long period: 4 × 500 = 2000."""
    rec = recommend_window(SIGNALS['sine_500'])
    expected = CYCLES_TO_CAPTURE * 500  # 2000
    assert rec.window_size == expected, f"Sine 500 should be {expected}, got {rec.window_size}"
    assert rec.method == 'period'


def test_white_noise_uses_acf():
    """White noise: ACF half-life = 1 → 4×1 = 4 → clamped to MIN_WINDOW."""
    rec = recommend_window(SIGNALS['white_noise'])
    assert rec.window_size == MIN_WINDOW, f"White noise should be {MIN_WINDOW}, got {rec.window_size}"
    assert rec.method == 'acf_half_life'


def test_random_walk_long_memory():
    """Random walk: ACF never decayed → long_memory rule."""
    rec = recommend_window(SIGNALS['random_walk'])
    assert rec.method == 'long_memory', f"Random walk method should be long_memory, got {rec.method}"
    # Non-stationary + long memory — should be reasonable size
    assert rec.window_size >= MIN_WINDOW
    assert rec.window_size <= MAX_WINDOW


def test_ar1_uses_acf_half_life():
    """AR(1) with slow decay: ACF half-life = 14 → 4×14 = 56."""
    rec = recommend_window(SIGNALS['ar1_slow'])
    expected = max(MIN_WINDOW, ACF_MULTIPLIER * 14)  # 56, but MIN_WINDOW=32 so 56
    assert rec.window_size == expected, f"AR1 should be {expected}, got {rec.window_size}"
    assert rec.method == 'acf_half_life'


def test_lorenz_uses_acf():
    """Lorenz: ACF half-life = 8 → 4×8 = 32 = MIN_WINDOW."""
    rec = recommend_window(SIGNALS['lorenz'])
    expected = max(MIN_WINDOW, ACF_MULTIPLIER * 8)
    assert rec.window_size == expected, f"Lorenz should be {expected}, got {rec.window_size}"


def test_turbofan_long_memory():
    """Turbofan: non-stationary + long memory, short signal (300 samples)."""
    rec = recommend_window(SIGNALS['turbofan'])
    assert rec.method == 'long_memory'
    # Should not exceed n_samples / 2
    assert rec.window_size <= 150, f"Turbofan window {rec.window_size} exceeds half of 300 samples"


def test_bearing_uses_period():
    """Bearing: PERIODIC with period=120 → 4×120 = 480."""
    rec = recommend_window(SIGNALS['bearing'])
    expected = CYCLES_TO_CAPTURE * 120
    assert rec.window_size == expected, f"Bearing should be {expected}, got {rec.window_size}"
    assert rec.method == 'period'


def test_short_signal_capped():
    """Short signal: window must not exceed n_samples // 2."""
    rec = recommend_window(SIGNALS['short_signal'])
    max_allowed = 100 // 2  # 50
    assert rec.window_size <= max_allowed, \
        f"Short signal window {rec.window_size} exceeds {max_allowed}"
    assert rec.window_size >= MIN_WINDOW


def test_quasi_periodic_uses_period():
    """QUASI_PERIODIC also uses period rule."""
    rec = recommend_window(SIGNALS['quasi_periodic'])
    expected = CYCLES_TO_CAPTURE * 75  # 300
    assert rec.window_size == expected, f"Quasi-periodic should be {expected}, got {rec.window_size}"
    assert rec.method == 'period'


def test_window_always_in_bounds():
    """All non-constant signals should produce windows in [MIN, MAX]."""
    for name, row in SIGNALS.items():
        rec = recommend_window(row)
        if row.get('continuity') == 'CONSTANT':
            assert rec.window_size == 0, f"{name} constant should be 0"
        else:
            assert rec.window_size >= MIN_WINDOW, \
                f"{name} window {rec.window_size} below MIN_WINDOW {MIN_WINDOW}"
            assert rec.window_size <= MAX_WINDOW, \
                f"{name} window {rec.window_size} above MAX_WINDOW {MAX_WINDOW}"


def test_confidence_set():
    """All recommendations must have a confidence level."""
    for name, row in SIGNALS.items():
        rec = recommend_window(row)
        assert rec.confidence in ('high', 'medium', 'low'), \
            f"{name} confidence '{rec.confidence}' not valid"


def test_method_set():
    """All recommendations must have a method."""
    valid_methods = {'constant', 'period', 'acf_half_life', 'long_memory',
                     'non_stationary_cap', 'default'}
    for name, row in SIGNALS.items():
        rec = recommend_window(row)
        assert rec.method in valid_methods, \
            f"{name} method '{rec.method}' not in {valid_methods}"


# ============================================================
# STRIDE TESTS
# ============================================================

def test_stride_50_pct():
    """Default 50% overlap."""
    assert compute_stride(128, 50.0) == 64
    assert compute_stride(100, 50.0) == 50


def test_stride_75_pct():
    """75% overlap for non-stationary."""
    assert compute_stride(128, 75.0) == 32
    assert compute_stride(100, 75.0) == 25


def test_stride_zero_window():
    """Zero window → zero stride."""
    assert compute_stride(0) == 0


def test_stride_min_one():
    """Stride should never be less than 1 for positive windows."""
    assert compute_stride(1, 95.0) >= 1


def test_recommend_stride_non_stationary():
    """Non-stationary signals get 75% overlap."""
    row = SIGNALS['random_walk']
    window = 128
    stride = recommend_stride(row, window)
    assert stride == 32, f"Non-stationary stride should be 32, got {stride}"


def test_recommend_stride_stationary():
    """Stationary signals get 50% overlap."""
    row = SIGNALS['white_noise']
    window = 128
    stride = recommend_stride(row, window)
    assert stride == 64, f"Stationary stride should be 64, got {stride}"


# ============================================================
# EIGENVALUE REFINEMENT TESTS
# ============================================================

def test_eigenvalue_stable():
    """Consistent eigenvalue proportions = stable window."""
    eigs = [
        [10.0, 5.0, 2.0, 1.0],
        [10.5, 4.8, 2.1, 1.1],
        [10.2, 5.1, 1.9, 1.0],
        [10.1, 4.9, 2.0, 1.05],
    ]
    result = eigenvalue_window_check(eigs)
    assert result['stable'] is True
    assert result['recommendation'] == 'window_ok'


def test_eigenvalue_unstable():
    """Wildly varying eigenvalue proportions = window too small."""
    eigs = [
        [10.0, 5.0, 2.0],
        [3.0, 8.0, 6.0],    # completely different structure
        [10.0, 1.0, 1.0],
        [5.0, 5.0, 5.0],
    ]
    result = eigenvalue_window_check(eigs, threshold=0.15)
    assert result['stable'] is False
    assert result['recommendation'] == 'window_too_small'


def test_eigenvalue_over_dominance():
    """First eigenvalue >95% = window possibly too large."""
    eigs = [
        [100.0, 0.5, 0.1],
        [101.0, 0.4, 0.1],
        [99.0, 0.6, 0.1],
    ]
    result = eigenvalue_window_check(eigs)
    assert result['recommendation'] == 'window_possibly_too_large'


def test_eigenvalue_insufficient_windows():
    """Less than 3 windows = can't assess."""
    eigs = [[10.0, 5.0], [11.0, 4.0]]
    result = eigenvalue_window_check(eigs)
    assert result['recommendation'] == 'insufficient_windows'


# ============================================================
# EDGE CASES
# ============================================================

def test_missing_all_fields():
    """Row with minimal fields should fall to default."""
    rec = recommend_window({'continuity': 'CONTINUOUS'})
    assert rec.window_size == DEFAULT_WINDOW
    assert rec.method == 'default'


def test_nan_acf():
    """NaN ACF half-life should not crash."""
    row = {
        'continuity': 'CONTINUOUS',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'RANDOM',
        'memory': 'SHORT_MEMORY',
        'acf_half_life': float('nan'),
        'n_samples': 1000,
    }
    rec = recommend_window(row)
    assert rec.window_size > 0  # Should fall through to default


def test_none_n_samples():
    """Missing n_samples should not crash."""
    row = {
        'continuity': 'CONTINUOUS',
        'temporal_pattern': 'PERIODIC',
        'seasonal_period': 50.0,
        'n_samples': None,
    }
    rec = recommend_window(row)
    assert rec.window_size == CYCLES_TO_CAPTURE * 50


def test_non_periodic_ignores_freq():
    """RANDOM signal should NOT use dominant_freq even if present."""
    row = {
        'continuity': 'CONTINUOUS',
        'temporal_pattern': 'RANDOM',
        'stationarity': 'STATIONARY',
        'memory': 'SHORT_MEMORY',
        'dominant_freq': 0.02,      # Present but irrelevant
        'seasonal_period': 50.0,    # Present but irrelevant
        'acf_half_life': 3.0,
        'acf_decayed': True,
        'n_samples': 5000,
    }
    rec = recommend_window(row)
    # Should use ACF, not period (because temporal_pattern is RANDOM)
    assert rec.method == 'acf_half_life', \
        f"RANDOM should use ACF not period, got {rec.method}"


# ============================================================
# PRINT SUMMARY & RUN
# ============================================================

def print_summary():
    """Print window recommendations for all benchmark signals."""
    print("=" * 70)
    print("WINDOW RECOMMENDER — BENCHMARK VALIDATION")
    print("=" * 70)

    for name, row in SIGNALS.items():
        rec = recommend_window(row)
        stride = recommend_stride(row, rec.window_size)
        n = row.get('n_samples', '?')
        n_windows = '?' if rec.window_size == 0 or stride == 0 else \
            max(1, (int(n) - rec.window_size) // stride + 1) if isinstance(n, int) else '?'

        print(f"\n  {name}")
        print(f"    Window: {rec.window_size:>5}  Stride: {stride:>4}  "
              f"≈{n_windows} windows from {n} samples")
        print(f"    Method: {rec.method} ({rec.confidence})")
        print(f"    {rec.reason}")

    print(f"\n{'=' * 70}")


def run_tests():
    """Run all tests."""
    tests = [
        # Rule tests
        test_constant_gets_zero,
        test_periodic_uses_period,
        test_short_period_hits_min,
        test_long_period,
        test_white_noise_uses_acf,
        test_random_walk_long_memory,
        test_ar1_uses_acf_half_life,
        test_lorenz_uses_acf,
        test_turbofan_long_memory,
        test_bearing_uses_period,
        test_short_signal_capped,
        test_quasi_periodic_uses_period,
        test_window_always_in_bounds,
        test_confidence_set,
        test_method_set,
        # Stride tests
        test_stride_50_pct,
        test_stride_75_pct,
        test_stride_zero_window,
        test_stride_min_one,
        test_recommend_stride_non_stationary,
        test_recommend_stride_stationary,
        # Eigenvalue refinement tests
        test_eigenvalue_stable,
        test_eigenvalue_unstable,
        test_eigenvalue_over_dominance,
        test_eigenvalue_insufficient_windows,
        # Edge cases
        test_missing_all_fields,
        test_nan_acf,
        test_none_n_samples,
        test_non_periodic_ignores_freq,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1

    print(f"\n  {passed}/{passed + failed} tests passed")
    return failed == 0


if __name__ == '__main__':
    print_summary()
    print()
    success = run_tests()
    sys.exit(0 if success else 1)
