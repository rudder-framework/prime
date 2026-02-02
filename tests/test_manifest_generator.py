"""
Test: Manifest Generator v2

Validates that the 10-dimension typology → engine mapping produces
correct engine lists for known signal types.

Each test creates a fake typology card for a known signal type and
checks that the manifest generator selects the right engines.

Run:
    python -m pytest tests/test_manifest_generator.py -v
    python tests/test_manifest_generator.py  # standalone
"""

import sys
from pathlib import Path

# Add parent to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from orthon.ingest.manifest_generator import (
    select_engines_for_signal,
    select_visualizations,
    compute_recommended_window,
    compute_derivative_depth,
    compute_eigenvalue_budget,
    compute_output_hints,
    DEPRECATED_ENGINES,
)


# ============================================================
# BENCHMARK TYPOLOGY CARDS
# ============================================================
# Each card represents a known signal type with expected behavior.
# These are the same signals used for typology validation.
# ============================================================

CARDS = {
    'white_noise': {
        'continuity': 'CONTINUOUS',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'RANDOM',
        'memory': 'SHORT_MEMORY',
        'complexity': 'HIGH',
        'distribution': 'GAUSSIAN',
        'amplitude': 'NOISY',
        'spectral': 'BROADBAND',
        'volatility': 'HOMOSCEDASTIC',
        'determinism': 'STOCHASTIC',
    },
    'sine_wave': {
        'continuity': 'CONTINUOUS',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'PERIODIC',
        'memory': 'SHORT_MEMORY',
        'complexity': 'LOW',
        'distribution': 'LIGHT_TAILED',
        'amplitude': 'SMOOTH',
        'spectral': 'NARROWBAND',
        'volatility': 'HOMOSCEDASTIC',
        'determinism': 'DETERMINISTIC',
    },
    'random_walk': {
        'continuity': 'CONTINUOUS',
        'stationarity': 'NON_STATIONARY',
        'temporal_pattern': 'TRENDING',
        'memory': 'LONG_MEMORY',
        'complexity': 'MEDIUM',
        'distribution': 'GAUSSIAN',
        'amplitude': 'SMOOTH',
        'spectral': 'ONE_OVER_F',
        'volatility': 'HETEROSCEDASTIC',
        'determinism': 'STOCHASTIC',
        'acf_half_life': 80,
    },
    'lorenz_attractor': {
        'continuity': 'CONTINUOUS',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'CHAOTIC',
        'memory': 'SHORT_MEMORY',
        'complexity': 'HIGH',
        'distribution': 'HEAVY_TAILED',
        'amplitude': 'MIXED',
        'spectral': 'BROADBAND',
        'volatility': 'HOMOSCEDASTIC',
        'determinism': 'DETERMINISTIC',
    },
    'degrading_turbofan': {
        'continuity': 'CONTINUOUS',
        'stationarity': 'NON_STATIONARY',
        'temporal_pattern': 'TRENDING',
        'memory': 'LONG_MEMORY',
        'complexity': 'LOW',
        'distribution': 'SKEWED_RIGHT',
        'amplitude': 'SMOOTH',
        'spectral': 'ONE_OVER_F',
        'volatility': 'HETEROSCEDASTIC',
        'determinism': 'MIXED',
        'acf_half_life': 120,
    },
    'bearing_vibration': {
        'continuity': 'CONTINUOUS',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'PERIODIC',
        'memory': 'SHORT_MEMORY',
        'complexity': 'MEDIUM',
        'distribution': 'HEAVY_TAILED',
        'amplitude': 'IMPULSIVE',
        'spectral': 'HARMONIC',
        'volatility': 'HOMOSCEDASTIC',
        'determinism': 'DETERMINISTIC',
    },
    'constant_sensor': {
        'continuity': 'CONSTANT',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'CONSTANT',
        'memory': None,
        'complexity': None,
        'distribution': 'CONSTANT',
        'amplitude': 'CONSTANT',
        'spectral': None,
        'volatility': None,
        'determinism': None,
    },
    'valve_position': {
        'continuity': 'DISCRETE',
        'stationarity': 'STATIONARY',
        'temporal_pattern': 'RANDOM',
        'memory': 'SHORT_MEMORY',
        'complexity': 'LOW',
        'distribution': 'LIGHT_TAILED',
        'amplitude': 'MIXED',
        'spectral': 'BROADBAND',
        'volatility': 'HOMOSCEDASTIC',
        'determinism': 'STOCHASTIC',
    },
}


# ============================================================
# TESTS
# ============================================================

def test_constant_gets_nothing():
    """CONSTANT signals should get zero engines."""
    engines = select_engines_for_signal(CARDS['constant_sensor'])
    assert engines == [], f"Constant got engines: {engines}"


def test_core_always_present():
    """Every non-constant signal must get kurtosis, skewness, crest_factor."""
    core = {'kurtosis', 'skewness', 'crest_factor'}
    for name, card in CARDS.items():
        if card['continuity'] == 'CONSTANT':
            continue
        engines = set(select_engines_for_signal(card))
        missing = core - engines
        assert not missing, f"{name} missing core engines: {missing}"


def test_no_deprecated_engines():
    """No signal should ever get a deprecated engine."""
    for name, card in CARDS.items():
        engines = set(select_engines_for_signal(card))
        bad = engines & DEPRECATED_ENGINES
        assert not bad, f"{name} has deprecated engines: {bad}"


def test_periodic_gets_spectral():
    """PERIODIC signals must get spectral and harmonics engines."""
    engines = set(select_engines_for_signal(CARDS['sine_wave']))
    assert 'spectral' in engines, "Sine wave missing spectral"

    engines_b = set(select_engines_for_signal(CARDS['bearing_vibration']))
    assert 'spectral' in engines_b, "Bearing missing spectral"
    assert 'harmonics' in engines_b, "Bearing missing harmonics"


def test_trending_gets_hurst():
    """TRENDING signals must get hurst and rate_of_change."""
    for name in ['random_walk', 'degrading_turbofan']:
        engines = set(select_engines_for_signal(CARDS[name]))
        assert 'hurst' in engines, f"{name} missing hurst"
        assert 'rate_of_change' in engines, f"{name} missing rate_of_change"


def test_chaotic_gets_dynamics():
    """CHAOTIC signals must get lyapunov and attractor."""
    engines = set(select_engines_for_signal(CARDS['lorenz_attractor']))
    assert 'lyapunov' in engines, "Lorenz missing lyapunov"
    assert 'attractor' in engines, "Lorenz missing attractor"


def test_non_stationary_gets_rolling():
    """NON_STATIONARY signals must get rolling engines."""
    engines = set(select_engines_for_signal(CARDS['random_walk']))
    rolling = {e for e in engines if e.startswith('rolling_')}
    assert len(rolling) > 0, "Random walk has no rolling engines"
    assert 'rolling_kurtosis' in engines, "Random walk missing rolling_kurtosis"


def test_stationary_no_unnecessary_rolling():
    """STATIONARY signals should not get rolling engines from stationarity dimension."""
    # Sine wave is STATIONARY + SMOOTH (smooth adds rolling_kurtosis via amplitude)
    # But should NOT get rolling from stationarity dimension
    engines = set(select_engines_for_signal(CARDS['white_noise']))
    # White noise is STATIONARY, NOISY, BROADBAND — no rolling needed from stationarity
    # (may get some from other dimensions)
    # Key check: no rolling_volatility (only comes from volatility dimension)
    assert 'rolling_volatility' not in engines, "White noise shouldn't have rolling_volatility"


def test_discrete_no_spectral():
    """DISCRETE signals should not get spectral engines."""
    engines = set(select_engines_for_signal(CARDS['valve_position']))
    assert 'spectral' not in engines, "Discrete got spectral"
    assert 'harmonics' not in engines, "Discrete got harmonics"
    assert 'frequency_bands' not in engines, "Discrete got frequency_bands"


def test_volatility_clustering_gets_garch():
    """VOLATILITY_CLUSTERING signals must get garch."""
    card = {**CARDS['random_walk'], 'volatility': 'VOLATILITY_CLUSTERING'}
    engines = set(select_engines_for_signal(card))
    assert 'garch' in engines, "Volatility clustering missing garch"


def test_window_from_acf():
    """Window size should be 4× ACF half-life when available."""
    w = compute_recommended_window(CARDS['random_walk'])
    assert w == 320, f"Random walk window should be 320 (4×80), got {w}"

    w2 = compute_recommended_window(CARDS['degrading_turbofan'])
    assert w2 == 480, f"Turbofan window should be 480 (4×120), got {w2}"


def test_window_memory_fallback():
    """Without ACF half-life, use memory-based heuristic."""
    card = {**CARDS['sine_wave']}  # SHORT_MEMORY, no acf_half_life
    w = compute_recommended_window(card)
    assert w == 128, f"Short memory default should be 128, got {w}"


def test_derivative_depth():
    """Trending/non-stationary → depth 2, stationary → depth 1, constant → 0."""
    assert compute_derivative_depth(CARDS['constant_sensor']) == 0
    assert compute_derivative_depth(CARDS['sine_wave']) == 1
    assert compute_derivative_depth(CARDS['random_walk']) == 2
    assert compute_derivative_depth(CARDS['degrading_turbofan']) == 2


def test_eigenvalue_budget():
    """LOW→3, MEDIUM→5, HIGH→8, CONSTANT→0."""
    assert compute_eigenvalue_budget(CARDS['constant_sensor']) == 0
    assert compute_eigenvalue_budget(CARDS['sine_wave']) == 3       # LOW
    assert compute_eigenvalue_budget(CARDS['bearing_vibration']) == 5  # MEDIUM
    assert compute_eigenvalue_budget(CARDS['white_noise']) == 8     # HIGH


def test_waterfall_visualization():
    """PERIODIC + HARMONIC signals should get waterfall recommendation."""
    engines = select_engines_for_signal(CARDS['bearing_vibration'])
    viz = select_visualizations(CARDS['bearing_vibration'], engines)
    assert 'waterfall' in viz, f"Bearing should get waterfall, got {viz}"


def test_phase_portrait_for_chaotic():
    """CHAOTIC signals should get phase_portrait recommendation."""
    engines = select_engines_for_signal(CARDS['lorenz_attractor'])
    viz = select_visualizations(CARDS['lorenz_attractor'], engines)
    assert 'phase_portrait' in viz, f"Lorenz should get phase_portrait, got {viz}"


def test_trend_overlay_for_trending():
    """TRENDING signals should get trend_overlay recommendation."""
    engines = select_engines_for_signal(CARDS['degrading_turbofan'])
    viz = select_visualizations(CARDS['degrading_turbofan'], engines)
    assert 'trend_overlay' in viz, f"Turbofan should get trend_overlay, got {viz}"


def test_waterfall_output_hints():
    """PERIODIC + HARMONIC should get per_bin spectral output."""
    engines = select_engines_for_signal(CARDS['bearing_vibration'])
    hints = compute_output_hints(CARDS['bearing_vibration'], engines)
    assert 'spectral' in hints, "Bearing should have spectral output hints"
    assert hints['spectral']['output_mode'] == 'per_bin', "Should be per_bin for waterfall"


def test_summary_spectral_for_broadband():
    """BROADBAND signals should get summary spectral output (not per_bin)."""
    engines = select_engines_for_signal(CARDS['white_noise'])
    hints = compute_output_hints(CARDS['white_noise'], engines)
    if 'spectral' in hints:
        assert hints['spectral']['output_mode'] == 'summary', "Broadband should be summary mode"


# ============================================================
# PRINT SUMMARY (standalone execution)
# ============================================================

def print_summary():
    """Print engine selection for all benchmark signals."""
    print("=" * 70)
    print("MANIFEST GENERATOR v2 — BENCHMARK VALIDATION")
    print("=" * 70)

    for name, card in CARDS.items():
        engines = select_engines_for_signal(card)
        viz = select_visualizations(card, engines)
        window = compute_recommended_window(card)
        depth = compute_derivative_depth(card)
        eig = compute_eigenvalue_budget(card)
        hints = compute_output_hints(card, engines)

        sig_eng = [e for e in engines if not e.startswith('rolling_')]
        roll_eng = [e for e in engines if e.startswith('rolling_')]

        print(f"\n{'─' * 70}")
        print(f"  {name}")
        print(f"{'─' * 70}")
        print(f"  Card:     {card.get('continuity')}, {card.get('stationarity')}, "
              f"{card.get('temporal_pattern')}, {card.get('memory')}")
        print(f"            {card.get('complexity')}, {card.get('distribution')}, "
              f"{card.get('amplitude')}, {card.get('spectral')}")
        print(f"            {card.get('volatility')}, {card.get('determinism')}")
        print(f"  Signal:   {sig_eng}")
        print(f"  Rolling:  {roll_eng}")
        print(f"  Window:   {window}  |  Depth: {depth}  |  Eigenvalues: {eig}")
        if viz:
            print(f"  Viz:      {viz}")
        if hints:
            for eng, cfg in hints.items():
                print(f"  Hint:     {eng} → {cfg.get('output_mode', '?')}")

    print(f"\n{'=' * 70}")


def run_tests():
    """Run all tests and print results."""
    tests = [
        test_constant_gets_nothing,
        test_core_always_present,
        test_no_deprecated_engines,
        test_periodic_gets_spectral,
        test_trending_gets_hurst,
        test_chaotic_gets_dynamics,
        test_non_stationary_gets_rolling,
        test_stationary_no_unnecessary_rolling,
        test_discrete_no_spectral,
        test_volatility_clustering_gets_garch,
        test_window_from_acf,
        test_window_memory_fallback,
        test_derivative_depth,
        test_eigenvalue_budget,
        test_waterfall_visualization,
        test_phase_portrait_for_chaotic,
        test_trend_overlay_for_trending,
        test_waterfall_output_hints,
        test_summary_spectral_for_broadband,
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
