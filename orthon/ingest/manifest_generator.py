"""
ORTHON Manifest Generator v2

Creates the complete order for PRISM from 10-dimension typology results.
Every engine selection decision lives here. PRISM receives the manifest
and executes exactly what it says — no interpretation, no classification.

ORTHON decides. PRISM executes.

Changes from v1:
    - Reads 10-dimension typology (continuity, stationarity, temporal_pattern,
      memory, complexity, distribution, amplitude, spectral, volatility, determinism)
    - Maps each dimension to real PRISM engine modules
    - Computes window sizes from ACF half-life
    - Sets eigenvalue budget from complexity
    - Sets derivative depth from stationarity + temporal pattern
    - Adds output format hints (waterfall, phase portrait, etc.)
    - Adds visualization recommendations per signal

Usage:
    python -m orthon.manifest_generator data/typology.parquet data/manifest.yaml
"""

import yaml
import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

from orthon.window_recommender import recommend_stride


# ============================================================
# ENGINE REGISTRY
# ============================================================
# Every engine name here maps to a real PRISM module.
# If it's not in prism/engines/signal/ or prism/engines/rolling/,
# it doesn't go in the manifest.
#
# Verified against ENGINE_INVENTORY.md and prism/engines/signal/__init__.py
# ============================================================

# Signal-level engines (one value per signal per window)
SIGNAL_ENGINES = {
    # Core (always run)
    'kurtosis', 'skewness', 'crest_factor',
    # Spectral / frequency
    'spectral', 'harmonics', 'frequency_bands',
    # Complexity / dynamics
    'entropy', 'hurst', 'lyapunov', 'garch', 'attractor', 'dmd',
    # Shape / rate
    'peak', 'rate_of_change', 'pulsation_index', 'time_constant',
    'cycle_counting', 'envelope',
    # Outlier / basin
    'basin', 'lof',
}

# Rolling engines (one value per I per signal)
ROLLING_ENGINES = {
    'rolling_kurtosis', 'rolling_skewness',
    'rolling_entropy', 'rolling_crest_factor',
    'rolling_hurst', 'rolling_lyapunov',
    'rolling_volatility', 'rolling_pulsation',
    # Deprecated but still exist in codebase:
    # 'rolling_rms', 'rolling_mean', 'rolling_std', 'rolling_range', 'rolling_envelope'
}

# Pair engines (directional: A → B)
PAIR_ENGINES = {'granger', 'transfer_entropy'}

# Symmetric pair engines (A ↔ B, computed once)
SYMMETRIC_PAIR_ENGINES = {'correlation', 'mutual_info', 'cointegration'}

# Windowed / cross-signal engines
WINDOWED_ENGINES = {'derivatives', 'manifold', 'stability'}

# Structure engines (system-level)
STRUCTURE_ENGINES = {
    'covariance_engine', 'eigenvalue_engine', 'koopman_engine',
    'spectral_engine', 'wavelet_engine',
}

# Dynamics engines
DYNAMICS_ENGINES = {
    'lyapunov_engine', 'attractor_engine', 'recurrence_engine',
    'bifurcation_engine',
}

# Scale-dependent — NEVER include in manifest
DEPRECATED_ENGINES = {
    'rms', 'peak', 'total_power', 'mean', 'std',
    'rolling_rms', 'rolling_mean', 'rolling_std', 'rolling_range',
    'rolling_envelope', 'envelope',
    'harmonic_2x', 'harmonic_3x',
    'band_low', 'band_mid', 'band_high',
}

# All valid engines (for validation)
ALL_VALID = SIGNAL_ENGINES | ROLLING_ENGINES | PAIR_ENGINES | SYMMETRIC_PAIR_ENGINES


# ============================================================
# DIMENSION → ENGINE MAPPING
# ============================================================
# Each typology dimension maps to engines that are relevant
# when a signal has that particular characteristic.
#
# The mapping is additive: engines accumulate across dimensions.
# Core engines always run. Deprecated engines are always stripped.
# ============================================================

ENGINE_RULES = {

    # --- Always run (scale-invariant core) ---
    'core': ['kurtosis', 'skewness', 'crest_factor'],

    # --- Dimension 2: STATIONARITY ---
    # Non-stationary signals need rolling engines to track changing statistics
    'stationarity': {
        'STATIONARY':            [],
        'TREND_STATIONARY':      ['rolling_kurtosis', 'rolling_entropy'],
        'DIFFERENCE_STATIONARY': ['rolling_kurtosis', 'rolling_entropy', 'rolling_crest_factor'],
        'NON_STATIONARY':        ['rolling_kurtosis', 'rolling_entropy', 'rolling_crest_factor',
                                  'rolling_skewness', 'rolling_volatility'],
    },

    # --- Dimension 3: TEMPORAL PATTERN ---
    # This is the big one — determines the primary analysis strategy
    'temporal_pattern': {
        'PERIODIC':       ['spectral', 'harmonics', 'frequency_bands'],
        'QUASI_PERIODIC': ['spectral', 'frequency_bands'],
        'TRENDING':       ['hurst', 'rate_of_change'],
        'MEAN_REVERTING': ['hurst'],
        'CHAOTIC':        ['lyapunov', 'attractor', 'entropy'],
        'RANDOM':         ['entropy'],
    },

    # --- Dimension 4: MEMORY ---
    # Long-memory signals need larger windows and persistence tracking
    'memory': {
        'LONG_MEMORY':     ['hurst', 'rate_of_change'],
        'SHORT_MEMORY':    [],
        'ANTI_PERSISTENT': ['hurst'],
    },

    # --- Dimension 5: COMPLEXITY ---
    # High-complexity signals need entropy and chaos tools
    'complexity': {
        'LOW':    [],
        'MEDIUM': ['entropy'],
        'HIGH':   ['entropy', 'attractor'],
    },

    # --- Dimension 6: DISTRIBUTION SHAPE ---
    # Tail behavior determines which statistical engines matter
    'distribution': {
        'GAUSSIAN':     [],
        'HEAVY_TAILED': ['peak'],       # peak_ratio from peak module
        'LIGHT_TAILED': [],
        'SKEWED_RIGHT': [],
        'SKEWED_LEFT':  [],
    },

    # --- Dimension 7: AMPLITUDE CHARACTER ---
    # Signal texture determines time-domain vs frequency-domain focus
    'amplitude': {
        'SMOOTH':    ['rolling_kurtosis', 'rolling_crest_factor'],
        'NOISY':     ['entropy'],
        'IMPULSIVE': ['peak', 'cycle_counting'],
        'MIXED':     ['entropy', 'peak'],
    },

    # --- Dimension 8: SPECTRAL CHARACTER ---
    # Where energy lives in frequency determines spectral engines
    'spectral': {
        'NARROWBAND': ['spectral'],
        'BROADBAND':  ['spectral', 'frequency_bands'],
        'HARMONIC':   ['spectral', 'harmonics'],
        'ONE_OVER_F': ['hurst', 'spectral'],
    },

    # --- Dimension 9: VOLATILITY ---
    # Changing variance needs GARCH and rolling variance tracking
    'volatility': {
        'HOMOSCEDASTIC':        [],
        'HETEROSCEDASTIC':      ['rolling_volatility', 'rolling_kurtosis'],
        'VOLATILITY_CLUSTERING': ['garch', 'rolling_volatility', 'rolling_kurtosis'],
    },

    # --- Dimension 10: DETERMINISM ---
    # Deterministic signals get attractor/dynamics; stochastic get entropy
    'determinism': {
        'DETERMINISTIC': ['lyapunov', 'attractor'],
        'STOCHASTIC':    ['entropy'],
        'MIXED':         ['entropy', 'lyapunov'],
    },
}

# The 10 dimension keys (minus continuity which is a gate, not an engine selector)
DIMENSION_KEYS = [
    'stationarity', 'temporal_pattern', 'memory', 'complexity',
    'distribution', 'amplitude', 'spectral', 'volatility', 'determinism',
]


# ============================================================
# VISUALIZATION RECOMMENDATIONS
# ============================================================
# When the typology card has certain combinations, specific
# visualizations become useful. The explorer can offer these
# automatically without knowing the domain.
# ============================================================

VIZ_RULES = {
    # Waterfall: frequency-axis signals where spectral evolution matters
    'waterfall': {
        'requires_any': {
            'temporal_pattern': ['PERIODIC', 'QUASI_PERIODIC'],
            'spectral':         ['HARMONIC', 'NARROWBAND'],
        },
        'requires_engine': 'spectral',
        'description': 'Frequency × amplitude × time waterfall plot',
    },

    # Phase portrait: deterministic signals with attractor structure
    'phase_portrait': {
        'requires_any': {
            'temporal_pattern': ['CHAOTIC'],
            'determinism':      ['DETERMINISTIC'],
        },
        'requires_engine': 'attractor',
        'description': 'State-space attractor reconstruction',
    },

    # Trend plot: monotonic drift with rolling statistics overlay
    'trend_overlay': {
        'requires_any': {
            'temporal_pattern': ['TRENDING'],
            'stationarity':     ['NON_STATIONARY', 'TREND_STATIONARY'],
        },
        'requires_engine': 'rate_of_change',
        'description': 'Time series with rolling mean/std overlay',
    },

    # Recurrence plot: deterministic or chaotic structure
    'recurrence': {
        'requires_any': {
            'determinism': ['DETERMINISTIC', 'MIXED'],
        },
        'requires_engine': 'attractor',
        'description': 'Recurrence plot showing system revisitation',
    },

    # Volatility map: heteroscedastic signals
    'volatility_map': {
        'requires_any': {
            'volatility': ['HETEROSCEDASTIC', 'VOLATILITY_CLUSTERING'],
        },
        'requires_engine': 'rolling_volatility',
        'description': 'Rolling variance heatmap showing calm/stormy periods',
    },

    # Spectral density: any signal with spectral analysis
    'spectral_density': {
        'requires_any': {
            'spectral': ['NARROWBAND', 'BROADBAND', 'HARMONIC', 'ONE_OVER_F'],
        },
        'requires_engine': 'spectral',
        'description': 'Power spectral density plot',
    },
}


# ============================================================
# ENGINE SELECTION
# ============================================================

def select_engines_for_signal(typology_row: Dict[str, Any]) -> List[str]:
    """
    Select engines for a single signal based on its 10-dimension typology.

    This is THE decision function. Lives in ORTHON, not PRISM.
    Walks each dimension and accumulates the engines that apply.
    Core engines always included. Deprecated engines always stripped.

    Args:
        typology_row: Dict with all 10 dimension values from typology.parquet

    Returns:
        Sorted, deduplicated list of engine names to run
    """
    continuity = typology_row.get('continuity', 'CONTINUOUS')

    # Gate: CONSTANT signals get nothing
    if continuity == 'CONSTANT':
        return []

    # Start with core
    engines: Set[str] = set(ENGINE_RULES['core'])

    # Walk each dimension
    for dim_key in DIMENSION_KEYS:
        dim_value = typology_row.get(dim_key)
        if dim_value and dim_key in ENGINE_RULES:
            dim_engines = ENGINE_RULES[dim_key].get(dim_value, [])
            engines.update(dim_engines)

    # Gate: EVENT/DISCRETE signals — no spectral or derivative engines
    if continuity in ('EVENT', 'DISCRETE'):
        spectral_set = {'spectral', 'harmonics', 'frequency_bands'}
        engines -= spectral_set

    # Safety: strip anything deprecated
    engines -= DEPRECATED_ENGINES

    # Safety: only include engines that actually exist
    engines &= ALL_VALID

    return sorted(engines)


def select_visualizations(typology_row: Dict[str, Any], engines: List[str]) -> List[str]:
    """
    Select recommended visualizations based on typology card.

    Args:
        typology_row: Dict with 10 dimension values
        engines: List of selected engines for this signal

    Returns:
        List of visualization names that apply
    """
    if typology_row.get('continuity') == 'CONSTANT':
        return []

    engine_set = set(engines)
    viz_list = []

    for viz_name, rule in VIZ_RULES.items():
        # Check that required engine is in the selected set
        required_engine = rule.get('requires_engine')
        if required_engine and required_engine not in engine_set:
            continue

        # Check that at least one dimension value matches
        requires_any = rule.get('requires_any', {})
        matched = False
        for dim_key, acceptable_values in requires_any.items():
            actual_value = typology_row.get(dim_key)
            if actual_value in acceptable_values:
                matched = True
                break

        if matched:
            viz_list.append(viz_name)

    return viz_list


# ============================================================
# WINDOW SIZE / DERIVATIVE DEPTH / EIGENVALUE BUDGET
# ============================================================

def compute_recommended_window(typology_row: Dict[str, Any], default: int = 128) -> int:
    """
    Compute recommended window size from ACF half-life and memory.

    Rule: window_size = max(64, 4 × acf_half_life)
    Falls back to memory-based heuristic if half-life unavailable.
    """
    acf_half_life = typology_row.get('acf_half_life')
    if acf_half_life is not None and acf_half_life > 16:
        return max(64, int(4 * acf_half_life))

    memory = typology_row.get('memory', 'SHORT_MEMORY')
    if memory == 'LONG_MEMORY':
        return 256
    elif memory == 'ANTI_PERSISTENT':
        return 64
    return default


def compute_derivative_depth(typology_row: Dict[str, Any]) -> int:
    """
    How many derivatives PRISM should compute (velocity, acceleration, jerk).

    0 = CONSTANT (nothing), 1 = stationary, 2 = non-stationary/trending
    """
    if typology_row.get('continuity') == 'CONSTANT':
        return 0

    stationarity = typology_row.get('stationarity', 'STATIONARY')
    temporal = typology_row.get('temporal_pattern', 'RANDOM')

    if stationarity in ('NON_STATIONARY', 'DIFFERENCE_STATIONARY'):
        return 2
    if temporal == 'TRENDING':
        return 2
    return 1


def compute_eigenvalue_budget(typology_row: Dict[str, Any]) -> int:
    """
    How many eigenvalues to retain in state_geometry.

    Based on complexity: LOW=3, MEDIUM=5, HIGH=8.
    Signals with low complexity concentrate their energy in few dimensions.
    """
    if typology_row.get('continuity') == 'CONSTANT':
        return 0

    complexity = typology_row.get('complexity', 'MEDIUM')
    return {'LOW': 3, 'MEDIUM': 5, 'HIGH': 8}.get(complexity, 5)


# ============================================================
# OUTPUT FORMAT HINTS
# ============================================================

def compute_output_hints(typology_row: Dict[str, Any], engines: List[str]) -> Dict[str, Any]:
    """
    Tell PRISM what output format to use for certain engines.

    For example, if spectral engine is selected and the signal is
    PERIODIC or HARMONIC, request per-bin output (waterfall-ready)
    instead of summary-only.

    Args:
        typology_row: Typology card
        engines: Selected engines

    Returns:
        Dict of engine_name → output config overrides
    """
    hints = {}
    engine_set = set(engines)

    # Spectral engine: request per-bin output for waterfall-capable signals
    if 'spectral' in engine_set:
        temporal = typology_row.get('temporal_pattern', '')
        spectral_char = typology_row.get('spectral', '')

        if temporal in ('PERIODIC', 'QUASI_PERIODIC') or spectral_char in ('HARMONIC', 'NARROWBAND'):
            hints['spectral'] = {
                'output_mode': 'per_bin',          # full frequency × amplitude per window
                'n_bins': 'auto',                   # let engine decide from window size
                'include_phase': False,             # phase adds compute, skip unless needed
                'note': 'waterfall-ready output',
            }
        else:
            hints['spectral'] = {
                'output_mode': 'summary',           # centroid, bandwidth, entropy, slope
            }

    # Harmonics engine: track individual harmonics for HARMONIC signals
    if 'harmonics' in engine_set:
        spectral_char = typology_row.get('spectral', '')
        if spectral_char == 'HARMONIC':
            hints['harmonics'] = {
                'n_harmonics': 5,                   # track up to 5th harmonic
                'include_thd': True,                # total harmonic distortion
            }

    # Attractor engine: full reconstruction for deterministic/chaotic
    if 'attractor' in engine_set:
        determinism = typology_row.get('determinism', '')
        if determinism == 'DETERMINISTIC':
            hints['attractor'] = {
                'output_mode': 'trajectory',        # full state-space points
                'embedding_dim': 'auto',
            }

    # GARCH: full model for volatility clustering
    if 'garch' in engine_set:
        hints['garch'] = {
            'output_mode': 'full',                  # omega, alpha, beta, persistence
            'include_conditional_variance': True,
        }

    return hints


# ============================================================
# MANIFEST GENERATION
# ============================================================

def generate_manifest(
    typology_path: str,
    output_path: str = "manifest.yaml",
    observations_path: str = "observations.parquet",
    job_name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate complete PRISM manifest from 10-dimension typology.

    Reads typology.parquet, selects engines per signal, computes window sizes,
    derivative depths, eigenvalue budgets, visualization recommendations,
    and output format hints. Writes manifest.yaml that PRISM executes verbatim.

    Args:
        typology_path: Path to typology.parquet (10 dimensions per signal)
        output_path: Where to write manifest.yaml
        observations_path: Path to observations.parquet
        job_name: Optional job identifier
        params: Optional parameter overrides
        verbose: Print progress

    Returns:
        The generated manifest dict
    """
    params = params or {}

    df = pl.read_parquet(typology_path)

    if verbose:
        print("=" * 60)
        print("ORTHON MANIFEST GENERATOR v2")
        print("10-dimension typology → PRISM engine selection")
        print("=" * 60)
        print(f"  Typology: {typology_path}")
        print(f"  Signals:  {len(df)}")
        print()

    # ---- Per-signal configuration (nested by cohort) ----
    cohorts_config: Dict[str, Dict[str, Any]] = {}
    all_signal_engines: Set[str] = set()
    all_rolling_engines: Set[str] = set()
    constant_count = 0

    for row in df.iter_rows(named=True):
        signal_id = row['signal_id']
        cohort = row.get('cohort') or row.get('unit_id') or 'default'

        # Select engines from typology card
        engines = select_engines_for_signal(row)
        window = compute_recommended_window(row, params.get('default_window', 128))
        stride = recommend_stride(row, window)  # Compute stride from stationarity/temporal
        depth = compute_derivative_depth(row)
        eigen_budget = compute_eigenvalue_budget(row)
        viz = select_visualizations(row, engines)
        output_hints = compute_output_hints(row, engines)

        # Categorize engines
        sig_eng = sorted([e for e in engines if e in SIGNAL_ENGINES])
        roll_eng = sorted([e for e in engines if e in ROLLING_ENGINES])

        all_signal_engines.update(sig_eng)
        all_rolling_engines.update(roll_eng)

        continuity = row.get('continuity', 'CONTINUOUS')
        if continuity == 'CONSTANT':
            constant_count += 1

        # Build signal entry
        entry = {
            'engines': sig_eng,
            'rolling_engines': roll_eng,
            'window_size': window,
            'stride': stride,  # Per-signal stride based on stationarity/temporal
            'derivative_depth': depth,
            'eigenvalue_budget': eigen_budget,
        }

        # Typology card (for debugging / downstream reference)
        card = {}
        for dim in ['continuity'] + DIMENSION_KEYS:
            val = row.get(dim)
            if val is not None:
                card[dim] = val
        entry['typology'] = card

        # Visualization recommendations
        if viz:
            entry['visualizations'] = viz

        # Output format hints
        if output_hints:
            entry['output_hints'] = output_hints

        # Nest under cohort
        if cohort not in cohorts_config:
            cohorts_config[cohort] = {}
        cohorts_config[cohort][signal_id] = entry

        if verbose:
            if continuity == 'CONSTANT':
                print(f"    {cohort}/{signal_id}: CONSTANT → skip")
            else:
                n_eng = len(sig_eng) + len(roll_eng)
                viz_str = f", viz=[{','.join(viz)}]" if viz else ""
                print(f"    {cohort}/{signal_id}: {n_eng} engines, w={window}, s={stride}, d={depth}, eig={eigen_budget}{viz_str}")

    # ---- Build manifest ----
    active_count = len(df) - constant_count
    total_cohorts = len(cohorts_config)

    # Build skip list (cohort/signal_id pairs where signal is CONSTANT)
    skip_list = []
    for cohort, signals in cohorts_config.items():
        for signal_id, cfg in signals.items():
            if cfg.get('typology', {}).get('continuity') == 'CONSTANT':
                skip_list.append(f"{cohort}/{signal_id}")

    manifest = {
        'version': '2.1',
        'job_id': job_name or f"orthon-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        'created_at': datetime.now().isoformat(),
        'generator': 'orthon.manifest_generator v2.1 (nested by cohort)',

        'paths': {
            'observations': str(observations_path),
            'typology': str(typology_path),
            'output_dir': params.get('output_dir', 'output/'),
        },

        'summary': {
            'total_signals': len(df),
            'total_cohorts': total_cohorts,
            'active_signals': active_count,
            'constant_signals': constant_count,
            'signal_engines': sorted(all_signal_engines),
            'rolling_engines': sorted(all_rolling_engines),
            'pair_engines': sorted(PAIR_ENGINES),
            'symmetric_pair_engines': sorted(SYMMETRIC_PAIR_ENGINES),
            'n_signal_engines': len(all_signal_engines),
            'n_rolling_engines': len(all_rolling_engines),
        },

        'params': {
            'default_window': params.get('default_window', 128),
            'default_stride': params.get('default_stride', 64),  # Per-signal strides override this
            'min_samples': params.get('min_samples', 64),
            'note': 'stride is computed per-signal from stationarity/temporal pattern',
        },

        # Per-cohort, per-signal configuration (nested structure)
        # cohorts.engine_1.sensor_01: {...}
        'cohorts': cohorts_config,

        # Pair engines run on all non-constant signal combinations
        'pair_engines': sorted(PAIR_ENGINES),
        'symmetric_pair_engines': sorted(SYMMETRIC_PAIR_ENGINES),

        # Skip list for quick filtering (cohort/signal_id format)
        'skip_signals': sorted(skip_list),
    }

    # ---- Write ----
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.dump(manifest, default_flow_style=False, sort_keys=False))

    if verbose:
        print()
        print(f"  Manifest: {output_path}")
        print(f"  Cohorts:  {total_cohorts}")
        print(f"  Active:   {active_count} signals")
        print(f"  Skipped:  {constant_count} constant signals")
        print(f"  Signal engines:  {sorted(all_signal_engines)}")
        print(f"  Rolling engines: {sorted(all_rolling_engines)}")
        print("=" * 60)

    return manifest


# ============================================================
# CLI
# ============================================================

def main():
    import sys

    usage = """
ORTHON Manifest Generator v2
============================
10-dimension typology → PRISM engine manifest

Usage:
    python -m orthon.manifest_generator <typology.parquet> [manifest.yaml]

The typology.parquet must contain columns for all 10 dimensions:
    continuity, stationarity, temporal_pattern, memory, complexity,
    distribution, amplitude, spectral, volatility, determinism

Plus raw values: acf_half_life, hurst, perm_entropy (for window/budget computation)

Examples:
    python -m orthon.manifest_generator data/typology.parquet
    python -m orthon.manifest_generator data/typology.parquet output/manifest.yaml
"""

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    typology_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "manifest.yaml"

    generate_manifest(typology_path, output_path)


if __name__ == "__main__":
    main()
