"""
ORTHON Manifest Generator v2

Creates the complete order for PRISM from 10-dimension typology results.
This is where engine selection decisions are made.

ORTHON decides. PRISM executes.

Changes from v1:
    - Reads from 10-dimension typology instead of 4-category approximation
    - Adds rules for memory, complexity, spectral, volatility, determinism
    - Window size from ACF half-life instead of heuristics
    - Eigenvalue budget from complexity dimension
    - Derivative depth from stationarity + temporal pattern

Usage:
    python -m orthon.manifest_generator data/typology.parquet data/manifest.yaml
"""

import yaml
import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


# ============================================================
# ENGINE SELECTION RULES — 10-DIMENSION MAPPING
# ============================================================

ENGINE_RULES = {
    # Always run these (scale-invariant core)
    'core': ['kurtosis', 'skewness', 'crest_factor'],

    # --- Dimension 2: STATIONARITY ---
    'stationarity': {
        'STATIONARY': [],
        'TREND_STATIONARY': ['rolling_kurtosis', 'rolling_entropy'],
        'DIFFERENCE_STATIONARY': ['rolling_kurtosis', 'rolling_entropy', 'rolling_crest_factor'],
        'NON_STATIONARY': ['rolling_kurtosis', 'rolling_entropy', 'rolling_crest_factor', 'rolling_skewness'],
    },

    # --- Dimension 3: TEMPORAL PATTERN ---
    'temporal_pattern': {
        'PERIODIC': ['harmonics_ratio', 'band_ratios', 'spectral_entropy', 'thd'],
        'QUASI_PERIODIC': ['band_ratios', 'spectral_entropy'],
        'TRENDING': ['hurst', 'rate_of_change_ratio'],
        'MEAN_REVERTING': ['hurst'],
        'CHAOTIC': ['lyapunov', 'sample_entropy'],
        'RANDOM': ['entropy', 'permutation_entropy'],
    },

    # --- Dimension 4: MEMORY ---
    'memory': {
        'LONG_MEMORY': ['hurst', 'rate_of_change_ratio'],
        'SHORT_MEMORY': [],
        'ANTI_PERSISTENT': ['hurst'],
    },

    # --- Dimension 5: COMPLEXITY ---
    'complexity': {
        'LOW': [],           # Core engines sufficient
        'MEDIUM': ['entropy', 'sample_entropy'],
        'HIGH': ['entropy', 'sample_entropy', 'permutation_entropy'],
    },

    # --- Dimension 6: DISTRIBUTION SHAPE ---
    'distribution': {
        'GAUSSIAN': [],
        'HEAVY_TAILED': ['crest_factor', 'peak_ratio'],
        'LIGHT_TAILED': [],
        'SKEWED_RIGHT': [],
        'SKEWED_LEFT': [],
    },

    # --- Dimension 7: AMPLITUDE CHARACTER ---
    'amplitude': {
        'SMOOTH': ['rolling_kurtosis', 'rolling_crest_factor'],
        'NOISY': ['entropy', 'sample_entropy'],
        'IMPULSIVE': ['crest_factor', 'peak_ratio'],
        'MIXED': ['entropy', 'crest_factor'],
    },

    # --- Dimension 8: SPECTRAL CHARACTER ---
    'spectral': {
        'NARROWBAND': ['spectral_entropy'],
        'BROADBAND': ['spectral_entropy', 'band_ratios'],
        'HARMONIC': ['harmonics_ratio', 'thd'],
        'ONE_OVER_F': ['hurst'],
    },

    # --- Dimension 9: VOLATILITY ---
    'volatility': {
        'HOMOSCEDASTIC': [],
        'HETEROSCEDASTIC': ['rolling_kurtosis', 'rolling_entropy'],
        'VOLATILITY_CLUSTERING': ['garch', 'rolling_kurtosis', 'rolling_entropy'],
    },

    # --- Dimension 10: DETERMINISM ---
    'determinism': {
        'DETERMINISTIC': ['lyapunov'],
        'STOCHASTIC': ['entropy', 'permutation_entropy'],
        'MIXED': ['sample_entropy'],
    },

    # Pair engines (always run on non-constant pairs)
    'pair': ['granger', 'transfer_entropy'],

    # Symmetric pair engines
    'symmetric_pair': ['correlation', 'mutual_info', 'cointegration'],

    # DEPRECATED - never use these (absolute values, not scale-invariant)
    'deprecated': [
        'rms', 'peak', 'total_power', 'mean', 'std',
        'rolling_rms', 'rolling_mean', 'rolling_std', 'rolling_range',
        'envelope', 'harmonic_2x', 'harmonic_3x',
        'band_low', 'band_mid', 'band_high',
    ],
}

# The 10 dimension keys that map to ENGINE_RULES
DIMENSION_KEYS = [
    'stationarity',
    'temporal_pattern',
    'memory',
    'complexity',
    'distribution',
    'amplitude',
    'spectral',
    'volatility',
    'determinism',
]


# ============================================================
# ENGINE SELECTION LOGIC
# ============================================================

def select_engines_for_signal(typology_row: Dict[str, Any]) -> List[str]:
    """
    Select engines for a single signal based on its 10-dimension typology.

    This is THE decision function. Lives in ORTHON, not PRISM.

    Args:
        typology_row: Dict with all 10 dimension values

    Returns:
        Sorted list of engine names to run
    """
    # Skip constant signals
    continuity = typology_row.get('continuity', 'CONTINUOUS')
    if continuity == 'CONSTANT':
        return []

    engines = set(ENGINE_RULES['core'])

    # Walk each dimension and add its engines
    for dim_key in DIMENSION_KEYS:
        dim_value = typology_row.get(dim_key)
        if dim_value and dim_key in ENGINE_RULES:
            dim_engines = ENGINE_RULES[dim_key].get(dim_value, [])
            engines.update(dim_engines)

    # EVENT/DISCRETE signals: remove spectral/derivative engines
    if continuity in ('EVENT', 'DISCRETE'):
        spectral_engines = {'harmonics_ratio', 'band_ratios', 'spectral_entropy', 'thd'}
        engines -= spectral_engines

    # Remove deprecated engines
    engines -= set(ENGINE_RULES['deprecated'])

    return sorted(list(engines))


def compute_recommended_window(typology_row: Dict[str, Any], default: int = 128) -> int:
    """
    Compute recommended window size from ACF half-life and memory.

    Rule: window_size = max(64, 4 × acf_half_life)
    Fallback to memory-based heuristic if acf_half_life unavailable.

    Args:
        typology_row: Dict with typology characteristics

    Returns:
        Recommended window size
    """
    acf_half_life = typology_row.get('acf_half_life')
    if acf_half_life is not None and acf_half_life > 16:
        return max(64, int(4 * acf_half_life))

    memory = typology_row.get('memory', 'SHORT_MEMORY')
    if memory == 'LONG_MEMORY':
        return 256
    elif memory == 'ANTI_PERSISTENT':
        return 64
    else:
        return default


def compute_derivative_depth(typology_row: Dict[str, Any]) -> int:
    """
    Determine derivative depth from stationarity and temporal pattern.

    Returns:
        0 for CONSTANT, 1 for STATIONARY, 2 for NON_STATIONARY/TRENDING
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
    Determine eigenvalue budget from complexity dimension.

    Returns:
        0 for CONSTANT, 3 for LOW, 5 for MEDIUM, 8 for HIGH
    """
    if typology_row.get('continuity') == 'CONSTANT':
        return 0

    complexity = typology_row.get('complexity', 'MEDIUM')
    if complexity == 'LOW':
        return 3
    elif complexity == 'HIGH':
        return 8
    return 5


# ============================================================
# MANIFEST GENERATION
# ============================================================

def generate_manifest(
    typology_path: str,
    output_path: str = "manifest.yaml",
    observations_path: str = "observations.parquet",
    job_name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate complete manifest from 10-dimension typology.

    Args:
        typology_path: Path to typology.parquet
        output_path: Where to write manifest.yaml
        observations_path: Path to observations.parquet
        job_name: Optional job name
        params: Optional override parameters
        verbose: Print progress

    Returns:
        Generated manifest dict
    """
    params = params or {}

    # Read typology
    df = pl.read_parquet(typology_path)

    if verbose:
        print(f"ORTHON Manifest Generator v2")
        print(f"  Typology: {typology_path}")
        print(f"  Signals: {len(df)}")

    # Generate per-signal configuration
    signals_config = {}
    all_engines = set()

    for row in df.iter_rows(named=True):
        signal_id = row['signal_id']
        engines = select_engines_for_signal(row)
        window = compute_recommended_window(row)
        depth = compute_derivative_depth(row)
        eigen_budget = compute_eigenvalue_budget(row)

        signals_config[signal_id] = {
            'engines': engines,
            'window_size': window,
            'derivative_depth': depth,
            'eigenvalue_budget': eigen_budget,
            'continuity': row.get('continuity', 'CONTINUOUS'),
            'stationarity': row.get('stationarity', 'STATIONARY'),
            'temporal_pattern': row.get('temporal_pattern', 'RANDOM'),
        }
        all_engines.update(engines)

        if verbose:
            continuity = row.get('continuity', '?')
            if continuity == 'CONSTANT':
                print(f"    {signal_id}: CONSTANT (skipped)")
            else:
                print(f"    {signal_id}: {len(engines)} engines, w={window}, d={depth}, eig={eigen_budget}")

    # Build manifest
    manifest = {
        'job_id': job_name or f"orthon-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        'created_at': datetime.now().isoformat(),
        'generator': 'orthon.manifest_generator_v2',
        'observations_path': str(observations_path),
        'typology_path': str(typology_path),

        # Global summary
        'summary': {
            'total_signals': len(df),
            'constant_signals': sum(1 for s in signals_config.values() if s['continuity'] == 'CONSTANT'),
            'active_signals': sum(1 for s in signals_config.values() if s['continuity'] != 'CONSTANT'),
            'unique_engines': sorted(list(all_engines)),
            'n_unique_engines': len(all_engines),
        },

        # Per-signal configuration
        'signals': signals_config,
    }

    # Write
    output = Path(output_path)
    output.write_text(yaml.dump(manifest, default_flow_style=False, sort_keys=False))

    if verbose:
        print(f"\n  Manifest written: {output_path}")
        print(f"  Active signals: {manifest['summary']['active_signals']}")
        print(f"  Unique engines: {manifest['summary']['n_unique_engines']}")

    return manifest


# CLI
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("ORTHON Manifest Generator v2")
        print("=" * 40)
        print("\nUsage:")
        print("  python -m orthon.manifest_generator_v2 <typology.parquet> [manifest.yaml]")
        sys.exit(1)

    typology_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "manifest.yaml"

    generate_manifest(typology_path, output_path)
