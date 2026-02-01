"""
ORTHON Manifest Generator

Creates the complete order for PRISM from typology results.
This is where engine selection decisions are made.

ORTHON decides. PRISM executes.

Usage:
    python -m orthon.manifest_generator data/typology.parquet data/manifest.yaml
"""

import yaml
import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


# ============================================================
# ENGINE SELECTION RULES
# ============================================================

ENGINE_RULES = {
    # Always run these (scale-invariant core)
    'core': ['kurtosis', 'skewness', 'crest_factor'],

    # By signal type
    'signal_type': {
        'SMOOTH': ['rolling_kurtosis', 'rolling_entropy', 'rolling_crest_factor'],
        'NOISY': ['entropy', 'sample_entropy'],
        'IMPULSIVE': ['crest_factor', 'peak_ratio'],
        'MIXED': ['entropy', 'crest_factor', 'sample_entropy'],
    },

    # By periodicity
    'periodicity': {
        'PERIODIC': ['harmonics_ratio', 'band_ratios', 'spectral_entropy', 'thd'],
        'QUASI_PERIODIC': ['band_ratios', 'spectral_entropy'],
        'APERIODIC': ['entropy', 'hurst', 'sample_entropy'],
    },

    # By tail behavior
    'tail_type': {
        'HEAVY_TAILS': ['kurtosis', 'crest_factor'],
        'LIGHT_TAILS': ['entropy', 'sample_entropy'],
        'NORMAL_TAILS': [],
    },

    # By stationarity
    'stationarity': {
        'STATIONARY': [],
        'NON_STATIONARY': ['rolling_kurtosis', 'rolling_entropy', 'rolling_crest_factor'],
        'VARIANCE_INCREASING': ['rolling_kurtosis', 'rolling_entropy'],
        'VARIANCE_DECREASING': ['rolling_kurtosis', 'rolling_entropy'],
    },

    # By memory (Hurst-like)
    'memory': {
        'TRENDING': ['hurst', 'rate_of_change_ratio'],
        'REVERTING': ['hurst'],
        'RANDOM': [],
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


# ============================================================
# ENGINE SELECTION LOGIC
# ============================================================

def select_engines_for_signal(typology_row: Dict[str, Any]) -> List[str]:
    """
    Select engines for a single signal based on its typology.

    This is THE decision function. Lives in ORTHON, not PRISM.

    Args:
        typology_row: Dict with typology characteristics

    Returns:
        Sorted list of engine names to run
    """
    # Skip constant signals
    if typology_row.get('is_constant') or typology_row.get('signal_type') == 'CONSTANT':
        return []

    engines = set(ENGINE_RULES['core'])

    # By signal type
    sig_type = typology_row.get('signal_type', 'MIXED')
    engines.update(ENGINE_RULES['signal_type'].get(sig_type, []))

    # By periodicity
    periodicity = typology_row.get('periodicity_type', 'APERIODIC')
    engines.update(ENGINE_RULES['periodicity'].get(periodicity, []))

    # By tail behavior
    tail_type = typology_row.get('tail_type', 'NORMAL_TAILS')
    engines.update(ENGINE_RULES['tail_type'].get(tail_type, []))

    # By stationarity
    stationarity = typology_row.get('stationarity_type', 'STATIONARY')
    engines.update(ENGINE_RULES['stationarity'].get(stationarity, []))

    # By memory proxy (Hurst-like interpretation)
    memory = typology_row.get('memory_proxy')
    if memory is not None:
        if memory < 0.45:
            engines.update(ENGINE_RULES['memory'].get('REVERTING', []))
        elif memory > 0.55:
            engines.update(ENGINE_RULES['memory'].get('TRENDING', []))

    # Remove deprecated engines
    engines -= set(ENGINE_RULES['deprecated'])

    return sorted(list(engines))


def compute_recommended_window(typology_row: Dict[str, Any], default: int = 50) -> int:
    """
    Compute recommended window size based on signal characteristics.

    Args:
        typology_row: Dict with typology characteristics
        default: Default window size

    Returns:
        Recommended window size
    """
    # Start with default
    window = default

    # Noisy signals need larger windows
    if typology_row.get('signal_type') == 'NOISY':
        window = max(window, 100)

    # Smooth signals can use smaller windows
    if typology_row.get('signal_type') == 'SMOOTH':
        smoothness = typology_row.get('smoothness', 0.5)
        if smoothness > 0.8:
            window = min(window, 30)

    # Non-stationary needs rolling windows
    if typology_row.get('stationarity_type') in ['NON_STATIONARY', 'VARIANCE_INCREASING', 'VARIANCE_DECREASING']:
        window = max(window, 50)

    return window


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
    Generate complete manifest from typology.

    The manifest is the complete order for PRISM:
    - Which signals to process
    - Which engines to run for each signal
    - What parameters to use

    Args:
        typology_path: Path to typology.parquet
        output_path: Output manifest path
        observations_path: Path to observations.parquet (for manifest)
        job_name: Optional job name
        params: Optional parameter overrides
        verbose: Print progress

    Returns:
        Complete manifest dict
    """
    typology = pl.read_parquet(typology_path)

    # Detect signal column
    signal_col = 'signal_name' if 'signal_name' in typology.columns else 'signal_id'

    # Get unique signals (typology may have multiple rows per signal if per-unit)
    if 'unit_id' in typology.columns:
        # Aggregate typology per signal (most common classification)
        signal_typology = (
            typology
            .group_by(signal_col)
            .agg([
                pl.col('signal_type').mode().first().alias('signal_type'),
                pl.col('periodicity_type').mode().first().alias('periodicity_type'),
                pl.col('tail_type').mode().first().alias('tail_type'),
                pl.col('stationarity_type').mode().first().alias('stationarity_type'),
                pl.col('smoothness').mean().alias('smoothness'),
                pl.col('memory_proxy').mean().alias('memory_proxy'),
            ])
        )
    else:
        signal_typology = typology

    # Build manifest
    manifest = {
        'version': '2.0',

        'job': {
            'id': f"prism_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'name': job_name or 'PRISM Analysis',
            'created_at': datetime.now().isoformat(),
        },

        'data': {
            'observations': observations_path,
            'typology': str(typology_path),
            'output_dir': 'output/',
        },

        'params': {
            'window_size': 50,
            'stride': 5,
            'min_samples': 20,
        },

        'signals': {},
        'skip_signals': [],
        'engines_required': {
            'signal': set(),
            'rolling': set(),
            'pair': list(ENGINE_RULES['pair']),
            'symmetric_pair': list(ENGINE_RULES['symmetric_pair']),
        },
    }

    # Override params if provided
    if params:
        manifest['params'].update(params)

    # Process each signal
    for row in signal_typology.iter_rows(named=True):
        signal = row[signal_col]
        is_constant = row.get('signal_type') == 'CONSTANT'

        if is_constant:
            manifest['skip_signals'].append(signal)
            manifest['signals'][signal] = {
                'is_constant': True,
                'engines': [],
            }
            continue

        # Select engines
        engines = select_engines_for_signal(row)

        # Compute recommended window
        rec_window = compute_recommended_window(row, manifest['params']['window_size'])

        # Categorize engines
        rolling_engines = [e for e in engines if e.startswith('rolling_')]
        signal_engines = [e for e in engines if not e.startswith('rolling_')]

        manifest['engines_required']['signal'].update(signal_engines)
        manifest['engines_required']['rolling'].update(rolling_engines)

        # Build signal entry
        manifest['signals'][signal] = {
            'is_constant': False,
            'signal_type': row.get('signal_type', 'MIXED'),
            'periodicity': row.get('periodicity_type', 'APERIODIC'),
            'tail_type': row.get('tail_type', 'NORMAL_TAILS'),
            'stationarity': row.get('stationarity_type', 'STATIONARY'),
            'smoothness': float(row.get('smoothness')) if row.get('smoothness') is not None else None,
            'memory_proxy': float(row.get('memory_proxy')) if row.get('memory_proxy') is not None else None,
            'recommended_window': rec_window,
            'engines': engines,
        }

    # Convert sets to sorted lists for YAML
    manifest['engines_required']['signal'] = sorted(list(manifest['engines_required']['signal']))
    manifest['engines_required']['rolling'] = sorted(list(manifest['engines_required']['rolling']))

    # Compute global window recommendation (minimum of all signal windows)
    active_signals = [s for s, v in manifest['signals'].items() if not v.get('is_constant')]
    if active_signals:
        windows = [manifest['signals'][s].get('recommended_window', 50) for s in active_signals]
        manifest['params']['window_size'] = min(windows)

    # Write manifest
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    if verbose:
        print("=" * 60)
        print("MANIFEST GENERATED")
        print("=" * 60)
        print(f"Output: {output_path}")
        print(f"Active signals: {len(active_signals)}")
        print(f"Skip signals: {len(manifest['skip_signals'])}")
        print(f"Signal engines: {manifest['engines_required']['signal']}")
        print(f"Rolling engines: {manifest['engines_required']['rolling']}")
        print(f"Window size: {manifest['params']['window_size']}")
        print("=" * 60)

    return manifest


# ============================================================
# CLI
# ============================================================

def main():
    import sys

    usage = """
ORTHON Manifest Generator

Creates the complete order for PRISM from typology results.

Usage:
    python -m orthon.manifest_generator <typology.parquet> [manifest.yaml]

Examples:
    python -m orthon.manifest_generator data/typology.parquet
    python -m orthon.manifest_generator data/typology.parquet data/manifest.yaml
"""

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    typology_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "manifest.yaml"

    generate_manifest(typology_path, output_path)


if __name__ == "__main__":
    main()
