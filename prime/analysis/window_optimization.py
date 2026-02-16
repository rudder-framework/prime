#!/usr/bin/env python3
"""
Prime Analysis: Window Optimization

THE missing script. Answers: what (window_size, stride) combination
maximizes the predictive power of geometric features?

Key hypothesis from prior work:
    Non-overlapping windows (stride = window_size) outperform overlapping
    because eigendecomposition requires independent observations.

This script PROVES it by sweeping a grid and measuring the 20/20 correlation
(early eff_dim vs lifecycle length) at each operating point.

Method:
    For each (window_size, stride) combination:
      1. For each cohort (engine): pivot observations -> wide matrix
      2. Slide windows -> compute eigenvalues per window
      3. Extract eff_dim trajectory per cohort
      4. Run 20/20 analysis (correlate early eff_dim with lifecycle)
      5. Also run eigenvalue_window_check for stability validation

    Reports: optimal (window, stride), heat map of correlations,
    proof that stride=window beats stride=window//2

Does NOT use the full Manifold pipeline -- computes eigendecomp directly
for speed. A full grid sweep runs in minutes, not hours.

Input:
    observations.parquet  -- canonical schema (cohort, signal_id, signal_0, value)
    typology.parquet      -- for constant signal filtering

Output:
    window_optimization.parquet  -- full grid results
    Console report with optimal parameters

Usage:
    python -m prime.analysis.window_optimization \
        --observations path/to/observations.parquet \
        --typology path/to/typology.parquet \
        --output path/to/window_optimization.parquet
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# FAST EIGENDECOMP (bypasses Manifold for speed)
# ============================================================

def fast_eigendecomp(matrix: np.ndarray) -> dict:
    """
    Minimal eigendecomposition. No frills -- just the numbers.

    Args:
        matrix: N_signals x D_features (signals as rows, features as columns)
                OR for our case: each row is one signal's value at this window

    Returns:
        Dict with effective_dim, total_variance, eigenvalues
    """
    n_rows, n_cols = matrix.shape

    if n_rows < 2 or n_cols < 2:
        return {'effective_dim': np.nan, 'total_variance': np.nan, 'eigenvalues': []}

    # Remove constant columns
    col_std = np.std(matrix, axis=0)
    valid_cols = col_std > 1e-10
    if valid_cols.sum() < 2:
        return {'effective_dim': np.nan, 'total_variance': np.nan, 'eigenvalues': []}

    matrix = matrix[:, valid_cols]

    # Center
    centered = matrix - np.mean(matrix, axis=0)

    # Covariance
    n = centered.shape[0]
    cov = (centered.T @ centered) / max(n - 1, 1)

    # Eigenvalues (symmetric matrix -> real eigenvalues)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]  # descending
    eigenvalues = np.maximum(eigenvalues, 0)  # clip numerical negatives

    total_variance = float(np.sum(eigenvalues))

    if total_variance < 1e-15:
        return {'effective_dim': np.nan, 'total_variance': 0.0, 'eigenvalues': eigenvalues.tolist()}

    # Effective dimension (participation ratio)
    p = eigenvalues / total_variance
    p = p[p > 1e-15]  # filter near-zero
    entropy = -np.sum(p * np.log(p))
    effective_dim = float(np.exp(entropy))

    return {
        'effective_dim': effective_dim,
        'total_variance': total_variance,
        'eigenvalues': eigenvalues.tolist(),
    }


# ============================================================
# EIGENVALUE WINDOW CHECK (from window_recommender.py logic)
# ============================================================

def eigenvalue_stability_check(eigenvalues_by_window: List[List[float]], threshold: float = 0.3) -> dict:
    """
    Check eigenvalue proportion stability across windows.

    Unstable proportions = window too small (each window sees different structure).
    Over-dominant first eigenvalue = window too large (averaged out).
    """
    if len(eigenvalues_by_window) < 3:
        return {'stable': True, 'recommendation': 'insufficient_windows', 'variation': 0.0}

    proportions = []
    for eigs in eigenvalues_by_window:
        eigs = [max(0, e) for e in eigs]
        total = sum(eigs)
        if total > 0:
            proportions.append([e / total for e in eigs])

    if len(proportions) < 3:
        return {'stable': True, 'recommendation': 'insufficient_data', 'variation': 0.0}

    first_eig_props = [p[0] for p in proportions if len(p) > 0]
    if not first_eig_props:
        return {'stable': True, 'recommendation': 'no_eigenvalues', 'variation': 0.0}

    mean_prop = np.mean(first_eig_props)
    variation = float(max(abs(p - mean_prop) for p in first_eig_props))

    if mean_prop > 0.95:
        return {'stable': True, 'recommendation': 'window_too_large', 'variation': variation}

    if variation > threshold:
        return {'stable': False, 'recommendation': 'window_too_small', 'variation': variation}

    return {'stable': True, 'recommendation': 'window_ok', 'variation': variation}


# ============================================================
# CORE: COMPUTE EFF_DIM TRAJECTORY AT GIVEN WINDOW/STRIDE
# ============================================================

def compute_effdim_trajectories(
    observations: pl.DataFrame,
    active_signals: List[str],
    window_size: int,
    stride: int,
) -> Dict[str, List[dict]]:
    """
    For each cohort, pivot to wide, slide windows, eigendecomp.

    Returns:
        Dict mapping cohort -> list of {signal_0, effective_dim, total_variance}
    """
    cohorts = sorted(observations['cohort'].unique().to_list())
    trajectories = {}

    for cohort in cohorts:
        cohort_obs = observations.filter(pl.col('cohort') == cohort)

        # Pivot: rows=signal_0, columns=signal_id, values=value
        wide = (
            cohort_obs
            .filter(pl.col('signal_id').is_in(active_signals))
            .pivot(
                on='signal_id',
                index='signal_0',
                values='value',
            )
            .sort('signal_0')
        )

        signal_0_values = wide['signal_0'].to_numpy()
        signal_cols = [c for c in wide.columns if c != 'signal_0']
        matrix = wide.select(signal_cols).to_numpy()

        n_rows = len(signal_0_values)

        if n_rows < window_size:
            continue

        windows = []
        eigenvalues_for_stability = []

        # Slide windows
        for win_end in range(window_size - 1, n_rows, stride):
            win_start = win_end - window_size + 1

            # Window matrix: each row = one time step, each col = one signal
            # We want eigendecomp of signals (signals as features observed at time steps)
            window_matrix = matrix[win_start:win_end + 1, :]

            # Remove rows/cols with NaN
            valid_rows = ~np.any(np.isnan(window_matrix), axis=1)
            window_clean = window_matrix[valid_rows]

            if len(window_clean) < 3:
                continue

            result = fast_eigendecomp(window_clean)

            if not np.isnan(result['effective_dim']):
                windows.append({
                    'signal_0': int(signal_0_values[win_end]),
                    'effective_dim': result['effective_dim'],
                    'total_variance': result['total_variance'],
                })
                eigenvalues_for_stability.append(result['eigenvalues'][:5])

        if len(windows) >= 3:
            trajectories[cohort] = {
                'windows': windows,
                'stability': eigenvalue_stability_check(eigenvalues_for_stability),
            }

    return trajectories


# ============================================================
# 20/20 ANALYSIS (inline for speed -- no file I/O)
# ============================================================

def fast_twenty_twenty(trajectories: Dict, lifecycles: Dict[str, int], early_pct: float = 0.20) -> dict:
    """
    Fast 20/20 correlation from in-memory trajectories.
    """
    shared_cohorts = sorted(set(trajectories.keys()) & set(lifecycles.keys()))

    if len(shared_cohorts) < 10:
        return {
            'r': np.nan, 'p': np.nan, 'n': len(shared_cohorts),
            'r_delta': np.nan, 'rho': np.nan,
        }

    early_dims = []
    deltas = []
    lives = []

    for cohort in shared_cohorts:
        windows = trajectories[cohort]['windows']
        n = len(windows)
        n_early = max(1, int(n * early_pct))
        n_late = max(1, int(n * early_pct))

        eff_dims = [w['effective_dim'] for w in windows]

        early_mean = np.mean(eff_dims[:n_early])
        late_mean = np.mean(eff_dims[-n_late:])

        early_dims.append(early_mean)
        deltas.append(early_mean - late_mean)
        lives.append(lifecycles[cohort])

    early_arr = np.array(early_dims)
    delta_arr = np.array(deltas)
    life_arr = np.array(lives)

    r, p = stats.pearsonr(early_arr, life_arr)
    r_delta, p_delta = stats.pearsonr(delta_arr, life_arr)
    rho, p_rho = stats.spearmanr(early_arr, life_arr)

    # Stability summary
    n_stable = sum(1 for c in shared_cohorts if trajectories[c]['stability']['recommendation'] == 'window_ok')
    n_too_small = sum(1 for c in shared_cohorts if trajectories[c]['stability']['recommendation'] == 'window_too_small')
    n_too_large = sum(1 for c in shared_cohorts if trajectories[c]['stability']['recommendation'] == 'window_too_large')

    return {
        'r': float(r),
        'p': float(p),
        'r_squared': float(r ** 2),
        'r_delta': float(r_delta),
        'rho': float(rho),
        'n': len(shared_cohorts),
        'n_stable': n_stable,
        'n_too_small': n_too_small,
        'n_too_large': n_too_large,
    }


# ============================================================
# GRID SWEEP
# ============================================================

DEFAULT_WINDOWS = [8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]
DEFAULT_OVERLAP_MODES = [
    ('non_overlapping', 1.0),   # stride = window (THE HYPOTHESIS)
    ('half_overlap', 0.5),      # stride = window // 2
    ('quarter_overlap', 0.25),  # stride = window // 4
]


def run_grid_sweep(
    observations: pl.DataFrame,
    active_signals: List[str],
    lifecycles: Dict[str, int],
    window_sizes: List[int] = None,
    overlap_modes: List[Tuple[str, float]] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Sweep (window_size, stride) grid. Returns results DataFrame.
    """
    if window_sizes is None:
        window_sizes = DEFAULT_WINDOWS
    if overlap_modes is None:
        overlap_modes = DEFAULT_OVERLAP_MODES

    total = len(window_sizes) * len(overlap_modes)

    if verbose:
        print(f"\n  Grid: {len(window_sizes)} windows x {len(overlap_modes)} overlap modes = {total} combos")
        print(f"  Windows: {window_sizes}")
        print(f"  Modes: {[m[0] for m in overlap_modes]}")
        print()

    rows = []
    best_r = -1
    best_combo = None

    for i, window_size in enumerate(window_sizes):
        for mode_name, overlap_ratio in overlap_modes:
            stride = max(1, int(window_size * (1 - overlap_ratio)))
            if mode_name == 'non_overlapping':
                stride = window_size  # exact

            t0 = time.time()

            # Compute trajectories
            trajectories = compute_effdim_trajectories(
                observations, active_signals, window_size, stride
            )

            # Run 20/20
            result = fast_twenty_twenty(trajectories, lifecycles)

            elapsed = time.time() - t0

            # Average windows per cohort
            avg_windows = np.mean([len(t['windows']) for t in trajectories.values()]) if trajectories else 0

            row = {
                'window_size': window_size,
                'stride': stride,
                'overlap_mode': mode_name,
                'overlap_pct': overlap_ratio * 100,
                'r': result['r'],
                'p': result['p'],
                'r_squared': result['r_squared'],
                'r_delta': result['r_delta'],
                'rho': result['rho'],
                'n_cohorts': result['n'],
                'n_stable': result.get('n_stable', 0),
                'n_too_small': result.get('n_too_small', 0),
                'n_too_large': result.get('n_too_large', 0),
                'avg_windows_per_cohort': float(avg_windows),
                'elapsed_sec': float(elapsed),
            }
            rows.append(row)

            if abs(result['r']) > abs(best_r):
                best_r = result['r']
                best_combo = (window_size, stride, mode_name)

            if verbose:
                stable_str = f"stable={result.get('n_stable', '?')}"
                print(f"  w={window_size:>3d} s={stride:>3d} ({mode_name:>17s})  "
                      f"r={result['r']:+.3f}  R²={result['r_squared']:.3f}  "
                      f"rho={result['rho']:+.3f}  "
                      f"n={result['n']:>3d}  {stable_str}  "
                      f"[{elapsed:.1f}s]")

    return pl.DataFrame(rows)


# ============================================================
# ANALYSIS AND REPORTING
# ============================================================

def analyze_results(results: pl.DataFrame) -> dict:
    """Analyze grid sweep results for paper-ready insights."""

    # Best overall
    best_idx = results['r_squared'].arg_max()
    best = results.row(best_idx, named=True)

    # Best per overlap mode
    best_per_mode = {}
    for mode in results['overlap_mode'].unique().to_list():
        mode_data = results.filter(pl.col('overlap_mode') == mode)
        idx = mode_data['r_squared'].arg_max()
        best_per_mode[mode] = mode_data.row(idx, named=True)

    # Non-overlapping vs overlapping comparison at each window size
    comparisons = []
    for ws in results['window_size'].unique().sort().to_list():
        ws_data = results.filter(pl.col('window_size') == ws)
        non_overlap = ws_data.filter(pl.col('overlap_mode') == 'non_overlapping')
        half_overlap = ws_data.filter(pl.col('overlap_mode') == 'half_overlap')

        if len(non_overlap) > 0 and len(half_overlap) > 0:
            comparisons.append({
                'window_size': ws,
                'r_non_overlapping': float(non_overlap['r'][0]),
                'r_half_overlap': float(half_overlap['r'][0]),
                'diff': float(non_overlap['r'][0]) - float(half_overlap['r'][0]),
                'non_overlap_wins': float(abs(non_overlap['r'][0])) > float(abs(half_overlap['r'][0])),
            })

    comparison_df = pl.DataFrame(comparisons) if comparisons else pl.DataFrame()

    # How often does non-overlapping win?
    if len(comparison_df) > 0:
        n_wins = int(comparison_df['non_overlap_wins'].sum())
        n_total = len(comparison_df)
    else:
        n_wins, n_total = 0, 0

    return {
        'best': best,
        'best_per_mode': best_per_mode,
        'comparison': comparison_df,
        'non_overlap_win_rate': n_wins / max(n_total, 1),
        'n_comparisons': n_total,
    }


def print_report(results: pl.DataFrame, analysis: dict):
    """Print paper-ready optimization report."""

    print()
    print("=" * 78)
    print("  WINDOW OPTIMIZATION REPORT")
    print("=" * 78)

    best = analysis['best']

    print()
    print("-- Optimal Operating Point --")
    print()
    print(f"  Window size: {best['window_size']}")
    print(f"  Stride:      {best['stride']}")
    print(f"  Mode:        {best['overlap_mode']}")
    print(f"  Pearson r:   {best['r']:.4f}")
    print(f"  R-squared:   {best['r_squared']:.4f}")
    print(f"  Spearman rho:{best['rho']:.4f}")
    print(f"  p-value:     {best['p']:.2e}")
    print(f"  Cohorts:     {best['n_cohorts']}")
    print(f"  Eigenvalue stability: {best['n_stable']} ok, {best['n_too_small']} too_small, {best['n_too_large']} too_large")

    # Best per mode
    print()
    print("-- Best Per Overlap Mode --")
    print()
    for mode, row in analysis['best_per_mode'].items():
        print(f"  {mode:>20s}:  w={row['window_size']:>3d}  s={row['stride']:>3d}  "
              f"r={row['r']:+.4f}  R-squared={row['r_squared']:.4f}")

    # Non-overlapping vs overlapping
    print()
    print("-- Non-Overlapping vs 50% Overlap (per window size) --")
    print()

    comp = analysis['comparison']
    if len(comp) > 0:
        for row in comp.iter_rows(named=True):
            winner = "NON-OVERLAP" if row['non_overlap_wins'] else "  overlap"
            print(f"  w={row['window_size']:>3d}:  non-overlap r={row['r_non_overlapping']:+.4f}  "
                  f"overlap r={row['r_half_overlap']:+.4f}  "
                  f"delta={row['diff']:+.4f}  {winner}")

        win_rate = analysis['non_overlap_win_rate']
        print()
        print(f"  Non-overlapping wins: {analysis['non_overlap_win_rate']*100:.0f}% "
              f"({int(win_rate * analysis['n_comparisons'])}/{analysis['n_comparisons']})")

    # Full grid sorted by R-squared
    print()
    print("-- Full Grid (sorted by R-squared) --")
    print()
    sorted_results = results.sort('r_squared', descending=True)
    for row in sorted_results.head(15).iter_rows(named=True):
        bar_len = int(row['r_squared'] * 40)
        bar = "#" * bar_len
        print(f"  w={row['window_size']:>3d} s={row['stride']:>3d} ({row['overlap_mode']:>17s})  "
              f"R-squared={row['r_squared']:.4f}  {bar}")

    # Publishable claim
    print()
    print("-- Publishable Claim --")
    print()

    best_no = analysis['best_per_mode'].get('non_overlapping', {})
    best_ho = analysis['best_per_mode'].get('half_overlap', {})

    if best_no and best_ho:
        r_no = best_no.get('r_squared', 0)
        r_ho = best_ho.get('r_squared', 0)

        if r_no > r_ho:
            improvement = (r_no - r_ho) / r_ho * 100 if r_ho > 0 else float('inf')
            print(f"  Non-overlapping windows (stride = window_size) improve predictive")
            print(f"  accuracy by {improvement:.1f}% over 50% overlapping windows")
            print(f"  (R-squared = {r_no:.4f} vs {r_ho:.4f}), confirming that eigendecomposition")
            print(f"  requires independent observations for reliable geometry estimation.")
        else:
            print(f"  Overlapping windows slightly outperform non-overlapping")
            print(f"  (R-squared = {r_ho:.4f} vs {r_no:.4f}) for this dataset.")
            print(f"  This may indicate the signal dynamics change faster than one window.")

    mode_desc = 'non-overlapping' if best['overlap_mode'] == 'non_overlapping' else f"{best['overlap_pct']:.0f}% overlap"
    print(f"\n  Optimal window: {best['window_size']} samples ({mode_desc})")

    print()
    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Prime Window Optimization')
    parser.add_argument('--observations', required=True, help='Path to observations.parquet')
    parser.add_argument('--typology', default=None, help='Path to typology.parquet (for constant filtering)')
    parser.add_argument('--output', default='window_optimization.parquet', help='Output path')
    parser.add_argument('--windows', default=None, help='Comma-separated window sizes (default: auto)')
    parser.add_argument('--subsample', type=int, default=None, help='Use N random cohorts (for speed)')
    args = parser.parse_args()

    print("=" * 78)
    print("  PRIME WINDOW OPTIMIZATION")
    print("=" * 78)

    # Load observations
    print(f"\nLoading: {args.observations}")
    observations = pl.read_parquet(args.observations)
    print(f"  {observations.shape[0]:,} rows, {observations['cohort'].n_unique()} cohorts, "
          f"{observations['signal_id'].n_unique()} signals")

    # Get active signals (exclude constants)
    all_signals = sorted(observations['signal_id'].unique().to_list())

    if args.typology:
        print(f"Loading typology: {args.typology}")
        typology = pl.read_parquet(args.typology)

        if 'is_constant' in typology.columns:
            constant_sigs = (
                typology
                .filter(pl.col('is_constant') == True)
                .select('signal_id')
                .unique()
                .to_series()
                .to_list()
            )
        elif 'temporal_primary' in typology.columns:
            constant_sigs = (
                typology
                .filter(pl.col('temporal_primary') == 'CONSTANT')
                .select('signal_id')
                .unique()
                .to_series()
                .to_list()
            )
        else:
            constant_sigs = []

        active_signals = [s for s in all_signals if s not in constant_sigs]
        print(f"  Active signals: {len(active_signals)} (excluded {len(constant_sigs)} constants)")
    else:
        active_signals = all_signals
        print(f"  All {len(active_signals)} signals (no typology filter)")

    # Get lifecycles
    first_sig = observations['signal_id'].unique().sort()[0]
    lifecycle_df = (
        observations
        .filter(pl.col('signal_id') == first_sig)
        .group_by('cohort')
        .agg(pl.col('signal_0').max() + 1)
        .sort('cohort')
    )
    lifecycles = dict(zip(lifecycle_df['cohort'].to_list(), lifecycle_df['signal_0'].to_list()))
    print(f"  Lifecycles: {min(lifecycles.values())} - {max(lifecycles.values())} cycles")

    # Optional subsample for speed
    if args.subsample and args.subsample < len(lifecycles):
        import random
        random.seed(42)
        # Stratified: pick from short, medium, and long-lived
        sorted_cohorts = sorted(lifecycles.keys(), key=lambda c: lifecycles[c])
        n = args.subsample
        indices = np.linspace(0, len(sorted_cohorts) - 1, n, dtype=int)
        selected = [sorted_cohorts[i] for i in indices]
        observations = observations.filter(pl.col('cohort').is_in(selected))
        lifecycles = {c: lifecycles[c] for c in selected}
        print(f"  Subsampled to {n} cohorts (stratified by lifecycle)")

    # Window sizes
    if args.windows:
        window_sizes = [int(x) for x in args.windows.split(',')]
    else:
        # Auto: scale to dataset
        min_life = min(lifecycles.values())
        # Don't exceed 1/3 of shortest lifecycle
        max_window = min_life // 3
        window_sizes = [w for w in DEFAULT_WINDOWS if w <= max_window]
        if not window_sizes:
            window_sizes = [8, 10, 15, 20]
        print(f"  Auto window range: {window_sizes} (max = {max_window} = shortest_life/3)")

    # Run grid sweep
    t0 = time.time()
    results = run_grid_sweep(observations, active_signals, lifecycles, window_sizes)
    total_time = time.time() - t0

    print(f"\n  Total sweep time: {total_time:.1f}s")

    # Analyze
    analysis = analyze_results(results)

    # Report
    print_report(results, analysis)

    # Save
    results.write_parquet(args.output)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
