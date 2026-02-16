#!/usr/bin/env python3
"""
Framework Analysis: 20/20 Early Detection

Reads Manifold's state_geometry.parquet (per-cohort eigenvalue trajectories)
and tests whether geometric deformation in the first 20% of life
predicts total lifecycle length.

This is the core publishable result.

Input:
    state_geometry.parquet  — from Manifold (cohort = engine)
    observations.parquet    — for lifecycle length per cohort

Output:
    twenty_twenty_results.parquet  — per-cohort early/late metrics + lifecycle
    Console summary with correlation statistics

Usage:
    python -m prime.analysis.twenty_twenty \
        --geometry path/to/state_geometry.parquet \
        --observations path/to/observations.parquet \
        --output path/to/twenty_twenty_results.parquet
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from scipy import stats


def get_lifecycle_per_cohort(observations: pl.DataFrame) -> pl.DataFrame:
    """
    Extract lifecycle length per cohort from observations.
    
    Lifecycle = max(signal_0) - min(signal_0) + 1 per cohort.
    Uses any single signal_id to count cycles (they all share the same signal_0 range).
    """
    # Pick one signal to count cycles (avoid double-counting)
    first_signal = observations['signal_id'].unique().sort()[0]
    
    lifecycle = (
        observations
        .filter(pl.col('signal_id') == first_signal)
        .group_by('cohort')
        .agg([
            pl.col('signal_0').min().alias('min_signal_0'),
            pl.col('signal_0').max().alias('max_signal_0'),
            pl.col('signal_0').n_unique().alias('n_cycles'),
        ])
        .with_columns(
            (pl.col('max_signal_0') - pl.col('min_signal_0') + 1).alias('lifecycle_length')
        )
        .sort('cohort')
    )
    
    return lifecycle


def run_twenty_twenty(
    geometry: pl.DataFrame,
    lifecycle: pl.DataFrame,
    engine_name: str = None,
    early_pct: float = 0.20,
    late_pct: float = 0.20,
) -> dict:
    """
    Core 20/20 analysis.
    
    For each cohort (engine):
      1. Sort windows by signal_0
      2. Take first 20% and last 20% of windows
      3. Compute mean effective_dim in each segment
      4. Correlate early effective_dim with lifecycle length
    
    Args:
        geometry: state_geometry.parquet as DataFrame
        lifecycle: lifecycle info per cohort
        engine_name: Filter to specific engine group (e.g., 'shape'). None = use all.
        early_pct: Fraction of life to call "early" (default 0.20)
        late_pct: Fraction of life to call "late" (default 0.20)
    
    Returns:
        Dict with results DataFrame and statistics
    """
    # Filter to specific engine group if requested
    if engine_name and 'engine' in geometry.columns:
        geo = geometry.filter(pl.col('engine') == engine_name)
    else:
        geo = geometry
    
    # Get cohorts that exist in both geometry and lifecycle
    geo_cohorts = set(geo['cohort'].unique().to_list())
    life_cohorts = set(lifecycle['cohort'].to_list())
    shared = sorted(geo_cohorts & life_cohorts)
    
    if len(shared) == 0:
        raise ValueError("No overlapping cohorts between geometry and lifecycle data")
    
    rows = []
    
    for cohort in shared:
        # This cohort's geometry trajectory
        cohort_geo = geo.filter(pl.col('cohort') == cohort).sort('signal_0')
        
        if len(cohort_geo) < 5:
            continue  # need enough windows to split
        
        n_windows = len(cohort_geo)
        n_early = max(1, int(n_windows * early_pct))
        n_late = max(1, int(n_windows * late_pct))
        
        eff_dims = cohort_geo['effective_dim'].to_numpy()
        total_vars = cohort_geo['total_variance'].to_numpy()
        signal_0_values = cohort_geo['signal_0'].to_numpy()
        
        # Early segment (first 20%)
        early_eff_dim = float(np.mean(eff_dims[:n_early]))
        early_total_var = float(np.mean(total_vars[:n_early]))
        
        # Late segment (last 20%)
        late_eff_dim = float(np.mean(eff_dims[-n_late:]))
        late_total_var = float(np.mean(total_vars[-n_late:]))
        
        # Delta
        delta_eff_dim = early_eff_dim - late_eff_dim
        delta_total_var = early_total_var - late_total_var
        
        # Trend (correlation of eff_dim with window index over full life)
        if len(eff_dims) >= 3:
            trend_r, trend_p = stats.pearsonr(np.arange(len(eff_dims)), eff_dims)
        else:
            trend_r, trend_p = np.nan, np.nan
        
        # Get lifecycle length
        life_row = lifecycle.filter(pl.col('cohort') == cohort)
        lifecycle_length = int(life_row['lifecycle_length'][0])
        
        # Eigenvalue concentration in early vs late
        if 'explained_1' in cohort_geo.columns:
            early_concentration = float(np.mean(
                cohort_geo['explained_1'].to_numpy()[:n_early]
            ))
            late_concentration = float(np.mean(
                cohort_geo['explained_1'].to_numpy()[-n_late:]
            ))
        else:
            early_concentration = np.nan
            late_concentration = np.nan
        
        rows.append({
            'cohort': cohort,
            'lifecycle_length': lifecycle_length,
            'n_windows': n_windows,
            'early_eff_dim': early_eff_dim,
            'late_eff_dim': late_eff_dim,
            'delta_eff_dim': delta_eff_dim,
            'early_total_var': early_total_var,
            'late_total_var': late_total_var,
            'delta_total_var': delta_total_var,
            'early_concentration': early_concentration,
            'late_concentration': late_concentration,
            'eff_dim_trend_r': float(trend_r),
            'eff_dim_trend_p': float(trend_p),
        })
    
    results = pl.DataFrame(rows)
    
    # ── Core statistics ──
    
    early = results['early_eff_dim'].to_numpy()
    lifecycle_arr = results['lifecycle_length'].to_numpy()
    delta = results['delta_eff_dim'].to_numpy()
    
    # Primary result: early eff_dim vs lifecycle length
    r_early_life, p_early_life = stats.pearsonr(early, lifecycle_arr)
    
    # Secondary: delta eff_dim vs lifecycle length  
    r_delta_life, p_delta_life = stats.pearsonr(delta, lifecycle_arr)
    
    # Spearman (rank correlation — robust to outliers)
    rho_early, p_rho_early = stats.spearmanr(early, lifecycle_arr)
    
    # R-squared
    r_squared = r_early_life ** 2
    
    # Bootstrap 95% CI for the primary correlation
    n_boot = 10000
    boot_r = np.zeros(n_boot)
    n = len(early)
    rng = np.random.default_rng(42)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_r[b] = np.corrcoef(early[idx], lifecycle_arr[idx])[0, 1]
    ci_low, ci_high = np.percentile(boot_r, [2.5, 97.5])
    
    # Collapse detection: how many cohorts show eff_dim decrease?
    n_collapsing = int(np.sum(delta > 0))  # positive delta = early > late = collapse
    
    statistics = {
        'n_cohorts': len(results),
        'r_early_lifecycle': float(r_early_life),
        'p_early_lifecycle': float(p_early_life),
        'r_squared': float(r_squared),
        'r_delta_lifecycle': float(r_delta_life),
        'p_delta_lifecycle': float(p_delta_life),
        'rho_early_lifecycle': float(rho_early),
        'p_rho_early_lifecycle': float(p_rho_early),
        'bootstrap_ci_low': float(ci_low),
        'bootstrap_ci_high': float(ci_high),
        'n_collapsing': n_collapsing,
        'pct_collapsing': float(n_collapsing / len(results) * 100),
        'engine': engine_name or 'all',
    }
    
    return {
        'results': results,
        'statistics': statistics,
    }


def print_summary(statistics: dict, results: pl.DataFrame):
    """Print human-readable summary to console."""
    
    print()
    print("=" * 70)
    print("20/20 EARLY DETECTION ANALYSIS")
    print("=" * 70)
    print()
    print(f"  Engine group:     {statistics['engine']}")
    print(f"  Cohorts analyzed: {statistics['n_cohorts']}")
    print()
    print("── Primary Result ──")
    print()
    print(f"  Pearson r (early eff_dim vs lifecycle):  {statistics['r_early_lifecycle']:.3f}")
    print(f"  p-value:                                 {statistics['p_early_lifecycle']:.2e}")
    print(f"  R²:                                      {statistics['r_squared']:.3f}")
    print(f"  95% Bootstrap CI:                        [{statistics['bootstrap_ci_low']:.3f}, {statistics['bootstrap_ci_high']:.3f}]")
    print()
    print(f"  Spearman ρ:                              {statistics['rho_early_lifecycle']:.3f}")
    print(f"  p-value:                                 {statistics['p_rho_early_lifecycle']:.2e}")
    print()
    print("── Dimensional Collapse ──")
    print()
    print(f"  Engines showing collapse (early > late): {statistics['n_collapsing']}/{statistics['n_cohorts']} ({statistics['pct_collapsing']:.1f}%)")
    print()
    print(f"  Δ eff_dim vs lifecycle r:                {statistics['r_delta_lifecycle']:.3f}")
    print()
    print("── Interpretation ──")
    print()
    
    r = statistics['r_early_lifecycle']
    if abs(r) > 0.7:
        print(f"  STRONG: Geometric state at 20% of life explains {statistics['r_squared']*100:.1f}% of lifecycle variance.")
        print(f"  Engines with lower initial eff_dim die younger.")
    elif abs(r) > 0.4:
        print(f"  MODERATE: Early geometry partially predicts lifecycle ({statistics['r_squared']*100:.1f}% variance explained).")
    else:
        print(f"  WEAK: Early geometry has limited predictive value (R²={statistics['r_squared']:.3f}).")
    
    print()
    
    # Show extremes
    sorted_df = results.sort('lifecycle_length')
    n = len(sorted_df)
    
    print("── Shortest-lived engines ──")
    for row in sorted_df.head(3).iter_rows(named=True):
        print(f"  {row['cohort']}: {row['lifecycle_length']} cycles, early eff_dim={row['early_eff_dim']:.2f}, Δ={row['delta_eff_dim']:.2f}")
    
    print()
    print("── Longest-lived engines ──")
    for row in sorted_df.tail(3).iter_rows(named=True):
        print(f"  {row['cohort']}: {row['lifecycle_length']} cycles, early eff_dim={row['early_eff_dim']:.2f}, Δ={row['delta_eff_dim']:.2f}")
    
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Framework 20/20 Early Detection Analysis')
    parser.add_argument('--geometry', required=True, help='Path to state_geometry.parquet')
    parser.add_argument('--observations', required=True, help='Path to observations.parquet')
    parser.add_argument('--output', default='twenty_twenty_results.parquet', help='Output path')
    parser.add_argument('--engine', default=None, help='Filter to specific engine group (shape, complexity, spectral)')
    parser.add_argument('--early-pct', type=float, default=0.20, help='Early life fraction (default 0.20)')
    parser.add_argument('--late-pct', type=float, default=0.20, help='Late life fraction (default 0.20)')
    args = parser.parse_args()
    
    print(f"Loading geometry: {args.geometry}")
    geometry = pl.read_parquet(args.geometry)
    
    print(f"Loading observations: {args.observations}")
    observations = pl.read_parquet(args.observations)
    
    # Get lifecycle info
    lifecycle = get_lifecycle_per_cohort(observations)
    print(f"  {len(lifecycle)} cohorts, lifecycle range: {lifecycle['lifecycle_length'].min()}-{lifecycle['lifecycle_length'].max()} cycles")
    
    # Run analysis
    output = run_twenty_twenty(
        geometry, lifecycle,
        engine_name=args.engine,
        early_pct=args.early_pct,
        late_pct=args.late_pct,
    )
    
    # Print summary
    print_summary(output['statistics'], output['results'])
    
    # Save
    output['results'].write_parquet(args.output)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
