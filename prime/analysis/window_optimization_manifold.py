#!/usr/bin/env python3
"""
Prime Analysis: Window Optimization (Full Manifold)

Option B -- the proper version. Runs the ACTUAL Manifold pipeline
(stages 01->02->03) at each (window_size, stride) combination,
then measures 20/20 predictive power on the real feature geometry.

Compare with window_optimization.py (Option A) which uses raw eigendecomp.
The difference between A and B IS the paper's proof that feature engineering
through the engine architecture creates predictive geometric structure
that raw covariance cannot capture.

Designed for AWS / high-compute environments.
Each grid point takes ~30-120s depending on core count.
Full sweep (8 windows x 3 modes = 24 points) ~ 15-45 min.

Requirements:
    - Manifold (engines/) must be importable
    - observations.parquet, typology.parquet, manifest.yaml in data_dir

Usage:
    python -m prime.analysis.window_optimization_manifold \
        --data-dir /path/to/FD001 \
        --output /path/to/window_optimization_manifold.parquet

    # Quick test with fewer grid points
    python -m prime.analysis.window_optimization_manifold \
        --data-dir /path/to/FD001 \
        --windows 10,20,40 \
        --output results.parquet

    # Run specific combos
    python -m prime.analysis.window_optimization_manifold \
        --data-dir /path/to/FD001 \
        --windows 10,12,15,20,25,30,40,50 \
        --modes non_overlapping,half_overlap \
        --output results.parquet
"""

import argparse
import copy
import json
import numpy as np
import polars as pl
import shutil
import sys
import time
import traceback
import warnings
import yaml
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')


# ============================================================
# MANIFEST SURGERY
# ============================================================

def patch_manifest(manifest: dict, window_size: int, stride: int) -> dict:
    """
    Create a copy of the manifest with new window/stride.

    Touches ONLY system.window and system.stride.
    Everything else (cohorts, engines, paths) stays identical.
    """
    patched = copy.deepcopy(manifest)
    patched['system']['window'] = window_size
    patched['system']['stride'] = stride
    return patched


# ============================================================
# PIPELINE RUNNER (stages 01 -> 02 -> 03)
# ============================================================

def run_manifold_stages(
    observations_path: str,
    typology_path: str,
    manifest: dict,
    work_dir: str,
    verbose: bool = False,
) -> Optional[str]:
    """
    Run Manifold stages 01-03 with given manifest.

    Returns path to state_geometry.parquet, or None on failure.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    sv_path = str(work / 'signal_vector.parquet')
    st_path = str(work / 'state_vector.parquet')
    sg_path = str(work / 'state_geometry.parquet')

    try:
        # Stage 01: Signal Vector
        from engines.entry_points.stage_01_signal_vector import run as run_stage_01

        # Load window factors from typology
        from engines.entry_points.stage_01_signal_vector import load_window_factors
        typ_path = Path(typology_path)
        window_factors = load_window_factors(typ_path) if typ_path.exists() else {}

        obs = pl.read_parquet(observations_path)

        run_stage_01(
            observations_path=observations_path,
            output_path=sv_path,
            manifest=manifest,
            verbose=verbose,
            typology_path=typology_path,
        )

        if not Path(sv_path).exists():
            return None

        # Stage 02: State Vector
        from engines.entry_points.stage_02_state_vector import compute_state_vector

        compute_state_vector(
            signal_vector_path=sv_path,
            typology_path=typology_path,
            output_path=st_path,
            verbose=verbose,
        )

        if not Path(st_path).exists():
            return None

        # Stage 03: State Geometry
        from engines.entry_points.stage_03_state_geometry import compute_state_geometry

        compute_state_geometry(
            signal_vector_path=sv_path,
            state_vector_path=st_path,
            output_path=sg_path,
            verbose=verbose,
        )

        if not Path(sg_path).exists():
            return None

        return sg_path

    except Exception as e:
        if verbose:
            print(f"  ERROR in pipeline: {e}")
            traceback.print_exc()
        return None


# ============================================================
# 20/20 FROM STATE GEOMETRY
# ============================================================

def run_twenty_twenty_from_geometry(
    geometry_path: str,
    lifecycles: Dict[str, int],
    engine_name: str = None,
    early_pct: float = 0.20,
) -> dict:
    """
    Run 20/20 analysis on a state_geometry.parquet file.
    Returns correlation statistics.
    """
    geo = pl.read_parquet(geometry_path)

    if engine_name and 'engine' in geo.columns:
        geo = geo.filter(pl.col('engine') == engine_name)

    if 'cohort' not in geo.columns:
        return {'r': np.nan, 'p': np.nan, 'n': 0, 'note': 'no cohort column'}

    cohorts = sorted(set(geo['cohort'].unique().to_list()) & set(lifecycles.keys()))

    if len(cohorts) < 10:
        return {'r': np.nan, 'p': np.nan, 'n': len(cohorts), 'note': 'too few cohorts'}

    early_dims = []
    deltas = []
    lives = []

    for cohort in cohorts:
        cg = geo.filter(pl.col('cohort') == cohort).sort('signal_0')

        if 'effective_dim' not in cg.columns or len(cg) < 3:
            continue

        eff_dims = cg['effective_dim'].to_numpy()
        n = len(eff_dims)
        n_early = max(1, int(n * early_pct))
        n_late = max(1, int(n * early_pct))

        early_mean = float(np.nanmean(eff_dims[:n_early]))
        late_mean = float(np.nanmean(eff_dims[-n_late:]))

        if np.isnan(early_mean) or np.isnan(late_mean):
            continue

        early_dims.append(early_mean)
        deltas.append(early_mean - late_mean)
        lives.append(lifecycles[cohort])

    if len(early_dims) < 10:
        return {'r': np.nan, 'p': np.nan, 'n': len(early_dims), 'note': 'too few valid cohorts'}

    early_arr = np.array(early_dims)
    delta_arr = np.array(deltas)
    life_arr = np.array(lives)

    r, p = stats.pearsonr(early_arr, life_arr)
    r_delta, _ = stats.pearsonr(delta_arr, life_arr)
    rho, _ = stats.spearmanr(early_arr, life_arr)

    return {
        'r': float(r),
        'p': float(p),
        'r_squared': float(r ** 2),
        'r_delta': float(r_delta),
        'rho': float(rho),
        'n': len(early_dims),
    }


# ============================================================
# PER-ENGINE BREAKDOWN
# ============================================================

def run_per_engine_analysis(
    geometry_path: str,
    lifecycles: Dict[str, int],
) -> Dict[str, dict]:
    """Run 20/20 for each engine group separately."""
    geo = pl.read_parquet(geometry_path)

    if 'engine' not in geo.columns:
        return {'all': run_twenty_twenty_from_geometry(geometry_path, lifecycles)}

    engines = geo['engine'].unique().sort().to_list()
    results = {}

    for engine in engines:
        results[engine] = run_twenty_twenty_from_geometry(
            geometry_path, lifecycles, engine_name=engine
        )

    # Also run on all engines combined
    results['all'] = run_twenty_twenty_from_geometry(geometry_path, lifecycles)

    return results


# ============================================================
# GRID SWEEP
# ============================================================

DEFAULT_WINDOWS = [8, 10, 12, 15, 20, 25, 30, 40, 50]
DEFAULT_MODES = {
    'non_overlapping': 0.0,    # stride = window (THE HYPOTHESIS)
    'half_overlap': 50.0,      # stride = window // 2
    'quarter_overlap': 75.0,   # stride = window // 4
}


def run_grid_sweep(
    data_dir: str,
    window_sizes: List[int] = None,
    overlap_modes: Dict[str, float] = None,
    verbose: bool = True,
    keep_outputs: bool = False,
) -> pl.DataFrame:
    """
    Full Manifold grid sweep.

    For each (window, stride): patch manifest -> run stages 01-03 -> measure 20/20.
    """
    data_dir = Path(data_dir)

    if window_sizes is None:
        window_sizes = DEFAULT_WINDOWS
    if overlap_modes is None:
        overlap_modes = DEFAULT_MODES

    # Load base files
    manifest_path = data_dir / 'manifest.yaml'
    observations_path = data_dir / 'observations.parquet'
    typology_path = data_dir / 'typology.parquet'

    for f in [manifest_path, observations_path, typology_path]:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")

    with open(manifest_path) as f:
        base_manifest = yaml.safe_load(f)

    # Get lifecycles
    obs = pl.read_parquet(str(observations_path))
    first_sig = obs['signal_id'].unique().sort()[0]
    lifecycle_df = (
        obs
        .filter(pl.col('signal_id') == first_sig)
        .group_by('cohort')
        .agg(pl.col('signal_0').max() + 1)
    )
    lifecycles = dict(zip(lifecycle_df['cohort'].to_list(), lifecycle_df['signal_0'].to_list()))

    total = len(window_sizes) * len(overlap_modes)

    if verbose:
        print(f"\n  Grid: {len(window_sizes)} windows x {len(overlap_modes)} modes = {total} combos")
        print(f"  Windows: {window_sizes}")
        print(f"  Modes: {list(overlap_modes.keys())}")
        print(f"  Cohorts: {len(lifecycles)}")
        print(f"  Lifecycle range: {min(lifecycles.values())} - {max(lifecycles.values())}")
        print()

    # Work directory for intermediate files
    work_root = data_dir / 'window_optimization_work'
    work_root.mkdir(exist_ok=True)

    rows = []
    best_r_sq = -1
    best_combo = None
    run_count = 0
    sweep_start = time.time()

    for window_size in window_sizes:
        for mode_name, overlap_pct in overlap_modes.items():
            run_count += 1

            if mode_name == 'non_overlapping':
                stride = window_size
            else:
                stride = max(1, int(window_size * (1 - overlap_pct / 100)))

            # Work directory for this run
            run_dir = work_root / f"w{window_size}_s{stride}"

            if verbose:
                print(f"  [{run_count}/{total}] w={window_size:>3d} s={stride:>3d} ({mode_name:>17s}) ...",
                      end='', flush=True)

            t0 = time.time()

            # Patch manifest
            patched = patch_manifest(base_manifest, window_size, stride)

            # Run pipeline
            sg_path = run_manifold_stages(
                str(observations_path),
                str(typology_path),
                patched,
                str(run_dir),
                verbose=False,
            )

            elapsed = time.time() - t0

            if sg_path is None:
                if verbose:
                    print(f"  FAILED [{elapsed:.1f}s]")
                rows.append({
                    'window_size': window_size,
                    'stride': stride,
                    'overlap_mode': mode_name,
                    'overlap_pct': overlap_pct,
                    'r': np.nan, 'p': np.nan, 'r_squared': np.nan,
                    'r_delta': np.nan, 'rho': np.nan,
                    'n_cohorts': 0, 'elapsed_sec': elapsed,
                    'status': 'failed',
                })
                continue

            # Run 20/20 analysis
            result = run_twenty_twenty_from_geometry(sg_path, lifecycles)

            # Per-engine breakdown
            engine_results = run_per_engine_analysis(sg_path, lifecycles)

            # Count windows per cohort
            sg = pl.read_parquet(sg_path)
            if 'cohort' in sg.columns:
                windows_per = sg.group_by('cohort').len()['len'].to_numpy()
                avg_windows = float(np.mean(windows_per))
            else:
                avg_windows = len(sg)

            row = {
                'window_size': window_size,
                'stride': stride,
                'overlap_mode': mode_name,
                'overlap_pct': overlap_pct,
                'r': result['r'],
                'p': result['p'],
                'r_squared': result['r_squared'],
                'r_delta': result.get('r_delta', np.nan),
                'rho': result.get('rho', np.nan),
                'n_cohorts': result['n'],
                'avg_windows_per_cohort': avg_windows,
                'elapsed_sec': elapsed,
                'status': 'ok',
            }

            # Add per-engine R-squared columns
            for eng_name, eng_result in engine_results.items():
                row[f'r_{eng_name}'] = eng_result.get('r', np.nan)
                row[f'r_squared_{eng_name}'] = eng_result.get('r_squared', np.nan)

            rows.append(row)

            if not np.isnan(result['r_squared']) and result['r_squared'] > best_r_sq:
                best_r_sq = result['r_squared']
                best_combo = (window_size, stride, mode_name)

            if verbose:
                eng_str = "  ".join(
                    f"{k}={v.get('r', 0):+.3f}"
                    for k, v in engine_results.items()
                    if k != 'all'
                )
                print(f"  r={result['r']:+.4f}  R-sq={result['r_squared']:.4f}  "
                      f"rho={result.get('rho', 0):+.3f}  "
                      f"[{elapsed:.1f}s]  {eng_str}")

            # Cleanup unless keeping
            if not keep_outputs:
                shutil.rmtree(run_dir, ignore_errors=True)

    total_elapsed = time.time() - sweep_start

    if verbose:
        print(f"\n  Total sweep: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
        if best_combo:
            print(f"  Best: w={best_combo[0]} s={best_combo[1]} ({best_combo[2]}) R-sq={best_r_sq:.4f}")

    # Cleanup work dir
    if not keep_outputs and work_root.exists():
        shutil.rmtree(work_root, ignore_errors=True)

    return pl.DataFrame(rows)


# ============================================================
# REPORTING (same structure as Option A for comparison)
# ============================================================

def analyze_results(results: pl.DataFrame) -> dict:
    """Analyze grid results."""
    valid = results.filter(pl.col('status') == 'ok')

    if len(valid) == 0:
        return {'best': None, 'best_per_mode': {}, 'comparison': pl.DataFrame()}

    # Best overall
    best_idx = valid['r_squared'].arg_max()
    best = valid.row(best_idx, named=True)

    # Best per mode
    best_per_mode = {}
    for mode in valid['overlap_mode'].unique().to_list():
        mode_data = valid.filter(pl.col('overlap_mode') == mode)
        if len(mode_data) > 0:
            idx = mode_data['r_squared'].arg_max()
            best_per_mode[mode] = mode_data.row(idx, named=True)

    # Non-overlapping vs overlapping comparison
    comparisons = []
    for ws in valid['window_size'].unique().sort().to_list():
        ws_data = valid.filter(pl.col('window_size') == ws)
        non_overlap = ws_data.filter(pl.col('overlap_mode') == 'non_overlapping')
        half_overlap = ws_data.filter(pl.col('overlap_mode') == 'half_overlap')

        if len(non_overlap) > 0 and len(half_overlap) > 0:
            r_no = float(non_overlap['r'][0])
            r_ho = float(half_overlap['r'][0])
            comparisons.append({
                'window_size': ws,
                'r_non_overlapping': r_no,
                'r_half_overlap': r_ho,
                'r_sq_non_overlapping': float(non_overlap['r_squared'][0]),
                'r_sq_half_overlap': float(half_overlap['r_squared'][0]),
                'diff': r_no - r_ho,
                'non_overlap_wins': abs(r_no) > abs(r_ho),
            })

    comparison_df = pl.DataFrame(comparisons) if comparisons else pl.DataFrame()

    n_wins = int(comparison_df['non_overlap_wins'].sum()) if len(comparison_df) > 0 else 0
    n_total = len(comparison_df)

    return {
        'best': best,
        'best_per_mode': best_per_mode,
        'comparison': comparison_df,
        'non_overlap_win_rate': n_wins / max(n_total, 1),
        'n_comparisons': n_total,
    }


def print_report(results: pl.DataFrame, analysis: dict):
    """Print paper-ready report."""

    print()
    print("=" * 78)
    print("  WINDOW OPTIMIZATION REPORT (FULL MANIFOLD)")
    print("=" * 78)

    best = analysis['best']
    if best is None:
        print("\n  No successful runs. Check pipeline errors.")
        return

    print()
    print("-- Optimal Operating Point --")
    print()
    print(f"  Window size:  {best['window_size']}")
    print(f"  Stride:       {best['stride']}")
    print(f"  Mode:         {best['overlap_mode']}")
    print(f"  Pearson r:    {best['r']:.4f}")
    print(f"  R-squared:    {best['r_squared']:.4f}")
    print(f"  Spearman rho: {best['rho']:.4f}")
    print(f"  p-value:      {best['p']:.2e}")
    print(f"  Cohorts:      {best['n_cohorts']}")
    print(f"  Avg windows:  {best['avg_windows_per_cohort']:.1f} per cohort")

    # Per-engine R-squared at optimal
    eng_cols = [c for c in results.columns if c.startswith('r_squared_') and c != 'r_squared']
    if eng_cols:
        print()
        print("  Per-engine R-squared at optimal:")
        for col in sorted(eng_cols):
            eng_name = col.replace('r_squared_', '')
            val = best.get(col, np.nan)
            if val is not None and not np.isnan(val):
                print(f"    {eng_name:>15s}: {val:.4f}")

    # Best per mode
    print()
    print("-- Best Per Overlap Mode --")
    print()
    for mode, row in analysis['best_per_mode'].items():
        print(f"  {mode:>20s}:  w={row['window_size']:>3d}  s={row['stride']:>3d}  "
              f"r={row['r']:+.4f}  R-sq={row['r_squared']:.4f}  [{row['elapsed_sec']:.1f}s]")

    # Non-overlapping vs overlapping
    comp = analysis['comparison']
    if len(comp) > 0:
        print()
        print("-- Non-Overlapping vs 50% Overlap --")
        print()
        for row in comp.iter_rows(named=True):
            winner = "NON-OVERLAP" if row['non_overlap_wins'] else "  overlap"
            print(f"  w={row['window_size']:>3d}:  "
                  f"R-sq_no={row['r_sq_non_overlapping']:.4f}  "
                  f"R-sq_ho={row['r_sq_half_overlap']:.4f}  "
                  f"delta_r={row['diff']:+.4f}  {winner}")

        print()
        win_rate = analysis['non_overlap_win_rate']
        n_wins = int(win_rate * analysis['n_comparisons'])
        print(f"  Non-overlapping wins: {win_rate*100:.0f}% ({n_wins}/{analysis['n_comparisons']})")

    # Full grid
    valid = results.filter(pl.col('status') == 'ok').sort('r_squared', descending=True)
    print()
    print("-- Full Grid (top 15 by R-squared) --")
    print()
    for row in valid.head(15).iter_rows(named=True):
        bar_len = max(0, int(row['r_squared'] * 50))
        bar = "#" * bar_len
        print(f"  w={row['window_size']:>3d} s={row['stride']:>3d} ({row['overlap_mode']:>17s})  "
              f"R-sq={row['r_squared']:.4f}  r={row['r']:+.4f}  "
              f"[{row['elapsed_sec']:.0f}s]  {bar}")

    # For paper
    print()
    print("-- For Paper --")
    print()

    best_no = analysis['best_per_mode'].get('non_overlapping', {})
    best_ho = analysis['best_per_mode'].get('half_overlap', {})

    if best_no and best_ho:
        r_sq_no = best_no.get('r_squared', 0)
        r_sq_ho = best_ho.get('r_squared', 0)

        if r_sq_no > r_sq_ho:
            pct = (r_sq_no - r_sq_ho) / max(r_sq_ho, 1e-6) * 100
            print(f"  Non-overlapping eigendecomposition (stride = window_size) yields")
            print(f"  R-squared = {r_sq_no:.4f} vs R-squared = {r_sq_ho:.4f} for 50% overlap,")
            print(f"  a {pct:.1f}% improvement in early-life predictive power.")
        else:
            print(f"  50% overlap slightly outperforms non-overlapping")
            print(f"  (R-squared = {r_sq_ho:.4f} vs {r_sq_no:.4f}).")

    mode_desc = 'non-overlapping' if best['overlap_mode'] == 'non_overlapping' else f"{best['overlap_pct']:.0f}% overlap"
    print(f"\n  Optimal configuration: window={best['window_size']}, stride={best['stride']} ({mode_desc})")
    print(f"  Use these values in manifest.yaml system.window and system.stride.")

    print()
    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prime Window Optimization (Full Manifold Pipeline)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full sweep
  python -m prime.analysis.window_optimization_manifold --data-dir data/FD001

  # Quick test
  python -m prime.analysis.window_optimization_manifold --data-dir data/FD001 --windows 10,20,40

  # Keep intermediate files for debugging
  python -m prime.analysis.window_optimization_manifold --data-dir data/FD001 --keep-outputs
        """
    )
    parser.add_argument('--data-dir', required=True, help='Directory with observations.parquet, typology.parquet, manifest.yaml')
    parser.add_argument('--output', default=None, help='Output path (default: data_dir/window_optimization_manifold.parquet)')
    parser.add_argument('--windows', default=None, help='Comma-separated window sizes')
    parser.add_argument('--modes', default=None, help='Comma-separated overlap modes (non_overlapping,half_overlap,quarter_overlap)')
    parser.add_argument('--keep-outputs', action='store_true', help='Keep intermediate pipeline outputs')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = args.output or str(data_dir / 'window_optimization_manifold.parquet')
    verbose = not args.quiet

    if verbose:
        print()
        print("#" * 78)
        print("  PRIME WINDOW OPTIMIZATION -- FULL MANIFOLD PIPELINE")
        print("#" * 78)
        print(f"\n  Data dir: {data_dir}")
        print(f"  Output:   {output_path}")

    # Parse window sizes
    window_sizes = None
    if args.windows:
        window_sizes = [int(x.strip()) for x in args.windows.split(',')]

    # Parse modes
    overlap_modes = None
    if args.modes:
        mode_names = [m.strip() for m in args.modes.split(',')]
        overlap_modes = {m: DEFAULT_MODES[m] for m in mode_names if m in DEFAULT_MODES}

    # Auto-cap window sizes to dataset
    if window_sizes is None:
        obs = pl.read_parquet(str(data_dir / 'observations.parquet'))
        first_sig = obs['signal_id'].unique().sort()[0]
        min_life = (
            obs.filter(pl.col('signal_id') == first_sig)
            .group_by('cohort')
            .agg(pl.col('signal_0').max() + 1)
            ['signal_0'].min()
        )
        max_window = min_life // 3
        window_sizes = [w for w in DEFAULT_WINDOWS if w <= max_window]
        if not window_sizes:
            window_sizes = [8, 10, 15]
        if verbose:
            print(f"  Auto windows: {window_sizes} (max={max_window}, shortest life={min_life})")

    # Run sweep
    results = run_grid_sweep(
        str(data_dir),
        window_sizes=window_sizes,
        overlap_modes=overlap_modes,
        verbose=verbose,
        keep_outputs=args.keep_outputs,
    )

    # Analyze
    analysis = analyze_results(results)

    # Report
    if verbose:
        print_report(results, analysis)

    # Save
    results.write_parquet(output_path)
    if verbose:
        print(f"Saved: {output_path}")

    # Also save analysis summary as JSON
    summary_path = output_path.replace('.parquet', '_summary.json')
    summary = {
        'best_window': analysis['best']['window_size'] if analysis['best'] else None,
        'best_stride': analysis['best']['stride'] if analysis['best'] else None,
        'best_mode': analysis['best']['overlap_mode'] if analysis['best'] else None,
        'best_r': analysis['best']['r'] if analysis['best'] else None,
        'best_r_squared': analysis['best']['r_squared'] if analysis['best'] else None,
        'non_overlap_win_rate': analysis['non_overlap_win_rate'],
        'n_grid_points': len(results),
        'n_successful': len(results.filter(pl.col('status') == 'ok')),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    if verbose:
        print(f"Saved: {summary_path}")


if __name__ == '__main__':
    main()
