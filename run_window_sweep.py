#!/usr/bin/env python3
"""
Window Optimization Sweep
=========================
For each window/stride combo, patches manifest.yaml, re-runs Manifold stages 01-03,
runs Prime analysis, and collects total_variance R² into a comparison table.
"""

import json
import subprocess
import sys
import yaml
import copy
from pathlib import Path

DATA_DIR = Path.home() / 'data' / 'FD001'
MANIFOLD_DIR = Path.home() / 'manifold'
PRIME_DIR = Path.home() / 'prime'
MANIFEST_PATH = DATA_DIR / 'manifest.yaml'
MANIFEST_BACKUP = DATA_DIR / 'manifest.yaml.orig'

PYTHON_MANIFOLD = str(MANIFOLD_DIR / 'venv' / 'bin' / 'python')
PYTHON_PRIME = str(PRIME_DIR / 'venv' / 'bin' / 'python')

WINDOWS = [10, 15, 20, 25, 30, 40]
STRIDE_MODES = ['full', 'half']  # full = window, half = window//2


def patch_manifest(manifest: dict, window: int, stride: int) -> dict:
    """Patch system + all per-signal window_size and stride values."""
    m = copy.deepcopy(manifest)

    # System level
    m['system']['window'] = window
    m['system']['stride'] = stride

    # Params level
    m['params']['default_window'] = window
    m['params']['default_stride'] = stride

    # Per-signal: set all window_size and stride
    for cohort_name, cohort_config in m.get('cohorts', {}).items():
        for signal_id, signal_config in cohort_config.items():
            if not isinstance(signal_config, dict):
                continue
            signal_config['window_size'] = window
            signal_config['stride'] = stride
            # Clear engine_window_overrides — let engines use the swept window
            # (engines with hard minimums will still gate internally)
            if 'engine_window_overrides' in signal_config:
                del signal_config['engine_window_overrides']

    # Clear global engine_windows too (let everything use swept value)
    # Keep note but remove numeric overrides
    if 'engine_windows' in m:
        note = m['engine_windows'].get('note', '')
        m['engine_windows'] = {'note': note} if note else {}

    return m


def run_stages_01_03(manifest_path: str, verbose: bool = True) -> bool:
    """Run Manifold stages 01-03."""
    env = {'PYTHONPATH': str(MANIFOLD_DIR)}
    import os
    full_env = {**os.environ, **env}

    cmd = [
        PYTHON_MANIFOLD, '-m', 'engines.entry_points.run_pipeline',
        manifest_path, '--stages', '01,02,03', '-q'
    ]

    result = subprocess.run(cmd, env=full_env, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        # Atlas assertion is expected — stages 01-03 still succeed
        if 'Incomplete atlas' in result.stderr:
            return True
        if verbose:
            print(f"    STDERR: {result.stderr[-200:]}")
        return False
    return True


def run_analysis(data_dir: str) -> dict:
    """Run Prime analysis, return summary."""
    cmd = [PYTHON_PRIME, str(PRIME_DIR / 'run_analysis.py'), '--data', data_dir]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    summary_path = Path(data_dir) / 'analysis' / 'analysis_summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {}


def main():
    print("=" * 70)
    print("  WINDOW OPTIMIZATION SWEEP")
    print("=" * 70)

    # Load original manifest
    with open(MANIFEST_PATH) as f:
        original_manifest = yaml.safe_load(f)

    # Backup original
    if not MANIFEST_BACKUP.exists():
        with open(MANIFEST_BACKUP, 'w') as f:
            yaml.dump(original_manifest, f, default_flow_style=False)
        print(f"\n  Backed up original manifest to {MANIFEST_BACKUP}")

    # Copy manifold outputs needed for analysis
    output_dir = DATA_DIR / 'output'

    results = []
    total = len(WINDOWS) * len(STRIDE_MODES)
    run_num = 0

    for window in WINDOWS:
        for stride_mode in STRIDE_MODES:
            stride = window if stride_mode == 'full' else max(1, window // 2)
            overlap_pct = int((1 - stride / window) * 100) if window > 0 else 0
            run_num += 1

            label = f"w={window:>3d}, s={stride:>3d} ({overlap_pct}% overlap)"
            print(f"\n  [{run_num:>2d}/{total}] {label}")

            # 1. Patch manifest
            patched = patch_manifest(original_manifest, window, stride)
            with open(MANIFEST_PATH, 'w') as f:
                yaml.dump(patched, f, default_flow_style=False)

            # 2. Run stages 01-03
            print(f"          Running stages 01-03...", end='', flush=True)
            ok = run_stages_01_03(str(MANIFEST_PATH))
            if not ok:
                print(" FAILED")
                results.append({
                    'window': window,
                    'stride': stride,
                    'overlap_pct': overlap_pct,
                    'stride_mode': stride_mode,
                    'status': 'FAILED',
                    'tv_r_squared': None,
                    'ed_r_squared': None,
                })
                continue
            print(" done", flush=True)

            # 3. Copy outputs for analysis
            for fname in ['signal_vector.parquet', 'state_vector.parquet', 'state_geometry.parquet']:
                src = output_dir / fname
                dst = DATA_DIR / fname
                if src.exists():
                    import shutil
                    shutil.copy2(src, dst)

            # 4. Run analysis
            print(f"          Running analysis...", end='', flush=True)
            summary = run_analysis(str(DATA_DIR))
            print(" done", flush=True)

            # 5. Collect results
            tv_r2 = None
            if 'total_variance' in summary:
                tv_r2 = summary['total_variance'].get('r_squared')

            ed_r2 = None
            if 'twenty_twenty' in summary:
                ed_r2 = summary['twenty_twenty'].get('r_squared')

            best_feat = summary.get('best_feature', '?')
            best_r2 = summary.get('best_r_squared')

            n_collapse = None
            if 'twenty_twenty' in summary:
                n_collapse = summary['twenty_twenty'].get('n_collapse')

            results.append({
                'window': window,
                'stride': stride,
                'overlap_pct': overlap_pct,
                'stride_mode': stride_mode,
                'status': 'OK',
                'tv_r_squared': tv_r2,
                'ed_r_squared': ed_r2,
                'best_feature': best_feat,
                'best_r_squared': best_r2,
                'n_collapse': n_collapse,
            })

            if tv_r2 is not None:
                print(f"          total_variance R²={tv_r2:.4f}  |  eff_dim R²={ed_r2:.4f}  |  collapse={n_collapse}/100")

    # Restore original manifest
    with open(MANIFEST_BACKUP) as f:
        orig = yaml.safe_load(f)
    with open(MANIFEST_PATH, 'w') as f:
        yaml.dump(orig, f, default_flow_style=False)
    print(f"\n  Restored original manifest.")

    # Print comparison table
    print()
    print("=" * 90)
    print("  WINDOW OPTIMIZATION RESULTS")
    print("=" * 90)
    print()
    print(f"  {'Window':>6s}  {'Stride':>6s}  {'Overlap':>7s}  {'TV R²':>8s}  {'ED R²':>8s}  {'Best Feature':>25s}  {'Best R²':>8s}  {'Collapse':>8s}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*25}  {'─'*8}  {'─'*8}")

    best_tv = None
    best_row = None

    for r in results:
        if r['status'] == 'FAILED':
            print(f"  {r['window']:>6d}  {r['stride']:>6d}  {r['overlap_pct']:>6d}%  {'FAILED':>8s}")
            continue

        tv_str = f"{r['tv_r_squared']:.4f}" if r['tv_r_squared'] is not None else "N/A"
        ed_str = f"{r['ed_r_squared']:.4f}" if r['ed_r_squared'] is not None else "N/A"
        best_str = f"{r['best_r_squared']:.4f}" if r.get('best_r_squared') is not None else "N/A"
        col_str = f"{r.get('n_collapse', '?')}/100"

        print(f"  {r['window']:>6d}  {r['stride']:>6d}  {r['overlap_pct']:>6d}%  {tv_str:>8s}  {ed_str:>8s}  {r.get('best_feature', '?'):>25s}  {best_str:>8s}  {col_str:>8s}")

        if r['tv_r_squared'] is not None:
            if best_tv is None or r['tv_r_squared'] > best_tv:
                best_tv = r['tv_r_squared']
                best_row = r

    if best_row:
        print()
        print(f"  BEST: window={best_row['window']}, stride={best_row['stride']} "
              f"({best_row['overlap_pct']}% overlap) → total_variance R²={best_tv:.4f}")

    # Save results
    results_path = DATA_DIR / 'window_sweep_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")
    print()


if __name__ == '__main__':
    main()
