#!/usr/bin/env python3
"""
Framework Study Runner

Runs all analyses on a single C-MAPSS dataset (or any dataset with
cohort-level observations and Manifold outputs).

Expects a directory with Manifold outputs:
    observations.parquet
    signal_vector.parquet
    state_vector.parquet
    state_geometry.parquet

Produces:
    twenty_twenty_results.parquet
    canary_results.parquet
    thermodynamics.parquet
    thermodynamics_transitions.parquet
    study_summary.txt

Usage:
    python -m prime.analysis.study --data path/to/dataset_dir
    python -m prime.analysis.study --data data/cmapss/FD001
"""

import argparse
import json
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime

from prime.analysis.twenty_twenty import (
    get_lifecycle_per_cohort,
    run_twenty_twenty,
    print_summary as print_twenty_twenty,
)
from prime.analysis.canary import (
    get_lifecycle_per_cohort as get_lifecycles_dict,
    analyze_signal_velocity,
    analyze_signal_collapse_correlation,
    analyze_single_signal_rul,
    print_summary as print_canary,
)
from prime.analysis.thermodynamics import (
    compute_thermodynamics,
    detect_phase_transitions,
    print_summary as print_thermo,
)


def verify_inputs(data_dir: Path) -> dict:
    """Check which Manifold outputs exist."""
    files = {
        'observations': data_dir / 'observations.parquet',
        'signal_vector': data_dir / 'signal_vector.parquet',
        'state_vector': data_dir / 'state_vector.parquet',
        'state_geometry': data_dir / 'state_geometry.parquet',
    }
    
    status = {}
    for name, path in files.items():
        exists = path.exists()
        status[name] = {'path': str(path), 'exists': exists}
        if exists:
            df = pl.read_parquet(path)
            status[name]['rows'] = len(df)
            status[name]['columns'] = df.columns
            if 'cohort' in df.columns:
                status[name]['n_cohorts'] = df['cohort'].n_unique()
    
    return status


def run_study(data_dir: Path, output_dir: Path = None, engine_name: str = None):
    """Run complete study on one dataset."""
    
    if output_dir is None:
        output_dir = data_dir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print()
    print("█" * 70)
    print("  FRAMEWORK STUDY")
    print(f"  Data:   {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█" * 70)
    
    # ── Verify inputs ──
    print("\n── Checking Manifold outputs ──\n")
    status = verify_inputs(data_dir)
    
    for name, info in status.items():
        icon = "✓" if info['exists'] else "✗"
        if info['exists']:
            cohort_str = f", {info.get('n_cohorts', '?')} cohorts" if 'n_cohorts' in info else ""
            print(f"  {icon} {name}: {info['rows']} rows{cohort_str}")
        else:
            print(f"  {icon} {name}: MISSING — {info['path']}")
    
    if not status['state_geometry']['exists']:
        print("\n  ERROR: state_geometry.parquet is required. Run Manifold first.")
        return
    
    if not status['observations']['exists']:
        print("\n  ERROR: observations.parquet is required for lifecycle info.")
        return
    
    # ── Load data ──
    print("\n── Loading data ──\n")
    
    observations = pl.read_parquet(status['observations']['path'])
    geometry = pl.read_parquet(status['state_geometry']['path'])
    
    has_signal_vector = status['signal_vector']['exists']
    signal_vector = pl.read_parquet(status['signal_vector']['path']) if has_signal_vector else None
    
    lifecycle = get_lifecycle_per_cohort(observations)
    lifecycles_dict = dict(zip(lifecycle['cohort'].to_list(), lifecycle['lifecycle_length'].to_list()))
    
    print(f"  Cohorts: {len(lifecycle)}")
    print(f"  Lifecycle range: {lifecycle['lifecycle_length'].min()} – {lifecycle['lifecycle_length'].max()} cycles")
    
    # Available engine groups
    if 'engine' in geometry.columns:
        engines = geometry['engine'].unique().sort().to_list()
        print(f"  Engine groups: {engines}")
    else:
        engines = [None]
    
    results = {}
    
    # ── 1. Twenty-Twenty Analysis ──
    print("\n" + "─" * 70)
    print("  ANALYSIS 1: 20/20 EARLY DETECTION")
    print("─" * 70)
    
    tt_output = run_twenty_twenty(geometry, lifecycle, engine_name=engine_name)
    print_twenty_twenty(tt_output['statistics'], tt_output['results'])
    
    tt_path = output_dir / 'twenty_twenty_results.parquet'
    tt_output['results'].write_parquet(str(tt_path))
    results['twenty_twenty'] = tt_output['statistics']
    
    # ── 2. Canary Analysis ──
    if has_signal_vector:
        print("\n" + "─" * 70)
        print("  ANALYSIS 2: CANARY SIGNAL IDENTIFICATION")
        print("─" * 70)
        
        print("\n  Method 1: Signal velocity...")
        velocity = analyze_signal_velocity(signal_vector, lifecycles_dict)
        
        print("  Method 2: Collapse correlation...")
        correlation = analyze_signal_collapse_correlation(signal_vector, geometry, engine_name)
        
        print("  Method 3: Single-signal RUL prediction...")
        rul = analyze_single_signal_rul(signal_vector, lifecycles_dict)
        
        print_canary(velocity, correlation, rul)
        
        # Save
        if len(velocity) > 0:
            canary_path = output_dir / 'canary_results.parquet'
            combined = velocity
            if len(rul) > 0:
                combined = combined.join(rul, on='signal_id', how='outer', suffix='_rul')
            combined.write_parquet(str(canary_path))
        
        results['canary'] = {
            'top_velocity': velocity.head(1)['signal_id'].to_list()[0] if len(velocity) > 0 else None,
            'top_rul': rul.head(1)['signal_id'].to_list()[0] if len(rul) > 0 else None,
            'top_rul_r_squared': float(rul.head(1)['best_r_squared'][0]) if len(rul) > 0 else None,
        }
    else:
        print("\n  SKIPPING canary analysis (no signal_vector.parquet)")
    
    # ── 3. Thermodynamic Analysis ──
    print("\n" + "─" * 70)
    print("  ANALYSIS 3: THERMODYNAMICS")
    print("─" * 70)
    
    thermo = compute_thermodynamics(geometry, engine_name)
    transitions = detect_phase_transitions(thermo)
    print_thermo(thermo, transitions)
    
    thermo_path = output_dir / 'thermodynamics.parquet'
    thermo.write_parquet(str(thermo_path))
    
    trans_path = output_dir / 'thermodynamics_transitions.parquet'
    transitions.write_parquet(str(trans_path))
    
    detected = transitions.filter(pl.col('first_detection_type').is_not_null())
    if len(detected) > 0:
        pct_life = detected['first_detection_pct_life'].drop_nulls().to_numpy()
        results['thermodynamics'] = {
            'n_transitions_detected': len(detected),
            'mean_detection_pct': float(np.mean(pct_life)) if len(pct_life) > 0 else None,
        }
    
    # ── Summary ──
    print("\n" + "█" * 70)
    print("  STUDY COMPLETE")
    print("█" * 70)
    print()
    
    print(f"  Files written to: {output_dir}")
    for f in sorted(output_dir.glob('*.parquet')):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name} ({size_kb:.1f} KB)")
    
    # Save summary JSON
    summary_path = output_dir / 'study_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"    study_summary.json")
    
    print()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Framework Study Runner')
    parser.add_argument('--data', required=True, help='Directory with Manifold outputs')
    parser.add_argument('--output', default=None, help='Output directory (default: data/analysis)')
    parser.add_argument('--engine', default=None, help='Filter to specific engine group')
    args = parser.parse_args()
    
    data_dir = Path(args.data)
    output_dir = Path(args.output) if args.output else None
    
    run_study(data_dir, output_dir, args.engine)


if __name__ == '__main__':
    main()
