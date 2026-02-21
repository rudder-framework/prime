"""
02: Typology Entry Point
========================

Pure orchestration - calls typology computation module.
Computes 27 raw typology measures per signal.

Stages: observations.parquet → typology_raw.parquet

If Manifold's typology_vector.parquet exists in the output directory,
uses it as a richer source of per-signal summary metrics (mean, std, cv,
varies). Otherwise falls back to compute_typology_raw() (full-signal).

These measures are used by 03_classify for signal classification.
"""

import polars as pl
from pathlib import Path
from typing import Optional

from prime.ingest.typology_raw import compute_typology_raw


def run(
    observations_path: str,
    output_path: str = "typology_raw.parquet",
    output_dir: str = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute raw typology measures.

    If Manifold's typology_vector.parquet (per-signal summary with _mean,
    _std, _cv, _varies columns) exists in output_dir, reads it directly.
    Otherwise falls back to full-signal computation via compute_typology_raw().

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for typology_raw.parquet
        output_dir: Optional output directory to check for typology_vector.parquet
        verbose: Print progress

    Returns:
        DataFrame with typology measures per signal
    """
    if verbose:
        print("=" * 70)
        print("02: TYPOLOGY - Computing raw measures")
        print("=" * 70)

    # Check for Manifold's per-signal typology_vector (summary, not windows)
    tv_path = None
    if output_dir:
        candidate = Path(output_dir) / 'signal' / 'typology_vector.parquet'
        if candidate.exists():
            tv_path = str(candidate)
    if tv_path is None:
        # Check sibling output dirs relative to observations
        obs_dir = Path(observations_path).parent
        for output_subdir in obs_dir.glob('output_*/signal/typology_vector.parquet'):
            tv_path = str(output_subdir)
            break

    if tv_path:
        if verbose:
            print(f"  Found Manifold typology_vector: {tv_path}")
            print("  Using pre-computed per-signal summary")
        typology_df = pl.read_parquet(tv_path)
        typology_df.write_parquet(output_path)
        if verbose:
            n_varies = 0
            varies_cols = [c for c in typology_df.columns if c.endswith('_varies')]
            for c in varies_cols:
                n_varies += typology_df[c].sum()
            print(f"  {typology_df.height} signals, {n_varies} metric-signal pairs vary")
            print(f"  Wrote {output_path}")
    else:
        if verbose:
            print("  No typology_vector found - computing full-signal measures")
        typology_df = compute_typology_raw(observations_path, output_path, verbose=verbose)

    return typology_df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="02: Typology Raw Computation")
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('--output', '-o', default='typology_raw.parquet', help='Output path')
    parser.add_argument('--output-dir', default=None, help='Output directory to check for typology_vector')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(args.observations, args.output, output_dir=args.output_dir, verbose=not args.quiet)


if __name__ == '__main__':
    main()
