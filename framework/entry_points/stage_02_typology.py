"""
02: Typology Entry Point
========================

Pure orchestration - calls typology computation module.
Computes 27 raw typology measures per signal.

Stages: observations.parquet → typology_raw.parquet

These measures are used by 03_classify for signal classification.
"""

import polars as pl
from pathlib import Path
from typing import Optional

from framework.ingest.typology_raw import compute_typology_raw


def run(
    observations_path: str,
    output_path: str = "typology_raw.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute raw typology measures.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for typology_raw.parquet
        verbose: Print progress

    Returns:
        DataFrame with 27 typology measures per signal
    """
    if verbose:
        print("=" * 70)
        print("02: TYPOLOGY - Computing raw measures")
        print("=" * 70)

    # compute_typology_raw expects a file path, not a DataFrame
    typology_df = compute_typology_raw(observations_path, output_path, verbose=verbose)

    return typology_df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="02: Typology Raw Computation")
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('--output', '-o', default='typology_raw.parquet', help='Output path')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(args.observations, args.output, verbose=not args.quiet)


if __name__ == '__main__':
    main()
