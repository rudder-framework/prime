"""
03: Classification Entry Point
==============================

Pure orchestration - calls classification modules.
Applies two-stage classification: discrete/sparse (PR5) → continuous (PR4).

Stages: typology_raw.parquet → typology.parquet

Classification output has 10 dimensions including temporal_pattern.
"""

import polars as pl
from pathlib import Path
from typing import Optional

from prime.typology.discrete_sparse import apply_discrete_sparse_classification
from prime.typology.level2_corrections import apply_corrections


def run(
    typology_raw_path: str,
    output_path: str = "typology.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Apply two-stage classification to typology measures.

    Stage 1: Discrete/Sparse Detection (PR5)
        - CONSTANT, BINARY, DISCRETE, IMPULSIVE, EVENT

    Stage 2: Continuous Classification (PR4)
        - PERIODIC, TRENDING, CHAOTIC, RANDOM, QUASI_PERIODIC, STATIONARY

    Args:
        typology_raw_path: Path to typology_raw.parquet
        output_path: Output path for typology.parquet
        verbose: Print progress

    Returns:
        DataFrame with classification columns added
    """
    if verbose:
        print("=" * 70)
        print("03: CLASSIFICATION - Two-stage classification")
        print("=" * 70)

    # Load typology_raw
    df = pl.read_parquet(typology_raw_path)
    if verbose:
        print(f"Loaded: {len(df)} signals from typology_raw")

    # Classification functions expect row dicts, not DataFrames.
    # Iterate rows: PR5 (discrete/sparse) first, then PR4 (continuous).
    rows = df.to_dicts()
    classified_rows = []

    if verbose:
        print("\nStage 1: Discrete/Sparse Detection (PR5)...")

    n_discrete = 0
    for row in rows:
        # PR5: Discrete/sparse classification (runs first)
        row = apply_discrete_sparse_classification(row)

        # PR4: Continuous classification (if not discrete/sparse)
        if not row.get('is_discrete_sparse', False):
            row = apply_corrections(row)
        else:
            n_discrete += 1

        # Ensure required columns exist
        if 'temporal_pattern' not in row:
            row['temporal_pattern'] = 'STATIONARY'
        if 'spectral' not in row:
            row['spectral'] = 'BROADBAND'

        classified_rows.append(row)

    if verbose:
        print(f"  Detected {n_discrete} discrete/sparse signals")
        print("\nStage 2: Continuous Classification (PR4)... done")

    # Rebuild DataFrame
    df = pl.DataFrame(classified_rows)

    if verbose:
        if 'temporal_pattern' in df.columns:
            print("\nClassification summary:")
            pattern_counts = df.group_by('temporal_pattern').count().sort('count', descending=True)
            for row in pattern_counts.to_dicts():
                print(f"  {row['temporal_pattern']}: {row['count']}")

    # Save
    df.write_parquet(output_path)
    if verbose:
        print(f"\nSaved: {output_path}")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="03: Classification")
    parser.add_argument('typology_raw', help='Path to typology_raw.parquet')
    parser.add_argument('--output', '-o', default='typology.parquet', help='Output path')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(args.typology_raw, args.output, verbose=not args.quiet)


if __name__ == '__main__':
    main()
