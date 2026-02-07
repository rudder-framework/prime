"""
11: Fetch Entry Point
======================

Pure orchestration - calls data reader and validation.
Reads data files, profiles them, and validates for PRISM format.

Stages: raw file (csv/parquet/tsv) → validated observations.parquet

Combines data reading, profiling, and observation validation.
"""

import polars as pl
from pathlib import Path
from typing import Optional, Dict, Any

from orthon.ingest.data_reader import DataReader, DataProfile
from orthon.ingest.validate_observations import validate_and_save, ValidationStatus


def run(
    input_path: str,
    output_path: str = "observations.parquet",
    entity_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    validate: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Fetch and prepare data for PRISM.

    Args:
        input_path: Path to raw data file (csv, parquet, tsv)
        output_path: Output path for observations.parquet
        entity_col: Column name for entity/cohort ID
        timestamp_col: Column name for timestamp
        validate: Run validation after reading
        verbose: Print progress

    Returns:
        Dict with profile and validation results
    """
    if verbose:
        print("=" * 70)
        print("11: FETCH - Read & Prepare Data")
        print("=" * 70)

    # Read data
    reader = DataReader()
    df = reader.read(Path(input_path))

    if verbose:
        print(f"  Read: {len(df):,} rows from {input_path}")

    # Profile
    profile = reader.profile_data(entity_col, timestamp_col)

    if verbose:
        print(f"  Entities: {profile.n_entities}")
        print(f"  Signals: {profile.n_signals}")
        print(f"  Lifecycle: {profile.min_lifecycle}-{profile.max_lifecycle}")
        print(f"  Sampling: {'regular' if profile.is_regular_sampling else 'irregular'}")
        if profile.has_nulls:
            print(f"  Nulls: {profile.null_pct:.1f}%")

    # Validate and save
    if validate:
        if verbose:
            print("\n  Validating observations format...")
        result = validate_and_save(input_path, output_path)

        if verbose:
            print(f"  Status: {result.status.value}")
            if result.repairs:
                for repair in result.repairs:
                    print(f"    Fixed: {repair}")
            if result.issues:
                for issue in result.issues:
                    print(f"    Issue: {issue}")
    else:
        df.write_parquet(output_path)
        result = None

    if verbose:
        print(f"\n  Saved: {output_path}")

    return {
        "profile": {
            "n_rows": profile.n_rows,
            "n_entities": profile.n_entities,
            "n_signals": profile.n_signals,
            "lifecycle": f"{profile.min_lifecycle}-{profile.max_lifecycle}",
            "sampling": "regular" if profile.is_regular_sampling else "irregular",
        },
        "validation": result.status.value if result else "skipped",
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="11: Fetch & Prepare Data")
    parser.add_argument('input', help='Path to raw data file')
    parser.add_argument('--output', '-o', default='observations.parquet',
                        help='Output path')
    parser.add_argument('--entity-col', help='Entity/cohort column name')
    parser.add_argument('--timestamp-col', help='Timestamp column name')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip validation')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()

    run(
        args.input,
        output_path=args.output,
        entity_col=args.entity_col,
        timestamp_col=args.timestamp_col,
        validate=not args.no_validate,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
