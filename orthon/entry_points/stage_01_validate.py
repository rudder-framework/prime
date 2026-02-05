"""
01: Validation Entry Point
==========================

Pure orchestration - calls validation module.
Removes constants, duplicates, and flags problematic signals.

Stages: observations.parquet → observations_validated.parquet

Principle: "Garbage in, REJECTED"
"""

import polars as pl
from pathlib import Path
from typing import Optional

# Import validation from core module
from orthon.core.validation import (
    SignalValidator,
    ValidationConfig,
    ValidationReport,
    validate_observations,
)


def run(
    observations_path: str,
    output_path: Optional[str] = None,
    strict: bool = True,
    verbose: bool = True,
) -> ValidationReport:
    """
    Run validation on observations.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for validated data (optional)
        strict: If True, exclude bad signals. If False, only warn.
        verbose: Print progress

    Returns:
        ValidationReport with validation results
    """
    if verbose:
        print("=" * 70)
        print("01: VALIDATION")
        print("Removing constants, duplicates, and flagging issues")
        print("=" * 70)

    # Load data
    df = pl.read_parquet(observations_path)
    if verbose:
        n_signals = df['signal_id'].n_unique() if 'signal_id' in df.columns else 0
        print(f"Loaded: {len(df):,} rows, {n_signals} signals")

    # Configure validation
    config = ValidationConfig.strict_mode() if strict else ValidationConfig.permissive()
    validator = SignalValidator(config)

    # Run validation
    validated_df, report = validator.validate(df)

    if verbose:
        print(f"\nValid: {report.valid_signals} signals")
        print(f"Excluded: {len(report.excluded)} signals")
        print(f"Warnings: {len(report.warnings)} signals")

        if report.excluded:
            print("\nExcluded signals:")
            for sv in report.excluded[:10]:
                print(f"  ✗ {sv.signal_id}: {sv.issue.value}")
            if len(report.excluded) > 10:
                print(f"  ... and {len(report.excluded) - 10} more")

    # Save validated data
    if output_path:
        validated_df.write_parquet(output_path)
        if verbose:
            print(f"\nSaved: {output_path}")

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="01: Validation")
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('--output', '-o', help='Output path for validated data')
    parser.add_argument('--permissive', action='store_true', help='Warn only, do not exclude')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        output_path=args.output,
        strict=not args.permissive,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
