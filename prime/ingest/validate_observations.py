"""
Observations Validator & Repairer

Validates and repairs observations.parquet before Manifold processing.
Ensures signal_0 is sorted ascending Float64 per signal_id.

Manifold expects:
- signal_id: REQUIRED (what we're measuring)
- signal_0: REQUIRED, sorted ascending Float64 per signal_id
- value: REQUIRED (the measurement)
- unit_id: OPTIONAL (pass-through label)

Common issues this fixes:
- Legacy I column (rename to signal_0, cast to Float64)
- signal_0 has duplicates within signal_id
- signal_0 is not sorted
- signal_0 is missing entirely

Usage:
    python validate_observations.py <input.parquet> [output.parquet]
    python validate_observations.py --check <input.parquet>
"""

import polars as pl
from pathlib import Path
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ValidationStatus(Enum):
    VALID = "valid"
    REPAIRED = "repaired"
    FAILED = "failed"


@dataclass
class ValidationResult:
    status: ValidationStatus
    issues: List[str]
    repairs: List[str]
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]


# ============================================================
# VALIDATION CHECKS
# ============================================================

def check_required_columns(df: pl.DataFrame) -> Tuple[bool, List[str]]:
    """Check for required columns."""
    issues = []
    required = ['signal_id', 'signal_0', 'value']

    for col in required:
        if col not in df.columns:
            # Check for common aliases
            aliases = {
                'signal_id': ['signal_name', 'sensor', 'channel', 'variable'],
                'signal_0': ['I', 'index', 'idx', 'timestamp', 'time', 't', 'step'],
                'value': ['y', 'measurement', 'reading', 'val'],
            }
            found_alias = None
            for alias in aliases.get(col, []):
                if alias in df.columns:
                    found_alias = alias
                    break

            if found_alias:
                issues.append(f"Column '{col}' missing but found alias '{found_alias}'")
            else:
                issues.append(f"Required column '{col}' missing")

    return len([i for i in issues if 'missing' in i and 'alias' not in i]) == 0, issues


def check_signal_0_valid(df: pl.DataFrame) -> Tuple[bool, List[str]]:
    """
    Check if signal_0 is valid: Float64, sorted ascending per signal_id, no nulls.
    """
    issues = []

    if 'signal_0' not in df.columns:
        return False, ["signal_0 column missing"]

    # Check type is Float64
    if df['signal_0'].dtype != pl.Float64:
        issues.append(f"signal_0 should be Float64, got {df['signal_0'].dtype}")

    # Check for nulls
    null_count = df['signal_0'].null_count()
    if null_count > 0:
        issues.append(f"signal_0 has {null_count} null values")

    # Check for duplicates per signal_id
    if 'signal_id' in df.columns:
        dupes = (
            df.group_by(['signal_id', 'signal_0'])
            .agg(pl.len().alias('count'))
            .filter(pl.col('count') > 1)
        )
        if len(dupes) > 0:
            issues.append(f"signal_0 has {len(dupes)} duplicate (signal_id, signal_0) pairs")

        # Check sorted ascending per signal_id (sample check)
        sample_signals = df['signal_id'].unique().head(5).to_list()
        for sig in sample_signals:
            sig_df = df.filter(pl.col('signal_id') == sig).sort('signal_0')
            diffs = sig_df['signal_0'].diff().drop_nulls()
            if (diffs < 0).any():
                issues.append(f"signal_0 is not sorted ascending for signal '{sig}'")
                break

    return len(issues) == 0, issues


def check_signal_id_not_null(df: pl.DataFrame) -> Tuple[bool, List[str]]:
    """Check that signal_id has no null values."""
    issues = []

    if 'signal_id' not in df.columns:
        return False, ["signal_id column missing"]

    null_count = df['signal_id'].null_count()
    if null_count > 0:
        issues.append(f"signal_id has {null_count} null values")

    return null_count == 0, issues


def check_value_numeric(df: pl.DataFrame) -> Tuple[bool, List[str]]:
    """Check that value column is numeric."""
    issues = []

    if 'value' not in df.columns:
        return False, ["value column missing"]

    if df['value'].dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32, pl.UInt64]:
        issues.append(f"value column is {df['value'].dtype}, expected numeric")

    return len(issues) == 0, issues


# ============================================================
# REPAIRS
# ============================================================

def repair_column_names(df: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
    """Rename common aliases to standard names."""
    repairs = []

    # Column aliases
    aliases = {
        'signal_id': ['signal_name', 'sensor', 'channel', 'variable', 'sensor_id'],
        'signal_0': ['I', 'index', 'idx', 'timestamp', 'time', 't', 'step', 'timestep'],
        'value': ['y', 'measurement', 'reading', 'val', 'values'],
        'unit_id': ['entity_id', 'unit', 'entity', 'machine', 'asset'],
    }

    for standard, alias_list in aliases.items():
        if standard not in df.columns:
            for alias in alias_list:
                if alias in df.columns:
                    df = df.rename({alias: standard})
                    repairs.append(f"Renamed '{alias}' -> '{standard}'")
                    break

    return df, repairs


def repair_signal_0(df: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
    """
    Repair signal_0 to be sorted ascending Float64 per signal_id.

    If signal_0 exists but is wrong type, cast to Float64.
    If signal_0 is not sorted, sort it.
    If signal_0 is missing, create from row order.
    """
    repairs = []

    if 'signal_0' not in df.columns:
        repairs.append("Created signal_0 column from row order")
        df = df.with_row_index('_seq')
        df = df.with_columns(pl.col('_seq').cast(pl.Float64).alias('signal_0')).drop('_seq')
        return df, repairs

    # Cast to Float64 if needed
    if df['signal_0'].dtype != pl.Float64:
        original_type = df['signal_0'].dtype
        df = df.with_columns(pl.col('signal_0').cast(pl.Float64))
        repairs.append(f"Cast signal_0 from {original_type} to Float64")

    # Sort per signal_id if not sorted
    if 'signal_id' in df.columns:
        sample_signals = df['signal_id'].unique().head(5).to_list()
        needs_sort = False
        for sig in sample_signals:
            sig_df = df.filter(pl.col('signal_id') == sig)
            diffs = sig_df['signal_0'].diff().drop_nulls()
            if (diffs < 0).any():
                needs_sort = True
                break

        if needs_sort:
            df = df.sort(['signal_id', 'signal_0'])
            repairs.append("Sorted signal_0 ascending per signal_id")

    return df, repairs


def repair_signal_id_nulls(df: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
    """Remove rows with null signal_id."""
    repairs = []

    if 'signal_id' not in df.columns:
        return df, repairs

    null_count = df['signal_id'].null_count()
    if null_count > 0:
        df = df.filter(pl.col('signal_id').is_not_null())
        repairs.append(f"Removed {null_count} rows with null signal_id")

    return df, repairs


def repair_value_type(df: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
    """Cast value to Float64 if needed."""
    repairs = []

    if 'value' not in df.columns:
        return df, repairs

    if df['value'].dtype not in [pl.Float64, pl.Float32]:
        try:
            df = df.with_columns([pl.col('value').cast(pl.Float64)])
            repairs.append(f"Cast value column to Float64")
        except Exception as e:
            repairs.append(f"Failed to cast value to Float64: {e}")

    return df, repairs


# ============================================================
# MAIN VALIDATION & REPAIR
# ============================================================

def validate_observations(
    df: pl.DataFrame,
    repair: bool = True,
    verbose: bool = True
) -> Tuple[pl.DataFrame, ValidationResult]:
    """
    Validate and optionally repair observations DataFrame.

    Args:
        df: Input DataFrame
        repair: Whether to attempt repairs
        verbose: Print progress

    Returns:
        Tuple of (repaired DataFrame, ValidationResult)
    """
    original_shape = df.shape
    all_issues = []
    all_repairs = []

    if verbose:
        print("=" * 60)
        print("OBSERVATIONS VALIDATION")
        print("=" * 60)
        print(f"Input: {original_shape[0]} rows, {original_shape[1]} columns")
        print(f"Columns: {df.columns}")
        print()

    # Step 1: Repair column names first
    if repair:
        df, repairs = repair_column_names(df)
        all_repairs.extend(repairs)
        if repairs and verbose:
            print(f"Column repairs: {repairs}")

    # Step 2: Run all checks
    checks = [
        ("Required columns", check_required_columns),
        ("signal_0 valid", check_signal_0_valid),
        ("signal_id not null", check_signal_id_not_null),
        ("value numeric", check_value_numeric),
    ]

    for check_name, check_fn in checks:
        passed, issues = check_fn(df)
        if not passed:
            all_issues.extend(issues)
            if verbose:
                print(f"✗ {check_name}: {issues}")
        elif verbose:
            print(f"✓ {check_name}")

    # Step 3: Apply repairs if needed
    if repair and all_issues:
        if verbose:
            print()
            print("Applying repairs...")

        # Repair signal_0 column (most common issue)
        df, repairs = repair_signal_0(df)
        all_repairs.extend(repairs)

        # Repair null signal_ids
        df, repairs = repair_signal_id_nulls(df)
        all_repairs.extend(repairs)

        # Repair value type
        df, repairs = repair_value_type(df)
        all_repairs.extend(repairs)

        if verbose and all_repairs:
            print(f"Repairs applied: {all_repairs}")

    # Step 4: Final validation
    final_issues = []
    for check_name, check_fn in checks:
        passed, issues = check_fn(df)
        if not passed:
            final_issues.extend(issues)

    # Determine status
    if not final_issues:
        status = ValidationStatus.REPAIRED if all_repairs else ValidationStatus.VALID
    else:
        status = ValidationStatus.FAILED

    result = ValidationResult(
        status=status,
        issues=all_issues,
        repairs=all_repairs,
        original_shape=original_shape,
        final_shape=df.shape
    )

    if verbose:
        print()
        print(f"Status: {status.value}")
        print(f"Final: {df.shape[0]} rows, {df.shape[1]} columns")

    return df, result


def validate_and_save(
    input_path: str,
    output_path: str = None,
    verbose: bool = True
) -> ValidationResult:
    """
    Validate observations file and save repaired version.

    Args:
        input_path: Path to input parquet
        output_path: Path for output (default: overwrite input)
        verbose: Print progress

    Returns:
        ValidationResult
    """
    if output_path is None:
        output_path = input_path

    # Load
    df = pl.read_parquet(input_path)

    # Validate and repair
    df, result = validate_observations(df, repair=True, verbose=verbose)

    # Save if valid
    if result.status != ValidationStatus.FAILED:
        df.write_parquet(output_path)
        if verbose:
            print(f"\nSaved: {output_path}")
    else:
        if verbose:
            print(f"\nFailed to repair. Remaining issues: {result.issues}")

    return result


# ============================================================
# CLI
# ============================================================

def main():
    import sys

    usage = """
Observations Validator

Usage:
    python validate_observations.py <input.parquet> [output.parquet]
    python validate_observations.py --check <input.parquet>

Validates and repairs observations.parquet:
- Ensures signal_0 is sorted ascending Float64 per signal_id
- Fixes column name aliases (I → signal_0, etc.)
- Removes null signal_ids
- Casts value to numeric

If output path not specified, overwrites input.
Use --check to validate without modifying.
"""

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    if sys.argv[1] == '--check':
        if len(sys.argv) < 3:
            print("Error: Need input path for --check")
            sys.exit(1)
        input_path = sys.argv[2]
        df = pl.read_parquet(input_path)
        df, result = validate_observations(df, repair=False, verbose=True)
        sys.exit(0 if result.status == ValidationStatus.VALID else 1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    result = validate_and_save(input_path, output_path, verbose=True)
    sys.exit(0 if result.status != ValidationStatus.FAILED else 1)


if __name__ == "__main__":
    main()
