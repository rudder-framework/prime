"""
Schema Enforcer

Validates and transforms observations.parquet to Manifold v2.0.0 schema.
Prime produces correct data. Manifold shouldn't fix bad orders.

v2.0.0 Schema:
- unit_id (String, optional) - blank is fine
- signal_id (String, required)
- I (UInt32, required) - sequential per unit+signal
- value (Float64, required)

Legacy columns that get transformed:
- entity_id → unit_id
- timestamp → I (converted to sequential index)
- index → I
- obs_date → I
- y → value

Usage:
    python -m prime.ingest.schema_enforcer <observations.parquet>
    python -m prime.ingest.schema_enforcer --fix <observations.parquet>
"""

import polars as pl
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass


# ============================================================
# SCHEMA DEFINITION
# ============================================================

REQUIRED_COLUMNS = ['signal_id', 'I', 'value']
OPTIONAL_COLUMNS = ['unit_id']
ALL_COLUMNS = OPTIONAL_COLUMNS + REQUIRED_COLUMNS

# Legacy column mappings
COLUMN_ALIASES = {
    'unit_id': ['entity_id', 'unit', 'entity', 'asset_id', 'machine_id'],
    'signal_id': ['indicator_id', 'signal_name', 'sensor_id', 'feature', 'variable'],
    'I': ['timestamp', 'index', 'obs_date', 'time', 'cycle', 't', 'step'],
    'value': ['y', 'measurement', 'reading', 'val'],
}

# Expected types
COLUMN_TYPES = {
    'unit_id': pl.String,
    'signal_id': pl.String,
    'I': pl.UInt32,
    'value': pl.Float64,
}


@dataclass
class SchemaReport:
    """Schema validation report."""
    valid: bool
    columns_found: List[str]
    columns_missing: List[str]
    columns_renamed: dict  # old_name -> new_name
    type_issues: List[str]
    fixes_applied: List[str]


def detect_column_mapping(df: pl.DataFrame) -> dict:
    """
    Detect which columns map to the Manifold v2.0.0 schema.

    Returns:
        Dict mapping v2.0.0 names to actual column names in df
    """
    mapping = {}

    for target_col, aliases in COLUMN_ALIASES.items():
        # Check if target already exists
        if target_col in df.columns:
            mapping[target_col] = target_col
            continue

        # Check aliases
        for alias in aliases:
            if alias in df.columns:
                mapping[target_col] = alias
                break

    return mapping


def validate_schema(path: str, verbose: bool = True) -> SchemaReport:
    """
    Validate observations.parquet against Manifold v2.0.0 schema.

    Args:
        path: Path to observations.parquet
        verbose: Print detailed output

    Returns:
        SchemaReport with validation results
    """
    df = pl.read_parquet(path)

    report = SchemaReport(
        valid=True,
        columns_found=[],
        columns_missing=[],
        columns_renamed={},
        type_issues=[],
        fixes_applied=[],
    )

    # Detect column mapping
    mapping = detect_column_mapping(df)

    if verbose:
        print("=" * 60)
        print("SCHEMA VALIDATION")
        print("=" * 60)
        print(f"File: {path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns}")
        print()

    # Check required columns
    for col in REQUIRED_COLUMNS:
        if col in mapping:
            actual_col = mapping[col]
            report.columns_found.append(col)
            if actual_col != col:
                report.columns_renamed[actual_col] = col
                if verbose:
                    print(f"  {col}: found as '{actual_col}' (needs rename)")
            else:
                if verbose:
                    print(f"  {col}: OK")
        else:
            report.columns_missing.append(col)
            report.valid = False
            if verbose:
                print(f"  {col}: MISSING")

    # Check optional columns
    for col in OPTIONAL_COLUMNS:
        if col in mapping:
            actual_col = mapping[col]
            report.columns_found.append(col)
            if actual_col != col:
                report.columns_renamed[actual_col] = col
                if verbose:
                    print(f"  {col}: found as '{actual_col}' (needs rename)")
            else:
                if verbose:
                    print(f"  {col}: OK")
        else:
            if verbose:
                print(f"  {col}: not present (will add blank)")

    # Check types
    for col, expected_type in COLUMN_TYPES.items():
        if col in mapping:
            actual_col = mapping[col]
            actual_type = df[actual_col].dtype
            if actual_type != expected_type:
                report.type_issues.append(f"{col}: {actual_type} -> {expected_type}")
                if verbose:
                    print(f"  {col} type: {actual_type} (needs cast to {expected_type})")

    # Check for null signal_id (signal_id CANNOT be null, unit_id CAN be null)
    if 'signal_id' in mapping:
        signal_col = mapping['signal_id']
        null_count = df[signal_col].null_count()
        if null_count > 0:
            report.type_issues.append(f"signal_id has {null_count} null values (will be dropped)")
            if verbose:
                print(f"  signal_id: {null_count} null values (will be dropped)")

    if verbose:
        print()
        if report.valid and not report.columns_renamed and not report.type_issues:
            print("RESULT: Schema is valid v2.0.0")
        elif report.valid:
            print("RESULT: Schema can be fixed automatically")
        else:
            print(f"RESULT: Schema INVALID - missing: {report.columns_missing}")
        print("=" * 60)

    return report


def enforce_schema(
    path: str,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[pl.DataFrame, SchemaReport]:
    """
    Transform observations.parquet to Manifold v2.0.0 schema.

    Args:
        path: Path to input observations.parquet
        output_path: Path for output (default: overwrite input)
        verbose: Print detailed output

    Returns:
        Tuple of (transformed DataFrame, SchemaReport)
    """
    df = pl.read_parquet(path)
    output_path = output_path or path

    report = SchemaReport(
        valid=True,
        columns_found=[],
        columns_missing=[],
        columns_renamed={},
        type_issues=[],
        fixes_applied=[],
    )

    if verbose:
        print("=" * 60)
        print("SCHEMA ENFORCEMENT")
        print("=" * 60)
        print(f"Input: {path}")
        print(f"Original columns: {df.columns}")

    # Detect column mapping
    mapping = detect_column_mapping(df)

    # Check for required columns
    for col in REQUIRED_COLUMNS:
        if col not in mapping:
            report.valid = False
            report.columns_missing.append(col)
            if verbose:
                print(f"  ERROR: Cannot find {col} or any alias")

    if not report.valid:
        if verbose:
            print(f"  FAILED: Missing required columns: {report.columns_missing}")
        return df, report

    # Apply renames
    for target_col in ['unit_id', 'signal_id', 'I', 'value']:
        if target_col in mapping:
            actual_col = mapping[target_col]
            if actual_col != target_col:
                df = df.rename({actual_col: target_col})
                report.columns_renamed[actual_col] = target_col
                report.fixes_applied.append(f"Renamed {actual_col} -> {target_col}")
                if verbose:
                    print(f"  Renamed: {actual_col} -> {target_col}")

    # Add missing unit_id
    if 'unit_id' not in df.columns:
        df = df.with_columns(pl.lit('').alias('unit_id'))
        report.fixes_applied.append("Added blank unit_id")
        if verbose:
            print("  Added: blank unit_id")

    # Drop null signal_ids (signal_id CANNOT be null, unit_id CAN be null)
    if 'signal_id' in df.columns:
        null_count = df['signal_id'].null_count()
        if null_count > 0:
            df = df.filter(pl.col('signal_id').is_not_null())
            report.fixes_applied.append(f"Dropped {null_count} rows with null signal_id")
            if verbose:
                print(f"  Dropped: {null_count} rows with null signal_id")

    # Convert I to sequential index (always recreate to ensure proper sequencing)
    if 'I' in df.columns:
        # Sort by unit, signal, then original I (whatever type it is)
        df = df.sort(['unit_id', 'signal_id', 'I'])
        # Create sequential I starting from 0
        df = df.with_columns(
            pl.lit(0).cum_count().over(['unit_id', 'signal_id']).alias('I_seq')
        )
        df = df.drop('I').rename({'I_seq': 'I'})
        report.fixes_applied.append("Recreated I as sequential index")
        if verbose:
            print("  Recreated: I as sequential index (0-based)")

    # Cast types
    df = df.with_columns([
        pl.col('unit_id').cast(pl.String),
        pl.col('signal_id').cast(pl.String),
        pl.col('I').cast(pl.UInt32),
        pl.col('value').cast(pl.Float64),
    ])
    report.fixes_applied.append("Cast columns to correct types")

    # Select final columns in order
    df = df.select(['unit_id', 'signal_id', 'I', 'value'])

    # Save
    df.write_parquet(output_path)

    if verbose:
        print(f"  Output: {output_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns}")
        print("=" * 60)

    return df, report


def enforce_all_in_directory(
    directory: str,
    recursive: bool = True,
    verbose: bool = True
) -> dict:
    """
    Find and enforce schema on all observations.parquet files in directory.

    Args:
        directory: Root directory to search
        recursive: Search subdirectories
        verbose: Print progress

    Returns:
        Dict of path -> SchemaReport
    """
    results = {}
    pattern = '**/observations.parquet' if recursive else 'observations.parquet'

    for path in Path(directory).glob(pattern):
        if verbose:
            print(f"\nProcessing: {path}")

        try:
            _, report = enforce_schema(str(path), verbose=False)
            results[str(path)] = report

            if verbose:
                if report.fixes_applied:
                    print(f"  Fixed: {report.fixes_applied}")
                else:
                    print("  Already valid")
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            results[str(path)] = None

    return results


# ============================================================
# CLI
# ============================================================

def main():
    import sys

    usage = """
Schema Enforcer

Validates and transforms observations.parquet to Manifold v2.0.0 schema.

Usage:
    python -m prime.ingest.schema_enforcer <observations.parquet>
    python -m prime.ingest.schema_enforcer --fix <observations.parquet>
    python -m prime.ingest.schema_enforcer --fix-all <directory>

v2.0.0 Schema:
    unit_id   (String, optional)
    signal_id (String, required)
    I         (UInt32, required)
    value     (Float64, required)
"""

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    if sys.argv[1] == '--fix':
        if len(sys.argv) < 3:
            print("Usage: --fix <observations.parquet>")
            sys.exit(1)
        enforce_schema(sys.argv[2])

    elif sys.argv[1] == '--fix-all':
        if len(sys.argv) < 3:
            print("Usage: --fix-all <directory>")
            sys.exit(1)
        enforce_all_in_directory(sys.argv[2])

    else:
        validate_schema(sys.argv[1])


if __name__ == "__main__":
    main()
