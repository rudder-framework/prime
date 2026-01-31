"""
Universal Data Transformer

Raw data (any format) -> observations.parquet (PRISM format)

Schema v2.0.0:
- REQUIRED: signal_id, I, value
- OPTIONAL: unit_id (just a label, blank is fine)
"""

import polars as pl
from pathlib import Path
from typing import Optional


# =============================================================================
# SCHEMA CONSTANTS
# =============================================================================

REQUIRED_COLUMNS = ["signal_id", "I", "value"]
OPTIONAL_COLUMNS = ["unit_id"]


# =============================================================================
# VALIDATION
# =============================================================================

def validate_prism_schema(df: pl.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate DataFrame meets PRISM requirements.

    Required: signal_id, I, value
    Optional: unit_id (just a label)
    """
    errors = []

    # Check required columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing REQUIRED column: {col}")

    if errors:
        return False, errors

    # Add unit_id if not present (blank is fine)
    if "unit_id" not in df.columns:
        df = df.with_columns(pl.lit("").alias("unit_id"))

    # Check types
    if df["signal_id"].dtype != pl.String:
        errors.append(f"signal_id must be String, got {df['signal_id'].dtype}")

    if df["I"].dtype not in [pl.UInt32, pl.UInt64, pl.Int32, pl.Int64]:
        errors.append(f"I must be integer, got {df['I'].dtype}")

    if df["value"].dtype not in [pl.Float64, pl.Float32]:
        errors.append(f"value must be Float, got {df['value'].dtype}")

    # Group columns for I checks
    group_cols = ["unit_id", "signal_id"] if "unit_id" in df.columns else ["signal_id"]

    # Check I is sequential per group
    i_check = (
        df.group_by(group_cols)
        .agg([
            pl.col("I").min().alias("min_i"),
            pl.col("I").max().alias("max_i"),
            pl.len().alias("count")
        ])
    )

    non_sequential = i_check.filter(
        (pl.col("max_i") - pl.col("min_i") + 1) != pl.col("count")
    )

    if len(non_sequential) > 0:
        errors.append(f"I is not sequential for {len(non_sequential)} groups")

    # Check min I is 0 per group
    min_i_check = df.group_by(group_cols).agg(pl.col("I").min().alias("min_i"))
    non_zero_start = min_i_check.filter(pl.col("min_i") != 0)

    if len(non_zero_start) > 0:
        errors.append(f"I does not start at 0 for {len(non_zero_start)} groups")

    # Check at least 2 signals
    n_signals = df["signal_id"].n_unique()
    if n_signals < 2:
        errors.append(f"Need >=2 signals, found {n_signals}")

    # Check for nulls in required columns
    for col in ["signal_id", "I"]:
        null_count = df[col].null_count()
        if null_count > 0:
            errors.append(f"{col} has {null_count} null values")

    return len(errors) == 0, errors


# =============================================================================
# TRANSFORM FUNCTIONS
# =============================================================================

def transform_wide_to_long(
    df: pl.DataFrame,
    signal_columns: list[str],
    unit_column: Optional[str] = None,
    index_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Transform wide format (signals as columns) to PRISM long format.

    Wide format:
        unit_id   | timestamp | acc_x | acc_y | temp
        bearing_1 | 0         | 1.23  | 4.56  | 25.0

    Long format:
        unit_id   | I | signal_id | value
        bearing_1 | 0 | acc_x     | 1.23
        bearing_1 | 0 | acc_y     | 4.56
        bearing_1 | 0 | temp      | 25.0

    Args:
        df: Input DataFrame (wide format)
        signal_columns: List of columns that are signals
        unit_column: Optional column to use as unit_id (blank if None)
        index_column: Optional column to use as I (auto-generate if None)
    """

    # Handle unit_id
    if unit_column and unit_column in df.columns:
        df = df.with_columns(pl.col(unit_column).cast(pl.String).alias("unit_id"))
    else:
        # No unit column - use blank (this is fine)
        df = df.with_columns(pl.lit("").alias("unit_id"))

    # Handle I (index)
    if index_column and index_column in df.columns:
        df = df.with_columns(pl.col(index_column).cast(pl.UInt32).alias("I"))
    else:
        # Generate sequential I per unit
        df = df.with_row_index("I")
        df = df.with_columns(pl.col("I").cast(pl.UInt32))

    # Melt wide to long
    id_vars = ["unit_id", "I"]
    df_subset = df.select(id_vars + signal_columns)

    df_long = df_subset.unpivot(
        index=id_vars,
        on=signal_columns,
        variable_name="signal_id",
        value_name="value"
    )

    # Ensure types
    df_long = df_long.with_columns([
        pl.col("unit_id").cast(pl.String),
        pl.col("I").cast(pl.UInt32),
        pl.col("signal_id").cast(pl.String),
        pl.col("value").cast(pl.Float64),
    ])

    # Sort
    df_long = df_long.sort(["unit_id", "I", "signal_id"])

    return df_long


def fix_sparse_index(df: pl.DataFrame) -> pl.DataFrame:
    """
    Fix sparse I values (0, 10, 20...) to sequential (0, 1, 2...).
    """
    group_cols = ["unit_id", "signal_id"] if "unit_id" in df.columns else ["signal_id"]

    df = df.with_columns(
        pl.col("I").rank("dense").over(group_cols).cast(pl.UInt32).alias("I_new")
    )

    df = df.with_columns(
        (pl.col("I_new") - 1).cast(pl.UInt32).alias("I")
    ).drop("I_new")

    return df


def transform_to_prism_format(
    input_path: Path,
    output_path: Path,
    unit_column: Optional[str] = None,
    index_column: Optional[str] = None,
    signal_columns: Optional[list[str]] = None,
    is_wide: bool = True,
    fix_sparse: bool = True,
) -> pl.DataFrame:
    """
    Main entry point: Transform any dataset to PRISM format.

    Args:
        input_path: Path to raw data (parquet, csv)
        output_path: Path for observations.parquet
        unit_column: Column name for unit_id (optional, blank if None)
        index_column: Column name for I (optional, auto-generate if None)
        signal_columns: List of signal column names (auto-detect if None)
        is_wide: True if signals are columns, False if already long
        fix_sparse: True to fix sparse indices

    Returns:
        Validated DataFrame in PRISM format
    """

    # Load data
    if input_path.suffix == ".parquet":
        df = pl.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        df = pl.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported format: {input_path.suffix}")

    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns}")

    # Transform based on format
    if is_wide:
        if signal_columns is None:
            # Auto-detect: numeric columns that aren't unit/index
            exclude = {unit_column, index_column, "timestamp", "date", "time", "unit_id", "I"}
            signal_columns = [
                c for c in df.columns
                if c not in exclude and c is not None
                and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
            print(f"Auto-detected signal columns: {signal_columns}")

        df = transform_wide_to_long(df, signal_columns, unit_column, index_column)
    else:
        # Already long format, ensure column names
        if "signal_id" not in df.columns:
            raise ValueError("Long format requires 'signal_id' column")
        if "I" not in df.columns and index_column:
            df = df.rename({index_column: "I"})
        if "unit_id" not in df.columns:
            if unit_column and unit_column in df.columns:
                df = df.rename({unit_column: "unit_id"})
            else:
                df = df.with_columns(pl.lit("").alias("unit_id"))

    # Fix sparse indices if needed
    if fix_sparse:
        df = fix_sparse_index(df)

    # Validate
    is_valid, errors = validate_prism_schema(df)

    if not is_valid:
        print("\n[X] VALIDATION FAILED:")
        for error in errors:
            print(f"   - {error}")
        raise ValueError(f"Data does not meet PRISM schema requirements: {errors}")

    print("\n[OK] VALIDATION PASSED")

    # Summary stats
    n_units = df["unit_id"].n_unique() if "unit_id" in df.columns else 1
    n_signals = df["signal_id"].n_unique()
    n_obs = df.group_by(["unit_id", "signal_id"]).agg(pl.len()).select(pl.len()).mean().item()

    print(f"   Units: {n_units}" + (" (blank)" if n_units == 1 and df["unit_id"][0] == "" else ""))
    print(f"   Signals: {n_signals}")
    print(f"   Avg observations per group: {n_obs:.0f}")
    print(f"   Total rows: {df.shape[0]:,}")

    # Select final columns in order
    df = df.select(["unit_id", "I", "signal_id", "value"])

    # Write output
    df.write_parquet(output_path)
    print(f"\n[OK] Written: {output_path}")

    return df


# =============================================================================
# Dataset-Specific Transforms
# =============================================================================

def transform_femto(raw_path: Path, output_path: Path) -> pl.DataFrame:
    """
    Transform FEMTO bearing dataset to PRISM format.

    FEMTO: unit_id = bearing, signals = acc_x, acc_y
    """
    df = pl.read_parquet(raw_path)
    print(f"FEMTO raw columns: {df.columns}")

    signal_columns = ["acc_x", "acc_y"]

    return transform_to_prism_format(
        input_path=raw_path,
        output_path=output_path,
        unit_column="entity_id",  # Original column name
        index_column="I",
        signal_columns=signal_columns,
        is_wide=True,
        fix_sparse=True,
    )


def transform_skab(raw_path: Path, output_path: Path) -> pl.DataFrame:
    """
    Transform SKAB dataset to PRISM format.

    SKAB: unit_id = experiment, signals = sensor columns
    """
    df = pl.read_parquet(raw_path)
    print(f"SKAB raw columns: {df.columns}")

    return transform_to_prism_format(
        input_path=raw_path,
        output_path=output_path,
        unit_column="entity_id",
        index_column="I",
        signal_columns=None,  # auto-detect
        is_wide=True,
        fix_sparse=True,
    )


def transform_fama_french(raw_path: Path, output_path: Path) -> pl.DataFrame:
    """
    Transform Fama-French dataset to PRISM format.

    Fama-French: unit_id = blank (single unit), signals = industries
    """
    return transform_to_prism_format(
        input_path=raw_path,
        output_path=output_path,
        unit_column=None,  # No unit column - blank is fine
        index_column=None,  # Auto-generate
        signal_columns=None,  # auto-detect
        is_wide=True,
        fix_sparse=True,
    )


def transform_cmapss(raw_path: Path, output_path: Path) -> pl.DataFrame:
    """
    Transform NASA C-MAPSS turbofan dataset to PRISM format.

    C-MAPSS: unit_id = engine, signals = sensors
    """
    signal_columns = [f"sensor_{i}" for i in range(1, 22)]

    return transform_to_prism_format(
        input_path=raw_path,
        output_path=output_path,
        unit_column="unit_id",
        index_column="cycle",
        signal_columns=signal_columns,
        is_wide=True,
        fix_sparse=True,
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transform data to PRISM format")
    parser.add_argument("input", type=Path, help="Input file (parquet or csv)")
    parser.add_argument("output", type=Path, help="Output observations.parquet")
    parser.add_argument("--unit", type=str, default=None, help="Unit column name (optional)")
    parser.add_argument("--index", type=str, default=None, help="Index column name")
    parser.add_argument("--signals", type=str, nargs="+", default=None, help="Signal column names")
    parser.add_argument("--long", action="store_true", help="Input is already long format")

    args = parser.parse_args()

    transform_to_prism_format(
        input_path=args.input,
        output_path=args.output,
        unit_column=args.unit,
        index_column=args.index,
        signal_columns=args.signals,
        is_wide=not args.long,
    )
