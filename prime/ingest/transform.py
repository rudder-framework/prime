"""
Universal Data Transformer

Raw data (any format) -> observations.parquet (canonical format)

Schema v3.0:
- REQUIRED: signal_id, signal_0, value
- OPTIONAL: cohort (grouping key, replaces legacy unit_id)
"""

import polars as pl
from pathlib import Path
from typing import Optional


# =============================================================================
# SCHEMA CONSTANTS
# =============================================================================

REQUIRED_COLUMNS = ["signal_id", "signal_0", "value"]
OPTIONAL_COLUMNS = ["cohort"]


# =============================================================================
# VALIDATION
# =============================================================================

def validate_manifold_schema(df: pl.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate DataFrame meets Manifold requirements.

    Required: signal_id, signal_0, value
    Optional: cohort (grouping key, replaces legacy unit_id)
    """
    errors = []

    # Check required columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing REQUIRED column: {col}")

    if errors:
        return False, errors

    # Determine grouping column (cohort is new, unit_id is legacy)
    if "cohort" in df.columns:
        group_col = "cohort"
    elif "unit_id" in df.columns:
        group_col = "unit_id"
    else:
        group_col = None

    # Check types
    if df["signal_id"].dtype not in [pl.String, pl.Utf8]:
        errors.append(f"signal_id must be String, got {df['signal_id'].dtype}")

    if df["signal_0"].dtype != pl.Float64:
        errors.append(f"signal_0 must be Float64, got {df['signal_0'].dtype}")

    if df["value"].dtype not in [pl.Float64, pl.Float32]:
        errors.append(f"value must be Float, got {df['value'].dtype}")

    # Group columns for signal_0 checks
    group_cols = [group_col, "signal_id"] if group_col else ["signal_id"]

    # Check signal_0 is sorted ascending per group (no nulls)
    sort_check = (
        df.group_by(group_cols)
        .agg([
            (pl.col("signal_0").diff().drop_nulls() < 0).any().alias("has_unsorted"),
            pl.col("signal_0").null_count().alias("null_count"),
        ])
    )

    unsorted = sort_check.filter(pl.col("has_unsorted"))
    if len(unsorted) > 0:
        errors.append(f"signal_0 is not sorted ascending for {len(unsorted)} groups")

    has_nulls = sort_check.filter(pl.col("null_count") > 0)
    if len(has_nulls) > 0:
        errors.append(f"signal_0 has nulls in {len(has_nulls)} groups")

    # Check at least 2 signals
    n_signals = df["signal_id"].n_unique()
    if n_signals < 2:
        errors.append(f"Need >=2 signals, found {n_signals}")

    # Check for nulls in required columns
    for col in ["signal_id", "signal_0"]:
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
    default_cohort: str = "",
) -> pl.DataFrame:
    """
    Transform wide format (signals as columns) to canonical long format.

    Wide format:
        cohort    | timestamp | acc_x | acc_y | temp
        bearing_1 | 0         | 1.23  | 4.56  | 25.0

    Long format:
        cohort    | signal_0 | signal_id | value
        bearing_1 | 0.0      | acc_x     | 1.23
        bearing_1 | 0.0      | acc_y     | 4.56
        bearing_1 | 0.0      | temp      | 25.0

    Args:
        df: Input DataFrame (wide format)
        signal_columns: List of columns that are signals
        unit_column: Optional column to use as cohort (blank if None)
        index_column: Optional column to use as signal_0 (auto-generate if None)
    """

    # Handle cohort
    if unit_column and unit_column in df.columns:
        df = df.with_columns(pl.col(unit_column).cast(pl.String).alias("cohort"))
    else:
        # No unit column — use default_cohort (derived from directory name or empty)
        df = df.with_columns(pl.lit(default_cohort).alias("cohort"))

    # Handle signal_0 (index)
    if index_column and index_column in df.columns:
        df = df.with_columns(pl.col(index_column).cast(pl.Float64).alias("signal_0"))
    else:
        # Generate sequential signal_0 per unit
        df = df.with_row_index("_seq")
        df = df.with_columns(pl.col("_seq").cast(pl.Float64).alias("signal_0")).drop("_seq")

    # Melt wide to long
    id_vars = ["cohort", "signal_0"]
    df_subset = df.select(id_vars + signal_columns)

    df_long = df_subset.unpivot(
        index=id_vars,
        on=signal_columns,
        variable_name="signal_id",
        value_name="value"
    )

    # Ensure types
    df_long = df_long.with_columns([
        pl.col("cohort").cast(pl.String),
        pl.col("signal_0").cast(pl.Float64),
        pl.col("signal_id").cast(pl.String),
        pl.col("value").cast(pl.Float64),
    ])

    # Sort
    df_long = df_long.sort(["cohort", "signal_0", "signal_id"])

    return df_long


def ensure_signal_0_sorted(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure signal_0 is sorted ascending per group. Drop exact duplicates.
    """
    if "cohort" in df.columns:
        group_cols = ["cohort", "signal_id"]
    elif "unit_id" in df.columns:
        group_cols = ["unit_id", "signal_id"]
    else:
        group_cols = ["signal_id"]

    df = df.unique(subset=group_cols + ["signal_0"], keep="first")
    df = df.sort(group_cols + ["signal_0"])

    return df


def transform_to_manifold_format(
    input_path: Path,
    output_path: Path,
    unit_column: Optional[str] = None,
    index_column: Optional[str] = None,
    signal_columns: Optional[list[str]] = None,
    is_wide: bool = True,
    fix_sparse: bool = True,
) -> pl.DataFrame:
    """
    Main entry point: Transform any dataset to canonical format.

    Args:
        input_path: Path to raw data (parquet, csv)
        output_path: Path for observations.parquet
        unit_column: Column name for cohort (optional, blank if None)
        index_column: Column name for signal_0 (optional, auto-generate if None)
        signal_columns: List of signal column names (auto-detect if None)
        is_wide: True if signals are columns, False if already long
        fix_sparse: True to sort and deduplicate signal_0

    Returns:
        Validated DataFrame in canonical format
    """

    # Load data
    if input_path.suffix == ".parquet":
        df = pl.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        df = pl.read_csv(input_path)
    elif input_path.suffix == ".txt":
        df = pl.read_csv(input_path, separator=" ", has_header=False, truncate_ragged_lines=True)
    else:
        raise ValueError(f"Unsupported format: {input_path.suffix}")

    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns}")

    # Derive default cohort from output directory name
    default_cohort = output_path.parent.name if output_path else ""

    # Transform based on format
    if is_wide:
        if signal_columns is None:
            # Auto-detect: numeric columns that aren't unit/index
            exclude = {unit_column, index_column, "timestamp", "date", "time", "cohort", "unit_id", "signal_0"}
            signal_columns = [
                c for c in df.columns
                if c not in exclude and c is not None
                and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
            print(f"Auto-detected signal columns: {signal_columns}")

        df = transform_wide_to_long(df, signal_columns, unit_column, index_column, default_cohort)
    else:
        # Already long format, ensure column names
        if "signal_id" not in df.columns:
            raise ValueError("Long format requires 'signal_id' column")
        if "signal_0" not in df.columns:
            if "I" in df.columns:
                df = df.rename({"I": "signal_0"})
            elif index_column:
                df = df.rename({index_column: "signal_0"})
        if "cohort" not in df.columns:
            if unit_column and unit_column in df.columns:
                df = df.rename({unit_column: "cohort"})
            elif "unit_id" in df.columns:
                df = df.rename({"unit_id": "cohort"})
            else:
                df = df.with_columns(pl.lit(default_cohort).alias("cohort"))

    # Ensure signal_0 is Float64
    if "signal_0" in df.columns and df["signal_0"].dtype != pl.Float64:
        df = df.with_columns(pl.col("signal_0").cast(pl.Float64))

    # Sort and deduplicate signal_0 if needed
    if fix_sparse:
        df = ensure_signal_0_sorted(df)

    # Validate
    is_valid, errors = validate_manifold_schema(df)

    if not is_valid:
        print("\n[X] VALIDATION FAILED:")
        for error in errors:
            print(f"   - {error}")
        raise ValueError(f"Data does not meet schema requirements: {errors}")

    print("\n[OK] VALIDATION PASSED")

    # Summary stats
    group_col = "cohort" if "cohort" in df.columns else "unit_id" if "unit_id" in df.columns else None
    if group_col:
        n_cohorts = df[group_col].n_unique()
        n_signals = df["signal_id"].n_unique()
        n_obs = df.group_by([group_col, "signal_id"]).agg(pl.len()).select(pl.len()).mean().item()
        print(f"   Cohorts: {n_cohorts}" + (" (blank)" if n_cohorts == 1 and df[group_col][0] == "" else ""))
    else:
        n_signals = df["signal_id"].n_unique()
        n_obs = df.group_by(["signal_id"]).agg(pl.len()).select(pl.len()).mean().item()

    print(f"   Signals: {n_signals}")
    print(f"   Avg observations per group: {n_obs:.0f}")
    print(f"   Total rows: {df.shape[0]:,}")

    # Ensure unit column exists (default to "" for dimensionless/unknown)
    if "unit" not in df.columns:
        df = df.with_columns(pl.lit("").alias("unit"))

    # Select final columns in order
    final_cols = ["cohort", "signal_0", "signal_id", "value", "unit"] if "cohort" in df.columns else ["signal_0", "signal_id", "value", "unit"]
    df = df.select(final_cols)

    # Write output
    df.write_parquet(output_path)
    print(f"\n[OK] Written: {output_path}")

    # Write signals.parquet alongside observations.parquet
    from prime.ingest.signal_metadata import write_signal_metadata
    write_signal_metadata(df, output_path.parent)

    return df


# =============================================================================
# Dataset-Specific Transforms
# =============================================================================

def transform_femto(raw_path: Path, output_path: Path) -> pl.DataFrame:
    """
    Transform FEMTO bearing dataset to canonical format.

    FEMTO: cohort = bearing, signals = acc_x, acc_y
    """
    df = pl.read_parquet(raw_path)
    print(f"FEMTO raw columns: {df.columns}")

    signal_columns = ["acc_x", "acc_y"]

    return transform_to_manifold_format(
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
    Transform SKAB dataset to canonical format.

    SKAB: cohort = experiment, signals = sensor columns
    """
    df = pl.read_parquet(raw_path)
    print(f"SKAB raw columns: {df.columns}")

    return transform_to_manifold_format(
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
    Transform Fama-French dataset to canonical format.

    Fama-French: cohort = blank (single cohort), signals = industries
    """
    return transform_to_manifold_format(
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
    Transform NASA C-MAPSS turbofan dataset to canonical format.

    C-MAPSS: cohort = engine, signals = sensors
    """
    signal_columns = [f"sensor_{i}" for i in range(1, 22)]

    return transform_to_manifold_format(
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

    parser = argparse.ArgumentParser(description="Transform data to canonical format")
    parser.add_argument("input", type=Path, help="Input file (parquet or csv)")
    parser.add_argument("output", type=Path, help="Output observations.parquet")
    parser.add_argument("--unit", type=str, default=None, help="Unit column name (optional)")
    parser.add_argument("--index", type=str, default=None, help="Index column name")
    parser.add_argument("--signals", type=str, nargs="+", default=None, help="Signal column names")
    parser.add_argument("--long", action="store_true", help="Input is already long format")

    args = parser.parse_args()

    transform_to_manifold_format(
        input_path=args.input,
        output_path=args.output,
        unit_column=args.unit,
        index_column=args.index,
        signal_columns=args.signals,
        is_wide=not args.long,
    )
