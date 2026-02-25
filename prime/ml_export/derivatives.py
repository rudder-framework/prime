"""
Backward-only causal derivatives for ML consumption.

All math is self-contained. No imports from packages/.
Reads D0 values from parquet, computes D1/D2 using backward finite differences.
"""

import polars as pl
from pathlib import Path


def compute_causal_derivatives(
    source_path: Path,
    dest_path: Path,
    cohort_col: str,
    window_col: str,
    signal_col: str = None,
) -> bool:
    """
    Read a parquet with D0 metric values, compute backward D1/D2 for every
    numeric column, write wide-format result to dest_path.

    Parameters
    ----------
    source_path : Path to the parquet with D0 values (e.g., cohort_geometry.parquet)
    dest_path   : Path to write the ml_ derivative parquet
    cohort_col  : Column name for cohort/engine grouping
    window_col  : Column name for window/time ordering
    signal_col  : If not None, group by this too (for signal-level derivatives)

    Output schema (wide format):
        (cohort_col, [signal_col], window_col, metric1_d0, metric1_d1, metric1_d2, ...)
    """
    if not source_path.exists():
        return False

    df = pl.read_parquet(source_path)

    # Identify grouping columns and numeric metric columns
    group_cols = [cohort_col]
    if signal_col and signal_col in df.columns:
        group_cols.append(signal_col)

    # Only keep group cols that actually exist in the dataframe
    group_cols = [c for c in group_cols if c in df.columns]

    id_cols = group_cols + [window_col]
    numeric_cols = [
        c for c in df.columns
        if c not in id_cols and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]

    if not numeric_cols:
        return False

    # Sort by group + window to ensure correct ordering
    sort_cols = [c for c in group_cols + [window_col] if c in df.columns]
    df = df.sort(sort_cols)

    # Compute D1 and D2 for each numeric column, per group
    new_columns = []
    for col in numeric_cols:
        d1_name = f"{col}_d1"
        d2_name = f"{col}_d2"

        if group_cols:
            # Backward difference within each group
            d1_expr = pl.col(col).diff(n=1).over(group_cols).alias(d1_name)
            d2_expr = pl.col(col).diff(n=1).diff(n=1).over(group_cols).alias(d2_name)
        else:
            # No grouping — global backward difference
            d1_expr = pl.col(col).diff(n=1).alias(d1_name)
            d2_expr = pl.col(col).diff(n=1).diff(n=1).alias(d2_name)

        new_columns.extend([d1_expr, d2_expr])

    existing_cols = [c for c in id_cols if c in df.columns]
    result = df.select(existing_cols + [pl.col(c) for c in numeric_cols] + new_columns)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(dest_path)
    return True
