"""
Axis selection: reorder observations by a user-specified signal.

Default axis is time (row order). When a different signal is chosen,
that signal's values become signal_0 and the old signal_0 becomes
a new signal.

Usage from pipeline:
    reaxis_observations(observations_path, "x", x_observations_path)
"""

import polars as pl
from pathlib import Path


def reaxis_observations(
    observations_path: Path,
    axis_signal: str,
    output_path: Path,
    time_name: str = "time",
) -> pl.DataFrame:
    """
    Reorder observations so that axis_signal becomes signal_0.

    Steps:
        1. Pivot long → wide (one row per original timestep)
        2. Sort by axis_signal column
        3. axis_signal values → new signal_0
        4. Old signal_0 → new signal named time_name
        5. axis_signal removed from signal list
        6. Pivot back to long format

    Args:
        observations_path: Path to observations.parquet
        axis_signal: Signal whose values become signal_0
        output_path: Where to write the reaxised parquet
        time_name: Name for the displaced signal_0 (default: "time")

    Returns:
        Reaxised DataFrame in canonical long format
    """
    df = pl.read_parquet(observations_path)

    # Validate axis_signal exists
    signals = df["signal_id"].unique().sort().to_list()
    if axis_signal not in signals:
        raise ValueError(
            f"Axis signal '{axis_signal}' not found. "
            f"Available signals: {signals}"
        )

    # Check time_name doesn't collide with existing signal
    remaining = [s for s in signals if s != axis_signal]
    if time_name in remaining:
        raise ValueError(
            f"time_name '{time_name}' collides with existing signal. "
            f"Choose a different name."
        )

    # Check axis signal has no nulls
    axis_nulls = df.filter(pl.col("signal_id") == axis_signal)["value"].null_count()
    if axis_nulls > 0:
        raise ValueError(
            f"Axis signal '{axis_signal}' has {axis_nulls} null values. "
            f"Cannot use as axis."
        )

    has_cohort = "cohort" in df.columns

    # Pivot to wide: one row per (cohort, signal_0), one column per signal
    pivot_index = ["cohort", "signal_0"] if has_cohort else ["signal_0"]
    df_wide = df.pivot(
        on="signal_id",
        index=pivot_index,
        values="value",
    )

    # Rename: old signal_0 → time_name, axis_signal → new signal_0
    df_wide = df_wide.rename({
        "signal_0": time_name,
        axis_signal: "signal_0",
    })

    # Ensure signal_0 is Float64
    df_wide = df_wide.with_columns(pl.col("signal_0").cast(pl.Float64))

    # Sort by new signal_0 (per cohort if applicable)
    sort_cols = ["cohort", "signal_0"] if has_cohort else ["signal_0"]
    df_wide = df_wide.sort(sort_cols)

    # Break ties: when signal_0 has duplicate values within a cohort,
    # add a fractional offset so each row has a unique signal_0.
    # This preserves ordering and real spacing while ensuring the
    # downstream pivot (signal_0 → row index) has no collisions.
    group_cols = ["cohort", "signal_0"] if has_cohort else ["signal_0"]
    df_wide = df_wide.with_columns(
        pl.col("signal_0").cum_count().over(group_cols).alias("_tie_rank"),
        pl.len().over(group_cols).alias("_tie_count"),
    )
    df_wide = df_wide.with_columns(
        (pl.col("signal_0") + pl.col("_tie_rank") / (pl.col("_tie_count") + 1))
        .alias("signal_0")
    ).drop(["_tie_rank", "_tie_count"])

    # Signal columns: everything except cohort and signal_0
    exclude = {"cohort", "signal_0"}
    signal_columns = [c for c in df_wide.columns if c not in exclude]

    # Pivot back to long
    id_vars = ["cohort", "signal_0"] if has_cohort else ["signal_0"]
    df_long = df_wide.unpivot(
        index=id_vars,
        on=signal_columns,
        variable_name="signal_id",
        value_name="value",
    )

    # Ensure canonical types
    cast_exprs = [
        pl.col("signal_0").cast(pl.Float64),
        pl.col("signal_id").cast(pl.String),
        pl.col("value").cast(pl.Float64),
    ]
    if has_cohort:
        cast_exprs.append(pl.col("cohort").cast(pl.String))
    df_long = df_long.with_columns(cast_exprs)

    # Sort canonically
    sort_cols = (
        ["cohort", "signal_id", "signal_0"] if has_cohort
        else ["signal_id", "signal_0"]
    )
    df_long = df_long.sort(sort_cols)

    # Write
    df_long.write_parquet(output_path)

    new_signals = df_long["signal_id"].unique().sort().to_list()
    n_per = df_long.group_by("signal_id").len()["len"][0]
    print(f"  Axis: {axis_signal} → signal_0")
    print(f"  Displaced: signal_0 → {time_name}")
    print(f"  Signals: {new_signals}")
    print(f"  Rows per signal: {n_per}")
    print(f"  → {output_path}")

    return df_long
