#!/usr/bin/env python3
"""
prime/ingest/from_raw.py — Convert raw data files to observations.parquet

Universal ingest: drop in a file, get observations.parquet out.

Usage:
    python -m prime.ingest.from_raw data/FD004/train.txt --format cmapss --output data/FD004/train/
    python -m prime.ingest.from_raw data/FD004/test.txt  --format cmapss --output data/FD004/test/
    python -m prime.ingest.from_raw data/mydata.csv      --format csv    --output data/mydata/

Supported formats:
    cmapss    NASA C-MAPSS turbofan (.txt, space-delimited, no headers)
    csv       Generic CSV (must have headers)
    tsv       Generic TSV (must have headers)
    parquet   Wide-format parquet (must have headers)
    auto      Detect from file extension (default)

For generic formats (csv/tsv/parquet), you must specify:
    --cohort-col    Column name for cohort grouping (e.g. unit_id, engine_id)
    --index-col     Column name for time index (e.g. cycle, timestamp)
    --signal-cols   Comma-separated signal columns, or "all" for everything else
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ─────────────────────────────────────────────────────────────────────
# CMAPSS sensor names (from NASA documentation)
# ─────────────────────────────────────────────────────────────────────
CMAPSS_COLUMNS = [
    "unit_id", "cycle",
    "op1", "op2", "op3",                          # operational settings
    "T2", "T24", "T30", "T50",                     # temperatures
    "P2", "P15", "P30",                            # pressures
    "Nf", "Nc",                                    # speeds
    "epr",                                         # engine pressure ratio
    "Ps30",                                        # static pressure
    "phi",                                         # fuel flow ratio
    "NRf", "NRc",                                  # corrected speeds
    "BPR",                                         # bypass ratio
    "farB",                                        # fuel-air ratio
    "htBleed",                                     # bleed enthalpy
    "Nf_dmd", "PCNfR_dmd",                         # demanded speeds
    "W31", "W32",                                  # coolant bleeds
]

# 21 sensors + 3 operational settings = 24 signals
CMAPSS_SIGNAL_COLS = CMAPSS_COLUMNS[2:]  # everything after unit_id and cycle


def ingest_cmapss(filepath: Path) -> pd.DataFrame:
    """Parse raw CMAPSS .txt → observations.parquet format.

    Input:  space-delimited .txt, no headers, 26 columns
    Output: long-format DataFrame with (signal_0, signal_id, value, cohort)
    """
    # Read space-delimited, no headers
    df_wide = pd.read_csv(
        filepath, sep=r"\s+", header=None,
        names=CMAPSS_COLUMNS,
        engine="python",
    )

    n_units = df_wide["unit_id"].nunique()
    n_rows = len(df_wide)
    print(f"  Read: {n_rows:,} rows, {n_units} units, 24 signals")

    # Melt from wide to long
    records = []
    for col in CMAPSS_SIGNAL_COLS:
        subset = df_wide[["unit_id", "cycle", col]].copy()
        subset.columns = ["unit_id", "cycle", "value"]
        subset["signal_id"] = col
        records.append(subset)

    df_long = pd.concat(records, ignore_index=True)

    # Map unit_id → cohort string
    df_long["cohort"] = "engine_" + df_long["unit_id"].astype(str)

    # signal_0 = cycle - 1 (CMAPSS cycles start at 1, signal_0 starts at 0)
    df_long["signal_0"] = (df_long["cycle"] - 1).astype("float64")

    # Final schema
    df_out = df_long[["signal_0", "signal_id", "value", "cohort"]].copy()
    df_out["signal_id"] = df_out["signal_id"].astype(str)
    df_out["value"] = df_out["value"].astype("float64")
    df_out["cohort"] = df_out["cohort"].astype(str)

    # Sort: signal_id, cohort, signal_0 (consistent ordering)
    df_out = df_out.sort_values(["signal_id", "cohort", "signal_0"]).reset_index(drop=True)

    return df_out


def ingest_generic(filepath: Path, fmt: str, cohort_col: str,
                   index_col: str, signal_cols: str) -> pd.DataFrame:
    """Parse generic tabular file → observations.parquet format.

    Input:  CSV, TSV, or wide parquet with headers
    Output: long-format DataFrame with (signal_0, signal_id, value, cohort)
    """
    if fmt == "csv":
        df_wide = pd.read_csv(filepath)
    elif fmt == "tsv":
        df_wide = pd.read_csv(filepath, sep="\t")
    elif fmt == "parquet":
        df_wide = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    # Validate required columns exist
    if cohort_col and cohort_col not in df_wide.columns:
        raise ValueError(f"Cohort column '{cohort_col}' not found. Columns: {list(df_wide.columns)}")
    if index_col not in df_wide.columns:
        raise ValueError(f"Index column '{index_col}' not found. Columns: {list(df_wide.columns)}")

    # Determine signal columns
    if signal_cols == "all":
        exclude = {cohort_col, index_col} if cohort_col else {index_col}
        sig_cols = [c for c in df_wide.columns if c not in exclude]
    else:
        sig_cols = [c.strip() for c in signal_cols.split(",")]
        missing = [c for c in sig_cols if c not in df_wide.columns]
        if missing:
            raise ValueError(f"Signal columns not found: {missing}")

    n_rows = len(df_wide)
    print(f"  Read: {n_rows:,} rows, {len(sig_cols)} signals")

    # Melt from wide to long
    records = []
    for col in sig_cols:
        if cohort_col:
            subset = df_wide[[cohort_col, index_col, col]].copy()
            subset.columns = ["cohort", "idx", "value"]
        else:
            subset = df_wide[[index_col, col]].copy()
            subset.columns = ["idx", "value"]
            subset["cohort"] = "default"
        subset["signal_id"] = col
        records.append(subset)

    df_long = pd.concat(records, ignore_index=True)

    # Build signal_0 per (signal_id, cohort) — sequential from 0
    df_long = df_long.sort_values(["signal_id", "cohort", "idx"]).reset_index(drop=True)
    df_long["signal_0"] = df_long.groupby(["signal_id", "cohort"]).cumcount().astype("float64")

    # Final schema
    df_out = df_long[["signal_0", "signal_id", "value", "cohort"]].copy()
    df_out["signal_id"] = df_out["signal_id"].astype(str)
    df_out["value"] = df_out["value"].astype("float64")
    df_out["cohort"] = df_out["cohort"].astype(str)

    return df_out


def detect_format(filepath: Path) -> str:
    """Guess format from file extension."""
    ext = filepath.suffix.lower()
    if ext == ".txt":
        # Check if it looks like CMAPSS (26 space-separated columns, no header)
        with open(filepath) as f:
            first_line = f.readline().strip()
        n_cols = len(first_line.split())
        if n_cols == 26:
            # Check if first field is numeric (no header)
            try:
                float(first_line.split()[0])
                return "cmapss"
            except ValueError:
                return "tsv"
        return "tsv"
    elif ext == ".csv":
        return "csv"
    elif ext == ".tsv":
        return "tsv"
    elif ext == ".parquet":
        return "parquet"
    else:
        raise ValueError(f"Cannot detect format for extension '{ext}'. Use --format to specify.")


def write_observations(df: pd.DataFrame, output_dir: Path) -> Path:
    """Write observations.parquet with proper schema."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "observations.parquet"

    # Enforce schema
    table = pa.table({
        "signal_0": pa.array(df["signal_0"].values, type=pa.float64()),
        "signal_id": pa.array(df["signal_id"].values, type=pa.string()),
        "value": pa.array(df["value"].values, type=pa.float64()),
        "cohort": pa.array(df["cohort"].values, type=pa.string()),
    })

    pq.write_table(table, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw data files to observations.parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CMAPSS turbofan (auto-detected from 26-column .txt)
  python -m prime.ingest.from_raw data/FD004/train.txt --output data/FD004/train/
  python -m prime.ingest.from_raw data/FD004/test.txt  --output data/FD004/test/

  # Generic CSV with cohort grouping
  python -m prime.ingest.from_raw mydata.csv --cohort-col unit_id --index-col timestamp --signal-cols "temp,pressure,vibration"

  # Generic CSV, all columns as signals (no cohort)
  python -m prime.ingest.from_raw mydata.csv --index-col time --signal-cols all
        """,
    )
    parser.add_argument("input", type=Path, help="Input file path")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--format", "-f", dest="fmt", default="auto",
                        choices=["auto", "cmapss", "csv", "tsv", "parquet"],
                        help="Input format (default: auto-detect)")

    # Generic format options
    parser.add_argument("--cohort-col", default=None,
                        help="Column for cohort grouping (e.g. unit_id)")
    parser.add_argument("--index-col", default=None,
                        help="Column for time index (e.g. cycle, timestamp)")
    parser.add_argument("--signal-cols", default="all",
                        help='Comma-separated signal columns, or "all" (default: all)')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    # Auto-detect format
    fmt = args.fmt
    if fmt == "auto":
        fmt = detect_format(args.input)
        print(f"  Detected format: {fmt}")

    # Default output directory
    if args.output is None:
        args.output = args.input.parent

    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}/observations.parquet")
    print()

    # Ingest
    if fmt == "cmapss":
        df = ingest_cmapss(args.input)
    else:
        if args.index_col is None:
            print("Error: --index-col is required for generic formats", file=sys.stderr)
            sys.exit(1)
        df = ingest_generic(args.input, fmt, args.cohort_col, args.index_col, args.signal_cols)

    # Write
    out_path = write_observations(df, args.output)

    # Summary
    n_signals = df["signal_id"].nunique()
    n_cohorts = df["cohort"].nunique()
    n_rows = len(df)
    max_signal_0 = df["signal_0"].max()

    print()
    print(f"  observations.parquet written: {out_path}")
    print(f"    Rows:        {n_rows:,}")
    print(f"    Signals:     {n_signals}")
    print(f"    Cohorts:     {n_cohorts}")
    print(f"    Max signal_0: {max_signal_0}")
    print(f"    Size:        {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
