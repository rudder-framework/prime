"""
Quick validation script for observations.parquet

Schema v2.0.0:
- REQUIRED: signal_id, I, value
- OPTIONAL: unit_id (just a label)

Run before PRISM to catch issues early.
"""

import polars as pl
from pathlib import Path
import sys


REQUIRED_COLUMNS = ["signal_id", "I", "value"]


def validate_observations(path: Path) -> bool:
    """
    Validate observations.parquet meets PRISM requirements.
    Prints detailed report.
    """

    print(f"\n{'='*60}")
    print(f"VALIDATING: {path}")
    print(f"{'='*60}\n")

    if not path.exists():
        print("[X] File does not exist!")
        return False

    df = pl.read_parquet(path)

    errors = []
    warnings = []

    # 1. Check required columns
    print("1. REQUIRED COLUMNS")
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            print(f"   [OK] {col}: {df[col].dtype}")
        else:
            print(f"   [X] {col}: MISSING")
            errors.append(f"Missing column: {col}")
    print()

    if errors:
        print("="*60)
        print("[X] VALIDATION FAILED - Missing required columns")
        return False

    # 2. Check optional unit_id
    print("2. OPTIONAL: unit_id")
    if "unit_id" not in df.columns:
        print("   [i] No unit_id column (this is fine - will use blank)")
        df = df.with_columns(pl.lit("").alias("unit_id"))
    else:
        n_units = df["unit_id"].n_unique()
        print(f"   [OK] unit_id: {n_units} unique values")
    print()

    # 3. Check signals
    print("3. SIGNALS")
    n_signals = df["signal_id"].n_unique()
    signals = df["signal_id"].unique().sort().to_list()
    print(f"   Count: {n_signals}")
    print(f"   Names: {signals}")

    if n_signals < 2:
        errors.append(f"Only {n_signals} signal(s). Need >=2 for pair engines.")
        print(f"   [X] Need >=2 signals for pair engines!")
    else:
        print(f"   [OK] Sufficient for pair engines")
    print()

    # 4. Check I sequentiality
    print("4. INDEX (I) SEQUENTIALITY")

    group_cols = ["unit_id", "signal_id"] if "unit_id" in df.columns else ["signal_id"]

    sample_group = df.group_by(group_cols).first()
    if len(sample_group) > 0:
        first_row = sample_group.row(0, named=True)
        sample_filter = df
        for col in group_cols:
            sample_filter = sample_filter.filter(pl.col(col) == first_row[col])

        sample_i = sample_filter.select("I").to_series().sort().to_list()
        expected_i = list(range(len(sample_i)))
        is_sequential = sample_i == expected_i

        print(f"   Sample group: {first_row}")
        print(f"   I range: {min(sample_i)} - {max(sample_i)}")
        print(f"   First 10 I values: {sample_i[:10]}")

        if is_sequential:
            print(f"   [OK] Sequential (0, 1, 2, 3...)")
        else:
            print(f"   [X] NOT sequential!")
            errors.append("I is not sequential")
    print()

    # 5. Check observations per group
    print("5. OBSERVATIONS PER GROUP")
    obs_per_group = df.group_by(group_cols).agg(pl.len().alias("n_obs"))

    min_obs = obs_per_group["n_obs"].min()
    max_obs = obs_per_group["n_obs"].max()
    mean_obs = obs_per_group["n_obs"].mean()

    print(f"   Min: {min_obs}")
    print(f"   Max: {max_obs}")
    print(f"   Mean: {mean_obs:.0f}")

    if min_obs < 50:
        warnings.append(f"Some groups have only {min_obs} observations")
        print(f"   [!] Some have <50 observations")
    else:
        print(f"   [OK] Sufficient observations")
    print()

    # 6. Check nulls
    print("6. NULL VALUES")
    for col in ["signal_id", "I"]:
        n_nulls = df[col].null_count()
        if n_nulls > 0:
            print(f"   [X] {col}: {n_nulls} nulls")
            errors.append(f"{col} has {n_nulls} nulls")
        else:
            print(f"   [OK] {col}: 0 nulls")

    value_nulls = df["value"].null_count()
    if value_nulls > 0:
        print(f"   [i] value: {value_nulls} nulls (allowed)")
    else:
        print(f"   [OK] value: 0 nulls")
    print()

    # Summary
    print("="*60)
    if errors:
        print("[X] VALIDATION FAILED")
        print("\nErrors:")
        for e in errors:
            print(f"   - {e}")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"   - {w}")
        print("\n[!!] DO NOT run PRISM on this data!")
        return False
    else:
        print("[OK] VALIDATION PASSED")
        if warnings:
            print("\nWarnings (non-fatal):")
            for w in warnings:
                print(f"   - {w}")
        print("\n[OK] Safe to run PRISM")
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m orthon.ingest.validate observations.parquet")
        sys.exit(1)

    path = Path(sys.argv[1])
    is_valid = validate_observations(path)
    sys.exit(0 if is_valid else 1)
