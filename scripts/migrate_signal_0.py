#!/usr/bin/env python3
"""
Dataset Migration: I → signal_0

Converts all observations.parquet files from the old schema (I: UInt32)
to the new schema (signal_0: Float64).

Run once after the schema change PR lands. Not committed to main.

Usage:
    python scripts/migrate_signal_0.py

Domains:
    cmapss (train+test): I.cast(Float64) → signal_0
    building_vibration/index_axis: (I / 256.0) → signal_0
    building_vibration/z_axis: rename t → signal_0
    rossler: I.cast(Float64) → signal_0
    calce: I.cast(Float64) → signal_0
    lumo: I.cast(Float64) → signal_0
    pendulum: I.cast(Float64) → signal_0
    lorenz: I.cast(Float64) → signal_0
    hydraulic: I.cast(Float64) → signal_0
"""

import polars as pl
from pathlib import Path


DOMAINS_DIR = Path.home() / "domains"


def migrate_file(obs_path: Path, transform: str = "cast") -> bool:
    """
    Migrate a single observations.parquet from I→signal_0.

    Args:
        obs_path: Path to observations.parquet
        transform: One of:
            - "cast": I.cast(Float64) → signal_0 (default for most domains)
            - "divide_256": (I / 256.0) → signal_0 (building_vibration/index_axis)
            - "rename_t": rename t → signal_0 (building_vibration/z_axis)

    Returns:
        True if migrated, False if already migrated or not found
    """
    if not obs_path.exists():
        print(f"  SKIP: {obs_path} (not found)")
        return False

    df = pl.read_parquet(obs_path)

    # Already migrated?
    if "signal_0" in df.columns and "I" not in df.columns:
        print(f"  SKIP: {obs_path} (already migrated)")
        return False

    original_cols = df.columns.copy()

    if transform == "cast":
        if "I" in df.columns:
            df = df.with_columns(pl.col("I").cast(pl.Float64).alias("signal_0")).drop("I")
        else:
            print(f"  SKIP: {obs_path} (no I column, columns: {df.columns})")
            return False

    elif transform == "divide_256":
        if "I" in df.columns:
            df = df.with_columns((pl.col("I").cast(pl.Float64) / 256.0).alias("signal_0")).drop("I")
        else:
            print(f"  SKIP: {obs_path} (no I column)")
            return False

    elif transform == "rename_t":
        if "t" in df.columns:
            df = df.rename({"t": "signal_0"})
            if df["signal_0"].dtype != pl.Float64:
                df = df.with_columns(pl.col("signal_0").cast(pl.Float64))
        elif "I" in df.columns:
            # Fallback: use I if t not found
            df = df.with_columns(pl.col("I").cast(pl.Float64).alias("signal_0")).drop("I")
        else:
            print(f"  SKIP: {obs_path} (no t or I column)")
            return False

    # Reorder columns: put signal_0 where I was
    cols = df.columns
    final_cols = []
    for c in cols:
        if c == "signal_0":
            continue
        final_cols.append(c)

    # Insert signal_0 in the position I was (or second position)
    if "cohort" in final_cols:
        insert_pos = 1  # After cohort
    else:
        insert_pos = 0
    final_cols.insert(insert_pos, "signal_0")

    df = df.select(final_cols)

    # Write back
    df.write_parquet(obs_path)
    print(f"  OK: {obs_path} ({original_cols} → {df.columns})")
    return True


def main():
    print("=" * 60)
    print("MIGRATING observations.parquet: I → signal_0")
    print("=" * 60)

    migrated = 0
    skipped = 0

    # Standard domains: simple cast
    standard_domains = [
        "cmapss/FD_001/train",
        "cmapss/FD_001/test",
        "cmapss/FD_002/train",
        "cmapss/FD_002/test",
        "cmapss/FD_003/train",
        "cmapss/FD_003/test",
        "cmapss/FD_004/train",
        "cmapss/FD_004/test",
        "fd004/train",
        "fd004/test",
        "rossler/train",
        "calce/train",
        "lumo/train",
        "pendulum/train",
        "lorenz/train",
        "hydraulic/train",
    ]

    print("\n--- Standard domains (I → signal_0 via cast) ---")
    for domain in standard_domains:
        obs_path = DOMAINS_DIR / domain / "observations.parquet"
        if migrate_file(obs_path, "cast"):
            migrated += 1
        else:
            skipped += 1

    # Building vibration: special transforms
    print("\n--- Building vibration (special transforms) ---")

    bv_index = DOMAINS_DIR / "building_vibration" / "index_axis" / "observations.parquet"
    if migrate_file(bv_index, "divide_256"):
        migrated += 1
    else:
        skipped += 1

    bv_z = DOMAINS_DIR / "building_vibration" / "z_axis" / "observations.parquet"
    if migrate_file(bv_z, "rename_t"):
        migrated += 1
    else:
        skipped += 1

    # Catch any others we might have missed
    print("\n--- Scanning for remaining observations.parquet files ---")
    for obs_path in DOMAINS_DIR.rglob("observations.parquet"):
        df = pl.read_parquet(obs_path)
        if "I" in df.columns and "signal_0" not in df.columns:
            print(f"  FOUND unmigrated: {obs_path}")
            if migrate_file(obs_path, "cast"):
                migrated += 1
            else:
                skipped += 1

    print(f"\n{'=' * 60}")
    print(f"MIGRATION COMPLETE: {migrated} migrated, {skipped} skipped")
    print(f"{'=' * 60}")

    # Verify a sample
    print("\n--- Verification ---")
    for domain in ["rossler/train", "fd004/train"]:
        obs_path = DOMAINS_DIR / domain / "observations.parquet"
        if obs_path.exists():
            df = pl.read_parquet(obs_path)
            assert "signal_0" in df.columns, f"signal_0 not in {obs_path}"
            assert df["signal_0"].dtype == pl.Float64, f"signal_0 not Float64 in {obs_path}"
            assert "I" not in df.columns, f"I still in {obs_path}"
            assert df["signal_0"].null_count() == 0, f"signal_0 has nulls in {obs_path}"
            print(f"  VERIFIED: {obs_path} ({df.columns}, {df['signal_0'].dtype})")


if __name__ == "__main__":
    main()
