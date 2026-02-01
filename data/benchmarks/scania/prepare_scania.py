#!/usr/bin/env python3
"""
SCANIA Component X Dataset Preparation
=======================================

Converts SCANIA IDA 2024 Challenge data to observations.parquet format.

Dataset: SCANIA Component X (IDA 2024 Industrial Challenge)
- https://www.ida2024.org/challenge
- 33K+ heavy-duty trucks
- Operational histograms and counters
- Anonymized component failure prediction

REQUIRES REGISTRATION AND DOWNLOAD - see README.md

DO NOT RUN PRISM - only prepare data.
"""

import polars as pl
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "raw"
OUTPUT_PATH = BASE_DIR / "observations.parquet"
MANIFEST_PATH = BASE_DIR / "manifest.yaml"


def prepare_scania():
    """Convert SCANIA data to observations.parquet."""

    # Check for data files
    csv_files = list(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print("ERROR: No CSV files found in raw directory.")
        print("This dataset requires registration at IDA 2024.")
        print("See README.md for instructions.")
        return None

    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  {f.name}")

    # Try to identify train and test files
    train_file = None
    test_file = None

    for f in csv_files:
        name_lower = f.name.lower()
        if "train" in name_lower:
            train_file = f
        elif "test" in name_lower:
            test_file = f

    if train_file is None:
        # Use largest file as main data
        train_file = max(csv_files, key=lambda f: f.stat().st_size)

    print(f"\nUsing as main data: {train_file}")

    # Read data
    print("Reading data (this may take a while for 33K+ vehicles)...")
    df = pl.read_csv(train_file, infer_schema_length=10000)

    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # SCANIA data structure is typically:
    # - First column(s): vehicle ID, target
    # - Remaining: histogram bins and counters

    # Identify ID and target columns
    id_col = None
    target_col = None

    for col in df.columns:
        col_lower = col.lower()
        if "id" in col_lower or col == df.columns[0]:
            id_col = col
        if "target" in col_lower or "class" in col_lower or "failure" in col_lower:
            target_col = col

    if id_col is None:
        id_col = df.columns[0]
        print(f"Using first column as ID: {id_col}")

    # Rename to standard
    if id_col != "entity_id":
        df = df.rename({id_col: "entity_id"})

    # Add observation index
    df = df.with_row_index("I")

    # Get entity info
    entities_info = {
        "total_vehicles": df["entity_id"].n_unique(),
        "columns": df.columns,
    }

    if target_col:
        pos_count = df.filter(pl.col(target_col) == 1).height
        neg_count = df.filter(pl.col(target_col) == 0).height
        entities_info["positive_class"] = pos_count
        entities_info["negative_class"] = neg_count
        print(f"\nTarget distribution: {pos_count} positive, {neg_count} negative")

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    df.write_parquet(OUTPUT_PATH)

    # Summary
    print("\n" + "=" * 60)
    print("SCANIA DATASET PREPARED")
    print("=" * 60)
    print(f"Total observations: {len(df):,}")
    print(f"Vehicles (entities): {df['entity_id'].n_unique()}")
    print(f"Features: {len(df.columns) - 2}")  # Minus entity_id and I

    # Create manifest
    create_manifest(df, entities_info, target_col)

    return df


def create_manifest(observations: pl.DataFrame, entities_info: dict, target_col):
    """Create manifest.yaml for PRISM."""

    # Feature columns (exclude ID, I, target)
    feature_cols = [c for c in observations.columns
                    if c not in ["entity_id", "I", target_col]]

    manifest = {
        "dataset": {
            "name": "SCANIA Component X",
            "source": "SCANIA / IDA 2024 Industrial Challenge",
            "domain": "truck_engine_component",
            "description": "Real-world fleet data from 33K+ SCANIA heavy-duty trucks. Anonymized component failure prediction.",
            "challenge": "https://www.ida2024.org/challenge",
        },
        "data": {
            "observations_path": "observations.parquet",
            "entity_column": "entity_id",
            "index_column": "I",
            "n_features": len(feature_cols),
            "feature_types": {
                "histogram": "Anonymized histogram bins from ECU",
                "counter": "Anonymized counter values from ECU",
            },
        },
        "ground_truth": {
            "type": "binary_classification",
            "target_column": target_col,
            "positive_class": entities_info.get("positive_class"),
            "negative_class": entities_info.get("negative_class"),
            "notes": "Component X failure within prediction horizon",
        },
        "entities": {
            "total_vehicles": entities_info["total_vehicles"],
        },
        "prism": {
            "window_size": 1,
            "stride": 1,
            "notes": "Single snapshot per vehicle. May need different analysis approach.",
        },
        "verification": {
            "total_observations": len(observations),
            "n_entities": observations["entity_id"].n_unique(),
            "n_features": len(feature_cols),
        },
    }

    with open(MANIFEST_PATH, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    print(f"\nManifest saved to {MANIFEST_PATH}")


if __name__ == "__main__":
    observations = prepare_scania()

    if observations is not None:
        print("\n" + "=" * 60)
        print("SUCCESS - DO NOT RUN PRISM YET")
        print("=" * 60)
        print("Note: SCANIA data is snapshot-based (one row per vehicle).")
        print("May require different analysis approach than time series.")
    else:
        print("\n" + "=" * 60)
        print("DOWNLOAD REQUIRED")
        print("=" * 60)
        print("See README.md for registration and download instructions")
