#!/usr/bin/env python3
"""
CARE Wind Turbine Dataset Preparation
======================================

Converts CARE Wind raw CSV files to observations.parquet format.

Dataset: CARE Wind Turbine Benchmark
- Paper: https://www.mdpi.com/2306-5729/9/12/138
- 36 turbines, 89 years cumulative data
- SCADA data (10-minute typical)
- Anomaly labels for turbine-periods

REQUIRES MANUAL DOWNLOAD - see README.md

DO NOT RUN PRISM - only prepare data.
"""

import polars as pl
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "raw"
OUTPUT_PATH = BASE_DIR / "observations.parquet"
MANIFEST_PATH = BASE_DIR / "manifest.yaml"

# Typical SCADA signals (adjust based on actual data)
EXPECTED_SIGNALS = [
    "wind_speed",
    "power",
    "rotor_speed",
    "pitch_angle",
    "nacelle_direction",
    "ambient_temp",
    "generator_temp",
    "gearbox_temp",
    "main_bearing_temp",
]


def prepare_care_wind():
    """Convert CARE Wind CSVs to observations.parquet."""

    # Check for data files
    csv_files = list(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print("ERROR: No CSV files found in raw directory.")
        print("This dataset requires manual download.")
        print("See README.md for instructions.")
        return None

    print(f"Found {len(csv_files)} CSV files")

    all_data = []
    entities_info = {}

    for csv_file in sorted(csv_files):
        # Skip label files
        if "label" in csv_file.name.lower():
            continue

        entity_id = csv_file.stem
        print(f"Processing {entity_id}...")

        try:
            df = pl.read_csv(csv_file, infer_schema_length=5000)

            # Add entity_id
            df = df.with_columns(pl.lit(entity_id).alias("entity_id"))

            # Add observation index within entity
            df = df.with_row_index("I_local")

            all_data.append(df)

            entities_info[entity_id] = {
                "n_observations": len(df),
                "columns": df.columns,
            }

            print(f"  -> {len(df):,} observations")

        except Exception as e:
            print(f"  Error: {e}")

    if not all_data:
        print("ERROR: No data loaded!")
        return None

    # Combine all entities
    print("\nCombining all entities...")
    observations = pl.concat(all_data, how="diagonal")  # diagonal handles different schemas

    # Create global I index
    observations = observations.with_row_index("I")

    # Save
    print(f"Saving to {OUTPUT_PATH}...")
    observations.write_parquet(OUTPUT_PATH)

    # Load labels if available
    labels = None
    label_files = list(RAW_DIR.glob("*label*.csv")) + list(RAW_DIR.glob("*Label*.csv"))
    if label_files:
        print(f"\nFound label file: {label_files[0]}")
        try:
            labels = pl.read_csv(label_files[0])
            print(f"Labels shape: {labels.shape}")
        except Exception as e:
            print(f"Warning: Could not read labels: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("CARE WIND DATASET PREPARED")
    print("=" * 60)
    print(f"Total observations: {len(observations):,}")
    print(f"Entities (turbines): {observations['entity_id'].n_unique()}")
    print(f"Columns: {observations.columns}")

    # Create manifest
    create_manifest(observations, entities_info, labels)

    return observations


def create_manifest(observations: pl.DataFrame, entities_info: dict, labels):
    """Create manifest.yaml for PRISM."""

    # Identify signal columns
    signal_cols = [c for c in observations.columns
                   if c not in ["entity_id", "I", "I_local", "timestamp", "datetime"]]

    signals_config = []
    for col in signal_cols[:20]:  # Limit to first 20 for manifest
        sig_info = {"name": col, "type": "unknown"}

        # Try to identify type from name
        col_lower = col.lower()
        if "speed" in col_lower:
            sig_info["type"] = "speed"
            if "wind" in col_lower:
                sig_info["unit"] = "m/s"
            else:
                sig_info["unit"] = "rpm"
        elif "power" in col_lower:
            sig_info["type"] = "power"
            sig_info["unit"] = "kW"
        elif "temp" in col_lower:
            sig_info["type"] = "temperature"
            sig_info["unit"] = "celsius"
        elif "angle" in col_lower or "direction" in col_lower:
            sig_info["type"] = "angle"
            sig_info["unit"] = "degrees"

        signals_config.append(sig_info)

    manifest = {
        "dataset": {
            "name": "CARE Wind Turbine",
            "source": "MDPI Data / Real wind farms (anonymized)",
            "domain": "wind_turbine",
            "description": "89 years cumulative operational data from 36 wind turbines. CARE score evaluation.",
            "paper": "https://www.mdpi.com/2306-5729/9/12/138",
        },
        "data": {
            "observations_path": "observations.parquet",
            "entity_column": "entity_id",
            "index_column": "I",
            "signals": signals_config,
        },
        "ground_truth": {
            "type": "labeled_anomalies",
            "anomaly_count": 44 if labels is None else None,
            "normal_count": 51 if labels is None else None,
            "evaluation": "CARE score",
            "notes": "Binary anomaly labels for each turbine-period",
        },
        "entities": {
            entity_id: info for entity_id, info in list(entities_info.items())[:10]
        },
        "prism": {
            "window_size": 144,  # 24 hours at 10-minute intervals
            "stride": 144,
            "notes": "Adjust window_size based on actual sampling rate",
        },
        "verification": {
            "total_observations": len(observations),
            "n_entities": observations["entity_id"].n_unique(),
            "n_columns": len(observations.columns),
        },
    }

    with open(MANIFEST_PATH, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    print(f"\nManifest saved to {MANIFEST_PATH}")


if __name__ == "__main__":
    observations = prepare_care_wind()

    if observations is not None:
        print("\n" + "=" * 60)
        print("SUCCESS - DO NOT RUN PRISM YET")
        print("=" * 60)
        print("Verify the data first!")
    else:
        print("\n" + "=" * 60)
        print("DOWNLOAD REQUIRED")
        print("=" * 60)
        print("See README.md for manual download instructions")
