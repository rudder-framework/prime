"""
Convert SFI benchmark datasets from (cohort, date, signal_id, value)
to PRISM format (cohort, signal_id, I, value).

These datasets are designed to test PRISM/ORTHON's detection of:
1. Community Fragmentation - Echo chamber formation in social networks
2. Perception-Reality Divergence - Cognitive bias in organizational surveys
3. Ecological Regime Shift - Lake ecosystem tipping point (Scheffer model)
4. Structural Health - Bridge corrosion degradation

Ground Truth Events (what PRISM should detect WITHOUT being told):
1. Echo chamber recruitment cascade starting month 15
2. Org restructuring at month 18, perception lags by ~4 months
3. Lake ecosystem tipping point at month 30 (critical slowing down from month 22)
4. Bridge corrosion onset day 120, acceleration day 200
"""

import polars as pl
from pathlib import Path
import shutil

# Source files
SOURCE_DIR = Path("/tmp/orthon_files")
OUTPUT_BASE = Path.home() / "Domains" / "sfi_benchmarks"

DATASETS = {
    "01_community_fragmentation.parquet": {
        "output_dir": "community_fragmentation",
        "description": "Social network echo chamber formation",
        "ground_truth": "Recruitment cascade starts month 15, completes month 22",
        "signals": "30 users × 5 metrics (activity, reciprocity, clustering, betweenness, sentiment)",
    },
    "02_perception_reality.parquet": {
        "output_dir": "perception_reality",
        "description": "Perception-reality divergence in organizations",
        "ground_truth": "Restructuring at month 18, perception lags ~4 months",
        "signals": "20 people × 8 metrics (4 actual + 4 perceived)",
    },
    "03_ecological_regime_shift.parquet": {
        "output_dir": "ecological_regime",
        "description": "Lake ecosystem tipping point (Scheffer model)",
        "ground_truth": "Tipping point month 30, critical slowing down from month 22",
        "signals": "12 species/metrics",
    },
    "04_structural_health.parquet": {
        "output_dir": "structural_health",
        "description": "Bridge structural degradation monitoring",
        "ground_truth": "Degradation onset day 120, acceleration day 200",
        "signals": "15 sensors (vibration, strain, displacement × 5 locations)",
    },
}


def convert_to_prism_format(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert from (cohort, date, signal_id, value) to (cohort, signal_id, I, value).

    Creates sequential I index per signal_id based on date ordering.
    """
    # Sort by date within each signal
    df = df.sort(["signal_id", "date"])

    # Create I as row number within each signal_id
    df = df.with_columns([
        pl.col("date").rank("ordinal").over("signal_id").cast(pl.UInt32).alias("I") - 1
    ])

    # Select and reorder columns to PRISM format
    return df.select(["cohort", "signal_id", "I", "value"])


def main():
    print("=" * 70)
    print("SFI Benchmark Dataset Conversion")
    print("Converting from (cohort, date, signal_id, value) to PRISM format")
    print("=" * 70)
    print()

    for source_file, config in DATASETS.items():
        source_path = SOURCE_DIR / source_file
        output_dir = OUTPUT_BASE / config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing: {source_file}")
        print(f"  Description: {config['description']}")
        print(f"  Ground Truth: {config['ground_truth']}")

        # Load and convert
        df = pl.read_parquet(source_path)
        df_prism = convert_to_prism_format(df)

        # Save
        output_path = output_dir / "observations.parquet"
        df_prism.write_parquet(output_path)

        print(f"  Input: {df.shape} -> Output: {df_prism.shape}")
        print(f"  Signals: {df_prism['signal_id'].n_unique()}")
        print(f"  Timesteps: {df_prism['I'].max() + 1}")
        print(f"  Saved: {output_path}")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Dataset':<30} {'Signals':>8} {'Steps':>8} {'Rows':>10}")
    print("-" * 60)

    for source_file, config in DATASETS.items():
        output_dir = OUTPUT_BASE / config["output_dir"]
        df = pl.read_parquet(output_dir / "observations.parquet")
        print(f"{config['output_dir']:<30} {df['signal_id'].n_unique():>8} {df['I'].max()+1:>8} {df.shape[0]:>10,}")

    print()
    print("All datasets ready for PRISM pipeline:")
    print("  python -m prism ~/Domains/sfi_benchmarks/<dataset>")


if __name__ == "__main__":
    main()
