"""
Transform 5G3E Dataset to PRISM Format

The 5G3E dataset contains 5G infrastructure metrics:
- dl_brate, ul_brate: Throughput in bits/s (rate signals)
- cpu_*: CPU utilization per core (bounded 0-100)
- sys_mem, proc_rmem: Memory utilization
- system_load: System load average
- nof_ue: Number of UEs (discrete)

This is ideal for testing typology classification:
- Rate signals (/s) → CONTINUOUS, potentially NON_STATIONARY
- CPU cores → CONTINUOUS, BOUNDED, potentially PERIODIC (scheduling)
- Memory → CONTINUOUS, potentially TRENDING
- nof_ue → DISCRETE

Run:
    python scripts/transform_5g3e.py
"""

import polars as pl
from pathlib import Path
from typing import List


def transform_5g3e(input_dir: Path, output_path: Path, max_files: int = 5) -> pl.DataFrame:
    """
    Transform 5G3E CSV files to PRISM observations format.

    Args:
        input_dir: Path to 5G3E SampleData directory
        output_path: Path for output observations.parquet
        max_files: Maximum number of files to process (for speed)

    Returns:
        DataFrame in PRISM format
    """
    rows = []

    # Find CSV files
    csv_files = list(input_dir.rglob("*.csv"))[:max_files]
    print(f"Found {len(csv_files)} CSV files, processing {min(len(csv_files), max_files)}")

    for csv_path in csv_files:
        # Unit ID = filename (enb0, ue1, etc.)
        unit_id = csv_path.stem

        print(f"  Processing {unit_id}...")

        # Read CSV with semicolon separator
        df = pl.read_csv(csv_path, separator=";", infer_schema_length=1000)

        # Select key signals (not all 32 CPU cores)
        signals = [
            'dl_brate',      # Downlink bit rate (bits/s)
            'ul_brate',      # Uplink bit rate (bits/s)
            'nof_ue',        # Number of UEs (discrete)
            'sys_mem',       # System memory %
            'system_load',   # System load
            'proc_rmem',     # Process resident memory %
            'cpu_0',         # Sample CPU core 0
            'cpu_15',        # Sample CPU core 15
        ]

        # Filter to existing columns
        available_signals = [s for s in signals if s in df.columns]

        for signal_id in available_signals:
            values = df[signal_id].to_list()
            for i, v in enumerate(values):
                if v is not None:
                    rows.append({
                        'unit_id': unit_id,
                        'signal_id': signal_id,
                        'I': i,
                        'value': float(v),
                    })

    result_df = pl.DataFrame(rows)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(output_path)

    return result_df


def main():
    """Transform 5G3E data."""
    input_dir = Path("/home/rudderjason/orthon/data/raw/5g3e/version1/SampleData")
    output_dir = Path("/home/rudderjason/orthon/data/test_domains/5g3e")
    output_path = output_dir / "observations.parquet"

    print("=" * 60)
    print("Transforming 5G3E Dataset")
    print("=" * 60)

    df = transform_5g3e(input_dir, output_path, max_files=10)

    # Summary
    n_signals = df['signal_id'].n_unique()
    n_units = df['unit_id'].n_unique()
    n_rows = len(df)

    print(f"\nOutput: {output_path}")
    print(f"Signals: {n_signals}")
    print(f"Units: {n_units}")
    print(f"Rows: {n_rows}")

    # Signal details
    signals = df['signal_id'].unique().to_list()
    print(f"Signal IDs: {signals}")

    # Sample counts per signal
    print("\nSamples per signal (first unit):")
    first_unit = df['unit_id'].unique()[0]
    for sig in signals:
        count = df.filter((pl.col('unit_id') == first_unit) & (pl.col('signal_id') == sig)).height
        print(f"  {sig}: {count}")


if __name__ == '__main__':
    main()
