#!/usr/bin/env python3
"""
Convert C-MAPSS raw text files to observations.parquet
======================================================

Converts NASA C-MAPSS train/test text files into the canonical
long-format observations schema: (cohort, signal_id, I, value).

Works for all datasets: FD001, FD002, FD003, FD004, PHM08.

Usage:
    python convert_cmapss.py ~/data/FD_001/train_FD001.txt ~/data/FD_001/train/observations.parquet
    python convert_cmapss.py ~/data/FD_001/test_FD001.txt  ~/data/FD_001/test/observations.parquet

    # Or batch-convert a whole dataset directory:
    python convert_cmapss.py --dir ~/data/FD_002/
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path

# NASA C-MAPSS column mapping (26 columns total):
# Col 0: unit number
# Col 1: time in cycles
# Col 2-4: operational settings
# Col 5-25: sensor measurements (21 sensors)
SIGNAL_NAMES = [
    'op1', 'op2', 'op3',           # cols 2-4: operational settings
    'T2', 'T24', 'T30', 'T50',     # cols 5-8: temperatures
    'P2', 'P15', 'P30',            # cols 9-11: pressures
    'Nf', 'Nc',                    # cols 12-13: speeds
    'epr',                         # col 14: engine pressure ratio
    'Ps30',                        # col 15: static pressure
    'phi',                         # col 16: corrected fan speed
    'NRf', 'NRc',                  # cols 17-18: corrected speeds
    'BPR',                         # col 19: bypass ratio
    'farB',                        # col 20: burner fuel-air ratio
    'htBleed',                     # col 21: bleed enthalpy
    'Nf_dmd',                      # col 22: demanded fan speed
    'PCNfR_dmd',                   # col 23: demanded corrected fan speed
    'W31', 'W32',                  # cols 24-25: component work
]

assert len(SIGNAL_NAMES) == 24, f"Expected 24 signals, got {len(SIGNAL_NAMES)}"


def convert_file(input_path: Path, output_path: Path) -> pl.DataFrame:
    """Convert a single C-MAPSS text file to observations.parquet."""
    raw = np.loadtxt(str(input_path))
    n_rows, n_cols = raw.shape
    assert n_cols == 26, f"Expected 26 columns, got {n_cols}"

    units = raw[:, 0].astype(int)
    cycles = raw[:, 1].astype(int)
    values = raw[:, 2:]  # 24 signal columns

    records = []
    for i in range(n_rows):
        unit = units[i]
        cycle = cycles[i] - 1  # 0-indexed
        cohort = f"engine_{unit}"
        for j, sig in enumerate(SIGNAL_NAMES):
            records.append({
                'cohort': cohort,
                'signal_id': sig,
                'I': cycle,
                'value': float(values[i, j]),
            })

    df = pl.DataFrame(records, schema={
        'cohort': pl.String,
        'signal_id': pl.String,
        'I': pl.UInt32,
        'value': pl.Float64,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(output_path))

    n_engines = df['cohort'].n_unique()
    n_signals = df['signal_id'].n_unique()
    n_cycles = n_rows // n_engines if n_engines > 0 else 0
    print(f"  {input_path.name} → {output_path}")
    print(f"    {n_engines} engines, {n_signals} signals, {len(df):,} rows")
    return df


def convert_directory(data_dir: Path):
    """Convert all raw files in a dataset directory."""
    # Find train file
    train_files = list(data_dir.glob('train_FD*.txt'))
    test_files = list(data_dir.glob('test_FD*.txt'))

    if not train_files and not test_files:
        print(f"No C-MAPSS files found in {data_dir}")
        return

    for f in sorted(train_files):
        out = data_dir / 'train' / 'observations.parquet'
        convert_file(f, out)

    for f in sorted(test_files):
        out = data_dir / 'test' / 'observations.parquet'
        convert_file(f, out)


def main():
    parser = argparse.ArgumentParser(description='Convert C-MAPSS text to observations.parquet')
    parser.add_argument('input', nargs='?', help='Input text file (or use --dir)')
    parser.add_argument('output', nargs='?', help='Output parquet path')
    parser.add_argument('--dir', help='Dataset directory (batch convert train + test)')
    args = parser.parse_args()

    if args.dir:
        convert_directory(Path(args.dir).expanduser())
    elif args.input and args.output:
        convert_file(Path(args.input).expanduser(), Path(args.output).expanduser())
    else:
        parser.error("Provide input/output paths or --dir")


if __name__ == '__main__':
    main()
