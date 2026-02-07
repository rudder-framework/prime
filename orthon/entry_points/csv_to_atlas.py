"""
CSV to Dynamical Atlas - One Command Pipeline
==============================================

The "stranger uploads a CSV and gets a dynamical atlas" entry point.

Usage:
    python -m orthon.entry_points.csv_to_atlas data.csv --output-dir output/
    python -m orthon.entry_points.csv_to_atlas data.xlsx --signals temp,pressure,flow
    python -m orthon.entry_points.csv_to_atlas data.parquet --cohort-col entity_id

Pipeline:
    1. Load data (CSV, Excel, Parquet, TSV, MATLAB)
    2. Auto-detect or use specified column mappings
    3. Transform to observations.parquet (PRISM format)
    4. Validate schema
    5. Compute typology (signal characterization)
    6. Generate manifest (engine selection per signal)
    7. Run PRISM pipeline (signal_vector → geometry → dynamics)
    8. Output: Complete dynamical atlas in output directory

Output Files:
    observations.parquet    - Canonical format data
    typology_raw.parquet    - 27 raw measures per signal
    typology.parquet        - Signal classification
    manifest.yaml           - Engine selection and parameters
    output/
        signal_vector.parquet
        state_vector.parquet
        state_geometry.parquet
        geometry_full.parquet
        velocity_field.parquet
        ftle_rolling.parquet
        ridge_proximity.parquet
        ...
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, List

import polars as pl


def run(
    input_path: str,
    output_dir: str = "output",
    signal_columns: Optional[List[str]] = None,
    cohort_column: Optional[str] = None,
    index_column: Optional[str] = None,
    skip_prism: bool = False,
    prism_stages: Optional[List[str]] = None,
    verbose: bool = True,
) -> dict:
    """
    Complete CSV-to-Atlas pipeline.

    Args:
        input_path: Path to input file (CSV, Excel, Parquet, etc.)
        output_dir: Output directory for all pipeline outputs
        signal_columns: Explicit list of signal column names (auto-detect if None)
        cohort_column: Column to use as cohort (entity/unit grouping)
        index_column: Column to use as I (sequential index)
        skip_prism: If True, stop after manifest generation (ORTHON only)
        prism_stages: Specific PRISM stages to run (None = full pipeline)
        verbose: Print progress

    Returns:
        dict with paths to all generated files
    """
    from orthon.ingest.upload import load_file
    from orthon.ingest.transform import transform_wide_to_long, validate_prism_schema, fix_sparse_index

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("CSV TO DYNAMICAL ATLAS")
        print("=" * 70)
        print(f"Input: {input_path}")
        print(f"Output: {output_dir}")

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    if verbose:
        print("\n[1/6] Loading data...")

    df = load_file(input_path)
    df = pl.from_pandas(df)

    if verbose:
        print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Columns: {df.columns[:10]}{'...' if len(df.columns) > 10 else ''}")

    # =========================================================================
    # STEP 2: AUTO-DETECT STRUCTURE
    # =========================================================================
    if verbose:
        print("\n[2/6] Detecting data structure...")

    # Detect signal columns
    if signal_columns is None:
        # Auto-detect: numeric columns that aren't obvious metadata
        exclude_patterns = {
            'id', 'index', 'timestamp', 'time', 'date', 'datetime',
            'entity', 'unit', 'cohort', 'cycle', 'row'
        }
        signal_columns = []
        for c in df.columns:
            if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                c_lower = c.lower()
                if not any(pat in c_lower for pat in exclude_patterns):
                    signal_columns.append(c)

        if verbose:
            print(f"  Auto-detected {len(signal_columns)} signal columns")
            if len(signal_columns) <= 10:
                print(f"  Signals: {signal_columns}")
            else:
                print(f"  Signals: {signal_columns[:5]} ... {signal_columns[-5:]}")

    if len(signal_columns) < 2:
        raise ValueError(f"Need at least 2 signals, found {len(signal_columns)}")

    # Detect cohort column
    if cohort_column is None:
        for candidate in ['cohort', 'unit_id', 'entity_id', 'entity', 'unit', 'engine', 'bearing']:
            for c in df.columns:
                if c.lower() == candidate:
                    cohort_column = c
                    break
            if cohort_column:
                break

    if verbose and cohort_column:
        n_cohorts = df[cohort_column].n_unique()
        print(f"  Cohort column: {cohort_column} ({n_cohorts} unique)")

    # Detect index column
    if index_column is None:
        for candidate in ['I', 'index', 'cycle', 'timestamp', 'time', 't']:
            for c in df.columns:
                if c.lower() == candidate:
                    index_column = c
                    break
            if index_column:
                break

    if verbose and index_column:
        print(f"  Index column: {index_column}")

    # =========================================================================
    # STEP 3: TRANSFORM TO OBSERVATIONS.PARQUET
    # =========================================================================
    if verbose:
        print("\n[3/6] Transforming to PRISM format...")

    # Prepare for melt
    if cohort_column and cohort_column in df.columns:
        df = df.with_columns(pl.col(cohort_column).cast(pl.Utf8).alias("cohort"))
    else:
        df = df.with_columns(pl.lit("default").alias("cohort"))

    if index_column and index_column in df.columns:
        df = df.with_columns(pl.col(index_column).cast(pl.UInt32).alias("I"))
    else:
        df = df.with_row_index("I")
        df = df.with_columns(pl.col("I").cast(pl.UInt32))

    # Melt wide to long
    id_vars = ["cohort", "I"]
    df_long = df.select(id_vars + signal_columns).unpivot(
        index=id_vars,
        on=signal_columns,
        variable_name="signal_id",
        value_name="value"
    )

    # Ensure types
    df_long = df_long.with_columns([
        pl.col("cohort").cast(pl.Utf8),
        pl.col("I").cast(pl.UInt32),
        pl.col("signal_id").cast(pl.Utf8),
        pl.col("value").cast(pl.Float64),
    ])

    # Fix sparse indices (ensure 0,1,2,3...)
    df_long = fix_sparse_index(df_long)

    # Validate
    is_valid, errors = validate_prism_schema(df_long)
    if not is_valid:
        # Try to fix common issues
        if "I does not start at 0" in str(errors):
            df_long = df_long.with_columns(
                (pl.col("I") - pl.col("I").min().over(["cohort", "signal_id"])).alias("I")
            )
            is_valid, errors = validate_prism_schema(df_long)

    if not is_valid:
        print(f"  WARNING: Validation issues: {errors}")

    # Sort and save
    df_long = df_long.sort(["cohort", "I", "signal_id"])
    observations_path = output_dir / "observations.parquet"
    df_long.write_parquet(observations_path)

    n_cohorts = df_long["cohort"].n_unique()
    n_signals = df_long["signal_id"].n_unique()
    n_obs = len(df_long)

    if verbose:
        print(f"  Saved: {observations_path}")
        print(f"  Cohorts: {n_cohorts}, Signals: {n_signals}, Observations: {n_obs:,}")

    # =========================================================================
    # STEP 4: COMPUTE TYPOLOGY
    # =========================================================================
    if verbose:
        print("\n[4/6] Computing typology (signal characterization)...")

    from orthon.ingest.typology_raw import compute_typology_raw
    from orthon.typology.discrete_sparse import apply_discrete_sparse_classification
    from orthon.typology.level2_corrections import apply_corrections

    # Compute raw typology measures
    typology_raw_path = output_dir / "typology_raw.parquet"
    typology_raw = compute_typology_raw(
        str(observations_path),
        str(typology_raw_path),
        verbose=verbose
    )

    if verbose:
        print(f"  Saved: {typology_raw_path}")

    # Apply classification
    typology_path = output_dir / "typology.parquet"

    # Stage 1: Discrete/sparse classification
    typology_df = typology_raw.to_pandas()
    for idx, row in typology_df.iterrows():
        row_dict = row.to_dict()
        result = apply_discrete_sparse_classification(row_dict)
        for k, v in result.items():
            typology_df.at[idx, k] = v

    # Stage 2: Continuous classification (for non-discrete signals)
    for idx, row in typology_df.iterrows():
        if row.get('temporal_pattern') in [None, 'UNKNOWN', '']:
            row_dict = row.to_dict()
            result = apply_corrections(row_dict)
            for k, v in result.items():
                typology_df.at[idx, k] = v

    # Save typology
    typology_pl = pl.from_pandas(typology_df)
    typology_pl.write_parquet(typology_path)

    if verbose:
        print(f"  Saved: {typology_path}")
        patterns = typology_pl.group_by("temporal_pattern").len().sort("len", descending=True)
        print("  Signal types:")
        for row in patterns.head(5).iter_rows(named=True):
            print(f"    {row['temporal_pattern']}: {row['len']}")

    # =========================================================================
    # STEP 5: GENERATE MANIFEST
    # =========================================================================
    if verbose:
        print("\n[5/6] Generating manifest (engine selection)...")

    from orthon.manifest.generator import build_manifest, save_manifest

    manifest = build_manifest(
        typology_df=typology_df,
        observations_path=str(observations_path),
        typology_path=str(typology_path),
        output_dir=str(output_dir / "output"),
    )

    manifest_path = output_dir / "manifest.yaml"
    save_manifest(manifest, str(manifest_path))

    if verbose:
        print(f"  Saved: {manifest_path}")
        print(f"  Active signals: {manifest.get('summary', {}).get('active_signals', 'N/A')}")
        print(f"  Engines: {len(manifest.get('summary', {}).get('signal_engines', []))}")

    # =========================================================================
    # STEP 6: RUN PRISM PIPELINE (optional)
    # =========================================================================
    result = {
        'observations': str(observations_path),
        'typology_raw': str(typology_raw_path),
        'typology': str(typology_path),
        'manifest': str(manifest_path),
        'output_dir': str(output_dir),
    }

    if skip_prism:
        if verbose:
            print("\n[6/6] Skipping PRISM (--skip-prism)")
        return result

    if verbose:
        print("\n[6/6] Running PRISM pipeline...")

    # Try to run PRISM
    prism_dir = Path(__file__).parent.parent.parent.parent / "prism"

    if not (prism_dir / "prism").exists():
        # Try alternative location
        prism_dir = Path.home() / "prism"

    if not (prism_dir / "prism").exists():
        print("  WARNING: PRISM not found. Run manually with:")
        print(f"    cd ~/prism && ./venv/bin/python -m prism.entry_points.run_pipeline {manifest_path}")
        return result

    prism_python = prism_dir / "venv" / "bin" / "python"

    # Run PRISM pipeline via run_pipeline entry point
    if prism_stages:
        # Run specific stages
        stages_str = ",".join(prism_stages)
        cmd = [str(prism_python), "-m", "prism.entry_points.run_pipeline",
               str(manifest_path), "--stages", stages_str]
    else:
        # Run core + atlas stages
        cmd = [str(prism_python), "-m", "prism.entry_points.run_pipeline",
               str(manifest_path), "--stages", "01,02,03,20,21,22,23"]

    if verbose:
        print(f"  Command: {' '.join(cmd)}")

    try:
        result_code = subprocess.run(
            cmd,
            cwd=str(prism_dir),
            capture_output=not verbose,
            text=True
        )
        if result_code.returncode != 0:
            print(f"  WARNING: Pipeline returned code {result_code.returncode}")
            if not verbose and result_code.stderr:
                print(f"  STDERR: {result_code.stderr[:500]}")
    except Exception as e:
        print(f"  ERROR running pipeline: {e}")

    # Collect output files
    prism_output_dir = output_dir / "output"
    if prism_output_dir.exists():
        for f in prism_output_dir.glob("*.parquet"):
            result[f.stem] = str(f)

    if verbose:
        print("\n" + "=" * 70)
        print("DYNAMICAL ATLAS COMPLETE")
        print("=" * 70)
        print(f"Output directory: {output_dir}")
        print("\nGenerated files:")
        for name, path in result.items():
            if Path(path).exists():
                size = Path(path).stat().st_size / 1024
                print(f"  {name}: {size:.1f} KB")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="CSV to Dynamical Atlas - Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - auto-detect everything
  python -m orthon.entry_points.csv_to_atlas data.csv

  # Specify output directory
  python -m orthon.entry_points.csv_to_atlas data.xlsx -o my_analysis/

  # Specify signal columns explicitly
  python -m orthon.entry_points.csv_to_atlas data.csv --signals temp pressure flow

  # Specify cohort/entity column
  python -m orthon.entry_points.csv_to_atlas data.csv --cohort-col engine_id

  # Just generate manifest, skip PRISM
  python -m orthon.entry_points.csv_to_atlas data.csv --skip-prism

Output:
  observations.parquet    - Canonical format data
  typology_raw.parquet    - 27 raw measures per signal
  typology.parquet        - Signal classification
  manifest.yaml           - Engine selection
  output/                 - PRISM outputs (if not --skip-prism)
"""
    )
    parser.add_argument('input', help='Input file (CSV, Excel, Parquet, TSV, MATLAB)')
    parser.add_argument('-o', '--output-dir', default='output',
                        help='Output directory (default: output/)')
    parser.add_argument('--signals', nargs='+', default=None,
                        help='Signal column names (auto-detect if not specified)')
    parser.add_argument('--cohort-col', default=None,
                        help='Column to use as cohort/entity grouping')
    parser.add_argument('--index-col', default=None,
                        help='Column to use as sequential index')
    parser.add_argument('--skip-prism', action='store_true',
                        help='Stop after manifest generation (skip PRISM)')
    parser.add_argument('--stages', nargs='+', default=None,
                        help='Specific PRISM stages to run')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    run(
        input_path=args.input,
        output_dir=args.output_dir,
        signal_columns=args.signals,
        cohort_column=args.cohort_col,
        index_column=args.index_col,
        skip_prism=args.skip_prism,
        prism_stages=args.stages,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
