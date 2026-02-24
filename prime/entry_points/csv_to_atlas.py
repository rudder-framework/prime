"""
CSV to Dynamical Atlas - One Command Pipeline
==============================================

The "stranger uploads a CSV and gets a dynamical atlas" entry point.

Usage:
    python -m prime.entry_points.csv_to_atlas data.csv --output-dir output/
    python -m prime.entry_points.csv_to_atlas data.xlsx --signals temp,pressure,flow
    python -m prime.entry_points.csv_to_atlas data.parquet --cohort-col entity_id

Pipeline:
    1. Load data (CSV, Excel, Parquet, TSV, MATLAB)
    2. Auto-detect or use specified column mappings
    3. Transform to observations.parquet (canonical format)
    4. Validate schema
    5. Compute typology (signal characterization)
    6. Generate manifest (engine selection per signal)
    7. Run ENGINES pipeline (signal_vector → geometry → dynamics)
    8. Output: Complete dynamical atlas in output directory

Output Files:
    observations.parquet    - Canonical format data
    typology_raw.parquet    - 27 raw measures per signal
    typology.parquet        - Signal classification
    output_time/
        manifest.yaml       - Engine selection and parameters
        system/             - Manifold output parquets
            signal_vector.parquet
            state_geometry.parquet
            ...
"""

import argparse
import subprocess
from pathlib import Path
from typing import Optional, List

import polars as pl


def run(
    input_path: str,
    output_dir: str = "output",
    signal_columns: Optional[List[str]] = None,
    cohort_column: Optional[str] = None,
    index_column: Optional[str] = None,
    skip_manifold: bool = False,
    manifold_stages: Optional[List[str]] = None,
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
        skip_manifold: If True, stop after manifest generation (Prime only)
        manifold_stages: Specific ENGINES stages to run (None = full pipeline)
        verbose: Print progress

    Returns:
        dict with paths to all generated files
    """
    from prime.ingest.upload import load_file
    from prime.ingest.normalize import normalize_observations
    from prime.ingest.transform import validate_manifold_schema, ensure_signal_0_sorted

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
        for candidate in ['signal_0', 'I', 'index', 'cycle', 'timestamp', 'time', 't']:
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
        print("\n[3/6] Transforming to canonical format...")

    observations_path = output_dir / "observations.parquet"

    # Check if data is already long format (has signal_id + value)
    if "signal_id" in df.columns and "value" in df.columns:
        # Already long — write as-is, normalize will handle renames
        df.write_parquet(observations_path)
    else:
        # Wide format — prepare columns then write, normalize will melt
        # Rename cohort/index columns to canonical names before writing
        if cohort_column and cohort_column in df.columns:
            df = df.rename({cohort_column: "cohort"})
        if index_column and index_column in df.columns:
            df = df.rename({index_column: "signal_0"})

        # Select only signal columns + metadata for writing
        keep_cols = list(signal_columns)
        if "cohort" in df.columns:
            keep_cols.append("cohort")
        if "signal_0" in df.columns:
            keep_cols.append("signal_0")
        df.select(keep_cols).write_parquet(observations_path)

    # Normalize schema (handles wide→long melt, renames, I creation)
    changed, repairs = normalize_observations(observations_path, verbose=verbose)
    if changed and verbose:
        print(f"  Schema normalized ({len(repairs)} repairs)")

    # Reload and validate
    df_long = pl.read_parquet(observations_path)

    # Ensure unit column exists (default to "" for dimensionless/unknown)
    if "unit" not in df_long.columns:
        df_long = df_long.with_columns(pl.lit("").alias("unit"))
        df_long.write_parquet(observations_path)

    is_valid, errors = validate_manifold_schema(df_long)
    if not is_valid:
        # Try to fix common issues - sort signal_0 if needed
        if "not sorted" in str(errors):
            df_long = ensure_signal_0_sorted(df_long)
            df_long.write_parquet(observations_path)
            is_valid, errors = validate_manifold_schema(df_long)

    if not is_valid:
        print(f"  WARNING: Validation issues: {errors}")

    n_cohorts = df_long["cohort"].n_unique() if "cohort" in df_long.columns else 1
    n_signals = df_long["signal_id"].n_unique()
    n_obs = len(df_long)

    # Write signals.parquet alongside observations.parquet
    from prime.ingest.signal_metadata import write_signal_metadata
    write_signal_metadata(df_long, output_dir)

    if verbose:
        print(f"  Saved: {observations_path}")
        print(f"  Cohorts: {n_cohorts}, Signals: {n_signals}, Observations: {n_obs:,}")

    # =========================================================================
    # STEP 4: COMPUTE TYPOLOGY
    # =========================================================================
    if verbose:
        print("\n[4/6] Computing typology (signal characterization)...")

    from prime.ingest.typology_raw import compute_typology_raw
    from prime.typology.discrete_sparse import apply_discrete_sparse_classification
    from prime.typology.level2_corrections import apply_corrections

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
        if row.get('temporal_primary', row.get('temporal_pattern')) in [None, 'UNKNOWN', '']:
            row_dict = row.to_dict()
            result = apply_corrections(row_dict)
            for k, v in result.items():
                typology_df.at[idx, k] = v

    # Save typology
    typology_pl = pl.from_pandas(typology_df)
    typology_pl.write_parquet(typology_path)

    if verbose:
        print(f"  Saved: {typology_path}")
        group_col = "temporal_primary" if "temporal_primary" in typology_pl.columns else "temporal_pattern"
        patterns = typology_pl.group_by(group_col).len().sort("len", descending=True)
        print("  Signal types:")
        for row in patterns.head(5).iter_rows(named=True):
            print(f"    {row[group_col]}: {row['len']}")

    # =========================================================================
    # STEP 5: GENERATE MANIFEST
    # =========================================================================
    if verbose:
        print("\n[5/6] Generating manifest (engine selection)...")

    from prime.manifest.generator import build_manifest, save_manifest

    manifest = build_manifest(
        typology_df=typology_df,
        observations_path=str(observations_path),
        typology_path=str(typology_path),
        output_dir=str(output_dir / "output_time"),
    )

    engines_output_dir = output_dir / "output_time"
    engines_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = engines_output_dir / "manifest.yaml"
    save_manifest(manifest, str(manifest_path))

    if verbose:
        print(f"  Saved: {manifest_path}")
        print(f"  Active signals: {manifest.get('summary', {}).get('active_signals', 'N/A')}")
        print(f"  Engines: {len(manifest.get('summary', {}).get('signal_engines', []))}")

    # =========================================================================
    # STEP 6: RUN MANIFOLD PIPELINE (optional)
    # =========================================================================
    result = {
        'observations': str(observations_path),
        'typology_raw': str(typology_raw_path),
        'typology': str(typology_path),
        'manifest': str(manifest_path),
        'output_dir': str(output_dir),
    }

    if skip_manifold:
        if verbose:
            print("\n[6/6] Skipping ENGINES (--skip-engines)")
        return result

    if verbose:
        print("\n[6/6] Running ENGINES pipeline...")

    # Try to run ENGINES
    engines_dir = Path(__file__).parent.parent.parent.parent / "engines"

    if not (engines_dir / "engines").exists():
        # Try alternative location
        engines_dir = Path.home() / "engines"

    if not (engines_dir / "engines").exists():
        print("  WARNING: ENGINES not found. Run manually with:")
        print(f"    cd ~/engines && ./venv/bin/python -m engines.entry_points.run_pipeline {manifest_path}")
        return result

    engines_python = engines_dir / "venv" / "bin" / "python"

    # Run ENGINES pipeline via run_pipeline entry point
    if manifold_stages:
        # Run specific stages
        stages_str = ",".join(manifold_stages)
        cmd = [str(engines_python), "-m", "engines.entry_points.run_pipeline",
               str(manifest_path), "--stages", stages_str]
    else:
        # Run core + atlas stages
        cmd = [str(engines_python), "-m", "engines.entry_points.run_pipeline",
               str(manifest_path), "--stages", "01,02,03,21,22,23"]

    if verbose:
        print(f"  Command: {' '.join(cmd)}")

    try:
        result_code = subprocess.run(
            cmd,
            cwd=str(engines_dir),
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
    if engines_output_dir.exists():
        for f in engines_output_dir.glob("*.parquet"):
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
  python -m prime.entry_points.csv_to_atlas data.csv

  # Specify output directory
  python -m prime.entry_points.csv_to_atlas data.xlsx -o my_analysis/

  # Specify signal columns explicitly
  python -m prime.entry_points.csv_to_atlas data.csv --signals temp pressure flow

  # Specify cohort/entity column
  python -m prime.entry_points.csv_to_atlas data.csv --cohort-col engine_id

  # Just generate manifest, skip ENGINES
  python -m prime.entry_points.csv_to_atlas data.csv --skip-engines

Output:
  observations.parquet    - Canonical format data
  typology_raw.parquet    - 27 raw measures per signal
  typology.parquet        - Signal classification
  output_time/            - Manifold outputs and manifest (if not --skip-engines)
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
    parser.add_argument('--skip-manifold', '--skip-engines', action='store_true',
                        help='Stop after manifest generation (skip ENGINES)')
    parser.add_argument('--stages', nargs='+', default=None,
                        help='Specific ENGINES stages to run')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    run(
        input_path=args.input,
        output_dir=args.output_dir,
        signal_columns=args.signals,
        cohort_column=args.cohort_col,
        index_column=args.index_col,
        skip_manifold=args.skip_manifold,
        manifold_stages=args.stages,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
