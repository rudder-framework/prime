"""
Full pipeline: domain path in, results out.
Every run is fresh. All intermediate files overwritten.
"""

import shutil
import sys
from pathlib import Path


def _check_dependencies():
    """Check required and optional dependencies."""
    # Fatal — typology cannot run without pmtvs
    try:
        import pmtvs
    except ImportError:
        print("FATAL: pmtvs not installed.")
        print("Run: pip install pmtvs")
        sys.exit(1)

    # Optional — pipeline degrades gracefully without manifold
    try:
        import manifold
        return True  # manifold available
    except ImportError:
        print("WARNING: manifold not installed. Skipping compute stage.")
        print("SQL reports will use observation data only.")
        print("Install: pip install -e ~/manifold")
        return False  # manifold not available


def run_pipeline(domain_path: Path, axis: str = "time"):
    """
    Run the complete Prime pipeline.

    Args:
        domain_path: Path to domain directory containing raw data
                     (or at minimum, observations.parquet).
        axis: Signal to use as ordering axis. Default "time" uses
              row order (identical to current behavior). Any other
              value selects that signal's values as signal_0.
    """
    has_manifold = _check_dependencies()

    run_dir = domain_path / axis
    observations_path = domain_path / "observations.parquet"
    typology_raw_path = run_dir / "typology_raw.parquet"
    typology_path = run_dir / "typology.parquet"
    manifest_path = run_dir / "manifest.yaml"

    print(f"=== PRIME: {domain_path.name} (axis={axis}) ===\n")

    # Confirm overwrite if run directory already exists
    if run_dir.exists():
        print(f"WARNING: Run directory already exists: {run_dir}")
        response = input("  Overwrite? [y/N] ")
        if response.lower() not in ('y', 'yes'):
            print("Aborted.")
            sys.exit(0)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Step 1: INGEST — raw files → observations.parquet
    # ----------------------------------------------------------
    print("[1/7] Ingesting raw data...")
    try:
        raw_file = _find_raw_file(domain_path)
        if raw_file is None:
            raise FileNotFoundError("No raw data files found")

        from prime.ingest.from_raw import detect_format, ingest_cmapss, write_observations

        fmt = detect_format(raw_file)
        if fmt == "cmapss":
            df = ingest_cmapss(raw_file)
            write_observations(df, domain_path)
        else:
            from prime.ingest.transform import transform_to_manifold_format
            transform_to_manifold_format(
                input_path=raw_file,
                output_path=observations_path,
            )
        print(f"  → {observations_path} (overwritten)")
    except Exception as e:
        if not observations_path.exists():
            print(f"  Ingest failed: {e}")
            print(f"  No observations.parquet found. Cannot continue.")
            sys.exit(1)
        print(f"  Using existing observations.parquet")

    # Axis selection (post-ingest)
    axis_observations_path = run_dir / f"{axis}_observations.parquet"
    if axis == "time":
        shutil.copy2(observations_path, axis_observations_path)
        print(f"  → {axis_observations_path} (axis=time)")
    else:
        print(f"  Applying axis selection (axis={axis})...")
        from prime.ingest.axis import reaxis_observations
        reaxis_observations(observations_path, axis, axis_observations_path)

    observations_path = axis_observations_path

    # ----------------------------------------------------------
    # Step 2: TYPOLOGY_RAW — observations → measures per signal
    # ----------------------------------------------------------
    print("[2/7] Computing typology (pmtvs)...")
    from prime.ingest.typology_raw import compute_typology_raw

    typology_raw = compute_typology_raw(
        str(observations_path),
        str(typology_raw_path),
        verbose=True,
    )
    n_signals = len(typology_raw)
    print(f"  → {typology_raw_path} ({n_signals} signals)")

    # ----------------------------------------------------------
    # Step 3: CLASSIFY — typology_raw → 10 classification dimensions
    # ----------------------------------------------------------
    print("[3/7] Classifying signals...")
    from prime.entry_points.stage_03_classify import run as classify

    typology = classify(
        str(typology_raw_path),
        str(typology_path),
        verbose=False,
    )
    print(f"  → {typology_path}")

    # ----------------------------------------------------------
    # Step 4: MANIFEST — typology → engine selection per signal
    # ----------------------------------------------------------
    print("[4/7] Generating manifest...")
    from prime.entry_points.stage_04_manifest import run as generate_manifest

    manifest = generate_manifest(
        str(typology_path),
        str(manifest_path),
        observations_path=str(observations_path),
        verbose=False,
        axis=axis,
    )
    summary = manifest.get('summary', {})
    print(f"  → {manifest_path}")
    print(f"    Active signals: {summary.get('active_signals', 'N/A')}")
    print(f"    Engines: {len(summary.get('signal_engines', []))}")

    # ----------------------------------------------------------
    # Step 5: COMPUTE — manifold.run()
    # ----------------------------------------------------------
    if has_manifold:
        print("[5/7] Running Manifold compute engine...")

        # Output dir derived from observations prefix — matches manifest.
        # Manifold wipes output_dir on startup — Prime's files at run_dir root are safe.
        output_dir = Path(manifest['paths']['output_dir'])

        from prime.core.manifold_client import run_manifold

        run_manifold(
            observations_path=observations_path,
            manifest_path=manifest_path,
            output_dir=output_dir,
            verbose=True,
        )
        manifold_files = list(output_dir.rglob("*.parquet"))
        print(f"  → {output_dir}/ ({len(manifold_files)} Manifold files)")
    else:
        print("[5/7] Skipping Manifold compute (not installed)")

    # ----------------------------------------------------------
    # Step 6: ANALYZE — SQL layers on parquets
    # ----------------------------------------------------------
    print("[6/7] Running SQL analysis...")
    from prime.sql.runner import run_sql_analysis

    try:
        run_sql_analysis(run_dir)
    except Exception as e:
        print(f"  SQL analysis: {e}")

    # ----------------------------------------------------------
    # Step 6b: PARAMETERIZATION COMPILATION (if 2+ runs exist)
    # ----------------------------------------------------------
    from prime.parameterization.compile import discover_runs, compile_parameterization

    if len(discover_runs(domain_path)) >= 2:
        print("[6b/7] Compiling cross-run parameterization...")
        compile_parameterization(domain_path, verbose=True)

    # ----------------------------------------------------------
    # Step 7: SUMMARY
    # ----------------------------------------------------------
    print(f"\n[7/7] Done. Run 'prime query {run_dir}' to explore results.\n")
    output_dir = Path(manifest['paths']['output_dir'])
    _print_summary(domain_path, typology_raw, typology, run_dir, output_dir)


def _find_raw_file(domain_path: Path) -> Path | None:
    """Find a raw data file in the domain directory."""
    skip_stems = {'observations', 'typology', 'typology_raw', 'validated'}
    for ext in ['*.csv', '*.parquet', '*.xlsx', '*.tsv', '*.txt']:
        candidates = [
            c for c in domain_path.glob(ext)
            if c.stem not in skip_stems
            and not c.stem.endswith('_observations')
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return max(candidates, key=lambda f: f.stat().st_size)
    return None


def _print_summary(domain_path, typology_raw, typology, run_dir, output_dir: Path):
    """Print key results after pipeline completion."""
    n_signals = len(typology_raw)
    cohort_col = 'cohort' if 'cohort' in typology_raw.columns else None
    n_cohorts = typology_raw[cohort_col].n_unique() if cohort_col else 1

    print(f"=== RESULTS: {domain_path.name} ===")
    print(f"  Signals:  {n_signals}")
    print(f"  Cohorts:  {n_cohorts}")
    print()

    # Classification summary
    if 'temporal_primary' in typology.columns:
        print("  Temporal patterns:")
        patterns = typology.group_by('temporal_primary').len().sort('len', descending=True)
        for row in patterns.iter_rows(named=True):
            print(f"    {row['temporal_primary']}: {row['len']}")
        print()

    manifold_files = list(output_dir.rglob("*.parquet")) if output_dir.exists() else []
    if manifold_files:
        print(f"  Manifold output files ({len(manifold_files)}):")
        for f in sorted(manifold_files):
            size_kb = f.stat().st_size / 1024
            rel = f.relative_to(run_dir)
            print(f"    {rel} ({size_kb:.1f} KB)")
    else:
        print("  No Manifold output files (manifold not installed)")

    # SQL reports
    sql_dir = run_dir / 'sql'
    if sql_dir.exists():
        md_files = list(sql_dir.glob('*.md'))
        if md_files:
            print(f"\n  SQL reports ({len(md_files)}): {sql_dir}")

    print()
    print(f"  Domain:  {domain_path}")
    print(f"  Run dir: {run_dir}")
