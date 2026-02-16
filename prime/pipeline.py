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


def run_pipeline(domain_path: Path):
    """
    Run the complete Prime pipeline.

    Args:
        domain_path: Path to domain directory containing raw data
                     (or at minimum, observations.parquet).
    """
    has_manifold = _check_dependencies()

    output_dir = domain_path / "output"

    observations_path = domain_path / "observations.parquet"
    typology_raw_path = domain_path / "typology_raw.parquet"
    typology_path = domain_path / "typology.parquet"
    manifest_path = domain_path / "manifest.yaml"

    print(f"=== PRIME: {domain_path.name} ===\n")

    # ----------------------------------------------------------
    # Step 1: INGEST — raw files → observations.parquet
    # ----------------------------------------------------------
    print("[1/7] Ingesting raw data...")
    try:
        raw_file = _find_raw_file(domain_path)
        if raw_file is None:
            raise FileNotFoundError("No raw data files found")

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
        output_dir=str(output_dir),
        verbose=False,
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

        # Wipe output directory — fresh start
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from prime.core.manifold_client import run_manifold

        run_manifold(
            observations_path=observations_path,
            manifest_path=manifest_path,
            output_dir=output_dir,
            verbose=True,
        )
        output_files = list(output_dir.rglob("*.parquet"))
        print(f"  → {output_dir}/ ({len(output_files)} files)")
    else:
        print("[5/7] Skipping Manifold compute (not installed)")
        output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Step 6: ANALYZE — SQL layers on parquets
    # ----------------------------------------------------------
    print("[6/7] Running SQL analysis...")
    from prime.sql.runner import run_sql_analysis

    try:
        run_sql_analysis(domain_path)
    except Exception as e:
        print(f"  SQL analysis: {e}")

    # ----------------------------------------------------------
    # Step 7: SUMMARY
    # ----------------------------------------------------------
    print(f"\n[7/7] Done. Run 'prime query {domain_path}' to explore results.\n")
    _print_summary(domain_path, typology_raw, typology, output_dir)


def _find_raw_file(domain_path: Path) -> Path | None:
    """Find a raw data file in the domain directory."""
    skip_stems = {'observations', 'typology', 'typology_raw', 'validated'}
    for ext in ['*.csv', '*.parquet', '*.xlsx', '*.tsv', '*.txt']:
        candidates = [c for c in domain_path.glob(ext) if c.stem not in skip_stems]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return max(candidates, key=lambda f: f.stat().st_size)
    return None


def _print_summary(domain_path, typology_raw, typology, output_dir):
    """Print key results after pipeline completion."""
    n_signals = len(typology_raw)
    cohort_col = 'cohort' if 'cohort' in typology_raw.columns else None
    n_cohorts = typology_raw[cohort_col].n_unique() if cohort_col else 1

    print(f"=== RESULTS: {domain_path.name} ===")
    print(f"  Signals:  {n_signals}")
    print(f"  Cohorts:  {n_cohorts}")
    print()

    # Classification summary
    if 'temporal_pattern' in typology.columns:
        print("  Temporal patterns:")
        patterns = typology.group_by('temporal_pattern').len().sort('len', descending=True)
        for row in patterns.iter_rows(named=True):
            print(f"    {row['temporal_pattern']}: {row['len']}")
        print()

    # Output files
    output_files = sorted(output_dir.rglob("*.parquet"))
    if output_files:
        print(f"  Output files ({len(output_files)}):")
        for f in output_files:
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name} ({size_kb:.1f} KB)")
    else:
        print("  No Manifold output files (manifold not installed)")

    # SQL reports
    sql_dir = output_dir / 'sql'
    if sql_dir.exists():
        md_files = list(sql_dir.glob('*.md'))
        if md_files:
            print(f"\n  SQL reports ({len(md_files)}): {sql_dir}")

    print()
    print(f"  Domain: {domain_path}")
    print(f"  Output: {output_dir}")
