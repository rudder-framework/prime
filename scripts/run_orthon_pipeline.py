"""
Run ORTHON Pipeline: typology_raw → SQL classification → manifest

For each domain:
1. typology_raw.parquet (already computed) → typology_v2.sql → typology.parquet
2. typology.parquet → manifest_generator → manifest.yaml

Run:
    python scripts/run_orthon_pipeline.py [domain]
    python scripts/run_orthon_pipeline.py --all
"""

import subprocess
import sys
from pathlib import Path
import duckdb
import polars as pl
import yaml


def run_sql_classification(domain_dir: Path) -> pl.DataFrame:
    """Run typology_v2.sql on typology_raw.parquet."""
    typology_raw_path = domain_dir / "typology_raw.parquet"
    typology_path = domain_dir / "typology.parquet"
    sql_path = Path(__file__).parent.parent / "orthon" / "sql" / "typology_v2.sql"

    if not typology_raw_path.exists():
        print(f"  ⚠ No typology_raw.parquet in {domain_dir}")
        return None

    print(f"  Running SQL classification...")

    # Read SQL
    sql = sql_path.read_text()

    # Execute with DuckDB
    con = duckdb.connect()

    # Register the input file
    con.execute(f"CREATE VIEW typology_raw AS SELECT * FROM read_parquet('{typology_raw_path}')")

    # Run classification
    result = con.execute(sql).pl()

    # Save
    result.write_parquet(typology_path)
    print(f"  → {typology_path} ({len(result)} signals)")

    con.close()
    return result


def run_manifest_generator(domain_dir: Path) -> dict:
    """Generate manifest from typology.parquet."""
    typology_path = domain_dir / "typology.parquet"
    observations_path = domain_dir / "observations.parquet"
    manifest_path = domain_dir / "manifest.yaml"

    if not typology_path.exists():
        print(f"  ⚠ No typology.parquet in {domain_dir}")
        return None

    print(f"  Generating manifest...")

    # Import generator
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from orthon.ingest.manifest_generator import generate_manifest

    # Generate manifest (function expects path strings)
    manifest = generate_manifest(
        typology_path=str(typology_path),
        output_path=str(manifest_path),
        observations_path=str(observations_path),
        verbose=False,
    )

    print(f"  → {manifest_path}")

    return manifest


def process_domain(domain_dir: Path):
    """Process a single domain directory."""
    print(f"\n{'='*60}")
    print(f"Processing: {domain_dir.name}")
    print(f"{'='*60}")

    # Step 1: SQL classification
    typology_df = run_sql_classification(domain_dir)

    if typology_df is None:
        return

    # Step 2: Generate manifest
    manifest = run_manifest_generator(domain_dir)

    if manifest is None:
        return

    # Summary
    print(f"\n  Summary for {domain_dir.name}:")
    print(f"    Signals: {len(manifest.get('signals', {}))}")
    print(f"    Skip signals: {len(manifest.get('skip_signals', []))}")

    # Show sample signal config
    signals = manifest.get('signals', {})
    if signals:
        sample_sig = list(signals.keys())[0]
        sample_config = signals[sample_sig]
        print(f"    Sample ({sample_sig}):")
        print(f"      Engines: {len(sample_config.get('engines', []))}")
        print(f"      Rolling: {len(sample_config.get('rolling_engines', []))}")
        print(f"      Window: {sample_config.get('window_size')}")
        print(f"      Visualizations: {sample_config.get('visualizations', [])}")


def main():
    """Process all domains or a specific one."""
    base_dir = Path(__file__).parent.parent / "data" / "test_domains"

    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            domains = [d for d in base_dir.iterdir() if d.is_dir()]
        else:
            domains = [base_dir / sys.argv[1]]
    else:
        # Default: all domains
        domains = [d for d in base_dir.iterdir() if d.is_dir()]

    for domain_dir in sorted(domains):
        if domain_dir.is_dir():
            process_domain(domain_dir)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
