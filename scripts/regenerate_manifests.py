#!/usr/bin/env python3
"""
Regenerate all manifests in Domains/ to v2.5 schema.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from framework.manifest.generator import build_manifest, save_manifest, validate_manifest

def regenerate_manifest(domain_path: Path) -> bool:
    """Regenerate manifest for a single domain."""
    typology_path = domain_path / 'typology.parquet'
    manifest_path = domain_path / 'manifest.yaml'

    if not typology_path.exists():
        print(f"  SKIP: No typology.parquet")
        return False

    try:
        # Load typology
        typology_df = pd.read_parquet(typology_path)

        # Ensure required columns exist
        if 'signal_id' not in typology_df.columns:
            print(f"  SKIP: Missing signal_id column")
            return False

        # Add cohort column if missing (use domain name)
        if 'cohort' not in typology_df.columns:
            typology_df['cohort'] = domain_path.name

        # Add defaults for missing columns
        if 'temporal_pattern' not in typology_df.columns:
            typology_df['temporal_pattern'] = 'STATIONARY'
        if 'spectral' not in typology_df.columns:
            typology_df['spectral'] = 'BROADBAND'
        if 'n_samples' not in typology_df.columns:
            typology_df['n_samples'] = 1000

        # Build manifest
        manifest = build_manifest(
            typology_df=typology_df,
            observations_path='observations.parquet',
            typology_path='typology.parquet',
            output_dir='output/',
            job_id=domain_path.name,
        )

        # Validate
        errors = validate_manifest(manifest)
        if errors:
            print(f"  WARN: Validation errors: {errors[:3]}")

        # Save
        save_manifest(manifest, str(manifest_path))

        n_signals = manifest['summary']['total_signals']
        n_active = manifest['summary']['active_signals']
        sys_window = manifest['system']['window']
        print(f"  OK: {n_signals} signals ({n_active} active), system_window={sys_window}")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    """Regenerate all manifests."""
    parser = argparse.ArgumentParser(
        description='Regenerate all manifests in Domains/ to v2.5 schema.'
    )
    parser.add_argument(
        '--domains-dir', '-d',
        default=str(Path.home() / 'Domains'),
        help='Root directory containing domain folders'
    )
    args = parser.parse_args()

    domains_root = Path(args.domains_dir)

    print("Regenerating manifests to v2.5 schema...")
    print("=" * 60)

    # Find all directories with typology.parquet
    typology_files = list(domains_root.glob('**/typology.parquet'))

    # Exclude test_domains and benchmarks (duplicates)
    typology_files = [
        f for f in typology_files
        if 'test_domains' not in str(f) and 'benchmarks' not in str(f)
    ]

    success = 0
    failed = 0

    for typology_path in sorted(typology_files):
        domain_path = typology_path.parent
        print(f"\n{domain_path.name}:")

        if regenerate_manifest(domain_path):
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Done: {success} succeeded, {failed} failed")


if __name__ == '__main__':
    main()
