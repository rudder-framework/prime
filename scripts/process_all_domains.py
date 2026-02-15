#!/usr/bin/env python3
"""
Process all domains: Compute typology + Generate manifest v2.5

Runs sequentially so that typology.parquet and manifest.yaml
are created in each domain directory before moving to the next.

Handles schema normalization automatically:
- Renames unit_id/entity_id → cohort
- Melts wide-format data (sensor columns) → long-format (signal_id, value)
- Ensures canonical schema: (cohort, signal_id, I, value)

Usage:
    python scripts/process_all_domains.py
    python scripts/process_all_domains.py --domains-dir /path/to/domains
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import Dict, Any, List, Optional, Tuple
import traceback

import polars as pl
import pandas as pd

from prime.ingest.normalize import normalize_observations
from prime.ingest.typology_raw import compute_typology_raw
from prime.typology.discrete_sparse import apply_discrete_sparse_classification
from prime.typology.level2_corrections import apply_corrections
from prime.typology.window_factor import add_window_factor
from prime.manifest.generator import build_manifest, save_manifest, validate_manifest


def validate_environment():
    """Fail fast if imports are broken."""
    try:
        from prime.core.data_reader import DataProfile
        from prime.config import recommender
        from prime.typology.discrete_sparse import classify_discrete_sparse
    except ImportError as e:
        print(f"ERROR: Import chain broken: {e}")
        print("Run: python scripts/update_imports.py")
        print("Then: python scripts/test_pipeline.py")
        sys.exit(1)


# Validate imports at module load
validate_environment()


def compute_full_typology(observations_path: Path, output_dir: Path, verbose: bool = True) -> Optional[pl.DataFrame]:
    """
    Compute full typology pipeline for a domain.

    Steps:
        1. Compute raw typology measures (27 metrics)
        2. Apply discrete/sparse classification (PR5)
        3. Apply continuous classification (PR4)
        4. Save typology.parquet

    Returns:
        DataFrame with full typology, or None on failure
    """
    typology_raw_path = output_dir / 'typology_raw.parquet'
    typology_path = output_dir / 'typology.parquet'

    try:
        # Step 1: Compute raw typology measures
        if verbose:
            print("  [1/3] Computing raw typology measures...")

        raw_df = compute_typology_raw(
            str(observations_path),
            str(typology_raw_path),
            verbose=False
        )

        if verbose:
            print(f"        → {len(raw_df)} signals, saved to typology_raw.parquet")

        # Step 2 & 3: Apply classifications
        if verbose:
            print("  [2/3] Applying PR5 (discrete/sparse) + PR4 (continuous) classification...")

        # Convert to list of dicts for classification
        rows = raw_df.to_dicts()
        classified_rows = []

        for row in rows:
            # PR5: Discrete/sparse classification (runs first)
            row = apply_discrete_sparse_classification(row)

            # PR4: Continuous classification (if not discrete/sparse)
            if not row.get('is_discrete_sparse', False):
                row = apply_corrections(row)

            # Ensure required columns exist
            if 'temporal_pattern' not in row:
                row['temporal_pattern'] = 'STATIONARY'
            if 'spectral' not in row:
                row['spectral'] = 'BROADBAND'

            classified_rows.append(row)

        # Create final typology DataFrame
        typology_df = pl.DataFrame(classified_rows)

        # Add window_factor for adaptive windowing
        typology_df = add_window_factor(typology_df)

        # Save typology.parquet
        typology_df.write_parquet(str(typology_path))

        if verbose:
            # Count by type
            type_counts = {}
            for row in classified_rows:
                tp = row.get('temporal_pattern', 'UNKNOWN')
                type_counts[tp] = type_counts.get(tp, 0) + 1

            type_str = ', '.join(f"{k}:{v}" for k, v in sorted(type_counts.items()))
            print(f"        → Classifications: {type_str}")
            print(f"        → Saved to typology.parquet")

        return typology_df

    except Exception as e:
        print(f"  ERROR computing typology: {e}")
        if verbose:
            traceback.print_exc()
        return None


def generate_manifest(typology_path: Path, output_dir: Path, domain_name: str, verbose: bool = True) -> bool:
    """
    Generate manifest.yaml from typology.parquet.

    Returns:
        True on success, False on failure
    """
    manifest_path = output_dir / 'manifest.yaml'

    try:
        if verbose:
            print("  [3/3] Generating manifest v2.5...")

        # Read typology (use pandas for manifest generator compatibility)
        typology_df = pd.read_parquet(str(typology_path))

        # Ensure required columns
        if 'signal_id' not in typology_df.columns:
            print("  ERROR: Missing signal_id column")
            return False

        if 'cohort' not in typology_df.columns:
            typology_df['cohort'] = domain_name

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
            job_id=domain_name,
        )

        # Validate
        errors = validate_manifest(manifest)
        if errors:
            print(f"  WARN: Validation issues: {errors[:3]}")

        # Save
        save_manifest(manifest, str(manifest_path))

        if verbose:
            n_signals = manifest['summary']['total_signals']
            n_active = manifest['summary']['active_signals']
            n_constant = manifest['summary']['constant_signals']
            sys_window = manifest['system']['window']
            print(f"        → {n_signals} signals ({n_active} active, {n_constant} constant)")
            print(f"        → system_window={sys_window}")
            print(f"        → Saved to manifest.yaml")

        return True

    except Exception as e:
        print(f"  ERROR generating manifest: {e}")
        if verbose:
            traceback.print_exc()
        return False


def process_domain(domain_path: Path, verbose: bool = True) -> bool:
    """
    Process a single domain: normalize → typology → manifest.

    Returns:
        True on success, False on failure
    """
    observations_path = domain_path / 'observations.parquet'

    if not observations_path.exists():
        if verbose:
            print(f"  SKIP: No observations.parquet")
        return False

    # Step 0: Normalize schema (wide→long, rename aliases)
    try:
        changed, repairs = normalize_observations(observations_path, verbose)
        if changed and verbose:
            print(f"  [0/3] Schema normalized ({len(repairs)} repairs)")
    except Exception as e:
        print(f"  ERROR normalizing schema: {e}")
        if verbose:
            traceback.print_exc()
        return False

    # Step 1-2: Compute typology
    typology_df = compute_full_typology(observations_path, domain_path, verbose)
    if typology_df is None:
        return False

    # Step 3: Generate manifest
    typology_path = domain_path / 'typology.parquet'
    success = generate_manifest(typology_path, domain_path, domain_path.name, verbose)

    return success


def find_domains(domains_root: Path, exclude_patterns: List[str] = None) -> List[Path]:
    """
    Find all directories with observations.parquet.
    """
    if exclude_patterns is None:
        exclude_patterns = ['test_domains', 'benchmarks', '.git', '__pycache__']

    domains = []

    for obs_path in sorted(domains_root.glob('**/observations.parquet')):
        domain_path = obs_path.parent

        # Check exclusions
        skip = False
        for pattern in exclude_patterns:
            if pattern in str(domain_path):
                skip = True
                break

        if not skip:
            domains.append(domain_path)

    return domains


def main():
    parser = argparse.ArgumentParser(
        description='Process all domains: compute typology + generate manifest v2.5'
    )
    parser.add_argument(
        '--domains-dir', '-d',
        default=str(Path.home() / 'Domains'),
        help='Root directory containing domain folders'
    )
    parser.add_argument(
        '--include-test',
        action='store_true',
        help='Include test_domains and benchmarks'
    )
    parser.add_argument(
        '--domain',
        help='Process only this specific domain (by name)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )

    args = parser.parse_args()

    domains_root = Path(args.domains_dir)
    if not domains_root.exists():
        print(f"ERROR: Domains directory not found: {domains_root}")
        return 1

    # Find domains
    exclude = [] if args.include_test else ['test_domains', 'benchmarks']
    domains = find_domains(domains_root, exclude)

    # Filter to specific domain if requested
    if args.domain:
        domains = [d for d in domains if d.name == args.domain]
        if not domains:
            print(f"ERROR: Domain '{args.domain}' not found")
            return 1

    verbose = not args.quiet

    print("=" * 60)
    print("PRIME Domain Processing Pipeline")
    print("=" * 60)
    print(f"Domains root: {domains_root}")
    print(f"Found {len(domains)} domains to process")
    print("=" * 60)

    success = 0
    failed = 0
    skipped = 0

    for i, domain_path in enumerate(domains, 1):
        print(f"\n[{i}/{len(domains)}] {domain_path.name}")
        print("-" * 40)

        result = process_domain(domain_path, verbose)

        if result:
            success += 1
            print(f"  ✓ SUCCESS")
        else:
            # Check if it was skipped vs failed
            if not (domain_path / 'observations.parquet').exists():
                skipped += 1
            else:
                failed += 1
                print(f"  ✗ FAILED")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {success} succeeded, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
