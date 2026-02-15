"""
04: Manifest Generation Entry Point
===================================

Pure orchestration - calls manifest generator module.
Generates manifest.yaml with engine selection per signal.

Stages: typology.parquet → manifest.yaml

The manifest tells Manifold exactly which engines to run per signal.
"""

import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional

# Import manifest generator
from prime.manifest.generator import (
    build_manifest,
    save_manifest,
    validate_manifest,
)


def run(
    typology_path: str,
    output_path: str = "manifest.yaml",
    observations_path: str = "observations.parquet",
    output_dir: str = "output/",
    verbose: bool = True,
) -> dict:
    """
    Generate manifest from typology.

    Args:
        typology_path: Path to typology.parquet
        output_path: Output path for manifest.yaml
        observations_path: Path to observations.parquet (relative, for manifest)
        output_dir: Output directory for Manifold (relative, for manifest)
        verbose: Print progress

    Returns:
        Manifest dict
    """
    if verbose:
        print("=" * 70)
        print("04: MANIFEST GENERATION")
        print("=" * 70)

    # Load typology
    typology_df = pd.read_parquet(typology_path)
    if verbose:
        print(f"Loaded: {len(typology_df)} signals from typology")

    # Build manifest
    manifest = build_manifest(
        typology_df,
        observations_path=observations_path,
        typology_path=str(Path(typology_path).name),
        output_dir=output_dir,
    )

    # Validate
    errors = validate_manifest(manifest)
    if errors:
        if verbose:
            print(f"\nValidation errors:")
            for err in errors:
                print(f"  ✗ {err}")
        raise ValueError(f"Manifest validation failed: {len(errors)} errors")

    if verbose:
        summary = manifest.get('summary', {})
        print(f"\nManifest summary:")
        print(f"  Total signals: {summary.get('total_signals', 0)}")
        print(f"  Active signals: {summary.get('active_signals', 0)}")
        print(f"  Constant signals: {summary.get('constant_signals', 0)}")
        print(f"  Cohorts: {summary.get('total_cohorts', 0)}")
        print(f"  Signal engines: {summary.get('n_signal_engines', 0)}")

        system = manifest.get('system', {})
        print(f"\nSystem window: {system.get('window', 'N/A')}")
        print(f"System stride: {system.get('stride', 'N/A')}")

    # Save
    save_manifest(manifest, output_path)
    if verbose:
        print(f"\nSaved: {output_path}")

    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(description="04: Manifest Generation")
    parser.add_argument('typology', help='Path to typology.parquet')
    parser.add_argument('--output', '-o', default='manifest.yaml', help='Output path')
    parser.add_argument('--observations', default='observations.parquet', help='Observations path (for manifest)')
    parser.add_argument('--output-dir', default='output/', help='Manifold output directory')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.typology,
        args.output,
        observations_path=args.observations,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
