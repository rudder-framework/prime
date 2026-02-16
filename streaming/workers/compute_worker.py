"""
Compute worker for streaming pipeline.

Runs Prime stages 2-6 on a single partition:
  [2] Typology raw (pmtvs measures)
  [3] Classification (discrete/sparse → continuous)
  [4] Manifest generation
  [5] Manifold compute
  [6] SQL analysis

Supports bootstrap typology reuse and partition boundary overlap.
"""

import shutil
from pathlib import Path
from typing import Optional

import polars as pl
import numpy as np


def compute_partition(
    partition_dir: Path,
    bootstrap_typology_path: Optional[Path] = None,
    bootstrap_typology_raw_path: Optional[Path] = None,
    previous_partition_dir: Optional[Path] = None,
    overlap_samples: int = 0,
    skip_manifold: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run Prime compute stages on a single partition.

    Args:
        partition_dir: Partition directory containing observations.parquet
        bootstrap_typology_path: If set, copy this typology.parquet instead of recomputing
        bootstrap_typology_raw_path: If set, copy this typology_raw.parquet too
        previous_partition_dir: Previous partition for boundary overlap
        overlap_samples: Number of trailing samples to prepend from previous partition
        skip_manifold: If True, stop after manifest generation (no Manifold compute)
        verbose: Print progress

    Returns:
        Dict with status and paths for each stage
    """
    obs_path = partition_dir / "observations.parquet"
    raw_path = partition_dir / "typology_raw.parquet"
    typ_path = partition_dir / "typology.parquet"
    manifest_path = partition_dir / "manifest.yaml"
    output_dir = partition_dir / "output"

    result = {"partition_dir": str(partition_dir), "stages": {}}

    if not obs_path.exists():
        raise FileNotFoundError(f"No observations.parquet in {partition_dir}")

    # Handle partition boundary overlap
    if previous_partition_dir and overlap_samples > 0:
        _prepend_overlap(obs_path, previous_partition_dir / "observations.parquet", overlap_samples, verbose)

    # [2] TYPOLOGY RAW
    if bootstrap_typology_raw_path and bootstrap_typology_raw_path.exists():
        if verbose:
            print(f"    [2/6] Typology raw: reusing bootstrap → {raw_path.name}")
        shutil.copy2(bootstrap_typology_raw_path, raw_path)
        result["stages"]["typology_raw"] = "bootstrap"
    else:
        if verbose:
            print(f"    [2/6] Typology raw: computing...")
        from prime.ingest.typology_raw import compute_typology_raw
        compute_typology_raw(str(obs_path), str(raw_path), verbose=verbose)
        result["stages"]["typology_raw"] = "computed"

    # [3] CLASSIFY
    if bootstrap_typology_path and bootstrap_typology_path.exists():
        if verbose:
            print(f"    [3/6] Classification: reusing bootstrap → {typ_path.name}")
        shutil.copy2(bootstrap_typology_path, typ_path)
        result["stages"]["classification"] = "bootstrap"
    else:
        if verbose:
            print(f"    [3/6] Classification: running...")
        from prime.entry_points.stage_03_classify import run as classify
        classify(str(raw_path), str(typ_path), verbose=verbose)
        result["stages"]["classification"] = "computed"

    # [4] MANIFEST
    if verbose:
        print(f"    [4/6] Manifest: generating...")
    from prime.entry_points.stage_04_manifest import run as generate_manifest
    manifest = generate_manifest(
        str(typ_path),
        str(manifest_path),
        observations_path=str(obs_path),
        output_dir=str(output_dir),
        verbose=verbose,
    )
    result["stages"]["manifest"] = "generated"

    # [5] MANIFOLD COMPUTE
    if skip_manifold:
        if verbose:
            print(f"    [5/6] Manifold: SKIPPED (--skip-manifold)")
        result["stages"]["manifold"] = "skipped"
    else:
        if verbose:
            print(f"    [5/6] Manifold: computing...")
        output_dir.mkdir(parents=True, exist_ok=True)
        from prime.core.manifold_client import run_manifold, manifold_available
        if not manifold_available():
            if verbose:
                print(f"    WARNING: Manifold not available, skipping compute")
            result["stages"]["manifold"] = "unavailable"
        else:
            run_manifold(obs_path, manifest_path, output_dir, verbose=verbose)
            result["stages"]["manifold"] = "computed"

    # [6] SQL ANALYSIS
    if not skip_manifold and result["stages"].get("manifold") == "computed":
        if verbose:
            print(f"    [6/6] SQL analysis: running...")
        from prime.sql.runner import run_sql_analysis
        run_sql_analysis(partition_dir)
        result["stages"]["sql"] = "completed"
    else:
        if verbose:
            print(f"    [6/6] SQL analysis: SKIPPED (no Manifold output)")
        result["stages"]["sql"] = "skipped"

    return result


def _prepend_overlap(
    obs_path: Path,
    prev_obs_path: Path,
    overlap_samples: int,
    verbose: bool = True,
) -> None:
    """
    Prepend trailing samples from previous partition to current observations.

    Overlap samples get negative I values so Manifold processes them but
    results can be identified and trimmed downstream.
    """
    if not prev_obs_path.exists():
        if verbose:
            print(f"    Overlap: no previous partition, skipping")
        return

    # Read trailing samples from previous partition per (cohort, signal_id)
    prev_df = pl.read_parquet(prev_obs_path)
    current_df = pl.read_parquet(obs_path)

    # Get the last N samples per signal from previous partition
    group_cols = ["cohort", "signal_id"] if "cohort" in prev_df.columns else ["signal_id"]
    overlap_frames = []
    for _keys, group in prev_df.group_by(group_cols):
        tail = group.sort("I").tail(overlap_samples)
        n_tail = tail.height
        # Negative I values mark overlap zone (cast to Int64 since UInt32 can't hold negatives)
        tail = tail.with_columns(
            pl.Series("I", np.arange(-n_tail, 0, dtype=np.int64))
        )
        overlap_frames.append(tail)

    if overlap_frames:
        overlap_df = pl.concat(overlap_frames)
        # Cast both to Int64 so negative overlap I coexists with non-negative partition I
        overlap_df = overlap_df.with_columns(pl.col("I").cast(pl.Int64))
        current_df = current_df.with_columns(pl.col("I").cast(pl.Int64))
        merged = pl.concat([overlap_df, current_df])
        merged.write_parquet(obs_path)
        if verbose:
            print(f"    Overlap: prepended {overlap_df.height:,} samples from previous partition")
