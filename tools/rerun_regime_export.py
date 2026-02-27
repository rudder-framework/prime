"""
Re-run regime ML export on an existing Prime output directory.

Uses already-computed observations.parquet + regime_detection.parquet
to regenerate ml_normalized_observations.parquet, ml_normalized_rt.parquet,
and other regime-normalized ML files WITHOUT re-running the full pipeline.

FIX APPLIED: regime_id alignment for test splits.
    KMeans assigns arbitrary cluster labels. Train regime 0 and test regime 0
    may represent DIFFERENT operating conditions. This tool aligns test regime
    IDs to training regime IDs before applying normalization stats.

Usage:
    cd /Users/jasonrudder/prime
    .venv/bin/python tools/rerun_regime_export.py ~/domains/cmapss/FD_004/Train
    .venv/bin/python tools/rerun_regime_export.py ~/domains/cmapss/FD_004/Test
"""

import json
import sys
from pathlib import Path

# Add prime package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

from prime.regime.normalization import normalize_per_regime, apply_regime_stats, align_regime_ids
from prime.ml_export.regime_export import run_regime_ml_export


def rerun_regime_export(domain_path: Path) -> list[Path]:
    """
    Re-run Steps 5c/5d from pipeline.py using existing data.

    Args:
        domain_path: The Train/ or Test/ directory (e.g., ~/domains/cmapss/FD_004/Train)

    Returns:
        List of Paths written.
    """
    domain_path = Path(domain_path).expanduser().resolve()
    output_dir  = domain_path / "output_time"

    observations_path = output_dir / "observations.parquet"
    regime_path       = output_dir / "regime_detection.parquet"

    if not observations_path.exists():
        raise FileNotFoundError(f"Missing: {observations_path}")
    if not regime_path.exists():
        raise FileNotFoundError(
            f"Missing: {regime_path}\n"
            f"Run 'prime {domain_path}' first to generate regime_detection.parquet."
        )

    split_name    = domain_path.name  # "Train", "Test", etc.
    is_test_split = split_name.lower() == "test"

    print(f"Domain:    {domain_path}")
    print(f"Split:     {split_name}  (is_test={is_test_split})")
    print(f"Output:    {output_dir / 'ml'}/")
    print()

    obs       = pl.read_parquet(observations_path)
    regimes   = pl.read_parquet(regime_path)
    n_regimes = regimes["regime_id"].n_unique()
    print(f"Loaded observations: {obs.shape[0]:,} rows, {obs['signal_id'].n_unique()} signals")
    print(f"Loaded regimes:      {regimes.shape[0]:,} rows, {n_regimes} regime(s)")

    # Check for sibling train-split regime stats (prevents test leakage)
    train_stats_path = None
    if is_test_split:
        for candidate in ("Train", "train"):
            p = domain_path.parent / candidate / "output_time" / "ml" / "ml_regime_stats.json"
            if p.exists():
                train_stats_path = p
                break

    if train_stats_path is not None:
        print(f"Using training regime stats from: {train_stats_path}")
        with open(train_stats_path) as f:
            train_regime_stats = json.load(f)

        # CRITICAL: Align test regime IDs to train regime IDs.
        # KMeans assigns arbitrary labels — test regime 0 may not equal train regime 0.
        print("  Aligning test regime IDs to training regime IDs...")
        regimes_aligned = align_regime_ids(regimes, obs, train_regime_stats)
        print(f"  Alignment complete.")

        normalized_obs = apply_regime_stats(obs, regimes_aligned, train_regime_stats)
        regime_stats   = train_regime_stats

        # Use aligned regime info for the ml_regime_info.parquet export
        regimes_for_export = regimes_aligned

    else:
        print("Fitting regime stats from this split (train mode)...")
        normalized_obs, regime_stats = normalize_per_regime(obs, regimes)
        regimes_for_export = regimes

    print()
    written = run_regime_ml_export(
        output_dir=output_dir,
        regimes=regimes_for_export,
        regime_stats=regime_stats,
        normalized_obs=normalized_obs,
    )

    print()
    print(f"Done. {len(written)} file(s) written to {output_dir / 'ml'}/")
    for p in written:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name}  ({size_kb:.0f} KB)")

    return written


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/rerun_regime_export.py <domain_path>")
        print("Example: python tools/rerun_regime_export.py ~/domains/cmapss/FD_004/Train")
        sys.exit(1)

    written = rerun_regime_export(sys.argv[1])
    print(f"\nRe-run complete: {len(written)} files.")
