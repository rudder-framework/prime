"""
Time enrichment for Manifold outputs.

Manifold outputs use I (window end index) as the time axis. This module
adds real-time columns (t, window_start_t, etc.) to all output parquets
using t = I / sample_rate.

Idempotent: skips files that already have a t column.
"""

import polars as pl
import yaml
from pathlib import Path
from typing import Optional


# Files that are aggregated/metadata with no I column — skip entirely
SKIP_PATTERNS = {
    "cohort_baseline",
    "cohort_thermodynamics",
    "information_flow",
    "info_flow_delta",
    "segment_comparison",
    "ftle",
    "ftle_backward",
    "lyapunov",
}


def _load_signal_windows(manifest_path: Path) -> tuple[int, dict[str, int]]:
    """Extract system.window and per-signal window_size from manifest."""
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    system_window = manifest["system"]["window"]

    signal_windows = {}
    for _cohort_id, signals in manifest.get("cohorts", {}).items():
        for signal_id, config in signals.items():
            signal_windows[signal_id] = config["window_size"]

    return system_window, signal_windows


def _stem(path: Path) -> str:
    """Get the stem without any suffix."""
    return path.stem


def enrich_output_with_time(
    output_dir: str | Path,
    manifest_path: str | Path,
    sample_rate: float,
    verbose: bool = True,
) -> dict:
    """
    Add time columns to all Manifold output parquets.

    For each output parquet:
    - Has I column: add t = I / sample_rate
    - Has window_start/window_end (ftle_rolling): add window_start_t, window_end_t
    - Has first_break_I (break_sequence): add first_break_t
    - Signal-level with signal_id: add window_start_t from per-signal window
    - System-level (no signal_id): add window_start_t from system.window
    - Aggregated/metadata files: skip

    Idempotent: skips files that already have t.

    Returns dict with counts of enriched/skipped files.
    """
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)

    system_window, signal_windows = _load_signal_windows(manifest_path)

    parquets = sorted(output_dir.rglob("*.parquet"))
    enriched = []
    skipped = []

    for pq in parquets:
        stem = _stem(pq)

        # Skip aggregated/metadata files
        if stem in SKIP_PATTERNS:
            skipped.append(str(pq.relative_to(output_dir)))
            if verbose:
                print(f"  skip  {pq.relative_to(output_dir)}")
            continue

        df = pl.read_parquet(pq)

        # Skip if already enriched
        if "t" in df.columns:
            skipped.append(str(pq.relative_to(output_dir)))
            if verbose:
                print(f"  skip  {pq.relative_to(output_dir)} (already has t)")
            continue

        # Skip if no I column
        if "I" not in df.columns:
            skipped.append(str(pq.relative_to(output_dir)))
            if verbose:
                print(f"  skip  {pq.relative_to(output_dir)} (no I column)")
            continue

        # Core enrichment: t = I / sample_rate
        df = df.with_columns(
            (pl.col("I").cast(pl.Float64) / sample_rate).alias("t")
        )

        # ftle_rolling: window_start_t, window_end_t from existing columns
        if "window_start" in df.columns:
            df = df.with_columns(
                (pl.col("window_start").cast(pl.Float64) / sample_rate).alias("window_start_t")
            )
        if "window_end" in df.columns:
            df = df.with_columns(
                (pl.col("window_end").cast(pl.Float64) / sample_rate).alias("window_end_t")
            )

        # break_sequence: first_break_t from first_break_I
        if "first_break_I" in df.columns:
            df = df.with_columns(
                (pl.col("first_break_I").cast(pl.Float64) / sample_rate).alias("first_break_t")
            )

        # window_start_t for windowed outputs (where I = window end index)
        # Only add if not already present (ftle_rolling gets it from window_start above)
        if "window_start_t" not in df.columns:
            if "signal_id" in df.columns and stem in _SIGNAL_LEVEL_FILES:
                # Per-signal window: window_start_t = (I - window_size + 1) / sr
                df = df.with_columns(
                    pl.col("signal_id").replace_strict(
                        signal_windows, default=system_window
                    ).alias("_ws")
                ).with_columns(
                    ((pl.col("I").cast(pl.Float64) - pl.col("_ws") + 1) / sample_rate).alias("window_start_t")
                ).drop("_ws")
            elif stem in _SYSTEM_LEVEL_FILES:
                # System window: window_start_t = (I - system_window + 1) / sr
                df = df.with_columns(
                    ((pl.col("I").cast(pl.Float64) - system_window + 1) / sample_rate).alias("window_start_t")
                )

        df.write_parquet(pq)
        enriched.append(str(pq.relative_to(output_dir)))
        if verbose:
            extra = []
            if "window_start_t" in df.columns:
                extra.append("window_start_t")
            if "window_end_t" in df.columns:
                extra.append("window_end_t")
            if "first_break_t" in df.columns:
                extra.append("first_break_t")
            cols = "+".join(["t"] + extra)
            print(f"  enrich {pq.relative_to(output_dir)}  [{cols}]")

    result = {"enriched": len(enriched), "skipped": len(skipped), "files": enriched}
    if verbose:
        print(f"\nEnriched {len(enriched)} files, skipped {len(skipped)}")
    return result


# Signal-level files that use per-signal windows for I stepping.
# Currently empty: Manifold steps ALL outputs at system.window, not per-signal.
# Per-signal window_size in manifest controls engine-internal computations only.
_SIGNAL_LEVEL_FILES: set[str] = set()

# System-level: I is the window end for system.window
# Includes signal_vector and signal_geometry because Manifold windows them
# at system.window (72) with system.stride (18), not at per-signal windows.
_SYSTEM_LEVEL_FILES = {
    "signal_vector",
    "signal_geometry",
    "state_vector",
    "state_geometry",
    "geometry_dynamics",
    "persistent_homology",
    "cohort_vector",
}

# Note: these files have I but window_start_t is not meaningful:
#   breaks            — I is the break location (sample index), not a window end
#   velocity_field    — I is per-observation, no windowing
#   velocity_field_components — same
#   signal_pairwise   — multi-signal, window_start_t would be ambiguous
#   observation_geometry — per-observation
#   signal_stability  — uses its own agg_window/stride, not system window
#   sensor_eigendecomp — uses its own agg_window/stride
#   state_geometry_loadings — sidecar, t is sufficient
#   state_geometry_feature_loadings — sidecar, t is sufficient
#   ridge_proximity   — indexed by FTLE rolling window, not system window
