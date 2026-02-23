"""
Orchestration integration — direct Python import.

Prime calls orchestration.run() with observations + manifest.
Orchestration runs compute stages, writes parquets to output_dir.
One import, one call. Prime doesn't know about stages, workers, or internals.
"""

import os
from pathlib import Path
from typing import Union

from orchestration import run as _orchestration_run

# Default output directory (configurable via env var)
OUTPUT_DIR = Path(os.environ.get("PRIME_OUTPUT_DIR", "data"))


def run_manifold(
    observations_path: Union[str, Path],
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
    verbose: bool = True,
) -> dict:
    """
    Run the compute pipeline via orchestration.

    Args:
        observations_path: Path to observations.parquet
        manifest_path: Path to manifest.yaml
        output_dir: Directory to write output parquets
        verbose: Print progress

    Returns:
        Dict with output metadata (stages, files, elapsed, etc.)
    """
    return _orchestration_run(
        observations_path=str(observations_path),
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        verbose=verbose,
    )


def manifold_available() -> bool:
    """Check if orchestration package is importable."""
    # Orchestration is always available — it's part of the repo.
    # If this import fails, it's a real error that should crash.
    return True


def manifold_status() -> dict:
    """Get orchestration availability status."""
    import orchestration
    return {
        "available": True,
        "version": getattr(orchestration, "__version__", "0.1.0"),
        "message": "Orchestration package available",
    }
