"""
Manifold integration — direct Python import.

Prime calls manifold.run() with observations + manifest.
Manifold computes, writes parquets to output_dir.
One import, one call. Prime doesn't know about stages, workers, or internals.
"""

import os
from pathlib import Path
from typing import Union, Optional

# Default output directory (configurable via env var)
OUTPUT_DIR = Path(os.environ.get("PRIME_OUTPUT_DIR", "data"))


def run_manifold(
    observations_path: Union[str, Path],
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
    verbose: bool = True,
) -> dict:
    """
    Call Manifold compute engine.

    Args:
        observations_path: Path to observations.parquet
        manifest_path: Path to manifest.yaml
        output_dir: Directory for Manifold to write output parquets
        verbose: Print progress

    Returns:
        Dict with output metadata (cohorts, files, elapsed, etc.)
    """
    from manifold import run as _manifold_run

    return _manifold_run(
        observations_path=str(observations_path),
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        verbose=verbose,
    )


def manifold_available() -> bool:
    """Check if manifold package is importable."""
    try:
        import manifold
        return True
    except ImportError:
        return False


def manifold_status() -> dict:
    """Get manifold availability status."""
    try:
        import manifold
        return {
            "available": True,
            "version": getattr(manifold, "__version__", "unknown"),
            "message": "Manifold library available",
        }
    except ImportError:
        return {
            "available": False,
            "message": "Manifold not installed. pip install manifold",
        }
