"""
Output Paths

Fixed output directory. No exceptions.

observations.parquet and manifest.yaml always go to:
/Users/jasonrudder/prism/data/
"""

from pathlib import Path

# Fixed output directory - NO EXCEPTIONS
OUTPUT_DIR = Path("/Users/jasonrudder/prism/data")

# Fixed output paths
OBSERVATIONS_PATH = OUTPUT_DIR / "observations.parquet"
MANIFEST_PATH = OUTPUT_DIR / "manifest.yaml"


def ensure_output_dir() -> Path:
    """Ensure output directory exists and return path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def get_observations_path() -> Path:
    """Get path for observations.parquet."""
    ensure_output_dir()
    return OBSERVATIONS_PATH


def get_manifest_path() -> Path:
    """Get path for manifest.yaml."""
    ensure_output_dir()
    return MANIFEST_PATH


def check_output_empty() -> bool:
    """Check if output directory is empty (courtesy check)."""
    if not OUTPUT_DIR.exists():
        return True

    existing = list(OUTPUT_DIR.glob("*"))
    # Ignore hidden files
    existing = [f for f in existing if not f.name.startswith(".")]
    return len(existing) == 0


def list_output_contents() -> list[str]:
    """List contents of output directory."""
    if not OUTPUT_DIR.exists():
        return []
    return [f.name for f in OUTPUT_DIR.iterdir() if not f.name.startswith(".")]
