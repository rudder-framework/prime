import shutil
from pathlib import Path


def passthrough_copy(source_path: Path, dest_path: Path) -> bool:
    """
    Copy a parquet file from analytical output to ml/ directory.
    Returns True if successful, False if source doesn't exist.
    """
    if not source_path.exists():
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)
    return True
