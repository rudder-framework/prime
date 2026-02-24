"""Pipeline output checksums for reproducibility verification."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def hash_file(filepath: str, algorithm: str = "sha256") -> str:
    """Hash a single file. Reads in chunks to handle large parquets."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def generate_checksums(output_dir: str) -> dict:
    """
    Generate checksums for all pipeline outputs.

    Call this as the LAST step of every pipeline run.

    Returns dict and writes checksums.json to output_dir.
    """
    output_path = Path(output_dir)

    # Collect all output files, sorted for deterministic ordering
    extensions = {".parquet", ".yaml", ".yml", ".md", ".json", ".csv", ".html"}
    files = sorted(
        f for f in output_path.rglob("*")
        if f.is_file()
        and f.suffix in extensions
        and f.name != "checksums.json"  # don't hash ourselves
    )

    file_checksums = {}
    for f in files:
        rel_path = str(f.relative_to(output_path))
        file_checksums[rel_path] = {
            "sha256": hash_file(str(f)),
            "size_bytes": f.stat().st_size,
        }

    # Combined hash: hash of all individual hashes in sorted key order
    combined = hashlib.sha256()
    for key in sorted(file_checksums.keys()):
        combined.update(file_checksums[key]["sha256"].encode())

    manifest = {
        "pipeline_checksum": combined.hexdigest(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_path),
        "n_files": len(file_checksums),
        "total_bytes": sum(v["size_bytes"] for v in file_checksums.values()),
        "files": file_checksums,
    }

    # Write to output directory
    checksum_path = output_path / "checksums.json"
    with open(checksum_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
