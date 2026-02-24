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


def compare_checksums(path_a: str, path_b: str) -> dict:
    """
    Compare two checksum manifests. Returns diff report.

    Usage:
        diff = compare_checksums(
            "domains/rossler/run_001/checksums.json",
            "domains/rossler/run_002/checksums.json"
        )
    """
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    files_a = set(a["files"].keys())
    files_b = set(b["files"].keys())

    added = files_b - files_a
    removed = files_a - files_b
    common = files_a & files_b

    changed = []
    unchanged = []
    for fname in sorted(common):
        if a["files"][fname]["sha256"] != b["files"][fname]["sha256"]:
            changed.append({
                "file": fname,
                "size_a": a["files"][fname]["size_bytes"],
                "size_b": b["files"][fname]["size_bytes"],
            })
        else:
            unchanged.append(fname)

    identical = (len(added) == 0 and len(removed) == 0 and len(changed) == 0)

    return {
        "identical": identical,
        "pipeline_checksum_a": a["pipeline_checksum"],
        "pipeline_checksum_b": b["pipeline_checksum"],
        "n_files_a": a["n_files"],
        "n_files_b": b["n_files"],
        "added": sorted(added),
        "removed": sorted(removed),
        "changed": changed,
        "unchanged_count": len(unchanged),
    }


def print_comparison(diff: dict) -> None:
    """Pretty-print a checksum comparison."""
    if diff["identical"]:
        print("IDENTICAL -- all files match")
        print(f"   Pipeline checksum: {diff['pipeline_checksum_a'][:16]}...")
        print(f"   Files: {diff['n_files_a']}")
        return

    print("DIFFERENCES FOUND")
    print(f"   Run A: {diff['pipeline_checksum_a'][:16]}... ({diff['n_files_a']} files)")
    print(f"   Run B: {diff['pipeline_checksum_b'][:16]}... ({diff['n_files_b']} files)")

    if diff["added"]:
        print(f"\n   Added ({len(diff['added'])}):")
        for f in diff["added"]:
            print(f"     + {f}")

    if diff["removed"]:
        print(f"\n   Removed ({len(diff['removed'])}):")
        for f in diff["removed"]:
            print(f"     - {f}")

    if diff["changed"]:
        print(f"\n   Changed ({len(diff['changed'])}):")
        for c in diff["changed"]:
            size_delta = c["size_b"] - c["size_a"]
            sign = "+" if size_delta >= 0 else ""
            print(f"     ~ {c['file']} ({sign}{size_delta} bytes)")

    print(f"\n   Unchanged: {diff['unchanged_count']} files")
