"""
AI Manifest Generator

Scans uploaded data folder, figures out the structure,
writes manifest.yaml automatically.

The grad student uploads data. AI does the rest.
"""

import polars as pl
import numpy as np
from pathlib import Path
import yaml
from collections import Counter
from typing import Optional


def generate_manifest(
    data_path: Path,
    output_manifest: Path = None,
    dataset_name: str = None,
) -> dict:
    """
    Scan a data folder and generate manifest.yaml automatically.

    Args:
        data_path: Folder containing raw data files
        output_manifest: Where to write manifest (default: data_path/manifest.yaml)
        dataset_name: Name for the dataset (default: folder name)

    Returns:
        Generated manifest dict
    """

    data_path = Path(data_path)
    output_manifest = output_manifest or data_path / "manifest.yaml"
    dataset_name = dataset_name or data_path.name

    print(f"Scanning {data_path}...")

    # Find all data files
    all_files = list(data_path.rglob("*"))
    data_files = [f for f in all_files if f.is_file() and not f.name.startswith(".")]

    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_path}")

    # Analyze file types
    extensions = Counter(f.suffix.lower() for f in data_files)
    primary_ext = extensions.most_common(1)[0][0]

    print(f"   Found {len(data_files)} files")
    print(f"   Primary type: {primary_ext or 'no extension'} ({extensions[primary_ext]} files)")

    # Build glob pattern
    file_pattern = _build_glob_pattern(data_files, data_path, primary_ext)
    print(f"   Pattern: {file_pattern}")

    # Sample a file to understand structure
    sample_files = [f for f in data_files if f.suffix.lower() == primary_ext][:3]
    columns_info, entity_strategy = _analyze_files(sample_files)

    print(f"   Columns: {columns_info.get('names', columns_info.get('count', 'unknown'))}")
    print(f"   Entity strategy: {entity_strategy}")

    # Build manifest
    manifest = {
        "dataset": {
            "name": dataset_name,
            "description": f"Auto-generated manifest for {dataset_name}",
            "domain": "unknown",  # User should fill this
        },
        "data": {
            "raw_path": str(data_path),
            "output_path": str(data_path / "observations.parquet"),
            "file_pattern": file_pattern,
            "files_per_flush": 50,
        }
    }

    # Add entity strategy
    if entity_strategy["type"] == "from_path":
        manifest["data"]["entity_from_path"] = entity_strategy["value"]
    elif entity_strategy["type"] == "from_column":
        manifest["data"]["entity_column"] = entity_strategy["value"]

    # Add column info
    if columns_info:
        manifest["data"]["columns"] = columns_info

    # Write manifest
    output_manifest.write_text(yaml.dump(manifest, default_flow_style=False, sort_keys=False))

    print(f"\nGenerated: {output_manifest}")
    print(f"\nPlease review and edit if needed:")
    print(f"   - dataset.domain: What kind of system is this?")
    print(f"   - Verify column names are correct")
    print(f"   - Verify entity_id extraction is correct")

    return manifest


def scan_and_report(data_path: Path) -> dict:
    """
    Scan data folder and return analysis without writing manifest.
    Used by AI to show user what was detected before confirming.
    """

    data_path = Path(data_path)

    all_files = list(data_path.rglob("*"))
    data_files = [f for f in all_files if f.is_file() and not f.name.startswith(".")]

    if not data_files:
        return {"error": f"No data files found in {data_path}"}

    # Analyze file types
    extensions = Counter(f.suffix.lower() for f in data_files)
    primary_ext = extensions.most_common(1)[0][0]

    # Build glob pattern
    file_pattern = _build_glob_pattern(data_files, data_path, primary_ext)

    # Sample files
    sample_files = [f for f in data_files if f.suffix.lower() == primary_ext][:3]
    columns_info, entity_strategy = _analyze_files(sample_files)

    # Estimate row count from sample
    sample_rows = 0
    if sample_files:
        try:
            if primary_ext == ".csv":
                df = pl.read_csv(sample_files[0])
                sample_rows = len(df)
            elif primary_ext == ".parquet":
                df = pl.read_parquet(sample_files[0])
                sample_rows = len(df)
        except:
            pass

    # Count entities (unique parent folders if using parent strategy)
    if entity_strategy["value"] == "parent":
        entities = set(f.parent.name for f in data_files if f.suffix.lower() == primary_ext)
        entity_count = len(entities)
    else:
        entity_count = len([f for f in data_files if f.suffix.lower() == primary_ext])

    return {
        "file_count": len(data_files),
        "primary_extension": primary_ext,
        "file_pattern": file_pattern,
        "columns": columns_info,
        "entity_strategy": entity_strategy,
        "estimated_entities": entity_count,
        "sample_rows_per_file": sample_rows,
        "estimated_total_rows": sample_rows * len(data_files) if sample_rows else "unknown",
    }


def _build_glob_pattern(files: list[Path], base_path: Path, ext: str) -> str:
    """Build a glob pattern that matches the file structure."""

    # Get relative paths
    rel_paths = [f.relative_to(base_path) for f in files if f.suffix.lower() == ext]

    if not rel_paths:
        return f"**/*{ext}" if ext else "**/*"

    # Check depth
    depths = [len(p.parts) for p in rel_paths]
    max_depth = max(depths)

    if max_depth == 1:
        # Files directly in folder
        return f"*{ext}" if ext else "*"
    else:
        # Files in subdirectories
        return f"**/*{ext}" if ext else "**/*"


def _analyze_files(sample_files: list[Path]) -> tuple[dict, dict]:
    """Analyze sample files to understand structure."""

    columns_info = {}
    entity_strategy = {"type": "from_path", "value": "filename"}

    if not sample_files:
        return columns_info, entity_strategy

    sample = sample_files[0]
    ext = sample.suffix.lower()

    try:
        # Try to read the file
        if ext == ".csv":
            df = pl.read_csv(sample, n_rows=5)
            columns_info["names"] = df.columns
            columns_info["count"] = len(df.columns)

            # Check for entity column
            for col in ["entity_id", "id", "sample_id", "subject_id", "unit_id", "machine_id", "vehicle_id"]:
                if col in df.columns:
                    entity_strategy = {"type": "from_column", "value": col}
                    break

        elif ext in [".txt", ".dat", ""]:
            # Try numeric
            data = np.loadtxt(sample, max_rows=5)
            if data.ndim == 1:
                columns_info["count"] = 1
            else:
                columns_info["count"] = data.shape[1]
            columns_info["names"] = [f"col_{i}" for i in range(columns_info["count"])]

        elif ext == ".parquet":
            df = pl.scan_parquet(sample).head(1).collect()
            columns_info["names"] = df.columns
            columns_info["count"] = len(df.columns)

    except Exception as e:
        print(f"   Could not analyze {sample.name}: {e}")

    # Determine entity strategy from folder structure
    if len(sample_files) > 1:
        parents = set(f.parent.name for f in sample_files)
        grandparents = set(f.parent.parent.name for f in sample_files if f.parent.parent != sample_files[0].parent.parent.parent)

        if len(parents) > 1:
            entity_strategy = {"type": "from_path", "value": "parent"}
        elif len(grandparents) > 1:
            entity_strategy = {"type": "from_path", "value": "grandparent_parent"}

    return columns_info, entity_strategy


# CLI entrypoint
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python manifest_generator.py /path/to/data/folder")
        sys.exit(1)
    generate_manifest(Path(sys.argv[1]))
