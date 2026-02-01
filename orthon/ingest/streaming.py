"""
ORTHON Universal Streaming Ingestor

One function. Any dataset. Any size. Constant memory.
No code required from user - just a manifest.yaml

Output always goes to: /Users/jasonrudder/prism/data/
"""

import polars as pl
import numpy as np
from pathlib import Path
import yaml
import tempfile
import shutil
from typing import Optional, Callable, Generator

from .paths import get_observations_path, OBSERVATIONS_PATH, OUTPUT_DIR


def ingest_from_manifest(manifest_path: Path) -> pl.LazyFrame:
    """
    Ingest ANY dataset from a simple manifest.

    User provides: manifest.yaml (10 lines)
    ORTHON provides: observations.parquet (any size)

    Output ALWAYS goes to: /Users/jasonrudder/prism/data/observations.parquet
    NO EXCEPTIONS.

    Memory: O(buffer_size), NOT O(total_data)
    """

    manifest = yaml.safe_load(Path(manifest_path).read_text())

    raw_path = Path(manifest["data"]["raw_path"])
    # FIXED OUTPUT PATH - ignore manifest output_path
    output_path = get_observations_path()
    file_pattern = manifest["data"]["file_pattern"]

    print(f"Output: {output_path} (fixed)")

    # Auto-detect reader from file extension
    sample_file = next(raw_path.glob(file_pattern), None)
    if not sample_file:
        raise FileNotFoundError(f"No files matching '{file_pattern}' in {raw_path}")

    reader = _get_reader(sample_file.suffix.lower())

    # Get config from manifest
    entity_from_path = manifest["data"].get("entity_from_path")
    entity_column = manifest["data"].get("entity_column")
    column_names = manifest["data"].get("columns", {}).get("names")
    files_per_flush = manifest["data"].get("files_per_flush", 50)

    # Stream process
    temp_dir = Path(tempfile.mkdtemp())

    try:
        files = sorted(raw_path.glob(file_pattern))
        total_files = len(files)
        print(f"Found {total_files} files")

        buffer = []
        chunk_idx = 0
        global_I = 0

        for file_idx, filepath in enumerate(files):
            try:
                # Read one file
                df = reader(filepath)

                # Handle numpy arrays
                if isinstance(df, np.ndarray):
                    if column_names:
                        df = pl.DataFrame(df, schema=column_names[:df.shape[1]])
                    else:
                        df = pl.DataFrame(df, schema=[f"col_{i}" for i in range(df.shape[1])])

                # Add unit_id (v2.0.0 schema, was entity_id)
                if entity_from_path:
                    unit_id = _extract_entity(filepath, entity_from_path)
                    df = df.with_columns(pl.lit(unit_id).alias("unit_id"))
                elif entity_column and entity_column in df.columns:
                    df = df.rename({entity_column: "unit_id"})
                else:
                    df = df.with_columns(pl.lit(filepath.stem).alias("unit_id"))

                # Add global index
                n_rows = len(df)
                df = df.with_columns(pl.arange(global_I, global_I + n_rows).alias("I"))
                global_I += n_rows

                buffer.append(df)

                # Flush periodically (memory management)
                if len(buffer) >= files_per_flush:
                    _flush(buffer, temp_dir, chunk_idx)
                    buffer = []
                    chunk_idx += 1
                    print(f"   {file_idx + 1}/{total_files} files -> chunk {chunk_idx}")

            except Exception as e:
                print(f"   {filepath.name}: {e}")

        # Final flush
        if buffer:
            _flush(buffer, temp_dir, chunk_idx)
            chunk_idx += 1

        # Lazy concat -> sink (memory efficient)
        print(f"Combining {chunk_idx} chunks...")
        chunks = sorted(temp_dir.glob("*.parquet"))

        if not chunks:
            raise RuntimeError("No data processed. Check file format and manifest.")

        combined = pl.concat([pl.scan_parquet(f) for f in chunks])
        combined.sink_parquet(output_path)

        # Verify
        result = pl.scan_parquet(output_path)
        row_count = result.select(pl.len()).collect().item()
        entity_count = result.select(pl.col("entity_id").n_unique()).collect().item()

        print(f"\nComplete: {output_path}")
        print(f"   Rows: {row_count:,}")
        print(f"   Entities: {entity_count}")
        print(f"   Schema: {result.schema}")

        return result

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def ingest_with_builder(
    raw_path: Path,
    output_path: Path = None,
    file_glob: str = "**/*",
    file_reader: Callable[[Path], np.ndarray | pl.DataFrame] = None,
    row_builder: Callable[[Path, np.ndarray | pl.DataFrame, int], Generator[dict, None, None]] = None,
    files_per_flush: int = 50,
) -> pl.LazyFrame:
    """
    Low-level streaming ingestor with custom row builder.

    For datasets needing custom logic (like IMS with 4 bearings per file).
    For simple datasets, use ingest_from_manifest() instead.

    Output ALWAYS goes to: /Users/jasonrudder/prism/data/observations.parquet
    NO EXCEPTIONS (output_path parameter is ignored).

    Args:
        raw_path: Directory containing raw files
        output_path: IGNORED - always writes to fixed path
        file_glob: Pattern to match files
        file_reader: Function to read one file -> numpy array or DataFrame
        row_builder: Generator yielding dicts for each row
        files_per_flush: How often to flush to disk

    Memory: O(files_per_flush * rows_per_file), NOT O(total_dataset)
    """

    # FIXED OUTPUT PATH - ignore parameter
    output_path = get_observations_path()
    print(f"Output: {output_path} (fixed)")
    temp_dir = Path(tempfile.mkdtemp())

    try:
        files = sorted(Path(raw_path).glob(file_glob))
        total_files = len(files)

        if total_files == 0:
            raise FileNotFoundError(f"No files matching '{file_glob}' in {raw_path}")

        print(f"Found {total_files} files")

        buffer = []
        chunk_idx = 0

        for file_idx, filepath in enumerate(files):
            try:
                data = file_reader(filepath)

                for row in row_builder(filepath, data, file_idx):
                    buffer.append(row)

                if (file_idx + 1) % files_per_flush == 0:
                    _flush_dicts(buffer, temp_dir, chunk_idx)
                    buffer = []
                    chunk_idx += 1
                    print(f"   {file_idx + 1}/{total_files} files ({chunk_idx} chunks)")

            except Exception as e:
                print(f"   Error {filepath}: {e}")

        if buffer:
            _flush_dicts(buffer, temp_dir, chunk_idx)
            chunk_idx += 1
            print(f"   {total_files}/{total_files} files (final)")

        print(f"Combining {chunk_idx} chunks...")
        chunks = sorted(temp_dir.glob("*.parquet"))

        if not chunks:
            raise RuntimeError("No chunks written")

        combined = pl.concat([pl.scan_parquet(f) for f in chunks])
        combined.sink_parquet(output_path)

        result = pl.scan_parquet(output_path)
        row_count = result.select(pl.len()).collect().item()

        print(f"\nComplete: {output_path}")
        print(f"   Rows: {row_count:,}")

        return result

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _get_reader(ext: str):
    """Auto-detect file reader from extension."""
    readers = {
        ".csv": lambda f: pl.read_csv(f),
        ".tsv": lambda f: pl.read_csv(f, separator="\t"),
        ".txt": lambda f: _try_csv_or_numeric(f),
        ".parquet": lambda f: pl.read_parquet(f),
        ".xlsx": lambda f: pl.read_excel(f),
        ".xls": lambda f: pl.read_excel(f),
        ".json": lambda f: pl.read_json(f),
        ".dat": lambda f: _read_numeric(f),
        "": lambda f: _read_numeric(f),
    }
    return readers.get(ext, _try_csv_or_numeric)


def _try_csv_or_numeric(filepath: Path):
    """Try CSV first, fall back to numeric."""
    try:
        return pl.read_csv(filepath)
    except:
        return _read_numeric(filepath)


def _read_numeric(filepath: Path) -> np.ndarray:
    """Read space/tab delimited numeric file."""
    return np.loadtxt(filepath)


def _extract_entity(filepath: Path, pattern: str) -> str:
    """Extract entity ID from filepath."""
    patterns = {
        "parent": lambda f: f.parent.name,
        "grandparent": lambda f: f.parent.parent.name,
        "filename": lambda f: f.stem,
        "parent_filename": lambda f: f"{f.parent.name}_{f.stem}",
        "grandparent_parent": lambda f: f"{f.parent.parent.name}_{f.parent.name}",
    }
    return patterns.get(pattern, lambda f: f.stem)(filepath)


def _flush(buffer: list[pl.DataFrame], temp_dir: Path, idx: int):
    """Flush DataFrame buffer to temp parquet."""
    if buffer:
        combined = pl.concat(buffer)
        combined.write_parquet(temp_dir / f"chunk_{idx:06d}.parquet")


def _flush_dicts(buffer: list[dict], temp_dir: Path, idx: int):
    """Flush dict buffer to temp parquet."""
    if buffer:
        df = pl.DataFrame(buffer)
        df.write_parquet(temp_dir / f"chunk_{idx:06d}.parquet")


# CLI entrypoint
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python streaming.py manifest.yaml")
        sys.exit(1)
    ingest_from_manifest(Path(sys.argv[1]))
