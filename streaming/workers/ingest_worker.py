"""
Ingest worker for streaming pipeline.

Converts raw files → observations.parquet for a single partition.
Follows the buffer → flush → lazy concat → sink pattern from
prime/ingest/streaming.py.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from streaming.converters.base_converter import BaseConverter, ConversionResult
from streaming.converters.mat_converter import MatConverter


def create_converter(config: Dict) -> BaseConverter:
    """Factory: build a converter from YAML config."""
    conv_cfg = config.get("converter", {})
    conv_type = conv_cfg.get("type", "mat")

    if conv_type == "mat":
        return MatConverter(
            signal_keys=conv_cfg.get("signal_keys"),
            exclude_keys=conv_cfg.get("exclude_keys"),
            cohort_from=conv_cfg.get("cohort_from", "config"),
            cohort_value=conv_cfg.get("cohort_value"),
            sampling_rate=conv_cfg.get("sampling_rate"),
        )
    else:
        raise ValueError(f"Unknown converter type: {conv_type}")


def ingest_partition(
    partition_id: str,
    files: List[Path],
    output_dir: Path,
    converter: BaseConverter,
    files_per_flush: int = 50,
    verbose: bool = True,
) -> Path:
    """
    Ingest raw files into a single partition's observations.parquet.

    Pattern: buffer files → flush to temp chunks → lazy concat → sink.
    Memory: O(files_per_flush * samples_per_file), not O(total_partition).

    Args:
        partition_id: Partition identifier (e.g. '2023-W01')
        files: Raw files belonging to this partition
        output_dir: Partition output directory (will contain observations.parquet)
        converter: File converter instance
        files_per_flush: How many files to buffer before flushing a chunk
        verbose: Print progress

    Returns:
        Path to the written observations.parquet
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = output_dir / "observations.parquet"

    if verbose:
        print(f"  Ingesting partition {partition_id}: {len(files)} files")

    temp_dir = Path(tempfile.mkdtemp(prefix=f"stream_{partition_id}_"))

    try:
        chunk_idx = 0
        buffer_frames: List[pl.DataFrame] = []

        for file_idx, filepath in enumerate(files):
            result = converter.convert_file(filepath)
            if result.df.height > 0:
                buffer_frames.append(result.df)

            # Flush buffer periodically
            if (file_idx + 1) % files_per_flush == 0 and buffer_frames:
                _flush_frames(buffer_frames, temp_dir, chunk_idx)
                buffer_frames = []
                chunk_idx += 1
                if verbose:
                    print(f"    Flushed chunk {chunk_idx} ({file_idx + 1}/{len(files)} files)")

        # Flush remaining
        if buffer_frames:
            _flush_frames(buffer_frames, temp_dir, chunk_idx)
            chunk_idx += 1
            if verbose:
                print(f"    Flushed chunk {chunk_idx} (final, {len(files)}/{len(files)} files)")

        # Lazy concat all chunks → reindex I per (cohort, signal_id) → sink
        chunk_files = sorted(temp_dir.glob("chunk_*.parquet"))
        if not chunk_files:
            if verbose:
                print(f"    WARNING: No data produced for partition {partition_id}")
            # Write empty observations with correct schema
            pl.DataFrame(
                schema={"cohort": pl.Utf8, "signal_id": pl.Utf8, "I": pl.UInt32, "value": pl.Float64}
            ).write_parquet(observations_path)
            return observations_path

        lazy = pl.concat([pl.scan_parquet(f) for f in chunk_files])

        # Reindex I to be 0-indexed sequential per (cohort, signal_id) across the partition
        lazy = lazy.with_columns(
            pl.int_range(pl.len()).over("cohort", "signal_id").cast(pl.UInt32).alias("I")
        )

        lazy.sink_parquet(observations_path)

        if verbose:
            final_df = pl.scan_parquet(observations_path)
            n_rows = final_df.select(pl.len()).collect().item()
            n_signals = final_df.select(pl.col("signal_id").n_unique()).collect().item()
            print(f"    Partition {partition_id}: {n_rows:,} rows, {n_signals} signals → {observations_path}")

        return observations_path

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _flush_frames(frames: List[pl.DataFrame], temp_dir: Path, chunk_idx: int) -> None:
    """Concatenate buffered frames and write a numbered parquet chunk."""
    df = pl.concat(frames)
    chunk_path = temp_dir / f"chunk_{chunk_idx:04d}.parquet"
    df.write_parquet(chunk_path)
