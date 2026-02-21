"""
Signal Metadata Writer

Writes signals.parquet alongside observations.parquet at ingest time.

Schema:
    signal_id    String    Signal identifier (matches signal_id in observations)
    unit         String    Unit string ("psi", "°F", "m/s", "rpm", "" for dimensionless)
    description  String    Human-readable description — nullable
    source_name  String    Original column name from raw data before renaming

One row per unique signal_id, plus an optional row for signal_0 (the index axis).
Always written — even if units are unknown (nulls are fine).
Downstream code should never check "does signals.parquet exist?" — it always exists.
"""

import polars as pl
from pathlib import Path
from typing import Optional, Dict


def write_signal_metadata(
    observations: pl.DataFrame,
    output_dir: Path,
    units: Optional[Dict[str, str]] = None,
    descriptions: Optional[Dict[str, str]] = None,
    source_names: Optional[Dict[str, str]] = None,
    signal_0_unit: Optional[str] = None,
    signal_0_description: Optional[str] = None,
) -> Path:
    """
    Write signals.parquet alongside observations.parquet.

    Args:
        observations: DataFrame with signal_id column
        output_dir: Directory to write signals.parquet
        units: Optional mapping signal_id -> unit string ("psi", "rpm", "" for dimensionless)
        descriptions: Optional mapping signal_id -> human-readable description
        source_names: Optional mapping signal_id -> original column name from raw data
        signal_0_unit: Optional unit for the index axis (e.g., "cycles", "s", "steps")
        signal_0_description: Optional description for the index axis

    Returns:
        Path to the written signals.parquet
    """
    signal_ids = observations["signal_id"].unique().sort().to_list()

    rows = []

    # signal_0 row (index axis metadata) — first row if provided
    if signal_0_unit is not None or signal_0_description is not None:
        rows.append({
            "signal_id": "signal_0",
            "unit": signal_0_unit or "",
            "description": signal_0_description,
            "source_name": "signal_0",
        })

    for sid in signal_ids:
        rows.append({
            "signal_id": sid,
            "unit": units.get(sid) if units else None,
            "description": descriptions.get(sid) if descriptions else None,
            "source_name": source_names.get(sid, sid) if source_names else sid,
        })

    df = pl.DataFrame(rows, schema={
        "signal_id": pl.String,
        "unit": pl.String,
        "description": pl.String,
        "source_name": pl.String,
    })

    path = Path(output_dir) / "signals.parquet"
    df.write_parquet(path)
    return path
