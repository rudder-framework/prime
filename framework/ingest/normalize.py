"""
Observation schema normalization.

Handles three schema variants:
1. Standard:  (cohort, signal_id, I, value) — no changes needed
2. Legacy:    (unit_id/entity_id, signal_id, I, value) — rename → cohort
3. Wide-format: no signal_id/value columns — melt sensor columns to long format

Memory: O(1) for renames (lazy scan + sink), O(single_signal) for melt.
"""

from pathlib import Path
from typing import List, Tuple

import polars as pl


# Columns that are never signal values (excluded from melting)
_METADATA_COLUMNS = frozenset({
    'I', 'timestamp', 'time', 't', 'date', 'step', 'index', 'idx',
    'entity_id', 'unit_id', 'cohort', 'unit', 'entity', 'machine', 'asset',
    'signal_id', 'value',
    'file_idx', 'sample_in_file', 'condition', 'is_training',
    'speed_rpm', 'load_n', 'load',
    'source', 'label', 'class', 'target', 'anomaly',
})

# GPS columns — not meaningful as time-series signals for typology
_GPS_COLUMNS = frozenset({
    'gpsLong', 'gpsLat', 'gpsSpeed', 'gpsQuality',
    'latitude', 'longitude', 'gps_speed', 'gps_quality',
})


def normalize_observations(
    observations_path: Path,
    verbose: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Normalize observations.parquet to canonical schema in-place.

    Memory: O(1) for renames (lazy scan + sink), O(single_signal) for melt.

    Handles three schema variants:
    1. Standard:  (cohort, signal_id, I, value) — no changes needed
    2. Legacy:    (unit_id, signal_id, I, value) — rename unit_id → cohort
    3. Wide-format: no signal_id/value columns — melt sensor columns to long format

    Returns:
        (changed, repairs) — whether the file was rewritten and what was done
    """
    # Lazy scan — reads only metadata, not data
    lazy = pl.scan_parquet(observations_path)
    cols = set(lazy.collect_schema().names())
    dtypes = lazy.collect_schema()
    repairs = []

    # ------------------------------------------------------------------
    # Case 1: Already has signal_id + value → standard or legacy schema
    # Lazy rename + sink — constant memory even for huge files.
    # ------------------------------------------------------------------
    if 'signal_id' in cols and 'value' in cols:
        rename_col = None
        if 'cohort' not in cols:
            for alias in ('unit_id', 'entity_id'):
                if alias in cols:
                    rename_col = alias
                    repairs.append(f"Renamed '{alias}' → 'cohort'")
                    break

        if repairs:
            import tempfile
            tmp = Path(tempfile.mktemp(suffix='.parquet', dir=observations_path.parent))
            lazy.rename({rename_col: 'cohort'}).sink_parquet(str(tmp))
            tmp.rename(observations_path)
            if verbose:
                for r in repairs:
                    print(f"    {r}")
        return bool(repairs), repairs

    # ------------------------------------------------------------------
    # Case 2: Wide-format — no signal_id column, sensor data in columns
    # Streams one column at a time to avoid OOM on large datasets.
    # ------------------------------------------------------------------
    if 'signal_id' not in cols:
        import tempfile, shutil

        _NUMERIC_TYPES = (pl.Float64, pl.Float32, pl.Int64, pl.Int32,
                          pl.UInt32, pl.UInt64, pl.Int16, pl.UInt16,
                          pl.Int8, pl.UInt8)

        # Identify the cohort column
        cohort_col = None
        for alias in ('cohort', 'unit_id', 'entity_id', 'unit', 'entity'):
            if alias in cols:
                cohort_col = alias
                break

        # Identify value columns to melt (numeric, not metadata/GPS)
        exclude = _METADATA_COLUMNS | _GPS_COLUMNS
        if cohort_col:
            exclude = exclude | {cohort_col}

        value_cols = [
            c for c in dtypes.names()
            if c not in exclude
            and dtypes[c] in _NUMERIC_TYPES
        ]

        if not value_cols:
            if verbose:
                print(f"    SKIP: No numeric value columns found to melt")
            return False, ["No numeric value columns found"]

        # Figure out renames needed
        i_alias = None
        if 'I' not in cols:
            for alias in ('timestamp', 'time', 't', 'step', 'index', 'idx'):
                if alias in cols:
                    i_alias = alias
                    repairs.append(f"Renamed '{alias}' → 'I'")
                    break

        cohort_rename = None
        if cohort_col and cohort_col != 'cohort':
            cohort_rename = cohort_col
            repairs.append(f"Renamed '{cohort_col}' → 'cohort'")

        domain_name = observations_path.parent.name

        # Stream melt: one value column at a time → temp parquet chunks
        temp_dir = Path(tempfile.mkdtemp())
        total_rows = 0

        try:
            for col_idx, vcol in enumerate(value_cols):
                # Lazy scan — only reads the columns we need
                select_cols = [vcol]
                if i_alias:
                    select_cols.append(i_alias)
                elif 'I' in cols:
                    select_cols.append('I')
                if cohort_col:
                    select_cols.append(cohort_col)

                chunk = pl.read_parquet(observations_path, columns=select_cols)

                # Rename columns
                if i_alias:
                    chunk = chunk.rename({i_alias: 'I'})
                if cohort_rename:
                    chunk = chunk.rename({cohort_rename: 'cohort'})

                # Create I if it didn't exist at all
                if 'I' not in chunk.columns:
                    chunk = chunk.with_row_index('I')

                # Build canonical long-format for this signal
                chunk = chunk.rename({vcol: 'value'})
                chunk = chunk.with_columns([
                    pl.lit(vcol).alias('signal_id'),
                    pl.col('value').cast(pl.Float64),
                    pl.col('I').cast(pl.UInt32),
                ])

                if 'cohort' not in chunk.columns:
                    chunk = chunk.with_columns(pl.lit(domain_name).alias('cohort'))

                chunk = chunk.select(['cohort', 'signal_id', 'I', 'value'])
                chunk.write_parquet(str(temp_dir / f"signal_{col_idx:04d}.parquet"))
                total_rows += len(chunk)

                if verbose:
                    print(f"    [{col_idx+1}/{len(value_cols)}] {vcol}: {len(chunk):,} rows")

                del chunk

            # Lazy concat all chunks → sink to final parquet (constant memory)
            chunks = sorted(temp_dir.glob("*.parquet"))
            combined = pl.concat([pl.scan_parquet(f) for f in chunks])
            combined.sink_parquet(str(observations_path))

            if i_alias is None and 'I' not in cols:
                repairs.append("Created I from row order")
            if cohort_col is None:
                repairs.append(f"Added cohort='{domain_name}'")

            repairs.append(f"Melted {len(value_cols)} columns → {len(value_cols)} signals ({total_rows:,} rows)")

            if verbose:
                for r in repairs:
                    print(f"    {r}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return True, repairs

    # ------------------------------------------------------------------
    # Case 3: Has signal_id but no value column — try to find it
    # Lazy rename + sink — constant memory.
    # ------------------------------------------------------------------
    for alias in ('y', 'measurement', 'reading', 'val'):
        if alias in cols:
            import tempfile
            repairs.append(f"Renamed '{alias}' → 'value'")
            tmp = Path(tempfile.mktemp(suffix='.parquet', dir=observations_path.parent))
            lazy.rename({alias: 'value'}).sink_parquet(str(tmp))
            tmp.rename(observations_path)
            if verbose:
                for r in repairs:
                    print(f"    {r}")
            return True, repairs

    return False, ["Cannot determine value column"]
