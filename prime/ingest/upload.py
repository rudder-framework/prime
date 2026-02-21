"""
Upload Handler
=====================

Handle file uploads for CSV, Parquet, TSV, Excel, and MATLAB files.
"""

from pathlib import Path
from typing import Union, BinaryIO, Optional, List, Dict, Any
import pandas as pd
import numpy as np


SUPPORTED_FORMATS = {
    '.csv': 'csv',
    '.tsv': 'tsv',
    '.txt': 'tsv',
    '.parquet': 'parquet',
    '.pq': 'parquet',
    '.xlsx': 'excel',
    '.xls': 'excel',
    '.mat': 'matlab',
}


def detect_format(filename: str) -> str:
    """
    Detect file format from filename.

    Returns:
        'csv', 'tsv', 'parquet', or 'excel'

    Raises:
        ValueError: If format not supported
    """
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {suffix}. "
            f"Supported: {', '.join(SUPPORTED_FORMATS.keys())}"
        )

    return SUPPORTED_FORMATS[suffix]


def load_file(
    source: Union[str, Path, BinaryIO],
    filename: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load data from file or file-like object.

    Args:
        source: File path or file-like object (from Streamlit upload)
        filename: Original filename (needed if source is file-like)
        **kwargs: Passed to pandas read function

    Returns:
        Loaded DataFrame
    """
    # Determine filename
    if isinstance(source, (str, Path)):
        filename = str(source)
    elif filename is None:
        raise ValueError("filename required when source is file-like object")

    # Detect format
    fmt = detect_format(filename)

    # Load based on format
    if fmt == 'parquet':
        df = pd.read_parquet(source, **kwargs)
    elif fmt == 'tsv':
        df = pd.read_csv(source, sep='\t', **kwargs)
    elif fmt == 'excel':
        # Excel files - uses openpyxl for .xlsx, xlrd for .xls
        df = pd.read_excel(source, **kwargs)
    elif fmt == 'matlab':
        df = load_matlab_file(source, **kwargs)
    else:  # csv
        # Support comment lines starting with #
        csv_kwargs = {'comment': '#'}
        csv_kwargs.update(kwargs)
        df = pd.read_csv(source, **csv_kwargs)

    return df


def preview_file(
    source: Union[str, Path, BinaryIO],
    filename: Optional[str] = None,
    n_rows: int = 10
) -> pd.DataFrame:
    """
    Load just a preview of the file (first n rows).

    More efficient for large files.
    """
    if isinstance(source, (str, Path)):
        filename = str(source)
    elif filename is None:
        raise ValueError("filename required when source is file-like object")

    fmt = detect_format(filename)

    if fmt == 'parquet':
        # Parquet doesn't support nrows directly
        df = pd.read_parquet(source)
        return df.head(n_rows)
    elif fmt == 'tsv':
        return pd.read_csv(source, sep='\t', nrows=n_rows)
    elif fmt == 'excel':
        return pd.read_excel(source, nrows=n_rows)
    else:
        return pd.read_csv(source, comment='#', nrows=n_rows)


def get_file_info(source: Union[str, Path, BinaryIO], filename: Optional[str] = None) -> dict:
    """
    Get file metadata without loading full data.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        return {
            'filename': path.name,
            'size_bytes': path.stat().st_size,
            'size_human': _human_size(path.stat().st_size),
            'format': detect_format(path.name),
        }
    else:
        # File-like object
        source.seek(0, 2)  # Seek to end
        size = source.tell()
        source.seek(0)  # Reset

        return {
            'filename': filename or 'unknown',
            'size_bytes': size,
            'size_human': _human_size(size),
            'format': detect_format(filename) if filename else 'unknown',
        }


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# =============================================================================
# MATLAB FILE SUPPORT
# =============================================================================

def load_matlab_file(
    source: Union[str, Path],
    cohort: Optional[str] = None,
    unit: str = 'g',
    sampling_rate: Optional[float] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load a MATLAB .mat file and convert to canonical schema.

    Handles common time-series formats:
    - Arrays named *_time or *_data are treated as signals
    - RPM values are extracted as metadata

    Args:
        source: Path to .mat file
        cohort: Cohort identifier (defaults to filename without extension)
        unit: Default unit for signals (default: 'g' for vibration)
        sampling_rate: Sampling rate in Hz (for generating time index)

    Returns:
        DataFrame with canonical schema: cohort, signal_id, signal_0, value
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy required for .mat files: pip install scipy")

    path = Path(source)
    data = loadmat(str(path))

    # Default cohort from filename
    if cohort is None:
        cohort = path.stem

    # Find signal arrays (skip metadata keys)
    rows = []
    for key, value in data.items():
        if key.startswith('__'):
            continue

        # Skip RPM and other scalar values
        if not hasattr(value, 'shape') or value.size <= 1:
            continue

        # Extract signal name from key (e.g., X097_DE_time -> DE)
        signal_id = _extract_signal_name(key)
        if signal_id is None:
            continue

        # Flatten array
        arr = value.flatten()
        n = len(arr)

        # Create rows for this signal
        for i, y in enumerate(arr):
            rows.append({
                'cohort': cohort,
                'signal_id': signal_id,
                'signal_0': float(i),  # 0-indexed
                'value': float(y),
                'unit': unit,
            })

    if not rows:
        raise ValueError(f"No signal arrays found in {path.name}")

    return pd.DataFrame(rows)


def _extract_signal_name(key: str) -> Optional[str]:
    """
    Extract signal name from MATLAB variable name.

    Common patterns:
    - X097_DE_time -> DE
    - X105_BA_time -> BA
    - drive_end_accel -> drive_end_accel
    - vibration -> vibration
    """
    # Skip RPM and other metadata
    if 'RPM' in key.upper():
        return None

    # CWRU pattern: X###_SENSOR_time
    if '_time' in key.lower():
        parts = key.replace('_time', '').split('_')
        if len(parts) >= 2:
            # Return sensor ID (DE, FE, BA, etc.)
            return parts[-1]

    # Generic: use full key name
    return key


def load_matlab_directory(
    directory: Union[str, Path],
    unit: str = 'g',
    sampling_rate: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load all .mat files from a directory into a single canonical DataFrame.

    Each file becomes an entity. All signals across files are combined.

    Args:
        directory: Directory containing .mat files
        unit: Default unit for all signals
        sampling_rate: Sampling rate in Hz
        output_path: If provided, save observations.parquet here

    Returns:
        Combined DataFrame with canonical schema
    """
    directory = Path(directory)
    mat_files = sorted(directory.glob('*.mat'))

    if not mat_files:
        raise ValueError(f"No .mat files found in {directory}")

    all_dfs = []
    for mat_file in mat_files:
        df = load_matlab_file(mat_file, unit=unit, sampling_rate=sampling_rate)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(output_path, index=False)

    return combined


def create_observations_parquet(
    source_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    unit: str = 'g',
) -> Path:
    """
    Create observations.parquet from a directory of data files.

    Automatically detects file types and processes accordingly:
    - .mat files: Load as MATLAB time-series
    - .csv/.xlsx: Load and transform to canonical schema
    - .parquet: Pass through if already canonical, transform otherwise

    Args:
        source_dir: Directory containing source files
        output_dir: Output directory (defaults to source_dir)
        unit: Default unit for signals

    Returns:
        Path to created observations.parquet
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir) if output_dir else source_dir
    output_path = output_dir / 'observations.parquet'

    # Check for raw subdirectory
    raw_dir = source_dir / 'raw'
    if raw_dir.exists():
        source_dir = raw_dir

    # Find data files
    mat_files = list(source_dir.glob('*.mat'))
    csv_files = list(source_dir.glob('*.csv'))
    xlsx_files = list(source_dir.glob('*.xlsx'))
    parquet_files = list(source_dir.glob('*.parquet'))

    # Process based on what we find
    if mat_files:
        df = load_matlab_directory(source_dir, unit=unit, output_path=output_path)
    elif csv_files or xlsx_files:
        # Load first file and transform
        src_file = csv_files[0] if csv_files else xlsx_files[0]
        df = load_file(src_file)
        # TODO: Auto-transform to canonical schema
        df.to_parquet(output_path, index=False)
    elif parquet_files:
        # Check if already canonical
        df = pd.read_parquet(parquet_files[0])
        required = {'cohort', 'signal_id', 'signal_0', 'value', 'unit'}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"Parquet not in canonical schema. Has: {df.columns.tolist()}")
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"No supported data files found in {source_dir}")

    return output_path
