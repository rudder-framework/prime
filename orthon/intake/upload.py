"""
ORTHON Upload Handler
=====================

Handle file uploads for CSV, Parquet, and TSV files.
"""

from pathlib import Path
from typing import Union, BinaryIO, Optional
import pandas as pd


SUPPORTED_FORMATS = {
    '.csv': 'csv',
    '.tsv': 'tsv',
    '.txt': 'tsv',
    '.parquet': 'parquet',
    '.pq': 'parquet',
}


def detect_format(filename: str) -> str:
    """
    Detect file format from filename.

    Returns:
        'csv', 'tsv', or 'parquet'

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
