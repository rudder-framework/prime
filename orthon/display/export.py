"""
ORTHON Export Functions
=======================

Export reports and data to various formats.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd


def to_json(
    report: Dict[str, Any],
    path: Optional[Union[str, Path]] = None,
    indent: int = 2,
) -> str:
    """
    Export report to JSON.

    Args:
        report: Report dictionary
        path: Optional file path to save to
        indent: JSON indentation

    Returns:
        JSON string
    """
    # Make JSON-serializable
    clean_report = _make_serializable(report)

    json_str = json.dumps(clean_report, indent=indent, default=str)

    if path:
        Path(path).write_text(json_str)

    return json_str


def to_csv(
    report: Dict[str, Any],
    path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Export signals summary to CSV.

    Args:
        report: Report dictionary
        path: Optional file path to save to

    Returns:
        CSV string
    """
    from .report import format_signals_table

    df = format_signals_table(report)
    csv_str = df.to_csv(index=False)

    if path:
        Path(path).write_text(csv_str)

    return csv_str


def to_parquet(
    df: pd.DataFrame,
    path: Union[str, Path],
    compression: str = 'snappy',
) -> Path:
    """
    Export DataFrame to Parquet.

    Args:
        df: DataFrame to export
        path: File path
        compression: Compression codec

    Returns:
        Path to saved file
    """
    path = Path(path)
    df.to_parquet(path, compression=compression, index=False)
    return path


def generate_filename(
    base: str = 'orthon',
    ext: str = 'json',
    timestamp: bool = True,
) -> str:
    """
    Generate a filename with optional timestamp.
    """
    if timestamp:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{base}_{ts}.{ext}"
    return f"{base}.{ext}"


def _make_serializable(obj: Any) -> Any:
    """
    Convert object to JSON-serializable form.
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    else:
        return obj


class ReportExporter:
    """
    Helper class for exporting reports in multiple formats.
    """

    def __init__(self, report: Dict[str, Any]):
        self.report = report

    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        return to_json(self.report, path)

    def to_csv(self, path: Optional[Union[str, Path]] = None) -> str:
        return to_csv(self.report, path)

    def to_text(self) -> str:
        from .report import format_report_text
        return format_report_text(self.report)

    def save_all(self, directory: Union[str, Path], base_name: str = 'orthon_report'):
        """Save report in all formats to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON
        json_path = directory / f"{base_name}_{ts}.json"
        self.to_json(json_path)

        # CSV (signals)
        csv_path = directory / f"{base_name}_{ts}_signals.csv"
        self.to_csv(csv_path)

        # Text
        txt_path = directory / f"{base_name}_{ts}.txt"
        txt_path.write_text(self.to_text())

        return {
            'json': json_path,
            'csv': csv_path,
            'txt': txt_path,
        }
