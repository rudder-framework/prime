"""
Results Validator
=================

Validate PRISM output before displaying:
- Check parquet files are valid (not corrupted)
- Check files are not empty
- Check expected columns exist
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class FileStatus:
    """Status of a single parquet file"""
    filename: str
    status: str  # 'ok', 'corrupted', 'empty', 'missing'
    row_count: int
    column_count: int
    columns: List[str]
    error: Optional[str]

    def to_dict(self):
        return asdict(self)


@dataclass
class ValidationResult:
    """Result of validating PRISM output"""

    valid: bool
    errors: List[str]
    warnings: List[str]

    # File status
    files: List[FileStatus]
    files_ok: List[str]
    files_corrupted: List[str]
    files_empty: List[str]

    # Summary
    summary: str

    def to_dict(self):
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "files": [f.to_dict() for f in self.files],
            "files_ok": self.files_ok,
            "files_corrupted": self.files_corrupted,
            "files_empty": self.files_empty,
            "summary": self.summary,
        }


def read_parquet_safe(filepath: Path) -> tuple:
    """
    Safely read a parquet file, detecting corruption.

    Returns: (df, error_message)
    """
    if HAS_POLARS:
        try:
            df = pl.read_parquet(filepath)
            return df, None
        except Exception as e:
            return None, str(e)
    elif HAS_PANDAS:
        try:
            df = pd.read_parquet(filepath)
            return df, None
        except Exception as e:
            return None, str(e)
    else:
        return None, "No parquet reader available (install polars or pandas)"


def validate_parquet(filepath: Path) -> FileStatus:
    """Validate a single parquet file."""

    if not filepath.exists():
        return FileStatus(
            filename=filepath.name,
            status='missing',
            row_count=0,
            column_count=0,
            columns=[],
            error='File not found',
        )

    # Check file size
    if filepath.stat().st_size == 0:
        return FileStatus(
            filename=filepath.name,
            status='empty',
            row_count=0,
            column_count=0,
            columns=[],
            error='File is empty (0 bytes)',
        )

    # Try to read
    df, error = read_parquet_safe(filepath)

    if error:
        return FileStatus(
            filename=filepath.name,
            status='corrupted',
            row_count=0,
            column_count=0,
            columns=[],
            error=error,
        )

    # Get info
    if HAS_POLARS and isinstance(df, pl.DataFrame):
        row_count = len(df)
        columns = df.columns
    else:
        row_count = len(df)
        columns = list(df.columns)

    # Check if effectively empty (no data rows or only ID columns)
    if row_count == 0:
        return FileStatus(
            filename=filepath.name,
            status='empty',
            row_count=0,
            column_count=len(columns),
            columns=columns,
            error='No data rows',
        )

    # Check for physics.parquet with only entity_id
    if filepath.name == 'physics.parquet':
        non_id_cols = [c for c in columns if c not in ['entity_id', 'signal_id', 'window_idx']]
        if len(non_id_cols) == 0:
            return FileStatus(
                filename=filepath.name,
                status='empty',
                row_count=row_count,
                column_count=len(columns),
                columns=columns,
                error='No computed physics values (only ID columns)',
            )

    return FileStatus(
        filename=filepath.name,
        status='ok',
        row_count=row_count,
        column_count=len(columns),
        columns=columns,
        error=None,
    )


def validate_results(results_path: str, expected_files: List[str] = None) -> ValidationResult:
    """
    Validate PRISM output before displaying.

    Args:
        results_path: Path to results directory
        expected_files: List of expected parquet filenames (optional)

    Returns:
        ValidationResult
    """
    results_path = Path(results_path)
    errors = []
    warnings = []
    files = []
    files_ok = []
    files_corrupted = []
    files_empty = []

    # Find all parquet files
    if results_path.is_dir():
        parquet_files = list(results_path.glob("*.parquet"))
    elif results_path.is_file() and results_path.suffix == '.parquet':
        parquet_files = [results_path]
    else:
        errors.append(f"Results path not found: {results_path}")
        return ValidationResult(
            valid=False,
            errors=errors,
            warnings=warnings,
            files=[],
            files_ok=[],
            files_corrupted=[],
            files_empty=[],
            summary="Results path not found",
        )

    if not parquet_files:
        errors.append("No parquet files found")
        return ValidationResult(
            valid=False,
            errors=errors,
            warnings=warnings,
            files=[],
            files_ok=[],
            files_corrupted=[],
            files_empty=[],
            summary="No parquet files found",
        )

    # Validate each file
    for filepath in parquet_files:
        status = validate_parquet(filepath)
        files.append(status)

        if status.status == 'ok':
            files_ok.append(status.filename)
        elif status.status == 'corrupted':
            files_corrupted.append(status.filename)
            errors.append(f"Corrupted: {status.filename} - {status.error}")
        elif status.status == 'empty':
            files_empty.append(status.filename)
            warnings.append(f"Empty: {status.filename} - {status.error}")

    # Check expected files
    if expected_files:
        found_names = {f.filename for f in files}
        for expected in expected_files:
            if expected not in found_names:
                warnings.append(f"Expected file missing: {expected}")

    # Overall validity
    valid = len(files_corrupted) == 0 and len(files_ok) > 0

    # Summary
    summary = f"OK: {len(files_ok)}, Corrupted: {len(files_corrupted)}, Empty: {len(files_empty)}"

    return ValidationResult(
        valid=valid,
        errors=errors,
        warnings=warnings,
        files=files,
        files_ok=files_ok,
        files_corrupted=files_corrupted,
        files_empty=files_empty,
        summary=summary,
    )
