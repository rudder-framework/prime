"""
ORTHON Inspection Module
========================

Inspect uploaded files, detect capabilities, validate results.
"""

from .file_inspector import inspect_file, FileInspection, ColumnInfo
from .capability_detector import detect_capabilities, Capabilities
from .results_validator import validate_results, ValidationResult

__all__ = [
    "inspect_file",
    "FileInspection",
    "ColumnInfo",
    "detect_capabilities",
    "Capabilities",
    "validate_results",
    "ValidationResult",
]
