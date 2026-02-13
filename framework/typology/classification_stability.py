"""
Classification Stability Checker

Validates that signal classification is consistent across different
windows of the signal. Classifies on first N rows, verifies against
last N rows. Flags if classification differs.

RUDDER Principle: Classification is a property of the signal, not the sample.
If classification changes between windows, that indicates regime change.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class StabilityStatus(Enum):
    """Classification stability outcomes."""
    STABLE = "stable"
    REGIME_CHANGE = "regime_change"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class ClassificationResult:
    """Result from a single window classification."""
    window_name: str
    start_row: int
    end_row: int
    classification: Dict[str, Any]
    confidence: float


@dataclass
class StabilityResult:
    """Result from stability check across windows."""
    status: StabilityStatus
    primary_classification: ClassificationResult
    verification_classification: ClassificationResult
    match: bool
    differences: Optional[Dict[str, Any]] = None
    message: str = ""


def extract_window(
    data: np.ndarray,
    start: int,
    size: int
) -> np.ndarray:
    """
    Extract a window of rows from the data.
    
    Args:
        data: Full signal array
        start: Starting row index
        size: Number of rows to extract
        
    Returns:
        Windowed subset of data
    """
    end = min(start + size, len(data))
    return data[start:end]


def compare_classifications(
    primary: Dict[str, Any],
    verification: Dict[str, Any],
    tolerance: float = 0.1
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Compare two classification results.
    
    Args:
        primary: Classification from first window
        verification: Classification from verification window
        tolerance: Acceptable difference for numeric values
        
    Returns:
        Tuple of (match: bool, differences: dict or None)
    """
    differences = {}
    
    # Compare classification type (categorical)
    if primary.get("signal_type") != verification.get("signal_type"):
        differences["signal_type"] = {
            "primary": primary.get("signal_type"),
            "verification": verification.get("signal_type")
        }
    
    # Compare dominant components (categorical)
    if primary.get("dominant_component") != verification.get("dominant_component"):
        differences["dominant_component"] = {
            "primary": primary.get("dominant_component"),
            "verification": verification.get("dominant_component")
        }
    
    # Compare numeric properties with tolerance
    numeric_keys = ["trend_strength", "periodicity_strength", "noise_ratio"]
    for key in numeric_keys:
        p_val = primary.get(key, 0)
        v_val = verification.get(key, 0)
        if abs(p_val - v_val) > tolerance:
            differences[key] = {
                "primary": p_val,
                "verification": v_val,
                "delta": abs(p_val - v_val)
            }
    
    match = len(differences) == 0
    return match, differences if differences else None


def check_classification_stability(
    data: np.ndarray,
    classify_fn: callable,
    window_size: int = 10000,
    tolerance: float = 0.1
) -> StabilityResult:
    """
    Check if signal classification is stable between first and last windows.
    
    This is the main entry point. Classifies the first N rows, then
    verifies against the last N rows. Flags if classification differs.
    
    Args:
        data: Full signal array (can be millions of rows)
        classify_fn: Function that takes data array and returns classification dict
        window_size: Number of rows for each window (default 10000)
        tolerance: Acceptable difference for numeric comparisons
        
    Returns:
        StabilityResult with status, classifications, and any differences
    """
    total_rows = len(data)
    
    # Check sufficient data
    if total_rows < window_size * 2:
        return StabilityResult(
            status=StabilityStatus.INSUFFICIENT_DATA,
            primary_classification=None,
            verification_classification=None,
            match=False,
            message=f"Need at least {window_size * 2} rows, got {total_rows}"
        )
    
    # Extract first window
    first_window = extract_window(data, start=0, size=window_size)
    
    # Extract last window
    last_start = total_rows - window_size
    last_window = extract_window(data, start=last_start, size=window_size)
    
    # Classify both windows
    primary_class = classify_fn(first_window)
    verification_class = classify_fn(last_window)
    
    # Build result objects
    primary_result = ClassificationResult(
        window_name="first",
        start_row=0,
        end_row=window_size,
        classification=primary_class,
        confidence=primary_class.get("confidence", 0.0)
    )
    
    verification_result = ClassificationResult(
        window_name="last",
        start_row=last_start,
        end_row=total_rows,
        classification=verification_class,
        confidence=verification_class.get("confidence", 0.0)
    )
    
    # Compare
    match, differences = compare_classifications(
        primary_class,
        verification_class,
        tolerance=tolerance
    )
    
    # Determine status
    if match:
        status = StabilityStatus.STABLE
        message = "Classification stable across signal"
    else:
        status = StabilityStatus.REGIME_CHANGE
        message = f"Classification changed: {list(differences.keys())}"
    
    return StabilityResult(
        status=status,
        primary_classification=primary_result,
        verification_classification=verification_result,
        match=match,
        differences=differences,
        message=message
    )


# ============================================================
# SQL/DuckDB Integration
# ============================================================
# Per RUDDER canonical principle: engines compute, SQL handles logic
# Below are helper functions to produce SQL-friendly outputs

def stability_result_to_record(result: StabilityResult) -> Dict[str, Any]:
    """
    Convert StabilityResult to a flat dict for SQL insertion.
    
    Returns dict ready for DuckDB/SQL insertion.
    """
    return {
        "status": result.status.value,
        "match": result.match,
        "message": result.message,
        "primary_window_start": result.primary_classification.start_row if result.primary_classification else None,
        "primary_window_end": result.primary_classification.end_row if result.primary_classification else None,
        "primary_signal_type": result.primary_classification.classification.get("signal_type") if result.primary_classification else None,
        "primary_confidence": result.primary_classification.confidence if result.primary_classification else None,
        "verification_window_start": result.verification_classification.start_row if result.verification_classification else None,
        "verification_window_end": result.verification_classification.end_row if result.verification_classification else None,
        "verification_signal_type": result.verification_classification.classification.get("signal_type") if result.verification_classification else None,
        "verification_confidence": result.verification_classification.confidence if result.verification_classification else None,
        "has_differences": result.differences is not None,
        "difference_keys": ",".join(result.differences.keys()) if result.differences else None
    }
