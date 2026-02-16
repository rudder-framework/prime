"""
Prime Baseline Discovery

Two modes for establishing baseline geometry:
1. first_n_percent: Traditional (industrial default) - assumes first N% is healthy
2. stable_discovery: Find most rigid geometry across all time (markets, unknown systems)

Problem:
- Industrial: First 20% is healthy baseline (known)
- Markets: When was it ever "healthy"? (unknown)
- Need mode to DISCOVER stable geometry, not assume it
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Any
import polars as pl


class BaselineMode(Enum):
    """Baseline selection strategy."""

    FIRST_N_PERCENT = "first_n_percent"
    """Use first N% of data as baseline. Default for industrial systems."""

    LAST_N_PERCENT = "last_n_percent"
    """Use last N% of data as baseline. For post-maintenance scenarios."""

    STABLE_DISCOVERY = "stable_discovery"
    """Discover most stable geometry across all time. For unknown healthy states."""

    REFERENCE_PERIOD = "reference_period"
    """Use specific time period as baseline. When healthy period is known."""

    ROLLING = "rolling"
    """Rolling baseline. Adapts to gradual drift."""


@dataclass
class BaselineResult:
    """Result of baseline calculation."""

    geometry: dict[str, float]
    """Baseline geometry values (averaged)."""

    source: str
    """How baseline was determined."""

    mode: BaselineMode
    """Mode used for baseline selection."""

    n_windows: int
    """Number of windows used in baseline."""

    confidence: Optional[dict[str, float]] = None
    """Standard deviation of baseline metrics (how consistent?)."""

    metadata: Optional[dict[str, Any]] = None
    """Additional metadata about baseline calculation."""


def discover_stable_baseline(
    geometry: pl.DataFrame,
    stability_metric: str = "spectral_entropy",
    top_n: int = 100,
) -> BaselineResult:
    """
    Find the most stable geometric structure in the data.

    Instead of: "Here's the baseline, find deviations"
    Do:         "Find the MOST STABLE geometry across all time"
    Then:       "Measure deviations from that ideal"

    Args:
        geometry: DataFrame with geometric metrics per window
        stability_metric: Which metric defines "stable"?
            - spectral_entropy: Lower = more structured = more stable
            - coherence: Higher = more stable
            - lyapunov: Lower = more stable (less chaotic)
            - determinism: Higher = more stable (more predictable)
        top_n: Number of most stable windows to average

    Returns:
        BaselineResult with discovered baseline geometry
    """

    if len(geometry) == 0:
        raise ValueError("Empty geometry DataFrame")

    # Ensure we don't request more windows than exist
    top_n = min(top_n, len(geometry))

    # Score each window for stability
    # Convention: higher score = more stable
    if stability_metric == "spectral_entropy":
        if "spectral_entropy" not in geometry.columns:
            raise ValueError(f"Column 'spectral_entropy' not found in geometry")
        # Lower entropy = more structured = more stable
        scores = -geometry["spectral_entropy"]

    elif stability_metric == "coherence":
        # Try different coherence column names
        coherence_col = None
        for col in ["coherence_mean", "coherence", "coherence_avg"]:
            if col in geometry.columns:
                coherence_col = col
                break
        if coherence_col is None:
            raise ValueError("No coherence column found in geometry")
        # Higher coherence = more stable
        scores = geometry[coherence_col]

    elif stability_metric == "lyapunov":
        # Try different lyapunov column names
        lyap_col = None
        for col in ["lyapunov_max", "lyapunov", "lyapunov_exponent"]:
            if col in geometry.columns:
                lyap_col = col
                break
        if lyap_col is None:
            raise ValueError("No lyapunov column found in geometry")
        # Lower Lyapunov = more stable (less chaotic)
        scores = -geometry[lyap_col]

    elif stability_metric == "determinism":
        if "determinism" not in geometry.columns:
            raise ValueError(f"Column 'determinism' not found in geometry")
        # Higher determinism = more stable
        scores = geometry["determinism"]

    elif stability_metric == "recurrence_rate":
        if "recurrence_rate" not in geometry.columns:
            raise ValueError(f"Column 'recurrence_rate' not found in geometry")
        # Higher recurrence = more stable
        scores = geometry["recurrence_rate"]

    else:
        # Try to use the metric directly
        if stability_metric not in geometry.columns:
            raise ValueError(f"Unknown stability metric: {stability_metric}")
        scores = geometry[stability_metric]

    # Handle NaN scores
    scores = scores.fill_nan(float("-inf"))

    # Add scores to dataframe and get top N indices
    geometry_scored = geometry.with_columns(
        pl.Series("_stability_score", scores)
    )

    stable_windows = geometry_scored.sort("_stability_score", descending=True).head(top_n)

    # Get numeric columns for averaging (exclude index columns and score)
    exclude_cols = {"entity_id", "signal_id", "window_idx", "signal_0", "timestamp", "_stability_score"}
    numeric_cols = [
        col for col in stable_windows.columns
        if col not in exclude_cols
        and stable_windows[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
    ]

    # Calculate baseline geometry (mean of stable windows)
    baseline_values = {}
    confidence_values = {}

    for col in numeric_cols:
        values = stable_windows[col].drop_nulls().drop_nans()
        if len(values) > 0:
            baseline_values[col] = float(values.mean())
            confidence_values[col] = float(values.std()) if len(values) > 1 else 0.0

    return BaselineResult(
        geometry=baseline_values,
        source="discovered",
        mode=BaselineMode.STABLE_DISCOVERY,
        n_windows=top_n,
        confidence=confidence_values,
        metadata={
            "stability_metric": stability_metric,
            "actual_windows_used": len(stable_windows),
            "total_windows": len(geometry),
        },
    )


def get_baseline(
    geometry: pl.DataFrame,
    mode: BaselineMode | str,
    **kwargs,
) -> BaselineResult:
    """
    Get baseline using specified mode.

    Args:
        geometry: DataFrame with geometric metrics per window
        mode: BaselineMode or string mode name
        **kwargs: Mode-specific parameters:
            - percent: For first_n_percent/last_n_percent (default: 20)
            - stability_metric: For stable_discovery (default: spectral_entropy)
            - top_n_windows: For stable_discovery (default: 100)
            - start_idx, end_idx: For reference_period
            - window_size: For rolling (default: 100)

    Returns:
        BaselineResult with baseline geometry
    """

    if isinstance(mode, str):
        mode = BaselineMode(mode)

    if len(geometry) == 0:
        raise ValueError("Empty geometry DataFrame")

    # Get numeric columns for averaging
    exclude_cols = {"entity_id", "signal_id", "window_idx", "signal_0", "timestamp"}
    numeric_cols = [
        col for col in geometry.columns
        if col not in exclude_cols
        and geometry[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
    ]

    if mode == BaselineMode.FIRST_N_PERCENT:
        percent = kwargs.get("percent", 20)
        n_windows = max(1, int(len(geometry) * percent / 100))
        baseline_data = geometry.head(n_windows)

        baseline_values = {}
        confidence_values = {}
        for col in numeric_cols:
            values = baseline_data[col].drop_nulls().drop_nans()
            if len(values) > 0:
                baseline_values[col] = float(values.mean())
                confidence_values[col] = float(values.std()) if len(values) > 1 else 0.0

        return BaselineResult(
            geometry=baseline_values,
            source="first_n_percent",
            mode=mode,
            n_windows=n_windows,
            confidence=confidence_values,
            metadata={"percent": percent},
        )

    elif mode == BaselineMode.LAST_N_PERCENT:
        percent = kwargs.get("percent", 20)
        n_windows = max(1, int(len(geometry) * percent / 100))
        baseline_data = geometry.tail(n_windows)

        baseline_values = {}
        confidence_values = {}
        for col in numeric_cols:
            values = baseline_data[col].drop_nulls().drop_nans()
            if len(values) > 0:
                baseline_values[col] = float(values.mean())
                confidence_values[col] = float(values.std()) if len(values) > 1 else 0.0

        return BaselineResult(
            geometry=baseline_values,
            source="last_n_percent",
            mode=mode,
            n_windows=n_windows,
            confidence=confidence_values,
            metadata={"percent": percent},
        )

    elif mode == BaselineMode.STABLE_DISCOVERY:
        return discover_stable_baseline(
            geometry,
            stability_metric=kwargs.get("stability_metric", "spectral_entropy"),
            top_n=kwargs.get("top_n_windows", 100),
        )

    elif mode == BaselineMode.REFERENCE_PERIOD:
        start_idx = kwargs.get("start_idx", 0)
        end_idx = kwargs.get("end_idx", len(geometry))

        if "window_idx" in geometry.columns:
            baseline_data = geometry.filter(
                (pl.col("window_idx") >= start_idx) & (pl.col("window_idx") < end_idx)
            )
        else:
            baseline_data = geometry.slice(start_idx, end_idx - start_idx)

        baseline_values = {}
        confidence_values = {}
        for col in numeric_cols:
            values = baseline_data[col].drop_nulls().drop_nans()
            if len(values) > 0:
                baseline_values[col] = float(values.mean())
                confidence_values[col] = float(values.std()) if len(values) > 1 else 0.0

        return BaselineResult(
            geometry=baseline_values,
            source="reference_period",
            mode=mode,
            n_windows=len(baseline_data),
            confidence=confidence_values,
            metadata={"start_idx": start_idx, "end_idx": end_idx},
        )

    elif mode == BaselineMode.ROLLING:
        window_size = kwargs.get("window_size", 100)
        # For rolling, return the most recent window as "current baseline"
        n_windows = min(window_size, len(geometry))
        baseline_data = geometry.tail(n_windows)

        baseline_values = {}
        confidence_values = {}
        for col in numeric_cols:
            values = baseline_data[col].drop_nulls().drop_nans()
            if len(values) > 0:
                baseline_values[col] = float(values.mean())
                confidence_values[col] = float(values.std()) if len(values) > 1 else 0.0

        return BaselineResult(
            geometry=baseline_values,
            source="rolling",
            mode=mode,
            n_windows=n_windows,
            confidence=confidence_values,
            metadata={"window_size": window_size},
        )

    else:
        raise ValueError(f"Unknown baseline mode: {mode}")


def compute_deviation(
    current_geometry: dict[str, float],
    baseline: BaselineResult,
) -> dict[str, float]:
    """
    Compute deviation of current geometry from baseline.

    Args:
        current_geometry: Current window's geometry values
        baseline: Baseline result from get_baseline()

    Returns:
        Dict of metric -> deviation ratio (current / baseline - 1)
    """

    deviations = {}

    for metric, current_value in current_geometry.items():
        if metric in baseline.geometry:
            baseline_value = baseline.geometry[metric]

            if abs(baseline_value) > 1e-10:
                deviation = (current_value - baseline_value) / abs(baseline_value)
            else:
                deviation = 0.0 if current_value == baseline_value else float("inf")

            deviations[metric] = deviation

    return deviations
