"""
Window/Stride Configuration — Auto-detect, validate, recommend

Lives in ORTHON. PRISM just reads window/stride from config.json.

Usage:
    from orthon.shared.window_config import auto_detect_window, validate_window, WindowConfig

    # Auto-detect from data
    window_cfg = auto_detect_window(observations, config)

    # Validate (user override or auto)
    errors = validate_window(window_cfg, observations, config)
    if errors:
        raise ValueError(errors)

    # Add to config before saving
    config.window = window_cfg.to_dict()
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import math


# =============================================================================
# DOMAIN DEFAULTS
# =============================================================================

# Domain-specific window/stride recommendations
# Based on typical data characteristics and analysis needs

DOMAIN_DEFAULTS = {
    "turbomachinery": {
        "window_fraction": 0.1,      # 10% of entity length
        "stride_fraction": 0.5,       # 50% overlap
        "min_window": 50,             # Minimum window size
        "max_window": 500,            # Maximum window size
        "rationale": "Turbomachinery degradation is gradual; moderate windows capture trends without noise",
    },
    "battery": {
        "window_fraction": 0.05,      # 5% of entity length (batteries have long cycle life)
        "stride_fraction": 0.5,
        "min_window": 20,
        "max_window": 200,
        "rationale": "Battery capacity fade is slow; smaller windows detect subtle changes",
    },
    "bearing": {
        "window_fraction": 0.02,      # 2% (bearings fail fast, need fine resolution)
        "stride_fraction": 0.25,      # 75% overlap for bearing fault detection
        "min_window": 100,            # Need enough samples for FFT
        "max_window": 2000,
        "rationale": "Bearing faults develop quickly; high overlap catches rapid changes",
    },
    "fluid": {
        "window_fraction": 0.2,       # 20% (spatial analysis, larger context)
        "stride_fraction": 0.5,
        "min_window": 10,
        "max_window": 1000,
        "rationale": "Spatial fields need larger context for vortex/structure detection",
    },
    "chemical": {
        "window_fraction": 0.1,
        "stride_fraction": 0.5,
        "min_window": 30,
        "max_window": 300,
        "rationale": "Batch processes have distinct phases; moderate windows capture transitions",
    },
}

# General defaults (no domain specified)
GENERAL_DEFAULTS = {
    "window_fraction": 0.1,
    "stride_fraction": 0.5,
    "min_window": 20,
    "max_window": 500,
    "rationale": "General-purpose defaults for unknown data characteristics",
}


# =============================================================================
# COMPUTE LIMITS
# =============================================================================

# Prevent users from killing their machines
COMPUTE_LIMITS = {
    "max_total_windows": 100_000,     # Total windows across all entities
    "max_windows_per_entity": 10_000, # Windows per single entity
    "min_windows_per_entity": 3,      # Need at least 3 for any meaningful analysis
    "min_window_size": 5,             # Absolute minimum (statistics need samples)
    "max_memory_gb": 8,               # Rough estimate of memory limit
}


# =============================================================================
# WINDOW CONFIG DATACLASS
# =============================================================================

@dataclass
class WindowConfig:
    """Window/stride configuration"""

    size: int                     # Window size (number of sequence points)
    stride: int                   # Stride between windows
    min_samples: int = 50         # Minimum samples required per window

    # Metadata (optional, PRISM can ignore)
    auto_detected: bool = True    # False if user specified
    detection_method: str = ""    # How it was determined
    domain_used: Optional[str] = None

    # Computed during validation
    total_windows: int = 0
    windows_per_entity: Dict[str, int] = field(default_factory=dict)
    estimated_memory_mb: float = 0.0

    def to_dict(self) -> dict:
        """For adding to PrismConfig - matches PRISM schema"""
        return {
            "size": self.size,
            "stride": self.stride,
            "min_samples": self.min_samples,
        }


# =============================================================================
# AUTO-DETECTION
# =============================================================================

def auto_detect_window(
    observations,  # polars or pandas DataFrame
    config,        # PrismConfig
) -> WindowConfig:
    """
    Auto-detect optimal window/stride based on data characteristics.

    Strategy:
    1. Get domain defaults (if domain specified)
    2. Calculate entity lengths
    3. Use shortest entity to determine window (so all entities have enough windows)
    4. Apply domain constraints (min/max)
    5. Set stride as fraction of window
    """

    # Get defaults
    domain = getattr(config, 'domain', None)
    if domain and domain in DOMAIN_DEFAULTS:
        defaults = DOMAIN_DEFAULTS[domain]
    else:
        defaults = GENERAL_DEFAULTS
        domain = None

    # Calculate entity lengths (number of unique index values per entity)
    entity_lengths = _get_entity_lengths(observations, config)

    if not entity_lengths:
        # Fallback: use total observation count
        entities = getattr(config, 'entities', ['default'])
        min_length = len(observations) // max(len(entities), 1)
    else:
        min_length = min(entity_lengths.values())

    # Calculate window size
    size = int(min_length * defaults["window_fraction"])

    # Apply constraints
    size = max(size, defaults["min_window"])
    size = min(size, defaults["max_window"])
    size = max(size, COMPUTE_LIMITS["min_window_size"])

    # Ensure window doesn't exceed shortest entity
    if min_length > 0:
        size = min(size, min_length - 1)

    # Calculate stride
    stride = max(1, int(size * defaults["stride_fraction"]))

    # Build config
    window_cfg = WindowConfig(
        size=size,
        stride=stride,
        auto_detected=True,
        detection_method=f"auto:{domain or 'general'}",
        domain_used=domain,
    )

    return window_cfg


def _get_entity_lengths(observations, config) -> Dict[str, int]:
    """Get number of sequence points per entity"""

    entity_lengths = {}
    entities = getattr(config, 'entities', ['default'])

    # Try polars
    try:
        import polars as pl
        if isinstance(observations, pl.DataFrame):
            for entity in entities:
                entity_data = observations.filter(pl.col("entity_id") == entity)
                # Count unique I values (not total rows, which includes all signals)
                # CANONICAL: Column is 'I' not 'index'
                n_indices = entity_data.select("I").unique().height
                entity_lengths[entity] = n_indices
            return entity_lengths
    except Exception:
        pass

    # Try pandas
    try:
        import pandas as pd
        if isinstance(observations, pd.DataFrame):
            for entity in entities:
                entity_data = observations[observations["entity_id"] == entity]
                n_indices = entity_data["I"].nunique()  # CANONICAL: Column is 'I'
                entity_lengths[entity] = n_indices
            return entity_lengths
    except Exception:
        pass

    return {}


# =============================================================================
# VALIDATION
# =============================================================================

def validate_window(
    window_cfg: WindowConfig,
    observations,
    config,
) -> List[str]:
    """
    Validate window configuration. Returns list of error messages.
    Empty list = valid.

    Checks:
    1. Window size is reasonable
    2. Stride is reasonable
    3. Each entity has enough windows
    4. Total compute is within limits
    5. Memory estimate is reasonable
    """

    errors = []

    size = window_cfg.size
    stride = window_cfg.stride
    entities = getattr(config, 'entities', ['default'])
    signals = getattr(config, 'signals', [])

    # Basic sanity
    if size < COMPUTE_LIMITS["min_window_size"]:
        errors.append(
            f"Window size {size} is too small. "
            f"Minimum is {COMPUTE_LIMITS['min_window_size']} for meaningful statistics."
        )

    if stride < 1:
        errors.append(f"Stride must be at least 1, got {stride}")

    if stride > size:
        errors.append(
            f"Stride ({stride}) is larger than window ({size}). "
            f"This creates gaps in analysis. Set stride <= window."
        )

    # Get entity lengths
    entity_lengths = _get_entity_lengths(observations, config)

    # Check each entity
    total_windows = 0
    windows_per_entity = {}

    for entity in entities:
        length = entity_lengths.get(entity, 0)

        if length == 0:
            errors.append(f"Entity '{entity}' has no data")
            continue

        if size >= length:
            errors.append(
                f"Window ({size}) >= entity '{entity}' length ({length}). "
                f"Reduce window size or check your data."
            )
            continue

        # Calculate number of windows
        n_windows = (length - size) // stride + 1
        windows_per_entity[entity] = n_windows
        total_windows += n_windows

        if n_windows < COMPUTE_LIMITS["min_windows_per_entity"]:
            errors.append(
                f"Entity '{entity}' only produces {n_windows} windows. "
                f"Minimum is {COMPUTE_LIMITS['min_windows_per_entity']}. "
                f"Reduce window/stride or add more data."
            )

        if n_windows > COMPUTE_LIMITS["max_windows_per_entity"]:
            errors.append(
                f"Entity '{entity}' produces {n_windows:,} windows. "
                f"Maximum is {COMPUTE_LIMITS['max_windows_per_entity']:,}. "
                f"Increase stride to reduce compute."
            )

    # Total compute check
    if total_windows > COMPUTE_LIMITS["max_total_windows"]:
        errors.append(
            f"Total windows ({total_windows:,}) exceeds limit ({COMPUTE_LIMITS['max_total_windows']:,}). "
            f"Increase stride or reduce number of entities."
        )

    # Memory estimate (rough: 8 bytes per float, window * signals * entities)
    n_signals = len(signals) if signals else 10
    bytes_per_window = size * n_signals * 8
    total_bytes = bytes_per_window * total_windows
    memory_gb = total_bytes / (1024**3)

    if memory_gb > COMPUTE_LIMITS["max_memory_gb"]:
        errors.append(
            f"Estimated memory ({memory_gb:.1f} GB) exceeds limit ({COMPUTE_LIMITS['max_memory_gb']} GB). "
            f"Increase stride or reduce window size."
        )

    # Update window config with computed values
    window_cfg.total_windows = total_windows
    window_cfg.windows_per_entity = windows_per_entity
    window_cfg.estimated_memory_mb = total_bytes / (1024**2)

    return errors


def get_recommendation(
    observations,
    config,
    user_size: Optional[int] = None,
    user_stride: Optional[int] = None,
) -> Tuple[WindowConfig, List[str]]:
    """
    Get window config with validation. Convenience function.

    Args:
        observations: Data
        config: PrismConfig
        user_size: User override (None = auto-detect)
        user_stride: User override (None = auto-detect)

    Returns:
        (WindowConfig, list of errors)
    """

    if user_size is not None and user_stride is not None:
        # User specified both
        window_cfg = WindowConfig(
            size=user_size,
            stride=user_stride,
            auto_detected=False,
            detection_method="user_specified",
        )
    elif user_size is not None:
        # User specified size, auto stride
        stride = max(1, int(user_size * 0.5))  # Default 50% overlap
        window_cfg = WindowConfig(
            size=user_size,
            stride=user_stride if user_stride else stride,
            auto_detected=False,
            detection_method="user_size_auto_stride",
        )
    else:
        # Full auto-detect
        window_cfg = auto_detect_window(observations, config)

    errors = validate_window(window_cfg, observations, config)

    return window_cfg, errors


# =============================================================================
# PRETTY ERROR MESSAGES (for ORTHON UI)
# =============================================================================

def format_errors_for_ui(errors: List[str]) -> str:
    """Format errors for ORTHON Streamlit display"""
    if not errors:
        return ""

    lines = ["## Window Configuration Errors\n"]
    for i, error in enumerate(errors, 1):
        lines.append(f"{i}. {error}\n")

    lines.append("\n### How to fix:")
    lines.append("- **Window too large**: Reduce window size or add more data")
    lines.append("- **Too many windows**: Increase stride (e.g., from 10 to 50)")
    lines.append("- **Not enough windows**: Decrease window size or stride")
    lines.append("- **Memory limit**: Increase stride significantly")

    return "\n".join(lines)


def format_config_summary(window_cfg: WindowConfig) -> str:
    """Format config for display"""
    lines = [
        f"**Window size**: {window_cfg.size} points",
        f"**Stride**: {window_cfg.stride} points",
        f"**Overlap**: {(1 - window_cfg.stride/window_cfg.size)*100:.0f}%",
        f"**Total windows**: {window_cfg.total_windows:,}",
        f"**Memory estimate**: {window_cfg.estimated_memory_mb:.1f} MB",
    ]

    if window_cfg.auto_detected:
        lines.append(f"*Auto-detected using {window_cfg.domain_used or 'general'} defaults*")
    else:
        lines.append("*User specified*")

    return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'WindowConfig',
    'auto_detect_window',
    'validate_window',
    'get_recommendation',
    'format_errors_for_ui',
    'format_config_summary',
    'DOMAIN_DEFAULTS',
    'GENERAL_DEFAULTS',
    'COMPUTE_LIMITS',
]
