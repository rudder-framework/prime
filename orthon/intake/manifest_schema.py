"""
ORTHON Manifest Schema
======================

Pydantic models for manifest.json - the complete job specification
that Orthon generates and PRISM executes.

Orthon = Brain (decides what to run)
PRISM = Muscle (just executes the manifest)
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from datetime import datetime
import json
import uuid

from ..shared.engine_registry import Granularity


# =============================================================================
# ENGINE MANIFEST ENTRY
# =============================================================================

class EngineManifestEntry(BaseModel):
    """
    Specification for a single engine execution.

    This tells PRISM exactly what to run and how.
    """

    name: str = Field(..., description="Engine name (matches ENGINE_SPECS)")

    output: str = Field(
        ...,
        description="Output parquet file: vector, dynamics, geometry, pairs, physics"
    )

    granularity: str = Field(
        ...,
        description="Output granularity: signal, observation, pair_directional, pair_symmetric, observation_cross_signal"
    )

    groupby: List[str] = Field(
        default_factory=list,
        description="Columns to group by before engine execution"
    )

    orderby: List[str] = Field(
        default_factory=lambda: ["I"],
        description="Columns to order by within each group"
    )

    input_columns: List[str] = Field(
        default_factory=lambda: ["I", "y"],
        description="Columns to pass to engine function"
    )

    output_columns: List[str] = Field(
        default_factory=list,
        description="Columns produced by engine"
    )

    function: str = Field(
        default="",
        description="PRISM function path (e.g., prism.engines.hurst.compute)"
    )

    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to engine function"
    )

    filter: Optional[str] = Field(
        default=None,
        description="Polars filter expression (e.g., 'col(\"unit\").is_in([\"g\", \"mm/s\"])')"
    )

    min_rows: int = Field(
        default=10,
        description="Minimum rows required to run this engine"
    )

    enabled: bool = Field(
        default=True,
        description="Whether this engine is enabled"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# MANIFEST METADATA
# =============================================================================

class ManifestMetadata(BaseModel):
    """
    Metadata about the input data, computed by DataAnalyzer.
    """

    entity_count: int = Field(0, description="Number of unique entities")
    signal_count: int = Field(0, description="Number of unique signals")
    observation_count: int = Field(0, description="Total observations")

    entities: List[str] = Field(default_factory=list, description="List of entity IDs")
    signals: List[str] = Field(default_factory=list, description="List of signal IDs")

    units_present: List[str] = Field(
        default_factory=list,
        description="Unique units found in data"
    )

    unit_categories: List[str] = Field(
        default_factory=list,
        description="Categories detected from units"
    )

    sampling_rate: Optional[float] = Field(
        None,
        description="Detected sampling rate (if uniform)"
    )

    I_min: Optional[float] = Field(None, description="Minimum I value")
    I_max: Optional[float] = Field(None, description="Maximum I value")
    I_range: Optional[float] = Field(None, description="Range of I values")

    y_min: Optional[float] = Field(None, description="Global minimum y value")
    y_max: Optional[float] = Field(None, description="Global maximum y value")


# =============================================================================
# WINDOW CONFIGURATION
# =============================================================================

class WindowManifest(BaseModel):
    """Window/stride configuration for PRISM analysis."""

    size: int = Field(100, description="Window size in sequence points")
    stride: int = Field(50, description="Stride between windows")
    min_samples: int = Field(50, description="Minimum samples required per window")


# =============================================================================
# PRISM MANIFEST (TOP-LEVEL)
# =============================================================================

class PrismManifest(BaseModel):
    """
    Complete PRISM job manifest.

    This is the single source of truth that PRISM executes.
    Orthon generates this; PRISM doesn't need to understand the data.
    """

    # =========================================================================
    # IDENTIFIERS
    # =========================================================================

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique job identifier"
    )

    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp when manifest was created"
    )

    orthon_version: str = Field(
        default="0.1.0",
        description="ORTHON version that created this manifest"
    )

    # =========================================================================
    # CALLBACK (optional)
    # =========================================================================

    callback_url: Optional[str] = Field(
        None,
        description="URL to POST results when job completes"
    )

    # =========================================================================
    # I/O PATHS
    # =========================================================================

    input_file: str = Field(
        ...,
        description="Path to observations.parquet"
    )

    output_dir: str = Field(
        ...,
        description="Directory for output parquets"
    )

    # =========================================================================
    # DATA METADATA
    # =========================================================================

    metadata: ManifestMetadata = Field(
        default_factory=ManifestMetadata,
        description="Metadata about the input data"
    )

    # =========================================================================
    # ENGINE MANIFEST
    # =========================================================================

    engines: List[EngineManifestEntry] = Field(
        default_factory=list,
        description="List of engines to execute"
    )

    # =========================================================================
    # ANALYSIS CONFIGURATION
    # =========================================================================

    window: WindowManifest = Field(
        default_factory=WindowManifest,
        description="Window/stride configuration"
    )

    constants: Dict[str, Any] = Field(
        default_factory=dict,
        description="Global constants for physics calculations"
    )

    # =========================================================================
    # METHODS
    # =========================================================================

    def to_json(self, path: Union[str, Path]) -> None:
        """Write manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "PrismManifest":
        """Load manifest from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.model_validate(data)

    def get_enabled_engines(self) -> List[EngineManifestEntry]:
        """Get list of enabled engines."""
        return [e for e in self.engines if e.enabled]

    def get_engines_by_output(self, output: str) -> List[EngineManifestEntry]:
        """Get engines that write to a specific output parquet."""
        return [e for e in self.engines if e.output == output and e.enabled]

    def engine_count(self) -> int:
        """Count of enabled engines."""
        return len(self.get_enabled_engines())

    def summary(self) -> str:
        """Human-readable summary of manifest."""
        lines = [
            "PrismManifest Summary",
            "=" * 50,
            f"Job ID: {self.job_id[:8]}...",
            f"Created: {self.created_at}",
            "",
            "Input/Output:",
            f"  Input: {self.input_file}",
            f"  Output: {self.output_dir}",
            "",
            "Data:",
            f"  Entities: {self.metadata.entity_count}",
            f"  Signals: {self.metadata.signal_count}",
            f"  Observations: {self.metadata.observation_count:,}",
            f"  Categories: {', '.join(self.metadata.unit_categories) or '(none)'}",
            "",
            f"Window: size={self.window.size}, stride={self.window.stride}",
            "",
            f"Engines: {self.engine_count()} enabled",
        ]

        # Group by output
        outputs = {}
        for engine in self.get_enabled_engines():
            outputs.setdefault(engine.output, []).append(engine.name)

        for output, names in outputs.items():
            lines.append(f"  {output}.parquet: {len(names)} engines")
            for name in names[:5]:
                lines.append(f"    - {name}")
            if len(names) > 5:
                lines.append(f"    ... and {len(names) - 5} more")

        if self.constants:
            lines.append("")
            lines.append(f"Constants: {len(self.constants)}")
            for k, v in list(self.constants.items())[:3]:
                lines.append(f"  {k}: {v}")
            if len(self.constants) > 3:
                lines.append(f"  ... and {len(self.constants) - 3} more")

        return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'EngineManifestEntry',
    'ManifestMetadata',
    'WindowManifest',
    'PrismManifest',
]
